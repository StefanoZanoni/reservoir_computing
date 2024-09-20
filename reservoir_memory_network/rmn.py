import torch
import numpy as np

from tqdm import tqdm

from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler

from utils.initialization import (sparse_tensor_init, sparse_recurrent_tensor_init, spectral_norm_scaling,
                                  sparse_eye_init, fast_spectral_rescaling, circular_tensor_init, legendre_tensor_init)


class RMNCell(torch.nn.Module):
    def __init__(self,
                 input_units: int,
                 non_linear_units: int,
                 memory_units: int,
                 *,
                 input_memory_scaling: float = 1.0,
                 input_non_linear_scaling: float = 1.0,
                 memory_non_linear_scaling: float = 1.0,
                 spectral_radius: float = 0.99,
                 leaky_rate: float = 1.0,
                 input_memory_connectivity: int = 1,
                 input_non_linear_connectivity: int = 1,
                 non_linear_connectivity: int = 1,
                 memory_non_linear_connectivity: int = 1,
                 bias: bool = True,
                 distribution: str = 'uniform',
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 bias_scaling: float | None,
                 circular_non_linear_kernel: bool = True,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3,
                 recurrent_scaling: float = 1e-2,
                 legendre: bool = False,
                 theta: float = 1.0,
                 just_memory: bool = False) -> None:

        super().__init__()

        self.just_memory = just_memory

        if input_units < 1:
            raise ValueError("Input units must be greater than 0.")
        self.input_units = input_units

        if not just_memory:
            if non_linear_units < 1:
                raise ValueError("Recurrent units must be greater than 0.")
            self.non_linear_units = non_linear_units
            self.input_non_linear_scaling = input_non_linear_scaling
            if input_non_linear_connectivity > non_linear_units or input_non_linear_connectivity < 1:
                raise ValueError("Input to non linear connectivity must be in [1, non_linear_units].")
            self.input_non_linear_connectivity = input_non_linear_connectivity
            if non_linear_connectivity > non_linear_units or non_linear_connectivity < 1:
                raise ValueError("Non linear connectivity must be in [1, non_linear_units].")

        if memory_units < 1:
            raise ValueError("Recurrent units must be greater than 0.")
        self.memory_units = memory_units
        self.input_memory_scaling = input_memory_scaling
        if input_memory_connectivity > memory_units or input_memory_connectivity < 1:
            raise ValueError("Input to memory connectivity must be in [1, memory_units].")
        self.input_memory_connectivity = input_memory_connectivity

        if not just_memory:
            if spectral_radius > 1 or spectral_radius < 0:
                raise ValueError("Spectral radius must be in [0, 1].")
            self.spectral_radius = spectral_radius
            if leaky_rate > 1 or leaky_rate <= 0:
                raise ValueError("Leaky rate must be in (0, 1].")
            self.leaky_rate = leaky_rate
            self.one_minus_leaky_rate = 1 - leaky_rate

        # Input to memory reservoir connectivity
        self.input_memory_kernel = sparse_tensor_init(input_units, memory_units,
                                                      C=input_memory_connectivity) * input_memory_scaling
        self.input_memory_kernel = torch.nn.Parameter(self.input_memory_kernel, requires_grad=False)

        if not just_memory:
            # Input to non-linear reservoir connectivity
            self.input_non_linear_kernel = sparse_tensor_init(input_units, non_linear_units,
                                                              C=input_non_linear_connectivity) * input_non_linear_scaling
            self.input_non_linear_kernel = torch.nn.Parameter(self.input_non_linear_kernel, requires_grad=False)

        if not just_memory:
            if euler:
                W = skewsymmetric(non_linear_units, recurrent_scaling)
                self.non_linear_kernel = W - gamma * torch.eye(non_linear_units)
                self.epsilon = epsilon
            else:
                # Non-linear reservoir connectivity
                if circular_non_linear_kernel:
                    W = circular_tensor_init(non_linear_units, distribution=distribution)
                else:
                    W = sparse_recurrent_tensor_init(non_linear_units, C=non_linear_units, distribution=distribution)

                # re-scale the weight matrix to control the effective spectral radius of the linearized system
                if effective_rescaling and leaky_rate != 1:
                    I = sparse_eye_init(non_linear_units)
                    W = W * leaky_rate + (I * (1 - leaky_rate))
                    W = spectral_norm_scaling(W, spectral_radius)
                    self.non_linear_kernel = (W + I * (leaky_rate - 1)) * (1 / leaky_rate)
                else:
                    if distribution == 'normal' and not non_linear_units == 1:
                        W = spectral_radius * W  # NB: W was already rescaled to 1 (circular_non_linear law)
                    elif (distribution == 'uniform' and non_linear_connectivity == non_linear_units
                          and not circular_non_linear_kernel and not non_linear_units == 1):  # fully connected uniform
                        W = fast_spectral_rescaling(W, spectral_radius)
                    else:  # sparse connections uniform
                        W = spectral_norm_scaling(W, spectral_radius)
                    self.non_linear_kernel = W

            self.non_linear_kernel = torch.nn.Parameter(self.non_linear_kernel, requires_grad=False)

        # Memory reservoir connectivity
        if legendre:
            M = legendre_tensor_init(memory_units, theta)
            self.memory_kernel = torch.matrix_exp(M)
        else:
            self.memory_kernel = circular_tensor_init(memory_units, distribution='fixed')
        self.memory_kernel = torch.nn.Parameter(self.memory_kernel, requires_grad=False)

        if not just_memory:
            # Memory to non-linear reservoir connectivity
            self.memory_non_linear_kernel = (sparse_tensor_init(memory_units, non_linear_units,
                                                                C=memory_non_linear_connectivity)
                                             * memory_non_linear_scaling)
            self.memory_non_linear_kernel = torch.nn.Parameter(self.memory_non_linear_kernel, requires_grad=False)

        if not just_memory:
            if bias:
                if bias_scaling is None:
                    self.bias_scaling = input_non_linear_scaling
                else:
                    self.bias_scaling = bias_scaling
                # uniform init in [-1, +1] times bias_scaling
                self.bias = (2 * torch.rand(self.non_linear_units) - 1) * self.bias_scaling
                self.bias = torch.nn.Parameter(self.bias, requires_grad=False)
            else:
                # zero bias
                self.bias = torch.zeros(self.non_linear_units)
                self.bias = torch.nn.Parameter(self.bias, requires_grad=False)

        if not just_memory:
            self.non_linearity = non_linearity
            self._non_linearity_function: Callable = torch.tanh if non_linearity == 'tanh' else lambda x: x
        self._memory_state = None
        if not just_memory:
            self._non_linear_state = None
        if not just_memory:
            self._forward_function: Callable = self._forward_euler if euler else self._forward_leaky_integrator
        else:
            self._forward_function: Callable = self._forward_memory

    def _forward_memory(self, xt: torch.Tensor) -> torch.FloatTensor:

        # memory part
        input_memory_part = torch.matmul(xt, self.input_memory_kernel)
        torch.matmul(self._memory_state, self.memory_kernel, out=self._memory_state)
        self._memory_state.add_(input_memory_part)

        return None, self._memory_state

    def _forward_leaky_integrator(self, xt: torch.Tensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:

        # memory part
        input_memory_part = torch.matmul(xt, self.input_memory_kernel)  # Vx * x(t)
        torch.matmul(self._memory_state, self.memory_kernel, out=self._memory_state)  # Vm * m(t-1)
        self._memory_state.add_(input_memory_part)  # m(t) = Vx * x(t) + Vm * m(t-1)

        # non-linear part
        input_non_linear_part = torch.matmul(xt, self.input_non_linear_kernel)  # Wx * x(t)
        non_linear_part = torch.matmul(self._non_linear_state, self.non_linear_kernel)  # Wh * h(t-1)
        memory_non_linear_part = torch.matmul(self._memory_state, self.memory_non_linear_kernel)  # Wm * m(t)
        # Wx * x(t) + Wh * h(t-1) + Wm * m(t) + b
        combined_input = input_non_linear_part.add_(non_linear_part).add_(memory_non_linear_part).add_(self.bias)
        # h(t) = (1 - alpha) * h(t-1) + alpha * f(Wx * x(t) + Wh * h(t-1) + Wm * m(t) + b)
        self._non_linear_state.mul_(self.one_minus_leaky_rate).add_(self.leaky_rate *
                                                                    self._non_linearity_function(combined_input))

        return self._non_linear_state, self._memory_state

    def _forward_euler(self, xt: torch.Tensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:

        # memory part
        input_memory_part = torch.matmul(xt, self.input_memory_kernel)  # Vx * x(t)
        torch.matmul(self._memory_state, self.memory_kernel, out=self._memory_state)  # Vm * m(t-1)
        self._memory_state.add_(input_memory_part)  # m(t) = Vx * x(t) + Vm * m(t-1)

        # non-linear part
        input_non_linear_part = torch.matmul(xt, self.input_non_linear_kernel)  # Wx * x(t)
        non_linear_part = torch.matmul(self._non_linear_state, self.non_linear_kernel)  # Wh * h(t-1)
        memory_non_linear_part = torch.matmul(self._memory_state, self.memory_non_linear_kernel)  # Wm * m(t)
        # Wx * x(t) + Wh * h(t-1) + Wm * m(t) + b
        combined_input = input_non_linear_part.add_(non_linear_part).add_(memory_non_linear_part).add_(self.bias)

        # h(t) = h(t-1) + epsilon * f(Wh * h(t-1) + Wx * x(t) + Wm * m(t) + b)
        self._non_linear_state.add_(self.epsilon * self._non_linearity_function(combined_input))

        return self._non_linear_state, self._memory_state

    def forward(self, xt: torch.Tensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:

        if self._memory_state is None:
            self._memory_state = torch.zeros((xt.shape[0], self.memory_units), dtype=torch.float32, device=xt.device,
                                             requires_grad=False)
        if not self.just_memory:
            if self._non_linear_state is None:
                self._non_linear_state = torch.zeros((xt.shape[0], self.non_linear_units), dtype=torch.float32,
                                                     device=xt.device, requires_grad=False)

        return self._forward_function(xt)


class ReservoirMemoryNetwork(torch.nn.Module):
    def __init__(self,
                 task: str,
                 input_units: int,
                 non_linear_units: int,
                 memory_units: int,
                 *,
                 initial_transients: int = 0,
                 input_memory_scaling: float = 1.0,
                 input_non_linear_scaling: float = 1.0,
                 memory_non_linear_scaling: float = 1.0,
                 spectral_radius: float = 0.99,
                 leaky_rate: float = 1.0,
                 input_memory_connectivity: int = 1,
                 input_non_linear_connectivity: int = 1,
                 non_linear_connectivity: int = 1,
                 memory_non_linear_connectivity: int = 1,
                 bias: bool = True,
                 distribution: str = 'uniform',
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 bias_scaling: float | None,
                 circular_non_linear_kernel: bool = True,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3,
                 recurrent_scaling: float = 1e-2,
                 alpha: float = 1.0,
                 max_iter: int = 1000,
                 tolerance: float = 1e-4,
                 legendre: bool = False,
                 theta: float = 1.0,
                 just_memory: bool = False) -> None:

        super().__init__()
        self.just_memory = just_memory
        self.scaler = None
        self.task = task
        self.initial_transients = initial_transients
        self.net = RMNCell(input_units,
                           non_linear_units,
                           memory_units,
                           input_memory_scaling=input_memory_scaling,
                           input_non_linear_scaling=input_non_linear_scaling,
                           memory_non_linear_scaling=memory_non_linear_scaling,
                           spectral_radius=spectral_radius,
                           leaky_rate=leaky_rate,
                           input_memory_connectivity=input_memory_connectivity,
                           input_non_linear_connectivity=input_non_linear_connectivity,
                           non_linear_connectivity=non_linear_connectivity,
                           memory_non_linear_connectivity=memory_non_linear_connectivity,
                           bias=bias,
                           distribution=distribution,
                           non_linearity=non_linearity,
                           effective_rescaling=effective_rescaling,
                           bias_scaling=bias_scaling,
                           circular_non_linear_kernel=circular_non_linear_kernel,
                           euler=euler,
                           epsilon=epsilon,
                           gamma=gamma,
                           recurrent_scaling=recurrent_scaling,
                           legendre=legendre,
                           theta=theta,
                           just_memory=just_memory,
                           )
        if task == 'classification':
            self.readout = RidgeClassifier(alpha=alpha, max_iter=max_iter, tol=tolerance)
        elif task == 'regression':
            self.readout = Ridge(alpha=alpha, max_iter=max_iter, tol=tolerance)

    import torch

    def forward(self, x: torch.Tensor) \
            -> tuple[torch.FloatTensor, torch.FloatTensor]:
        if not self.just_memory:
            non_linear_states = torch.empty((x.shape[0], x.shape[1], self.net.non_linear_units), dtype=torch.float32,
                                            device=x.device, requires_grad=False)
        memory_states = torch.empty((x.shape[0], x.shape[1], self.net.memory_units), dtype=torch.float32,
                                    device=x.device, requires_grad=False)

        is_dim_2 = x.dim() == 2

        with torch.no_grad():
            for t in range(x.shape[1]):
                xt = x[:, t].unsqueeze(1) if is_dim_2 else x[:, t]
                if not self.just_memory:
                    non_linear_state, memory_state = self.net(xt)
                    non_linear_states[:, t, :].copy_(non_linear_state)
                    memory_states[:, t, :].copy_(memory_state)
                else:
                    _, memory_state = self.net(xt)
                    memory_states[:, t, :].copy_(memory_state)

        if not self.just_memory:
            non_linear_states = non_linear_states[:, self.initial_transients:, :]
        memory_states = memory_states[:, self.initial_transients:, :]

        if not self.just_memory:
            return non_linear_states, memory_states
        else:
            return None, memory_states

    def fit(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
            use_last_state: bool = True, disable_progress_bar: bool = False) -> None:
        states, ys = [], []
        for x, y in tqdm(data, desc='Fitting', disable=disable_progress_bar):
            x, y = x.to(device), y.to(device)
            state = self(x)[1 if self.just_memory else 0]
            if use_last_state:
                state = state[:, -1, :]
            states.append(state.cpu().numpy())
            ys.append(y.cpu().numpy())
        states = np.concatenate(states, axis=0)
        ys = np.concatenate(ys, axis=0)

        if not use_last_state:
            states = states.reshape(-1, states.shape[2])
            if len(ys.shape) == 1:
                ys = np.repeat(ys, states.shape[0] // ys.shape[0], axis=0)
            else:
                ys = ys.T

        if standardize:
            self.scaler = StandardScaler().fit(states)
            states = self.scaler.transform(states)

        self.readout.fit(states, ys)

    def score(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
              use_last_state: bool = True, disable_progress_bar: bool = False) \
            -> float:
        states, ys = [], []
        for x, y in tqdm(data, desc='Scoring', disable=disable_progress_bar):
            x, y = x.to(device), y.to(device)
            state = self(x)[1 if self.just_memory else 0]
            if use_last_state:
                state = state[:, -1, :]
            states.append(state.cpu().numpy())
            ys.append(y.cpu().numpy())
        states = np.concatenate(states, axis=0)
        ys = np.concatenate(ys, axis=0)

        if not use_last_state:
            states = states.reshape(-1, states.shape[2])
            if len(ys.shape) == 1:
                ys = np.repeat(ys, states.shape[0] // ys.shape[0], axis=0)
            else:
                ys = ys.T

        if standardize:
            if self.scaler is None:
                raise ValueError('Standardization is enabled but the model has not been fitted yet.')
            states = self.scaler.transform(states)

        return self.readout.score(states, ys)

    def predict(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
                use_last_state: bool = True, disable_progress_bar: bool = False) -> np.ndarray:
        states = []
        for x, _ in tqdm(data, desc='Predicting', disable=disable_progress_bar):
            x = x.to(device)
            state = self(x)[1 if self.just_memory else 0]
            if use_last_state:
                state = state[:, -1, :]
            states.append(state.cpu().numpy())
        states = np.concatenate(states, axis=0)

        if not use_last_state:
            states = states.reshape(-1, states.shape[2])

        if standardize:
            if self.scaler is None:
                raise ValueError('Standardization is enabled but the model has not been fitted yet.')
            states = self.scaler.transform(states)

        return self.readout.predict(states)

    def reset_state(self) -> None:
        self.net._non_linear_state = None
        self.net._memory_state = None
