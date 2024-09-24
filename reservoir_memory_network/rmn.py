import torch
import numpy as np

from tqdm import tqdm

from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler

from utils.initialization import init_memory_kernel, init_input_kernel, init_non_linear_kernel, init_bias


def validate_params(just_memory, input_units, memory_units, non_linear_units, spectral_radius, leaky_rate,
                    memory_non_linear_connectivity, input_non_linear_connectivity, non_linear_connectivity,
                    input_memory_connectivity, distribution, non_linearity):
    if input_units < 1:
        raise ValueError("Input units must be greater than 0.")
    if memory_units < 1:
        raise ValueError("Memory units must be greater than 0.")
    if not (1 <= input_memory_connectivity <= memory_units):
        raise ValueError("Input to memory connectivity must be in [1, memory_units].")
    if distribution not in ['uniform', 'normal', 'fixed']:
        raise ValueError("Distribution must be 'uniform', 'normal', or 'fixed'.")
    if non_linearity not in ['tanh', 'identity']:
        raise ValueError("Non-linearity must be 'tanh' or 'identity'.")

    if not just_memory:
        if non_linear_units < 1:
            raise ValueError("Non linear units must be greater than 0.")
        if not (0 <= spectral_radius <= 1):
            raise ValueError("Spectral radius must be in [0, 1].")
        if not (0 < leaky_rate <= 1):
            raise ValueError("Leaky rate must be in (0, 1].")
        if not (1 <= input_non_linear_connectivity <= non_linear_units):
            raise ValueError("Input to non linear connectivity must be in [1, non_linear_units].")
        if not (1 <= non_linear_connectivity <= non_linear_units):
            raise ValueError("Non linear connectivity must be in [1, non_linear_units].")
        if not (1 <= memory_non_linear_connectivity <= non_linear_units):
            raise ValueError("Memory to non linear connectivity must be in [1, non_linear_units].")


class RMNCell(torch.nn.Module):
    def __init__(self,
                 input_units: int,
                 non_linear_units: int,
                 memory_units: int,
                 *,
                 memory_scaling: float = 1.0,
                 non_linear_scaling: float = 1.0,
                 input_memory_scaling: float = 1.0,
                 input_non_linear_scaling: float = 1.0,
                 memory_non_linear_scaling: float = 1.0,
                 spectral_radius: float = 0.9,
                 leaky_rate: float = 0.5,
                 input_memory_connectivity: int = 1,
                 input_non_linear_connectivity: int = 1,
                 non_linear_connectivity: int = 1,
                 memory_non_linear_connectivity: int = 1,
                 bias: bool = True,
                 bias_scaling: float = None,
                 distribution: str = 'uniform',
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 circular_non_linear_kernel: bool = False,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3,
                 legendre: bool = False,
                 theta: float = 1.0,
                 just_memory: bool = False) -> None:

        super().__init__()

        validate_params(just_memory, input_units, memory_units, non_linear_units, spectral_radius, leaky_rate,
                        memory_non_linear_connectivity, input_non_linear_connectivity, non_linear_connectivity,
                        input_memory_connectivity, distribution, non_linearity)

        self.just_memory = just_memory
        self.input_units = input_units
        self.memory_units = memory_units
        self.memory_scaling = memory_scaling
        self.input_memory_scaling = input_memory_scaling

        # Input to memory reservoir kernel
        self.input_memory_kernel = init_input_kernel(input_units, memory_units, input_memory_connectivity,
                                                     input_memory_scaling, distribution)
        # Memory reservoir kernel
        self.memory_kernel = init_memory_kernel(memory_units, theta, legendre, memory_scaling)
        self._memory_state = None
        self._forward_function: Callable = self._forward_memory

        if not just_memory:
            self.non_linear_units = non_linear_units
            self.input_non_linear_scaling = input_non_linear_scaling
            self.input_non_linear_connectivity = input_non_linear_connectivity
            self.spectral_radius = spectral_radius
            self.leaky_rate = leaky_rate
            self.one_minus_leaky_rate = 1 - leaky_rate
            # Input to non-linear reservoir kernel
            self.input_non_linear_kernel = init_input_kernel(input_units, non_linear_units,
                                                             input_non_linear_connectivity, input_non_linear_scaling,
                                                             distribution)
            # Non-linear reservoir kernel
            self.non_linear_kernel = init_non_linear_kernel(non_linear_units, non_linear_connectivity, distribution,
                                                            spectral_radius, leaky_rate, effective_rescaling,
                                                            circular_non_linear_kernel, euler, gamma,
                                                            non_linear_scaling)
            self.epsilon = epsilon
            # Memory to non-linear reservoir connectivity
            self.memory_non_linear_kernel = init_input_kernel(memory_units, non_linear_units,
                                                              memory_non_linear_connectivity, memory_non_linear_scaling,
                                                              distribution)
            self.bias = init_bias(bias, non_linear_units, input_non_linear_scaling, bias_scaling)

            self.non_linearity = non_linearity
            self._non_linearity_function: Callable = torch.tanh if non_linearity == 'tanh' else lambda x: x

            self._non_linear_state = None
            self._forward_function: Callable = self._forward_euler if euler else self._forward_leaky_integrator

    @torch.no_grad()
    def _forward_memory(self, xt: torch.Tensor) -> torch.FloatTensor:

        # memory part
        input_memory_part = torch.matmul(xt, self.input_memory_kernel)
        self._memory_state = torch.addmm(input_memory_part, self._memory_state, self.memory_kernel)

        return None, self._memory_state

    @torch.no_grad()
    def _forward_leaky_integrator(self, xt: torch.Tensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:

        # memory part
        input_memory_part = torch.matmul(xt, self.input_memory_kernel)  # Vx * x(t)
        # m(t) = Vx * x(t) + Vm * m(t-1)
        self._memory_state = torch.addmm(input_memory_part, self._memory_state, self.memory_kernel)

        # non-linear part
        combined_input = torch.addmm(self.bias, self._non_linear_state, self.non_linear_kernel)  # Wh * h(t-1) + b
        combined_input.addmm_(xt, self.input_non_linear_kernel)  # Wx * x(t)
        combined_input.addmm_(self._memory_state, self.memory_non_linear_kernel)  # Wm * m(t)

        # h(t) = (1 - alpha) * h(t-1) + alpha * f(Wx * x(t) + Wh * h(t-1) + Wm * m(t) + b)
        (self._non_linear_state.mul_(self.one_minus_leaky_rate).add_
         (self.leaky_rate * self._non_linearity_function(combined_input)))

        return self._non_linear_state, self._memory_state

    @torch.no_grad()
    def _forward_euler(self, xt: torch.Tensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:

        # memory part
        input_memory_part = torch.matmul(xt, self.input_memory_kernel)  # Vx * x(t)
        # m(t) = Vx * x(t) + Vm * m(t-1)
        self._memory_state = torch.addmm(input_memory_part, self._memory_state, self.memory_kernel)

        # non-linear part
        combined_input = torch.addmm(self.bias, self._non_linear_state, self.non_linear_kernel)  # Wh * h(t-1) + b
        combined_input.addmm_(xt, self.input_non_linear_kernel)  # Wx * x(t)
        combined_input.addmm_(self._memory_state, self.memory_non_linear_kernel)  # Wm * m(t)

        # h(t) = h(t-1) + epsilon * f(Wh * h(t-1) + Wx * x(t) + Wm * m(t) + b)
        self._non_linear_state.add_(self.epsilon * self._non_linearity_function(combined_input))

        return self._non_linear_state, self._memory_state

    @torch.no_grad()
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
                 memory_scaling: float = 1.0,
                 non_linear_scaling: float = 1.0,
                 input_memory_scaling: float = 1.0,
                 input_non_linear_scaling: float = 1.0,
                 memory_non_linear_scaling: float = 1.0,
                 spectral_radius: float = 0.9,
                 leaky_rate: float = 0.5,
                 input_memory_connectivity: int = 1,
                 input_non_linear_connectivity: int = 1,
                 non_linear_connectivity: int = 1,
                 memory_non_linear_connectivity: int = 1,
                 bias: bool = True,
                 bias_scaling: float = None,
                 distribution: str = 'uniform',
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 circular_non_linear_kernel: bool = False,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3,
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
        self.net = RMNCell(input_units, non_linear_units, memory_units, memory_scaling=memory_scaling,
                           non_linear_scaling=non_linear_scaling, input_memory_scaling=input_memory_scaling,
                           input_non_linear_scaling=input_non_linear_scaling,
                           memory_non_linear_scaling=memory_non_linear_scaling, spectral_radius=spectral_radius,
                           leaky_rate=leaky_rate, input_memory_connectivity=input_memory_connectivity,
                           input_non_linear_connectivity=input_non_linear_connectivity,
                           non_linear_connectivity=non_linear_connectivity,
                           memory_non_linear_connectivity=memory_non_linear_connectivity, bias=bias,
                           bias_scaling=bias_scaling, distribution=distribution, non_linearity=non_linearity,
                           effective_rescaling=effective_rescaling,
                           circular_non_linear_kernel=circular_non_linear_kernel, euler=euler, epsilon=epsilon,
                           gamma=gamma, legendre=legendre, theta=theta, just_memory=just_memory)
        if task == 'classification':
            self.readout = RidgeClassifier(alpha=alpha, max_iter=max_iter, tol=tolerance)
        elif task == 'regression':
            self.readout = Ridge(alpha=alpha, max_iter=max_iter, tol=tolerance)

    import torch

    @torch.no_grad()
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

    @torch.no_grad()
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

    @torch.no_grad()
    def score(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
              use_last_state: bool = True, disable_progress_bar: bool = False) \
            -> float:

        if standardize:
            if self.scaler is None:
                raise ValueError('Standardization is enabled but the model has not been fitted yet.')

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
            states = self.scaler.transform(states)

        return self.readout.score(states, ys)

    @torch.no_grad()
    def predict(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
                use_last_state: bool = True, disable_progress_bar: bool = False) -> np.ndarray:

        if standardize:
            if self.scaler is None:
                raise ValueError('Standardization is enabled but the model has not been fitted yet.')

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
            states = self.scaler.transform(states)

        return self.readout.predict(states)

    def reset_state(self) -> None:
        self.net._non_linear_state = None
        self.net._memory_state = None
