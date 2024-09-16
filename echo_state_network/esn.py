from typing import Callable

import torch
import numpy as np

from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils.initialization import (sparse_tensor_init, sparse_recurrent_tensor_init, spectral_norm_scaling,
                                  sparse_eye_init, fast_spectral_rescaling, circular_tensor_init, skewsymmetric)


class ReservoirCell(torch.nn.Module):
    def __init__(self,
                 input_units: int,
                 recurrent_units: int,
                 *,
                 input_scaling: float = 1.0,
                 spectral_radius: float = 0.99,
                 leaky_rate: float = 1.0,
                 input_connectivity: int = 1,
                 recurrent_connectivity: int = 1,
                 bias: bool = True,
                 distribution: str = 'uniform',
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 bias_scaling: float | None,
                 circular_recurrent_kernel: bool = True,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3,
                 recurrent_scaling: float = 1e-2,
                 ) -> None:

        super().__init__()

        if input_units < 1:
            raise ValueError("Input units must be greater than 0.")
        self.input_units = input_units
        if recurrent_units < 1:
            raise ValueError("Recurrent units must be greater than 0.")
        self.recurrent_units = recurrent_units
        self.input_scaling = input_scaling
        if spectral_radius > 1 or spectral_radius < 0:
            raise ValueError("Spectral radius must be in [0, 1].")
        self.spectral_radius = spectral_radius
        if leaky_rate > 1 or leaky_rate <= 0:
            raise ValueError("Leaky rate must be in (0, 1].")
        self.leaky_rate = leaky_rate
        self.one_minus_leaky_rate = 1 - leaky_rate
        self.input_connectivity = input_connectivity
        if recurrent_connectivity > recurrent_units or recurrent_connectivity < 1:
            raise ValueError("Recurrent connectivity must be in [1, non_linear_units].")
        self.recurrent_connectivity = recurrent_connectivity

        self.input_kernel = sparse_tensor_init(input_units, recurrent_units, C=input_connectivity) * input_scaling
        self.input_kernel = torch.nn.Parameter(self.input_kernel, requires_grad=False)

        if euler:
            W = skewsymmetric(recurrent_units, recurrent_scaling)
            self.recurrent_kernel = W - gamma * torch.eye(recurrent_units)
            self.epsilon = epsilon
        else:
            if circular_recurrent_kernel:
                W = circular_tensor_init(recurrent_units, distribution=distribution)
            else:
                W = sparse_recurrent_tensor_init(recurrent_units, C=recurrent_connectivity, distribution=distribution)

            # re-scale the weight matrix to control the effective spectral radius of the linearized system
            if effective_rescaling and leaky_rate != 1:
                I = sparse_eye_init(recurrent_units)
                W = W * leaky_rate + (I * (1 - leaky_rate))
                W = spectral_norm_scaling(W, spectral_radius)
                self.recurrent_kernel = (W + I * (leaky_rate - 1)) * (1 / leaky_rate)
            else:
                if distribution == 'normal':
                    W = spectral_radius * W  # NB: W was already rescaled to 1 (circular law)
                elif distribution == 'uniform' and recurrent_connectivity == recurrent_units:  # fully connected uniform
                    W = fast_spectral_rescaling(W, spectral_radius)
                else:  # sparse connections uniform
                    W = spectral_norm_scaling(W, spectral_radius)
                self.recurrent_kernel = W

        self.recurrent_kernel = torch.nn.Parameter(self.recurrent_kernel, requires_grad=False)

        if bias:
            if bias_scaling is None:
                self.bias_scaling = input_scaling
            else:
                self.bias_scaling = bias_scaling
            # uniform init in [-1, +1] times bias_scaling
            self.bias = (2 * torch.rand(self.recurrent_units) - 1) * self.bias_scaling
            self.bias = torch.nn.Parameter(self.bias, requires_grad=False)
        else:
            # zero bias
            self.bias = torch.zeros(self.recurrent_units)
            self.bias = torch.nn.Parameter(self.bias, requires_grad=False)

        self.non_linearity = non_linearity
        self._non_linear_function: Callable = torch.tanh if non_linearity == 'tanh' else lambda x: x
        self._state = None
        self._forward_function: Callable = self._forward_euler if euler else self._forward_leaky_integrator

    def _forward_leaky_integrator(self, xt: torch.Tensor) -> torch.FloatTensor:
        input_part = torch.matmul(xt, self.input_kernel)
        state_part = torch.matmul(self._state, self.recurrent_kernel)
        output = self._non_linear_function(input_part.add_(state_part).add_(self.bias))

        self._state.mul_(self.one_minus_leaky_rate).add_(output.mul_(self.leaky_rate))

        return self._state

    def _forward_euler(self, xt: torch.Tensor) -> torch.FloatTensor:
        input_part = torch.matmul(xt, self.input_kernel)
        state_part = torch.matmul(self._state, self.recurrent_kernel)
        output = self._non_linear_function(input_part.add_(state_part).add_(self.bias))
        self._state.add_(output.mul_(self.epsilon))

        return self._state

    def forward(self, xt: torch.Tensor) -> torch.FloatTensor:
        if self._state is None:
            self._state = torch.zeros((xt.shape[0], self.recurrent_units), dtype=torch.float32, requires_grad=False,
                                      device=xt.device)

        return self._forward_function(xt)


class EchoStateNetwork(torch.nn.Module):
    def __init__(self,
                 task: str,
                 input_units: int,
                 recurrent_units: int,
                 *,
                 initial_transients: int = 0,
                 input_scaling: float = 1.0,
                 spectral_radius: float = 0.99,
                 leaky_rate: float = 1.0,
                 input_connectivity: int = 1,
                 recurrent_connectivity: int = 1,
                 bias: bool = True,
                 distribution: str = 'uniform',
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 bias_scaling: float | None,
                 circular_recurrent_kernel: bool = True,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3,
                 recurrent_scaling: float = 1e-2,
                 alpha: float = 1.0,
                 max_iter: int = 1000,
                 tolerance: float = 1e-4,
                 ) -> None:

        super().__init__()
        self.scaler = None
        self.initial_transients = initial_transients
        self.task = task
        self.net = ReservoirCell(input_units,
                                 recurrent_units,
                                 input_scaling=input_scaling,
                                 spectral_radius=spectral_radius,
                                 leaky_rate=leaky_rate,
                                 input_connectivity=input_connectivity,
                                 recurrent_connectivity=recurrent_connectivity,
                                 bias=bias,
                                 distribution=distribution,
                                 non_linearity=non_linearity,
                                 effective_rescaling=effective_rescaling,
                                 bias_scaling=bias_scaling,
                                 circular_recurrent_kernel=circular_recurrent_kernel,
                                 euler=euler,
                                 epsilon=epsilon,
                                 gamma=gamma,
                                 recurrent_scaling=recurrent_scaling)
        if task == 'classification':
            self.readout = RidgeClassifier(alpha=alpha, max_iter=max_iter, tol=tolerance)
        elif task == 'regression':
            self.readout = Ridge(alpha=alpha, max_iter=max_iter, tol=tolerance)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = torch.empty((x.shape[0], x.shape[1], self.net.recurrent_units), dtype=torch.float32,
                             requires_grad=False, device=x.device)

        is_dim_2 = x.dim() == 2

        with torch.no_grad():
            for t in range(x.shape[1]):
                xt = x[:, t].unsqueeze(1) if is_dim_2 else x[:, t]
                state = self.net(xt)
                states[:, t, :].copy_(state)

        states = states[:, self.initial_transients:, :]

        return states

    def fit(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
            use_last_state: bool = True, show_progress_bar: bool | None = True) -> None:
        states, ys = [], []
        for x, y in tqdm(data, desc='Fitting', disable=not show_progress_bar):
            x, y = x.to(device), y.to(device)
            state = self(x)
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
              use_last_state: bool = True, show_progress_bar: bool | None = True) -> float:
        states, ys = [], []
        for x, y in tqdm(data, desc='Scoring', disable=not show_progress_bar):
            x, y = x.to(device), y.to(device)
            state = self(x)
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
                use_last_state: bool = True, show_progress_bar: bool | None = True) -> np.ndarray:
        states = []
        for x, _ in tqdm(data, desc='Predicting', disable=not show_progress_bar):
            x = x.to(device)
            state = self(x)
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
        self.net._state = None
