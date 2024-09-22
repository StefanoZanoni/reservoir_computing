from typing import Callable

import torch
import numpy as np

from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils.initialization import init_input_kernel, init_non_linear_kernel, init_bias


def validate_params(input_units, recurrent_units, spectral_radius, leaky_rate, recurrent_connectivity,
                    distribution, non_linearity):
    if input_units < 1:
        raise ValueError("Input units must be greater than 0.")
    if recurrent_units < 1:
        raise ValueError("Recurrent units must be greater than 0.")
    if not (0 <= spectral_radius <= 1):
        raise ValueError("Spectral radius must be in [0, 1].")
    if not (0 < leaky_rate <= 1):
        raise ValueError("Leaky rate must be in (0, 1].")
    if not (1 <= recurrent_connectivity <= recurrent_units):
        raise ValueError("Recurrent connectivity must be in [1, recurrent_units].")
    if distribution not in ['uniform', 'normal', 'fixed']:
        raise ValueError("Distribution must be 'uniform', 'normal', or 'fixed'.")
    if non_linearity not in ['tanh', 'identity']:
        raise ValueError("Non-linearity must be 'tanh' or 'identity'.")


class ReservoirCell(torch.nn.Module):
    def __init__(self,
                 input_units: int,
                 recurrent_units: int,
                 *,
                 input_scaling: float = 1.0,
                 spectral_radius: float = 0.9,
                 leaky_rate: float = 1.0,
                 input_connectivity: int = 1,
                 recurrent_connectivity: int = 1,
                 bias: bool = True,
                 distribution: str = 'uniform',
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 bias_scaling: float | None = None,
                 circular_recurrent_kernel: bool = True,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3,
                 recurrent_scaling: float = 1e-2,
                 ) -> None:
        super().__init__()

        validate_params(input_units, recurrent_units, spectral_radius, leaky_rate, recurrent_connectivity,
                        distribution, non_linearity)

        self.input_units = input_units
        self.recurrent_units = recurrent_units
        self.input_scaling = input_scaling
        self.spectral_radius = spectral_radius
        self.leaky_rate = leaky_rate
        self.one_minus_leaky_rate = 1 - leaky_rate
        self.input_connectivity = input_connectivity
        self.recurrent_connectivity = recurrent_connectivity

        self.input_kernel = init_input_kernel(input_units, recurrent_units, input_connectivity, input_scaling)
        self.recurrent_kernel = init_non_linear_kernel(recurrent_units, recurrent_connectivity, distribution,
                                                       spectral_radius, leaky_rate, effective_rescaling,
                                                       circular_recurrent_kernel, euler, gamma, recurrent_scaling)
        self.bias = init_bias(bias, recurrent_units, input_scaling, bias_scaling)

        self.epsilon = epsilon
        self.non_linearity = non_linearity
        self._non_linear_function: Callable = torch.tanh if non_linearity == 'tanh' else lambda x: x
        self._state = None
        self._forward_function: Callable = self._forward_euler if euler else self._forward_leaky_integrator

    @torch.no_grad()
    def _forward_leaky_integrator(self, xt: torch.Tensor) -> torch.FloatTensor:
        self._state.mul_(self.one_minus_leaky_rate).add_(
            self._non_linear_function(
                torch.matmul(xt, self.input_kernel).add_(  # input part
                    torch.matmul(self._state, self.recurrent_kernel)  # state part
                ).add_(self.bias)
            ).mul_(self.leaky_rate)
        )
        return self._state

    @torch.no_grad()
    def _forward_euler(self, xt: torch.Tensor) -> torch.FloatTensor:
        self._state.add_(
            self._non_linear_function(
                torch.matmul(xt, self.input_kernel).add_(
                    torch.matmul(self._state, self.recurrent_kernel)
                ).add_(self.bias)
            ).mul_(self.epsilon)
        )
        return self._state

    @torch.no_grad()
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

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        states = torch.empty((x.shape[0], x.shape[1], self.net.recurrent_units), dtype=torch.float32,
                             requires_grad=False, device=x.device)

        is_dim_2 = x.dim() == 2

        for t in range(x.shape[1]):
            xt = x[:, t].unsqueeze(1) if is_dim_2 else x[:, t]
            state = self.net(xt)
            states[:, t, :].copy_(state)

        states = states[:, self.initial_transients:, :]

        return states

    @torch.no_grad()
    def fit(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
            use_last_state: bool = True, disable_progress_bar: bool = False) -> None:
        states, ys = [], []
        for x, y in tqdm(data, desc='Fitting', disable=disable_progress_bar):
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

    @torch.no_grad()
    def score(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
              use_last_state: bool = True, disable_progress_bar: bool = False) -> float:
        states, ys = [], []
        for x, y in tqdm(data, desc='Scoring', disable=disable_progress_bar):
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

    @torch.no_grad()
    def predict(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
                use_last_state: bool = True, disable_progress_bar: bool = False) -> np.ndarray:
        states = []
        for x, _ in tqdm(data, desc='Predicting', disable=disable_progress_bar):
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
