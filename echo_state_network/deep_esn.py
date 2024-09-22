import torch
import numpy as np
from numpy.ma.core import shape

from tqdm import tqdm

from echo_state_network import EchoStateNetwork

from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler


class DeepEchoStateNetwork(torch.nn.Module):
    def __init__(self,
                 task: str,
                 input_units: int,
                 total_units: int,
                 *,
                 number_of_layers: int = 1,
                 initial_transients: int = 0,
                 input_scaling: float = 1.0,
                 inter_scaling: float = 1.0,
                 spectral_radius: float = 0.9,
                 leaky_rate: float = 1.0,
                 input_connectivity: int = 1,
                 recurrent_connectivity: int = 1,
                 inter_connectivity: int = 1,
                 bias: bool = True,
                 distribution: str = 'uniform',
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 bias_scaling: float | None,
                 concatenate: bool = False,
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
        self.task = task
        self.number_of_layers = number_of_layers
        self.total_units = total_units
        self.concatenate = concatenate
        self.batch_first = True  # DeepReservoir only supports batch_first

        # In case in which all the reservoir layers are concatenated, each level
        # contains units/layers neurons. This is done to keep the number of
        # _state variables projected to the next layer fixed;
        # i.e., the number of trainable parameters does not depend on concatenate_non_linear
        if concatenate:
            self.recurrent_units = max(1, int(total_units / number_of_layers))
            input_connectivity = max(1, int(input_connectivity / number_of_layers))
            recurrent_connectivity = max(1, int(recurrent_connectivity / number_of_layers))
            inter_connectivity = max(1, int(inter_connectivity / number_of_layers))
        else:
            self.recurrent_units = total_units

        # creates a list of reservoirs
        # the first:
        reservoir_layers = [
            EchoStateNetwork(
                task,
                input_units,
                self.recurrent_units,
                initial_transients=initial_transients,
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
                recurrent_scaling=recurrent_scaling,
                max_iter=max_iter,
                alpha=alpha,
                tolerance=tolerance,
            )
        ]

        # all the others:
        # last_h_size may be different for the first layer
        # because of the remainder if concatenate_non_linear=True
        if total_units != self.recurrent_units:
            last_h_size = self.recurrent_units + total_units % number_of_layers
        else:
            last_h_size = self.recurrent_units
        for _ in range(number_of_layers - 1):
            reservoir_layers.append(
                EchoStateNetwork(
                    task,
                    last_h_size,
                    self.recurrent_units,
                    input_scaling=inter_scaling,
                    spectral_radius=spectral_radius,
                    leaky_rate=leaky_rate,
                    input_connectivity=inter_connectivity,
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
                    recurrent_scaling=recurrent_scaling,
                    max_iter=max_iter,
                    alpha=alpha,
                    tolerance=tolerance,
                )
            )
            last_h_size = self.recurrent_units
        self.reservoir = torch.nn.ModuleList(reservoir_layers)
        if task == 'classification':
            self.readout = RidgeClassifier(alpha=alpha, max_iter=max_iter, tol=tolerance)
        elif task == 'regression':
            self.readout = Ridge(alpha=alpha, max_iter=max_iter, tol=tolerance)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple:
        # list of all the states in all the layers
        states = []
        # List of the states in all the layers for the last time step.
        # states_last is a list because different layers may have different sizes.
        states_last = []

        layer_input = x.clone()

        for res_idx, reservoir_layer in enumerate(self.reservoir):
            state = reservoir_layer(layer_input)
            states.append(state)
            states_last.append(state[:, -1, :])
            layer_input = state

        if self.concatenate:
            states = torch.cat(states, dim=2)
        else:
            states = states[-1]

        return states, states_last

    @torch.no_grad()
    def fit(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
            use_last_state: bool = True, disable_progress_bar: bool = False) -> None:
        states, ys = [], []
        for x, y in tqdm(data, desc='Fitting', disable=disable_progress_bar):
            x, y = x.to(device), y.to(device)
            if use_last_state:
                state = self(x)[1][-1]
            else:
                state = self(x)[0]
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
            if use_last_state:
                state = self(x)[1][-1]
            else:
                state = self(x)[0]
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
            if use_last_state:
                state = self(x)[1][-1]
            else:
                state = self(x)[0]
            states.append(state.cpu().numpy())
        states = np.concatenate(states, axis=0)

        if not use_last_state:
            states = states.reshape(-1, states.shape[2])

        if standardize:
            if self.scaler is None:
                raise ValueError('Standardization is enabled but the model has not been fitted yet.')
            states = self.scaler.transform(states)

        return self.readout.predict(states)

    def reset_state(self):
        for reservoir in self.reservoir:
            reservoir.reset_state()
