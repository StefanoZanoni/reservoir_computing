import torch
import numpy as np

from tqdm import tqdm

from echo_state_network import EchoStateNetwork

from sklearn.linear_model import RidgeClassifier
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
                 spectral_radius: float = 0.99,
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
            self.recurrent_units = np.int(total_units / number_of_layers)
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
                recurrent_scaling=recurrent_scaling
            )
        ]

        # all the others:
        # last_h_size may be different for the first layer
        # because of the remainder if concatenate_non_linear=True
        last_h_size = self.recurrent_units + total_units % number_of_layers
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
                    recurrent_scaling=recurrent_scaling
                )
            )
            last_h_size = self.recurrent_units
        self.reservoir = torch.nn.ModuleList(reservoir_layers)
        if task == 'classification':
            self.readout = RidgeClassifier(alpha=alpha, max_iter=max_iter, tol=tolerance)

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

    def fit(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False) -> None:
        states, ys = [], []
        for x, y in tqdm(data, desc='Fitting'):
            x, y = x.to(device), y.to(device)
            if self.task == 'classification':
                state = self(x)[1][-1]
            states.append(state.cpu().numpy())
            ys.append(y.cpu().numpy())
        states = np.concatenate(states, axis=0)
        ys = np.concatenate(ys, axis=0)

        if standardize:
            self.scaler = StandardScaler().fit(states)
            states = self.scaler.transform(states)

        self.readout.fit(states, ys)

    def score(self, data: torch.utils.data.DataLoader, device: torch.device) -> float:
        states, ys = [], []
        for x, y in tqdm(data, desc='Scoring'):
            x, y = x.to(device), y.to(device)
            if self.task == 'classification':
                state = self(x)[1][-1]
            states.append(state.cpu().numpy())
            ys.append(y.cpu().numpy())
        states = np.concatenate(states, axis=0)
        ys = np.concatenate(ys, axis=0)

        if standardize:
            if self.scaler is None:
                raise ValueError('Standardization is enabled but the model has not been fitted yet.')
            states = self.scaler.transform(states)

        return self.readout.score(states, ys)

    def reset_state(self):
        for reservoir in self.reservoir:
            reservoir.reset_state()
