import torch
import numpy as np

from tqdm import tqdm

from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler

from reservoir_memory_network import ReservoirMemoryNetwork


class DeepReservoirMemoryNetwork(torch.nn.Module):
    def __init__(self,
                 task: str,
                 input_units: int,
                 total_non_linear_units: int,
                 total_memory_units: int,
                 *,
                 number_of_layers: int = 1,
                 initial_transients: int = 0,
                 input_memory_scaling: float = 1.0,
                 input_non_linear_scaling: float = 1.0,
                 memory_non_linear_scaling: float = 1.0,
                 inter_non_linear_scaling: float = 1.0,
                 inter_memory_scaling: float = 1.0,
                 spectral_radius: float = 0.99,
                 leaky_rate: float = 1.0,
                 input_memory_connectivity: int = 1,
                 input_non_linear_connectivity: int = 1,
                 non_linear_connectivity: int = 1,
                 memory_non_linear_connectivity: int = 1,
                 inter_non_linear_connectivity: int = 1,
                 inter_memory_connectivity: int = 1,
                 bias: bool = True,
                 distribution: str = 'uniform',
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 bias_scaling: float | None,
                 concatenate_non_linear: bool = False,
                 concatenate_memory: bool = False,
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
                 just_memory: bool = False,
                 ) -> None:

        super().__init__()
        self.just_memory = just_memory
        self.scaler = None
        self.task = task
        self.number_of_layers = number_of_layers
        self.total_non_linear_units = total_non_linear_units
        self.concatenate_non_linear = concatenate_non_linear
        self.total_memory_units = total_memory_units
        self.concatenate_memory = concatenate_memory
        self.batch_first = True  # DeepReservoir only supports batch_first

        # In case in which all the reservoir layers are concatenated, each level
        # contains units/layers neurons. This is done to keep the number of
        # _state variables projected to the next layer fixed;
        # i.e., the number of trainable parameters does not depend on concatenate_non_linear
        if concatenate_non_linear:
            self.non_linear_units = max(1, int(total_non_linear_units / number_of_layers))
            input_non_linear_connectivity = max(1, int(input_non_linear_connectivity / number_of_layers))
            inter_non_linear_connectivity = max(1, int(inter_non_linear_connectivity / number_of_layers))
            non_linear_connectivity = max(1, int(non_linear_connectivity / number_of_layers))
            memory_non_linear_connectivity = max(1, int(memory_non_linear_connectivity / number_of_layers))
        else:
            self.non_linear_units = total_non_linear_units
        if concatenate_memory:
            self.memory_units = max(1, int(total_memory_units / number_of_layers))
            input_memory_connectivity = max(1, int(input_memory_connectivity / number_of_layers))
            inter_memory_connectivity = max(1, int(inter_memory_connectivity / number_of_layers))
        else:
            self.memory_units = total_memory_units

        # creates a list of reservoirs
        # the first:
        reservoir_layers = [
            ReservoirMemoryNetwork(
                task,
                input_units,
                self.non_linear_units,
                self.memory_units,
                initial_transients=initial_transients,
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
                alpha=alpha,
                max_iter=max_iter,
                tolerance=tolerance,
                legendre=legendre,
                theta=theta,
            )
        ]

        # all the others:
        # last_h_size may be different for the first layer
        # because of the remainder if concatenate_non_linear=True
        if total_non_linear_units != self.non_linear_units:
            last_h_size = self.non_linear_units + total_non_linear_units % number_of_layers
        else:
            last_h_size = self.non_linear_units
        for _ in range(number_of_layers - 1):
            reservoir_layers.append(
                ReservoirMemoryNetwork(
                    task,
                    last_h_size,
                    self.non_linear_units,
                    self.memory_units,
                    input_memory_scaling=inter_memory_scaling,
                    input_non_linear_scaling=inter_non_linear_scaling,
                    memory_non_linear_scaling=memory_non_linear_scaling,
                    spectral_radius=spectral_radius,
                    leaky_rate=leaky_rate,
                    input_memory_connectivity=inter_memory_connectivity,
                    input_non_linear_connectivity=inter_non_linear_connectivity,
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
                    alpha=alpha,
                    max_iter=max_iter,
                    tolerance=tolerance,
                    legendre=legendre,
                    theta=theta,
                )
            )
            last_h_size = self.non_linear_units
        self.reservoir = torch.nn.ModuleList(reservoir_layers)
        if task == 'classification':
            self.readout = RidgeClassifier(alpha=alpha, max_iter=max_iter, tol=tolerance)
        elif task == 'regression':
            self.readout = Ridge(alpha=alpha, max_iter=max_iter, tol=tolerance)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple:

        if not self.just_memory:
            non_linear_states = []
            non_linear_states_last = []
        memory_states = []
        memory_states_last = []

        layer_input = x.clone()

        for res_idx, reservoir_layer in enumerate(self.reservoir):
            if not self.just_memory:
                non_linear_state, memory_state = reservoir_layer(layer_input)
            else:
                _, memory_state = reservoir_layer(layer_input)
            if not self.just_memory:
                non_linear_states.append(non_linear_state)
                non_linear_states_last.append(non_linear_state[:, -1, :])
            memory_states.append(memory_state)
            memory_states_last.append(memory_state[:, -1, :])
            if not self.just_memory:
                layer_input = non_linear_state
            else:
                layer_input = memory_state

        if not self.just_memory:
            if self.concatenate_non_linear:
                non_linear_states = torch.cat(non_linear_states, dim=2)
            else:
                non_linear_states = non_linear_states[-1]
        if self.concatenate_memory:
            memory_states = torch.cat(memory_states, dim=2)
        else:
            memory_states = memory_states[-1]

        if not self.just_memory:
            return non_linear_states, non_linear_states_last, memory_states, memory_states_last
        else:
            return None, None, memory_states, memory_states_last

    @torch.no_grad()
    def fit(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
            use_last_state: bool = True, disable_progress_bar: bool = False) -> None:
        states, ys = [], []
        for x, y in tqdm(data, desc='Fitting', disable=disable_progress_bar):
            x, y = x.to(device), y.to(device)
            state = self(x)[3 if self.just_memory else 1][-1] if use_last_state\
                else self(x)[2 if self.just_memory else 0]
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
            state = self(x)[3 if self.just_memory else 1][-1] if use_last_state \
                else self(x)[2 if self.just_memory else 0]
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
            state = self(x)[3 if self.just_memory else 1][-1] if use_last_state \
                else self(x)[2 if self.just_memory else 0]
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
        for reservoir in self.reservoir:
            reservoir.reset_state()
