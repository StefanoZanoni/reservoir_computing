import torch
import numpy as np

from tqdm import tqdm

from .esn import EchoStateNetwork

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
                 recurrent_scaling: float = 1.0,
                 inter_scaling: float = 1.0,
                 spectral_radius: float = 0.9,
                 leaky_rate: float = 0.5,
                 input_connectivity: int = 1,
                 recurrent_connectivity: int = 1,
                 inter_connectivity: int = 1,
                 bias: bool = True,
                 bias_scaling: float = None,
                 distribution: str = 'uniform',
                 signs_from: str | None = None,
                 fixed_input_kernel: bool = False,
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 concatenate: bool = False,
                 circular_recurrent_kernel: bool = False,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3,
                 alpha: float = 1.0,
                 max_iter: int = 1000,
                 tolerance: float = 1e-4,
                 ) -> None:

        super().__init__()
        self._initial_transients = initial_transients
        self._scaler = None
        self._concatenate = concatenate
        self._initial_transients = initial_transients
        self._total_units = total_units

        # In case in which all the reservoir layers are concatenated, each level
        # contains units/layers neurons. This is done to keep the number of
        # _state variables projected to the next layer fixed;
        # i.e., the number of trainable parameters does not depend on concatenate_non_linear
        if concatenate:
            self._recurrent_units = max(1, int(total_units / number_of_layers))
            input_connectivity = max(1, int(input_connectivity / number_of_layers))
            recurrent_connectivity = max(1, int(recurrent_connectivity / number_of_layers))
            inter_connectivity = max(1, int(inter_connectivity / number_of_layers))
        else:
            self._recurrent_units = total_units

        # creates a list of reservoirs
        # the first:
        reservoir_layers = [
            EchoStateNetwork(
                input_units,
                self._recurrent_units + total_units % number_of_layers,
                input_scaling=input_scaling,
                spectral_radius=spectral_radius,
                leaky_rate=leaky_rate,
                input_connectivity=input_connectivity,
                recurrent_connectivity=recurrent_connectivity,
                bias=bias,
                distribution=distribution,
                signs_from=signs_from,
                fixed_input_kernel=fixed_input_kernel,
                non_linearity=non_linearity,
                effective_rescaling=effective_rescaling,
                bias_scaling=bias_scaling,
                circular_recurrent_kernel=circular_recurrent_kernel,
                euler=euler,
                epsilon=epsilon,
                gamma=gamma,
                recurrent_scaling=recurrent_scaling,
            )
        ]

        # all the others:
        # last_h_size may be different for the first layer
        # because of the remainder if concatenate_non_linear=True
        if total_units != self._recurrent_units:
            last_h_size = self._recurrent_units + total_units % number_of_layers
        else:
            last_h_size = self._recurrent_units
        for _ in range(number_of_layers - 1):
            reservoir_layers.append(
                EchoStateNetwork(
                    last_h_size,
                    self._recurrent_units,
                    input_scaling=inter_scaling,
                    spectral_radius=spectral_radius,
                    leaky_rate=leaky_rate,
                    input_connectivity=inter_connectivity,
                    recurrent_connectivity=recurrent_connectivity,
                    bias=bias,
                    distribution=distribution,
                    signs_from=signs_from,
                    fixed_input_kernel=fixed_input_kernel,
                    non_linearity=non_linearity,
                    effective_rescaling=effective_rescaling,
                    bias_scaling=bias_scaling,
                    circular_recurrent_kernel=circular_recurrent_kernel,
                    euler=euler,
                    epsilon=epsilon,
                    gamma=gamma,
                    recurrent_scaling=recurrent_scaling,
                )
            )
            last_h_size = self._recurrent_units
        self.reservoir = torch.nn.ModuleList(reservoir_layers)
        if task == 'classification':
            self.readout = RidgeClassifier(alpha=alpha, max_iter=max_iter, tol=tolerance)
        elif task == 'regression':
            self.readout = Ridge(alpha=alpha, max_iter=max_iter, tol=tolerance)
        self._trained = False

    def _reset_state(self, batch_size: int, device: torch.device) -> None:
        for layer in self.reservoir:
            layer.net.reset_state(batch_size, device)

    @torch.no_grad()
    def _forward_core(self, x: torch.Tensor) -> tuple:
        layer_input = x
        states = []

        for idx, reservoir_layer in enumerate(self.reservoir):
            state = reservoir_layer(layer_input)
            states.append(state[:, self._initial_transients:, :])
            layer_input = state

        states = torch.cat(states, dim=2) if self._concatenate else states[-1]

        return states, states[:, -1, :]

    def forward(self, x: torch.Tensor) -> tuple:

        if not self._trained:
            raise RuntimeError('Model has not been trained yet.')

        return self._forward_core(x)

    def _forward(self, x: torch.Tensor) -> tuple:

        return self._forward_core(x)

    @torch.no_grad()
    def fit(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
            use_last_state: bool = True, disable_progress_bar: bool = False) -> None:

        batch_size = data.batch_size
        self._reset_state(batch_size, device)

        num_batches = len(data)
        state_size = self._total_units

        states = np.empty((num_batches * batch_size, data.dataset.data.shape[0] - self._initial_transients,
                           state_size), dtype=np.float32) if not use_last_state \
            else np.empty((num_batches * batch_size, state_size), dtype=np.float32)
        ys = np.empty((num_batches * batch_size, data.dataset.target.shape[0]), dtype=np.float32)

        self._trained = True
        idx = 0
        try:
            for x, y in tqdm(data, desc='Fitting', disable=disable_progress_bar):
                x, y = x.to(device), y.to(device)
                states[idx:idx + batch_size] = self._forward(x)[1].cpu().numpy() if use_last_state \
                    else self._forward(x)[0].cpu().numpy()
                ys[idx:idx + batch_size] = y.cpu().numpy()
                idx += batch_size

            if not use_last_state:
                states = np.concatenate(states, axis=0)
                ys = np.repeat(ys, states.shape[0] // ys.shape[0], axis=0) if len(ys.shape) == 1 else ys.T

            if standardize:
                self._scaler = StandardScaler().fit(states)
                states = self._scaler.transform(states)

            self.readout.fit(states, ys)
        except Exception as e:
            self._trained = False
            raise e

    @torch.no_grad()
    def score(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
              use_last_state: bool = True, disable_progress_bar: bool = False) -> float:

        if not self._trained:
            raise RuntimeError('Model has not been trained yet.')
        if standardize:
            if self._scaler is None:
                raise ValueError('Standardization is enabled but the model has not been fitted yet.')

        batch_size = data.batch_size
        self._reset_state(batch_size, device)

        num_batches = len(data)
        state_size = self._total_units

        states = np.empty((num_batches * batch_size, data.dataset.data.shape[0] - self._initial_transients,
                           state_size), dtype=np.float32) if not use_last_state \
            else np.empty((num_batches * batch_size, state_size), dtype=np.float32)
        ys = np.empty((num_batches * batch_size, data.dataset.target.shape[0]), dtype=np.float32)

        idx = 0
        for x, y in tqdm(data, desc='Fitting', disable=disable_progress_bar):
            x, y = x.to(device), y.to(device)
            states[idx:idx + batch_size] = self._forward(x)[1].cpu().numpy() if use_last_state \
                else self._forward(x)[0].cpu().numpy()
            ys[idx:idx + batch_size] = y.cpu().numpy()
            idx += batch_size

        if not use_last_state:
            states = np.concatenate(states, axis=0)
            ys = np.repeat(ys, states.shape[0] // ys.shape[0], axis=0) if len(ys.shape) == 1 else ys.T

        if standardize:
            states = self._scaler.transform(states)

        return self.readout.score(states, ys)

    @torch.no_grad()
    def predict(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
                use_last_state: bool = True, disable_progress_bar: bool = False) -> np.ndarray:

        if not self._trained:
            raise RuntimeError('Model has not been trained yet.')
        if standardize:
            if self._scaler is None:
                raise ValueError('Standardization is enabled but the model has not been fitted yet.')

        batch_size = data.batch_size
        self._reset_state(batch_size, device)

        num_batches = len(data)
        state_size = self._total_units

        states = np.empty((num_batches * batch_size, data.dataset.data.shape[0] - self._initial_transients,
                           state_size), dtype=np.float32) if not use_last_state \
            else np.empty((num_batches * batch_size, state_size), dtype=np.float32)

        idx = 0
        for x, _ in tqdm(data, desc='Fitting', disable=disable_progress_bar):
            x = x.to(device)
            states[idx:idx + batch_size] = self._forward(x)[1].cpu().numpy() if use_last_state \
                else self._forward(x)[0].cpu().numpy()
            idx += batch_size

        if not use_last_state:
            states = np.concatenate(states, axis=0)

        if standardize:
            states = self._scaler.transform(states)

        return self.readout.predict(states)
