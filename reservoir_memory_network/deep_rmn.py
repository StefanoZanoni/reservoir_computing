import torch
import numpy as np
from scipy.spatial.distance import num_obs_y
from torch.xpu import device

from tqdm import tqdm

from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler

from .rmn import MemoryCell, NonLinearCell


class DeepReservoirMemoryNetwork(torch.nn.Module):
    """
    A deep reservoir memory network for time series prediction tasks.

    Attributes:
        _just_memory (bool): Flag indicating whether to use only memory states.
        _scaler (StandardScaler): Scaler for standardizing the states.
        task (str): Task type ('classification' or 'regression').
        number_of_layers (int): Number of reservoir layers.
        total_non_linear_units (int): Total number of non-linear units.
        _concatenate_non_linear (bool): Flag indicating whether to concatenate non-linear states.
        total_memory_units (int): Total number of memory units.
        _concatenate_memory (bool): Flag indicating whether to concatenate memory states.
        batch_first (bool): Flag indicating whether the batch dimension is first.
        reservoir (torch.nn.ModuleList): List of reservoir layers.
        readout (RidgeClassifier or Ridge): Readout layer for the network.
    """

    def __init__(self,
                 task: str,
                 input_units: int,
                 total_non_linear_units: int,
                 total_memory_units: int,
                 *,
                 number_of_non_linear_layers: int = 1,
                 number_of_memory_layers: int = 1,
                 initial_transients: int = 0,
                 memory_scaling: float = 1.0,
                 non_linear_scaling: float = 1.0,
                 input_memory_scaling: float = 1.0,
                 input_non_linear_scaling: float = 1.0,
                 memory_non_linear_scaling: float = 1.0,
                 inter_non_linear_scaling: float = 1.0,
                 inter_memory_scaling: float = 1.0,
                 spectral_radius: float = 0.9,
                 leaky_rate: float = 0.5,
                 input_memory_connectivity: int = 1,
                 input_non_linear_connectivity: int = 1,
                 non_linear_connectivity: int = 1,
                 memory_non_linear_connectivity: int = 1,
                 inter_non_linear_connectivity: int = 1,
                 inter_memory_connectivity: int = 1,
                 bias: bool = True,
                 bias_scaling: float = None,
                 distribution: str = 'uniform',
                 signs_from: str | None = None,
                 fixed_input_kernel: bool = False,
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 concatenate_non_linear: bool = False,
                 concatenate_memory: bool = False,
                 circular_non_linear_kernel: bool = False,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3,
                 alpha: float = 1.0,
                 max_iter: int = 1000,
                 tolerance: float = 1e-4,
                 legendre: bool = False,
                 theta: float = 1.0,
                 just_memory: bool = False,
                 ) -> None:
        """
        Initializes the DeepReservoirMemoryNetwork.

        Args:
            task (str): Task type ('classification' or 'regression').
            input_units (int): Number of input units.
            total_non_linear_units (int): Total number of non-linear units.
            total_memory_units (int): Total number of memory units.
            number_of_layers (int): Number of reservoir layers.
            initial_transients (int): Number of initial transient states to discard.
            memory_scaling (float): Scaling factor for memory weights.
            non_linear_scaling (float): Scaling factor for non-linear weights.
            input_memory_scaling (float): Scaling factor for input memory weights.
            input_non_linear_scaling (float): Scaling factor for input non-linear weights.
            memory_non_linear_scaling (float): Scaling factor for memory non-linear weights.
            inter_non_linear_scaling (float): Scaling factor for inter non-linear weights.
            inter_memory_scaling (float): Scaling factor for inter memory weights.
            spectral_radius (float): Spectral radius of the recurrent weight matrix.
            leaky_rate (float): Leaky integration rate.
            input_memory_connectivity (int): Number of connections in the input memory weight matrix.
            input_non_linear_connectivity (int): Number of connections in the input non-linear weight matrix.
            non_linear_connectivity (int): Number of connections in the non-linear weight matrix.
            memory_non_linear_connectivity (int): Number of connections in the memory non-linear weight matrix.
            inter_non_linear_connectivity (int): Number of connections in the inter non-linear weight matrix.
            inter_memory_connectivity (int): Number of connections in the inter memory weight matrix.
            bias (bool): Whether to use a bias term.
            bias_scaling (float, optional): Scaling factor for the bias term.
            distribution (str): Distribution type for weight initialization.
            signs_from (str, optional): Source for signs of weights.
            fixed_input_kernel (bool): Whether to use a fixed input kernel.
            non_linearity (str): Non-linearity function to use.
            effective_rescaling (bool): Whether to use effective rescaling.
            concatenate_non_linear (bool): Whether to concatenate non-linear states.
            concatenate_memory (bool): Whether to concatenate memory states.
            circular_non_linear_kernel (bool): Whether to use a circular non-linear kernel.
            euler (bool): Whether to use Euler integration.
            epsilon (float): Euler integration step size.
            gamma (float): Scaling factor for recurrent weights.
            alpha (float): Regularization strength for the readout layer.
            max_iter (int): Maximum number of iterations for the readout layer.
            tolerance (float): Tolerance for the readout layer.
            legendre (bool): Whether to use Legendre polynomials.
            theta (float): Scaling factor for Legendre polynomials.
            just_memory (bool): Whether to use only memory states.
        """
        super().__init__()
        self._total_non_linear_units = total_non_linear_units
        self._total_memory_units = total_memory_units
        self._initial_transients = initial_transients
        self._just_memory = just_memory
        self._scaler = None
        self._concatenate_non_linear = concatenate_non_linear
        self._concatenate_memory = concatenate_memory

        # In case in which all the reservoir layers are concatenated, each level
        # contains units/layers neurons. This is done to keep the number of
        # _state variables projected to the next layer fixed;
        # i.e., the number of trainable parameters does not depend on concatenate_non_linear
        if concatenate_non_linear:
            self._non_linear_units = max(1, int(total_non_linear_units / number_of_non_linear_layers))
            input_non_linear_connectivity = max(1, int(input_non_linear_connectivity / number_of_non_linear_layers))
            inter_non_linear_connectivity = max(1, int(inter_non_linear_connectivity / number_of_non_linear_layers))
            non_linear_connectivity = max(1, int(non_linear_connectivity / number_of_non_linear_layers))
            memory_non_linear_connectivity = max(1, int(memory_non_linear_connectivity / number_of_non_linear_layers))
        else:
            self._non_linear_units = total_non_linear_units
        if concatenate_memory:
            self._memory_units = max(1, int(total_memory_units / number_of_memory_layers))
            input_memory_connectivity = max(1, int(input_memory_connectivity / number_of_memory_layers))
            inter_memory_connectivity = max(1, int(inter_memory_connectivity / number_of_memory_layers))
        else:
            self._memory_units = total_memory_units

        memory_layers = [
            MemoryCell(input_units, self._memory_units + total_memory_units % number_of_memory_layers,
                       memory_scaling=memory_scaling,
                       input_memory_scaling=input_memory_scaling,
                       input_memory_connectivity=input_memory_connectivity,
                       distribution=distribution,
                       signs_from=signs_from,
                       fixed_input_kernel=fixed_input_kernel,
                       legendre=legendre,
                       theta=theta)
        ]
        if total_memory_units != self._memory_units:
            last_h_memory_size = self._memory_units + total_memory_units % number_of_memory_layers
        else:
            last_h_memory_size = self._memory_units
        for _ in range(number_of_memory_layers - 1):
            memory_layers.append(
                MemoryCell(last_h_memory_size, self._memory_units,
                           memory_scaling=memory_scaling,
                           input_memory_scaling=inter_memory_scaling,
                           input_memory_connectivity=inter_memory_connectivity,
                           distribution=distribution,
                           signs_from=signs_from,
                           fixed_input_kernel=fixed_input_kernel,
                           legendre=legendre,
                           theta=theta)
            )
        self.memory_layers = torch.nn.ModuleList(memory_layers)

        if not just_memory:
            non_linear_layers = [
                NonLinearCell(input_units,
                              self._non_linear_units + total_non_linear_units % number_of_non_linear_layers,
                              last_h_memory_size,
                              non_linear_scaling=non_linear_scaling,
                              input_non_linear_scaling=input_non_linear_scaling,
                              memory_non_linear_scaling=memory_non_linear_scaling,
                              non_linear_connectivity=non_linear_connectivity,
                              input_non_linear_connectivity=input_non_linear_connectivity,
                              memory_non_linear_connectivity=memory_non_linear_connectivity,
                              spectral_radius=spectral_radius,
                              leaky_rate=leaky_rate,
                              bias=bias,
                              bias_scaling=bias_scaling,
                              distribution=distribution,
                              signs_from=signs_from,
                              fixed_input_kernel=fixed_input_kernel,
                              non_linearity=non_linearity,
                              effective_rescaling=effective_rescaling,
                              circular_non_linear_kernel=circular_non_linear_kernel,
                              euler=euler,
                              epsilon=epsilon,
                              gamma=gamma)
            ]

            if total_non_linear_units != self._non_linear_units:
                last_h_non_linear_size = self._non_linear_units + total_non_linear_units % number_of_non_linear_layers
            else:
                last_h_non_linear_size = self._non_linear_units
            for _ in range(number_of_non_linear_layers - 1):
                non_linear_layers.append(
                    NonLinearCell(last_h_non_linear_size, self._non_linear_units, last_h_memory_size,
                                  non_linear_scaling=non_linear_scaling,
                                  input_non_linear_scaling=inter_non_linear_scaling,
                                  memory_non_linear_scaling=memory_non_linear_scaling,
                                  non_linear_connectivity=non_linear_connectivity,
                                  input_non_linear_connectivity=inter_non_linear_connectivity,
                                  memory_non_linear_connectivity=memory_non_linear_connectivity,
                                  spectral_radius=spectral_radius,
                                  leaky_rate=leaky_rate,
                                  bias=bias,
                                  bias_scaling=bias_scaling,
                                  distribution=distribution,
                                  signs_from=signs_from,
                                  fixed_input_kernel=fixed_input_kernel,
                                  non_linearity=non_linearity,
                                  effective_rescaling=effective_rescaling,
                                  circular_non_linear_kernel=circular_non_linear_kernel,
                                  euler=euler,
                                  epsilon=epsilon,
                                  gamma=gamma)
                )
            self.non_linear_layers = torch.nn.ModuleList(non_linear_layers)

        if task == 'classification':
            self.readout = RidgeClassifier(alpha=alpha, max_iter=max_iter, tol=tolerance)
        elif task == 'regression':
            self.readout = Ridge(alpha=alpha, max_iter=max_iter, tol=tolerance)
        self._trained = False

    def _reset_state(self, batch_size: int, device: torch.device) -> None:
        """
        Resets the internal state of the reservoir.

        Args:
            batch_size (int): Batch size.
            device (torch.device): Device to perform computations on.
        """

        for memory_layer in self.memory_layers:
            memory_layer.reset_state(batch_size, device)
        if not self._just_memory:
            for non_linear_layer in self.non_linear_layers:
                non_linear_layer.reset_state(batch_size, device)

    @torch.no_grad()
    def _forward_core(self, x: torch.Tensor) \
            -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        seq_len = x.shape[1]

        memory_states = []

        last_memory_state = x
        for idx, memory_layer in enumerate(self.memory_layers):
            layer_states = torch.empty((x.shape[0], seq_len, memory_layer.memory_kernel.shape[0]),
                                       device=x.device, requires_grad=False, dtype=torch.float32)
            is_dim2 = last_memory_state.dim() == 2
            for t in range(seq_len):
                xt = last_memory_state[:, t].unsqueeze(1) if is_dim2 else last_memory_state[:, t]
                layer_states[:, t, :].copy_(memory_layer(xt))
            memory_states.append(layer_states)
            last_memory_state = layer_states

        if not self._just_memory:
            non_linear_states = []
            last_non_linear_state = x
            for idx, non_linear_layer in enumerate(self.non_linear_layers):
                layer_states = torch.empty((x.shape[0], seq_len, non_linear_layer.non_linear_kernel.shape[0]),
                                           device=x.device, requires_grad=False, dtype=torch.float32)
                is_dim2 = last_non_linear_state.dim() == 2
                for t in range(seq_len):
                    xt = last_non_linear_state[:, t].unsqueeze(1) if is_dim2 else last_non_linear_state[:, t]
                    layer_states[:, t, :].copy_(non_linear_layer(xt, last_memory_state[:, t, :]))
                non_linear_states.append(layer_states)
                last_non_linear_state = layer_states

        if not self._just_memory:
            if self._concatenate_non_linear:
                non_linear_states = torch.cat(non_linear_states, dim=-1)
            else:
                non_linear_states = non_linear_states[-1]

        if self._concatenate_memory:
            memory_states = torch.cat(memory_states, dim=-1)
        else:
            memory_states = memory_states[-1]

        if not self._just_memory:
            return (non_linear_states[:, self._initial_transients:, :], non_linear_states[:, -1, :],
                    memory_states[:, self._initial_transients:, :], memory_states[:, -1, :])
        else:
            return None, None, memory_states[:, self._initial_transients:, :], memory_states[:, -1, :]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) \
            -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        if not self._trained:
            raise ValueError('The model has not been trained yet. Use the fit method to train the model.')

        return self._forward_core(x)

    @torch.no_grad()
    def _forward(self, x: torch.Tensor) \
            -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:

        return self._forward_core(x)

    @torch.no_grad()
    def fit(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
            use_last_state: bool = True, disable_progress_bar: bool = False) -> None:
        """
        Fits the readout layer of the deep reservoir memory network.

        Args:
            data (torch.utils.data.DataLoader): DataLoader for the training data.
            device (torch.device): Device to perform computations on.
            standardize (bool): Whether to standardize the states.
            use_last_state (bool): Whether to use the last state for fitting.
            disable_progress_bar (bool): Whether to disable the progress bar.
        """

        batch_size = data.batch_size
        self._reset_state(batch_size, device)

        num_batches = len(data)
        state_size = self._total_non_linear_units if not self._just_memory else self._total_memory_units

        states = np.empty((num_batches * batch_size, data.dataset.data.shape[0] - self._initial_transients,
                           state_size), dtype=np.float32) if not use_last_state \
            else np.empty((num_batches * batch_size, state_size), dtype=np.float32)
        ys = np.empty((num_batches * batch_size, data.dataset.target.shape[0]), dtype=np.float32)

        self._trained = True
        idx = 0
        try:
            for x, y in tqdm(data, desc='Fitting', disable=disable_progress_bar):
                x, y = x.to(device), y.to(device)
                states[idx:idx + batch_size] = self._forward(x)[3 if self._just_memory else 1].cpu().numpy()\
                    if use_last_state else self._forward(x)[2 if self._just_memory else 0].cpu().numpy()
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
        """
        Scores the deep reservoir memory network on the given data.

        Args:
            data (torch.utils.data.DataLoader): DataLoader for the test data.
            device (torch.device): Device to perform computations on.
            standardize (bool): Whether to standardize the states.
            use_last_state (bool): Whether to use the last state for scoring.
            disable_progress_bar (bool): Whether to disable the progress bar.

        Returns:
            float: Score of the network.
        """

        if not self._trained:
            raise ValueError('The model has not been trained yet. Use the fit method to train the model.')
        if standardize:
            if self._scaler is None:
                raise ValueError('Standardization is enabled but the model has not been fitted yet.')

        batch_size = data.batch_size
        self._reset_state(batch_size, device)

        num_batches = len(data)
        state_size = self._total_non_linear_units if not self._just_memory else self._total_memory_units

        states = np.empty((num_batches * batch_size, data.dataset.data.shape[0] - self._initial_transients,
                           state_size), dtype=np.float32) if not use_last_state \
            else np.empty((num_batches * batch_size, state_size), dtype=np.float32)
        ys = np.empty((num_batches * batch_size, data.dataset.target.shape[0]), dtype=np.float32)

        idx = 0
        for x, y in tqdm(data, desc='Scoring', disable=disable_progress_bar):
            x, y = x.to(device), y.to(device)
            states[idx:idx + batch_size] = self._forward(x)[3 if self._just_memory else 1].cpu().numpy()\
                if use_last_state else self._forward(x)[2 if self._just_memory else 0].cpu().numpy()
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
        """
        Predicts the output for the given data.

        Args:
            data (torch.utils.data.DataLoader): DataLoader for the input data.
            device (torch.device): Device to perform computations on.
            standardize (bool): Whether to standardize the states.
            use_last_state (bool): Whether to use the last state for prediction.
            disable_progress_bar (bool): Whether to disable the progress bar.

        Returns:
            np.ndarray: Predicted output.
        """

        if not self._trained:
            raise ValueError('The model has not been trained yet. Use the fit method to train the model.')
        if standardize:
            if self._scaler is None:
                raise ValueError('Standardization is enabled but the model has not been fitted yet.')

        batch_size = data.batch_size
        self._reset_state(batch_size, device)

        num_batches = len(data)
        state_size = self._total_non_linear_units if not self._just_memory else self._total_memory_units

        states = np.empty((num_batches * batch_size, data.dataset.data.shape[0] - self._initial_transients,
                           state_size), dtype=np.float32) if not use_last_state \
            else np.empty((num_batches * batch_size, state_size), dtype=np.float32)

        idx = 0
        for x, _ in tqdm(data, desc='Predicting', disable=disable_progress_bar):
            x = x.to(device)
            states[idx:idx + batch_size] = self._forward(x)[3 if self._just_memory else 1].cpu().numpy()\
                if use_last_state else self._forward(x)[2 if self._just_memory else 0].cpu().numpy()
            idx += batch_size

        if not use_last_state:
            states = np.concatenate(states, axis=0)

        if standardize:
            states = self._scaler.transform(states)

        return self.readout.predict(states)

    def reset_state(self) -> None:
        """
        Resets the internal state of the reservoir.
        """

        for reservoir in self.reservoir:
            reservoir.reset_state()
