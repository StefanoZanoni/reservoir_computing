import torch
import numpy as np

from tqdm import tqdm

from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler

from .rmn import MemoryCell, NonLinearCell


class DeepReservoirMemoryNetwork(torch.nn.Module):
    """
    Deep Reservoir Memory Network.
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
        Initializes the Deep Reservoir Memory Network.

        :param task: Task to perform. Either 'classification' or 'regression'.
        :param input_units: Number of input units.
        :param total_non_linear_units: Total number of non-linear units.
        :param total_memory_units: Total number of memory units.
        :param number_of_non_linear_layers: Number of non-linear layers.
        :param number_of_memory_layers: Number of memory layers.
        :param initial_transients: Number of initial transients.
        :param memory_scaling: Scaling factor for the memory kernel.
        :param non_linear_scaling: Non-linear scaling factor.
        :param input_memory_scaling: Input-memory scaling factor.
        :param input_non_linear_scaling: Input-non-linear scaling factor.
        :param memory_non_linear_scaling: Memory-non-linear scaling factor.
        :param inter_non_linear_scaling: Inter-non-linear scaling factor.
        :param inter_memory_scaling: Inter-memory scaling factor.
        :param spectral_radius: Desired spectral radius.
        :param leaky_rate: Leaky integration rate.
        :param input_memory_connectivity: Input-memory connectivity.
        :param input_non_linear_connectivity: Input-non-linear connectivity.
        :param non_linear_connectivity: Non-linear connectivity.
        :param memory_non_linear_connectivity: Memory-non-linear connectivity.
        :param inter_non_linear_connectivity: Inter-non-linear connectivity.
        :param inter_memory_connectivity: Inter-memory connectivity.
        :param bias: Whether to use bias.
        :param bias_scaling: Bias scaling factor.
        :param distribution: Distribution of the weights.
        :param signs_from: Source of the signs of the weights.
        :param fixed_input_kernel: Whether to use a fixed input kernel.
        :param non_linearity: Non-linearity function.
        :param effective_rescaling: Whether to rescale the recurrent weights according to the leaky rate.
        :param concatenate_non_linear: Whether to concatenate the non-linear layers.
        :param concatenate_memory: Whether to concatenate the memory layers.
        :param circular_non_linear_kernel: Whether to use a circular non-linear kernel.
        :param euler: Whether to use the Euler method.
        :param epsilon: Euler integration step size.
        :param gamma: Diffusion coefficient for the Euler recurrent kernel.
        :param alpha: Regularization strength.
        :param max_iter: Maximum number of iterations for the readout layer.
        :param tolerance: Tolerance for the readout layer.
        :param legendre: Whether to use Legendre memory kernel.
        :param theta: Legendre memory kernel parameter.
        :param just_memory: Whether to use only the memory layers.
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
            if alpha > 0:
                self.readout = Ridge(alpha=alpha, max_iter=max_iter, tol=tolerance)
            else:
                self.readout = LinearRegression()
        self._trained = False

    def _reset_state(self, batch_size: int, device: torch.device) -> None:
        """
        Resets the internal state of the reservoir.

        :param batch_size: The batch size.
        :param device: The device to perform computations on.
        """

        for memory_layer in self.memory_layers:
            memory_layer.reset_state(batch_size, device)
        if not self._just_memory:
            for non_linear_layer in self.non_linear_layers:
                non_linear_layer.reset_state(batch_size, device)

    @torch.no_grad()
    def _forward_core(self, x: torch.Tensor) \
            -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Core forward method for the Deep Reservoir Memory Network.

        :param x: Input tensor.

        :return: The non-linear states, the non-linear state at the last time step,
        the memory states, and the memory state at the last time step.
        """

        seq_len = x.shape[1]

        memory_states = []

        # iterate over the memory layers and compute the states
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

        # iterate over the non-linear layers and compute the states
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

    def forward(self, x: torch.Tensor) \
            -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Forward method for the Deep Reservoir Memory Network.
        This method is not meant to be used without the fit method.

        :param x: The input tensor.

        :return: The non-linear states, the non-linear state at the last time step,
        the memory states, and the memory state at the last time step.
        """

        if not self._trained:
            raise ValueError('The model has not been trained yet. Use the fit method to train the model.')

        return self._forward_core(x)

    def _forward(self, x: torch.Tensor) \
            -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Forward method for the Deep Reservoir Memory Network.

        :param x: The input tensor.

        :return: The non-linear states, the non-linear state at the last time step,
        the memory states, and the memory state at the last time step.
        """

        return self._forward_core(x)

    @torch.no_grad()
    def fit(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
            use_last_state: bool = True, disable_progress_bar: bool = False) -> None:
        """
        Fits the deep reservoir memory network on the given data.

        :param data: The DataLoader for the training data.
        :param device: The device to perform computations on.
        :param standardize: Whether to standardize the states before fitting the readout layer.
        :param use_last_state: Whether to use the state at the last time step for fitting the readout layer.
        :param disable_progress_bar: Whether to disable the progress bar.
        """

        batch_size = data.batch_size
        self._reset_state(batch_size, device)

        num_batches = len(data)
        state_size = self._total_non_linear_units if not self._just_memory else self._total_memory_units

        # pre-allocate the states and the target values
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

        :param data: The DataLoader for the input data.
        :param device: The device to perform computations on.
        :param standardize: Whether to standardize the states before scoring.
        :param use_last_state: Whether to use the state at the last time step for scoring.
        :param disable_progress_bar: Whether to disable the progress bar.

        :return: The score of the deep reservoir memory network.
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
        Predicts the target values of the deep reservoir memory network on the given data.

        :param data: The DataLoader for the input data.
        :param device: The device to perform computations on.
        :param standardize: Whether to standardize the states before predicting.
        :param use_last_state: Whether to use the state at the last time step for predicting.
        :param disable_progress_bar: Whether to disable the progress bar.

        :return: The predicted target values.
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
