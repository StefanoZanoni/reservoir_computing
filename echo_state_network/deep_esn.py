import torch
import numpy as np

from tqdm import tqdm

from .esn import ReservoirCell

from sklearn.linear_model import RidgeClassifier, Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler


class DeepEchoStateNetwork(torch.nn.Module):
    """
    Deep Echo State Network model. It is composed of a stack of Echo State Networks (ESNs) and a readout layer.
    """

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
                 input_to_all: bool = False,
                 ) -> None:
        """
        Initializes the Deep Echo State Network model.

        :param task: The task to perform. It can be either 'classification' or 'regression'.
        :param input_units: Number of input units.
        :param total_units: Number of total units in the reservoir.
        :param number_of_layers: Number of layers in the deep ESN.
        :param initial_transients: Number of initial transients to discard.
        :param input_scaling: Input scaling factor.
        :param recurrent_scaling: Recurrent scaling factor.
        :param inter_scaling: Interlayer scaling factor.
        :param spectral_radius: Spectral radius of the recurrent weight matrix.
        :param leaky_rate: Leaky rate of the neurons.
        :param input_connectivity: Input connectivity.
        :param recurrent_connectivity: Recurrent connectivity.
        :param inter_connectivity: Interlayer connectivity.
        :param bias: Whether to use bias.
        :param bias_scaling: Bias scaling factor.
        :param distribution: Distribution of the weights.
        :param signs_from: Source of the signs of the weights.
        :param fixed_input_kernel: Whether to use a fixed input kernel.
        :param non_linearity: Non-linearity function.
        :param effective_rescaling: Whether to scale the weights considering the leaky rate.
        :param concatenate: Whether to concatenate the reservoir layers states.
        :param circular_recurrent_kernel: Whether to use a circular recurrent kernel.
        :param euler: Whether to use the Euler method.
        :param epsilon: Euler method integration step size.
        :param gamma: Diffusion coefficient for the Euler recurrent kernel.
        :param alpha: Regularization strength for the readout layer.
        :param max_iter: Maximum number of iterations for the readout layer.
        :param tolerance: Tolerance for the readout layer.
        """

        super().__init__()
        self._input_to_all = input_to_all
        self._initial_transients = initial_transients
        self._scaler = None
        self._concatenate = concatenate
        self._initial_transients = initial_transients
        self._total_units = total_units

        # In case in which all the reservoir layers are concatenated, each level
        # contains units/layers neurons. This is done to keep the number of
        # state variables projected to the next layer fixed;
        # i.e., the number of trainable parameters does not depend on concatenate_non_linear.
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
            ReservoirCell(
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
                ReservoirCell(
                    last_h_size + input_units if input_to_all else last_h_size,
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
            if alpha > 0:
                self.readout = Ridge(alpha=alpha, max_iter=max_iter, tol=tolerance)
            else:
                self.readout = LinearRegression()
        self._trained = False

    def _reset_state(self, batch_size: int, device: torch.device) -> None:
        """
        Resets the state of the reservoir layers.

        :param batch_size: The batch size.
        :param device: The device to use.
        """

        for layer in self.reservoir:
            layer.reset_state(batch_size, device)

    @torch.no_grad()
    def _forward_core(self, x: torch.Tensor) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Forward pass of the model.

        :param x: The input tensor.

        :return: The state of the deep reservoir for all the time steps and the state for the last time step.
        """

        non_linear_states = []
        last_non_linear_state = x
        seq_len = x.shape[1]
        x_is_dim2 = x.dim() == 2

        for idx, non_linear_layer in enumerate(self.reservoir):
            layer_states = torch.empty((x.shape[0], seq_len, non_linear_layer.recurrent_kernel.shape[0]),
                                       device=x.device, requires_grad=False, dtype=torch.float32)
            is_dim2 = last_non_linear_state.dim() == 2
            concatenate_state_input = self._input_to_all and idx > 0
            for t in range(seq_len):
                last_state_t = last_non_linear_state[:, t].unsqueeze(1) if is_dim2 else last_non_linear_state[:, t]
                if concatenate_state_input:
                    xt = x[:, t].unsqueeze(1) if x_is_dim2 else x[:, t]
                    last_state_t = torch.cat([last_state_t, xt], dim=-1)
                layer_states[:, t, :].copy_(non_linear_layer(last_state_t))
            non_linear_states.append(layer_states)
            last_non_linear_state = layer_states

        if self._concatenate:
            non_linear_states = torch.cat(non_linear_states, dim=-1)
        else:
            non_linear_states = non_linear_states[-1]

        return non_linear_states[:, self._initial_transients:, :], non_linear_states[:, -1, :]

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the model.
        This method is meant to be used after the fit method has been called.

        :param x: The input tensor.

        :return: The state of the deep reservoir for all the time steps and the state for the last time step.
        """

        if not self._trained:
            raise RuntimeError('Model has not been trained yet.')

        return self._forward_core(x)

    def _forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass of the model.

        :param x: The input tensor.

        :return: The state of the deep reservoir for all the time steps and the state for the last time step.
        """

        return self._forward_core(x)

    @torch.no_grad()
    def fit(self, data: torch.utils.data.DataLoader, device: torch.device, standardize: bool = False,
            use_last_state: bool = True, disable_progress_bar: bool = False) -> None:
        """
        Fits the model to the data.

        :param data: The data to fit the model to.
        :param device: The device to use.
        :param standardize: Whether to standardize the data before training the readout.
        :param use_last_state: Whether to use just the state at the last time step as input to the readout.
        :param disable_progress_bar: Whether to disable the progress bar.
        """

        batch_size = data.batch_size
        self._reset_state(batch_size, device)

        num_batches = len(data)
        state_size = self._total_units

        # pre-allocate memory for the states and the targets
        states = np.empty((num_batches * batch_size, data.dataset.data.shape[0] - self._initial_transients,
                           state_size), dtype=np.float32) if not use_last_state \
            else np.empty((num_batches * batch_size, state_size), dtype=np.float32)
        ys = np.empty((num_batches * batch_size, data.dataset.target.shape[0]), dtype=np.float32)

        self._trained = True
        idx = 0
        try:
            # iterate over the data
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
        """
        Scores the model on the data.

        :param data: The data to score the model on.
        :param device: The device to use.
        :param standardize: Whether to standardize the data before scoring the readout.
        :param use_last_state: Whether to use just the state at the last time step as input to the readout.
        :param disable_progress_bar: Whether to disable the progress bar.

        :return: The score of the model on the data.
        """

        if not self._trained:
            raise RuntimeError('Model has not been trained yet.')
        if standardize:
            if self._scaler is None:
                raise ValueError('Standardization is enabled but the model has not been fitted yet.')

        batch_size = data.batch_size
        self._reset_state(batch_size, device)

        num_batches = len(data)
        state_size = self._total_units

        # pre-allocate memory for the states and the targets
        states = np.empty((num_batches * batch_size, data.dataset.data.shape[0] - self._initial_transients,
                           state_size), dtype=np.float32) if not use_last_state \
            else np.empty((num_batches * batch_size, state_size), dtype=np.float32)
        ys = np.empty((num_batches * batch_size, data.dataset.target.shape[0]), dtype=np.float32)

        idx = 0
        # iterate over the data
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
        """
        Predicts the target values of the data.

        :param data: The data to predict the target values of.
        :param device: The device to use.
        :param standardize: Whether to standardize the data before predicting the target values.
        :param use_last_state: Whether to use just the state at the last time step as input to the readout.
        :param disable_progress_bar: Whether to disable the progress bar.

        :return: The predicted target values of the data.
        """

        if not self._trained:
            raise RuntimeError('Model has not been trained yet.')
        if standardize:
            if self._scaler is None:
                raise ValueError('Standardization is enabled but the model has not been fitted yet.')

        batch_size = data.batch_size
        self._reset_state(batch_size, device)

        num_batches = len(data)
        state_size = self._total_units

        # pre-allocate memory for the states
        states = np.empty((num_batches * batch_size, data.dataset.data.shape[0] - self._initial_transients,
                           state_size), dtype=np.float32) if not use_last_state \
            else np.empty((num_batches * batch_size, state_size), dtype=np.float32)

        idx = 0
        # iterate over the data
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
