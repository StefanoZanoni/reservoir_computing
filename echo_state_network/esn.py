import torch
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler
from typing import Callable
from utils.initialization import init_input_kernel, init_non_linear_kernel, init_bias


def validate_params(input_units: int, recurrent_units: int, leaky_rate: float, recurrent_connectivity: int,
                    input_connectivity: int, distribution: str, non_linearity: str, signs_from: str) -> None:
    """
    Validates the parameters for the ReservoirCell.

    :param input_units: Number of input units.
    :param recurrent_units: Number of recurrent units.
    :param leaky_rate: Leaky integration rate.
    :param recurrent_connectivity: Number of connections in the recurrent weight matrix.
    :param input_connectivity: Number of connections in the input weight matrix.
    :param distribution: Distribution type for weight initialization.
    :param non_linearity: Non-linearity function to use.
    :param signs_from: Source for signs of weights.

    :raises ValueError: If any of the parameters are invalid.
    """

    if input_units < 1:
        raise ValueError("Input units must be greater than 0.")
    if recurrent_units < 1:
        raise ValueError("Recurrent units must be greater than 0.")
    if not (0 < leaky_rate <= 1):
        raise ValueError("Leaky rate must be in (0, 1].")
    if not (1 <= recurrent_connectivity <= recurrent_units):
        raise ValueError("Recurrent connectivity must be in [1, recurrent_units].")
    if not (1 <= input_connectivity <= recurrent_connectivity):
        raise ValueError("Input connectivity must be in [1, recurrent_units].")
    if distribution not in ['uniform', 'normal']:
        raise ValueError("Distribution must be 'uniform', or 'normal'.")
    if non_linearity not in ['tanh', 'identity']:
        raise ValueError("Non-linearity must be 'tanh' or 'identity'.")
    if signs_from not in [None, 'random', 'pi', 'e', 'logistic']:
        raise ValueError("Signs from must be None, 'random', 'pi', 'e', or 'logistic'.")


class ReservoirCell(torch.nn.Module):
    """
    A single reservoir cell for the Echo State Network.
    It is a recurrent neural network cell with a leaky integrator or Euler integration.
    The update rule for the reservoir cell with leaky integrator neurons is given by:
     x(t) = (1 - α) * x(t-1) + α * f(W_in * u(t) + W * x(t-1) + b),
    where f is the non-linearity function, W_in is the input weight matrix, W is the recurrent weight matrix, b is the
    bias term, and α is the leaky integration rate.
    The update rule for the reservoir cell with Euler integrator is given by:
     x(t) = x(t-1) + epsilon * f(W_in * u(t) + (W - gamma * I) * x(t-1) + b), where I is the identity matrix, and
    epsilon is the Euler integration step size, and gamma is the diffusion coefficient for the recurrent weights.
    """

    def __init__(self,
                 input_units: int,
                 recurrent_units: int,
                 *,
                 input_scaling: float = 1.0,
                 recurrent_scaling: float = 1.0,
                 spectral_radius: float = 0.9,
                 leaky_rate: float = 0.5,
                 input_connectivity: int = 1,
                 recurrent_connectivity: int = 1,
                 bias: bool = True,
                 bias_scaling: float = None,
                 distribution: str = 'uniform',
                 signs_from: str | None = None,
                 fixed_input_kernel: bool = False,
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 circular_recurrent_kernel: bool = False,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3,
                 ) -> None:
        """
        Initializes the ReservoirCell.

        :param input_units: Number of input units.
        :param recurrent_units: Number of recurrent units.
        :param input_scaling: Scaling factor for input weights.
        :param recurrent_scaling: Scaling factor for recurrent weights.
        :param spectral_radius: Spectral radius of the recurrent weight matrix.
        :param leaky_rate: Leaky integration rate.
        :param input_connectivity: Number of connections in the input weight matrix.
        :param recurrent_connectivity: Number of connections in the recurrent weight matrix.
        :param bias: Whether to use a bias term.
        :param bias_scaling: Scaling factor for the bias term.
        :param distribution: Distribution type for weight initialization.
        :param signs_from: Source for signs of weights.
        :param fixed_input_kernel: Whether to use a fixed input kernel.
        :param non_linearity: Non-linearity function to use.
        :param effective_rescaling: Whether to rescale the recurrent weights according to the leaky rate.
        :param circular_recurrent_kernel: Whether to use a circular recurrent kernel.
        :param euler: Whether to use Euler integration.
        :param epsilon: Euler integration step size.
        :param gamma: Diffusion coefficient for the Euler recurrent kernel.
        """

        super().__init__()

        validate_params(input_units, recurrent_units, leaky_rate, recurrent_connectivity, input_connectivity,
                        distribution, non_linearity, signs_from)

        self.recurrent_units = recurrent_units
        self._leaky_rate = leaky_rate
        self._one_minus_leaky_rate = 1 - leaky_rate

        self.input_kernel = init_input_kernel(input_units, recurrent_units, input_connectivity, input_scaling,
                                              'fixed' if circular_recurrent_kernel and fixed_input_kernel
                                              else distribution,
                                              signs_from=signs_from if circular_recurrent_kernel and fixed_input_kernel
                                              else None)
        self.recurrent_kernel = init_non_linear_kernel(recurrent_units, recurrent_connectivity, distribution,
                                                       spectral_radius, leaky_rate, effective_rescaling,
                                                       circular_recurrent_kernel, euler, gamma, recurrent_scaling)
        self.bias = init_bias(bias, recurrent_units, input_scaling, bias_scaling)

        self._epsilon = epsilon
        self._non_linear_function: Callable = torch.tanh if non_linearity == 'tanh' else lambda x: x
        self._state = None
        self._forward_function: Callable = self._forward_euler if euler else self._forward_leaky_integrator

    @torch.no_grad()
    def _forward_leaky_integrator(self, xt: torch.Tensor) -> torch.FloatTensor:
        """
        Forward pass using the leaky integrator method.

        :param xt: Input tensor at time t.

        :return : Updated state tensor.
        """

        # x(t) = (1 - α) * x(t-1) + α * f(W_in * u(t) + W * x(t-1) + b)
        self._state.mul_(self._one_minus_leaky_rate).add_(
            self._non_linear_function(
                torch.matmul(xt, self.input_kernel)
                .addmm_(self._state, self.recurrent_kernel)
                .add_(self.bias)
            )
            .mul_(self._leaky_rate)
        )

        return self._state

    @torch.no_grad()
    def _forward_euler(self, xt: torch.Tensor) -> torch.FloatTensor:
        """
        Forward pass using the Euler integration method.

        :param xt: Input tensor at time t.

        :return : Updated state tensor.
        """

        # x(t) = x(t-1) + ε * f(W_in * u(t) + (W - γ * I) * x(t-1) + b)
        self._state.add_(
            self._non_linear_function(
                torch.matmul(xt, self.input_kernel)
                .addmm_(self._state, self.recurrent_kernel)
                .add_(self.bias)
            )
            .mul_(self._epsilon)
        )

        return self._state

    def forward(self, xt: torch.Tensor) -> torch.FloatTensor:
        """
        Forward pass through the reservoir cell.

        :param xt: Input tensor at time t.

        :return : Updated state tensor.
        """

        return self._forward_function(xt)

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        """
        Resets the state of the reservoir cell.

        :param batch_size: The batch size.
        :param device: The device to use.
        """

        self._state = torch.zeros((batch_size, self.recurrent_units), dtype=torch.float32,
                                  requires_grad=False, device=device)


class EchoStateNetwork(torch.nn.Module):
    """
    An Echo State Network (ESN) for time series prediction/classification tasks.
    """

    def __init__(self,
                 input_units: int,
                 recurrent_units: int,
                 *,
                 initial_transients: int = 0,
                 input_scaling: float = 1.0,
                 recurrent_scaling: float = 1.0,
                 spectral_radius: float = 0.9,
                 leaky_rate: float = 0.5,
                 input_connectivity: int = 1,
                 recurrent_connectivity: int = 1,
                 bias: bool = True,
                 bias_scaling: float = None,
                 distribution: str = 'uniform',
                 signs_from: str | None = None,
                 fixed_input_kernel: bool = False,
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 circular_recurrent_kernel: bool = False,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3,
                 ) -> None:
        """
        Initializes the EchoStateNetwork.

        :param task: Task type ('classification' or 'regression').
        :param input_units: Number of input units.
        :param recurrent_units: Number of recurrent units.
        :param initial_transients: Number of initial transient states to discard.
        :param input_scaling: Scaling factor for input weights.
        :param recurrent_scaling: Scaling factor for recurrent weights.
        :param spectral_radius: Spectral radius of the recurrent weight matrix.
        :param leaky_rate: Leaky integration rate.
        :param input_connectivity: Number of connections in the input weight matrix.
        :param recurrent_connectivity: Number of connections in the recurrent weight matrix.
        :param bias: Whether to use a bias term.
        :param bias_scaling: Scaling factor for the bias term.
        :param distribution: Distribution type for weight initialization.
        :param signs_from: Source for signs of weights.
        :param fixed_input_kernel: Whether to use a fixed input kernel.
        :param non_linearity: Non-linearity function to use.
        :param effective_rescaling: Whether to use effective rescaling.
        :param circular_recurrent_kernel: Whether to use a circular recurrent kernel.
        :param euler: Whether to use Euler integration.
        :param epsilon: Euler integration step size.
        :param gamma: Diffusion coefficient for the Euler recurrent kernel.
        :param alpha: Regularization strength for the readout layer.
        :param max_iter: Maximum number of iterations for the readout layer.
        :param tolerance: Tolerance for the readout layer.
        """

        super().__init__()
        self._initial_transients = initial_transients
        self.net = ReservoirCell(input_units,
                                 recurrent_units,
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
                                 recurrent_scaling=recurrent_scaling)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Echo State Network.

        :param x: Input tensor.

        :return : States tensor after passing through the network.
        """

        seq_len = x.shape[1]
        states = torch.empty((x.shape[0], seq_len, self.net.recurrent_units), dtype=torch.float32,
                             requires_grad=False, device=x.device)

        is_dim_2 = x.dim() == 2

        for t in range(seq_len):
            xt = x[:, t].unsqueeze(1) if is_dim_2 else x[:, t]
            states[:, t, :].copy_(self.net(xt))

        return states[:, self._initial_transients:, :]
