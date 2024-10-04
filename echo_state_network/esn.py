import torch
import numpy as np
from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import Callable
from utils.initialization import init_input_kernel, init_non_linear_kernel, init_bias


def validate_params(input_units, recurrent_units, spectral_radius, leaky_rate, recurrent_connectivity,
                    distribution, non_linearity):
    """
    Validates the parameters for the ReservoirCell.

    Args:
        input_units (int): Number of input units.
        recurrent_units (int): Number of recurrent units.
        spectral_radius (float): Spectral radius of the recurrent weight matrix.
        leaky_rate (float): Leaky integration rate.
        recurrent_connectivity (int): Number of connections in the recurrent weight matrix.
        distribution (str): Distribution type for weight initialization.
        non_linearity (str): Non-linearity function to use.
    
    Raises:
        ValueError: If any of the parameters are invalid.
    """

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
    if distribution not in ['uniform', 'normal']:
        raise ValueError("Distribution must be 'uniform', or 'normal'.")
    if non_linearity not in ['tanh', 'identity']:
        raise ValueError("Non-linearity must be 'tanh' or 'identity'.")


class ReservoirCell(torch.nn.Module):
    """
    A single reservoir cell for the Echo State Network.

    Attributes:
        input_kernel (torch.Tensor): Input weight matrix.
        recurrent_kernel (torch.Tensor): Recurrent weight matrix.
        bias (torch.Tensor): Bias term.
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

        Args:
            input_units (int): Number of input units.
            recurrent_units (int): Number of recurrent units.
            input_scaling (float): Scaling factor for input weights.
            recurrent_scaling (float): Scaling factor for recurrent weights.
            spectral_radius (float): Spectral radius of the recurrent weight matrix.
            leaky_rate (float): Leaky integration rate.
            input_connectivity (int): Number of connections in the input weight matrix.
            recurrent_connectivity (int): Number of connections in the recurrent weight matrix.
            bias (bool): Whether to use a bias term.
            bias_scaling (float, optional): Scaling factor for the bias term.
            distribution (str): Distribution type for weight initialization.
            signs_from (str, optional): Source for signs of weights.
            fixed_input_kernel (bool): Whether to use a fixed input kernel.
            non_linearity (str): Non-linearity function to use.
            effective_rescaling (bool): Whether to use effective rescaling.
            circular_recurrent_kernel (bool): Whether to use a circular recurrent kernel.
            euler (bool): Whether to use Euler integration.
            epsilon (float): Euler integration step size.
            gamma (float): Scaling factor for recurrent weights.
        """

        super().__init__()

        validate_params(input_units, recurrent_units, spectral_radius, leaky_rate, recurrent_connectivity,
                        distribution, non_linearity)

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

        Args:
            xt (torch.Tensor): Input tensor at time t.

        Returns:
            torch.FloatTensor: Updated state tensor.
        """

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

        Args:
            xt (torch.Tensor): Input tensor at time t.

        Returns:
            torch.FloatTensor: Updated state tensor.
        """

        self._state.add_(
            self._non_linear_function(
                torch.matmul(xt, self.input_kernel)
                .addmm_(self._state, self.recurrent_kernel)
                .add_(self.bias)
            )
            .mul_(self._epsilon)
        )

        return self._state

    @torch.no_grad()
    def forward(self, xt: torch.Tensor) -> torch.FloatTensor:
        """
        Forward pass through the reservoir cell.

        Args:
            xt (torch.Tensor): Input tensor at time t.

        Returns:
            torch.FloatTensor: Updated state tensor.
        """

        return self._forward_function(xt)

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        self._state = torch.zeros((batch_size, self.recurrent_kernel.shape[0]), dtype=torch.float32,
                                  requires_grad=False, device=device)


class EchoStateNetwork(torch.nn.Module):
    """
    An Echo State Network (ESN) for time series prediction/classification tasks.

    Attributes:
        scaler (StandardScaler): Scaler for standardizing the states.
        _initial_transients (int): Number of initial transient states to discard.
        task (str): Task type ('classification' or 'regression').
        net (ReservoirCell): Reservoir cell of the ESN.
        readout (RidgeClassifier or Ridge): Readout layer for the ESN.
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

        Args:
            task (str): Task type ('classification' or 'regression').
            input_units (int): Number of input units.
            recurrent_units (int): Number of recurrent units.
            initial_transients (int): Number of initial transient states to discard.
            input_scaling (float): Scaling factor for input weights.
            recurrent_scaling (float): Scaling factor for recurrent weights.
            spectral_radius (float): Spectral radius of the recurrent weight matrix.
            leaky_rate (float): Leaky integration rate.
            input_connectivity (int): Number of connections in the input weight matrix.
            recurrent_connectivity (int): Number of connections in the recurrent weight matrix.
            bias (bool): Whether to use a bias term.
            bias_scaling (float, optional): Scaling factor for the bias term.
            distribution (str): Distribution type for weight initialization.
            signs_from (str, optional): Source for signs of weights.
            fixed_input_kernel (bool): Whether to use a fixed input kernel.
            non_linearity (str): Non-linearity function to use.
            effective_rescaling (bool): Whether to use effective rescaling.
            circular_recurrent_kernel (bool): Whether to use a circular recurrent kernel.
            euler (bool): Whether to use Euler integration.
            epsilon (float): Euler integration step size.
            gamma (float): Scaling factor for recurrent weights.
            alpha (float): Regularization strength for the readout layer.
            max_iter (int): Maximum number of iterations for the readout layer.
            tolerance (float): Tolerance for the readout layer.
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

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: States tensor after passing through the network.
        """

        seq_len = x.shape[1]
        states = torch.empty((x.shape[0], seq_len, self.net.recurrent_kernel.shape[0]), dtype=torch.float32,
                             requires_grad=False, device=x.device)

        is_dim_2 = x.dim() == 2

        for t in range(seq_len):
            xt = x[:, t].unsqueeze(1) if is_dim_2 else x[:, t]
            state = self.net(xt)
            states[:, t, :].copy_(state)

        states = states[:, self._initial_transients:, :]

        return states
