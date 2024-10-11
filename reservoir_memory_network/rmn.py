import torch
import numpy as np

from tqdm import tqdm

from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler

from utils.initialization import init_memory_kernel, init_input_kernel, init_non_linear_kernel, init_bias


def validate_params_memory(input_units: int, memory_units: int, input_memory_connectivity: int, distribution: str,
                           signs_from: str, theta: float) -> None:
    """
    Validate the parameters for the memory cell.

    :param input_units: Number of input units.
    :param memory_units: Number of memory units.
    :param input_memory_connectivity: Input to memory connectivity.
    :param distribution: Distribution of the weights.
    :param signs_from: Source of weight signs.
    :param theta: Theta parameter for the Legendre polynomial kernel.
    """

    if input_units < 1:
        raise ValueError("Input units must be greater than 0.")
    if memory_units < 1:
        raise ValueError("Memory units must be greater than 0.")
    if not (1 <= input_memory_connectivity <= memory_units):
        raise ValueError("Input to memory connectivity must be in [1, memory_units].")
    if distribution not in ['uniform', 'normal']:
        raise ValueError("Distribution must be 'uniform', or 'normal'.")
    if signs_from not in [None, 'random', 'pi', 'e', 'logistic']:
        raise ValueError("Signs from must be None, 'random', 'pi', 'e', or 'logistic'.")
    if theta <= 0:
        raise ValueError("Theta must be greater than 0.")


def validate_params_non_linear(input_units: int, non_linear_units: int, memory_units: int, leaky_rate: float,
                               memory_non_linear_connectivity: int, input_non_linear_connectivity: int,
                               non_linear_connectivity: int, distribution: str, non_linearity: str,
                               signs_from: str) -> None:
    """
    Validate the parameters for the non-linear cell.

    :param input_units: Number of input units.
    :param non_linear_units: Number of non-linear units.
    :param leaky_rate: Leaky integrator rate.
    :param memory_non_linear_connectivity: Memory to non-linear connectivity.
    :param input_non_linear_connectivity: Input to non-linear connectivity.
    :param non_linear_connectivity: Non-linear connectivity.
    :param distribution: Distribution of the weights.
    :param non_linearity: Non-linearity function.
    :param signs_from: Source of weight signs.
    """

    if input_units < 1:
        raise ValueError("Input units must be greater than 0.")
    if non_linear_units < 1:
        raise ValueError("Non-linear units must be greater than 0.")
    if memory_units < 1:
        raise ValueError("Memory units must be greater than 0.")
    if not (0 < leaky_rate <= 1):
        raise ValueError("Leaky rate must be in (0, 1].")
    if not (1 <= memory_non_linear_connectivity <= non_linear_units):
        raise ValueError("Memory to non-linear connectivity must be in [1, non_linear_units].")
    if not (1 <= input_non_linear_connectivity <= non_linear_units):
        raise ValueError("Input to non-linear connectivity must be in [1, non_linear_units].")
    if not (1 <= non_linear_connectivity <= non_linear_units):
        raise ValueError("Non-linear connectivity must be in [1, non_linear_units].")
    if distribution not in ['uniform', 'normal']:
        raise ValueError("Distribution must be 'uniform', or 'normal'.")
    if non_linearity not in ['tanh', 'identity']:
        raise ValueError("Non-linearity must be 'tanh' or 'identity'.")
    if signs_from not in [None, 'random', 'pi', 'e', 'logistic']:
        raise ValueError("Signs from must be None, 'random', 'pi', 'e', or 'logistic'.")


class MemoryCell(torch.nn.Module):
    """
    Memory cell for the RMN model.
    """

    def __init__(self,
                 input_units: int,
                 memory_units: int,
                 *,
                 input_memory_scaling: float = 1.0,
                 memory_scaling: float = 1.0,
                 input_memory_connectivity: int = 1,
                 theta: float = 1.0,
                 legendre: bool = False,
                 distribution: str = 'uniform',
                 signs_from: str | None = None,
                 fixed_input_kernel: bool = False, ) \
            -> None:
        """
        Initialize the memory cell.

        :param input_units: Number of input units.
        :param memory_units: Number of memory units.
        :param input_memory_scaling: Input to memory scaling.
        :param memory_scaling: Memory scaling.
        :param input_memory_connectivity: Input to memory connectivity.
        :param theta: Legendre polynomial kernel parameter.
        :param legendre: Whether to use Legendre polynomial kernel.
        :param distribution: Distribution of the weights.
        :param signs_from: Source of weight signs.
        :param fixed_input_kernel: Whether to use fixed input kernel.
        """

        super().__init__()

        validate_params_memory(input_units, memory_units, input_memory_connectivity, distribution, signs_from, theta)

        self.input_memory_kernel = init_input_kernel(
            input_units, memory_units, input_memory_connectivity,
            input_memory_scaling, 'fixed' if fixed_input_kernel else distribution,
            signs_from=signs_from
        )

        self.memory_kernel = init_memory_kernel(memory_units, theta, legendre, memory_scaling)
        self._memory_state = None

    @torch.no_grad()
    def forward(self, xt: torch.Tensor) -> torch.FloatTensor:
        """
        Forward pass for the memory cell.

        :param xt: Input tensor at time t.

        :return: The memory state at time t.
        """

        # m(t) = Vm * m(t-1) + Vu * u(t)
        torch.addmm(torch.matmul(xt, self.input_memory_kernel),
                    self._memory_state, self.memory_kernel, out=self._memory_state)

        return self._memory_state

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        """
        Reset the memory state.

        :param batch_size: The batch size.
        :param device: The device to use.
        """

        self._memory_state = torch.zeros((batch_size, self.memory_kernel.shape[0]), dtype=torch.float32,
                                         device=device, requires_grad=False)


class NonLinearCell(torch.nn.Module):
    """
    Non-linear cell for the RMN model.
    """

    def __init__(self,
                 input_units: int,
                 non_linear_units: int,
                 memory_units: int,
                 *,
                 input_non_linear_scaling: float = 1.0,
                 non_linear_scaling: float = 1.0,
                 memory_non_linear_scaling: float = 1.0,
                 input_non_linear_connectivity: int = 1,
                 non_linear_connectivity: int = 1,
                 memory_non_linear_connectivity: int = 1,
                 spectral_radius: float = 0.9,
                 leaky_rate: float = 0.5,
                 bias: bool = True,
                 bias_scaling: float = None,
                 distribution: str = 'uniform',
                 signs_from: str | None = None,
                 fixed_input_kernel: bool = False,
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 circular_non_linear_kernel: bool = False,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3) -> None:
        """
        Initialize the non-linear cell.

        :param input_units: Number of input units.
        :param non_linear_units: Number of non-linear units.
        :param memory_units: NUmber of memory units.
        :param input_non_linear_scaling: Input to non-linear scaling.
        :param non_linear_scaling: Non-linear scaling.
        :param memory_non_linear_scaling: Memory to non-linear scaling.
        :param input_non_linear_connectivity: Input to non-linear connectivity.
        :param non_linear_connectivity: Non-linear connectivity.
        :param memory_non_linear_connectivity: Memory to non-linear connectivity.
        :param spectral_radius: Desired spectral radius.
        :param leaky_rate: Leaky integrator rate.
        :param bias: Whether to use bias.
        :param bias_scaling: Bias scaling.
        :param distribution: Distribution of the weights.
        :param signs_from: Source of weight signs.
        :param fixed_input_kernel: Whether to use fixed input kernel.
        :param non_linearity: Non-linearity function.
        :param effective_rescaling: Whether to rescale the weights according to the leaky rate.
        :param circular_non_linear_kernel: Whether to use circular non-linear kernel.
        :param euler: Where to use Euler integration.
        :param epsilon: Euler integration step size.
        :param gamma: Diffusion coefficient for the Euler recurrent kernel.
        """

        super().__init__()

        validate_params_non_linear(input_units, non_linear_units, memory_units, leaky_rate,
                                   memory_non_linear_connectivity, input_non_linear_connectivity,
                                   non_linear_connectivity, distribution, non_linearity, signs_from)

        self._leaky_rate = leaky_rate
        self._one_minus_leaky_rate = 1 - leaky_rate

        self.input_non_linear_kernel = init_input_kernel(
            input_units, non_linear_units, input_non_linear_connectivity,
            input_non_linear_scaling, 'fixed' if circular_non_linear_kernel and fixed_input_kernel
            else distribution,
            signs_from=signs_from if circular_non_linear_kernel and fixed_input_kernel else None
        )
        self.non_linear_kernel = init_non_linear_kernel(non_linear_units, non_linear_connectivity, distribution,
                                                        spectral_radius, leaky_rate, effective_rescaling,
                                                        circular_non_linear_kernel, euler, gamma,
                                                        non_linear_scaling)
        self._epsilon = epsilon
        self.memory_non_linear_kernel = init_input_kernel(memory_units, non_linear_units,
                                                          memory_non_linear_connectivity, memory_non_linear_scaling,
                                                          distribution)
        self.bias = init_bias(bias, non_linear_units, input_non_linear_scaling, bias_scaling)

        self._non_linear_function: Callable = torch.tanh if non_linearity == 'tanh' else lambda x: x

        self._non_linear_state = None
        self._forward_function: Callable = self._forward_euler if euler else self._forward_leaky_integrator

    @torch.no_grad()
    def _forward_leaky_integrator(self, xt: torch.Tensor, memory_state: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass for the non-linear cell using the leaky integrator.

        :param xt: Input tensor at time t.
        :param memory_state: Memory state at time t.
        :return: The non-linear state at time t.
        """

        # x(t) = (1 - a) * x(t-1) + a * f(Wx * x(t-1) + Wm * m(t) + Wu * u(t) + b)
        self._non_linear_state.mul_(self._one_minus_leaky_rate).add_(
            self._non_linear_function(
                torch.matmul(xt, self.input_non_linear_kernel)
                .addmm_(memory_state, self.memory_non_linear_kernel)
                .addmm_(self._non_linear_state, self.non_linear_kernel)
                .add_(self.bias)
            )
            .mul_(self._leaky_rate)
        )

        return self._non_linear_state

    @torch.no_grad()
    def _forward_euler(self, xt: torch.Tensor, memory_state: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass for the non-linear cell using Euler integration.

        :param xt: Input tensor at time t.
        :param memory_state: Memory state at time t.
        :return: The non-linear state at time t.
        """

        # x(t) = x(t-1) + ε * f(Wu * u(t) + Wm * m(t) + (W - γ * I) * x(t-1) + b)
        self._non_linear_state.add_(
            self._non_linear_function(
                torch.matmul(xt, self.input_non_linear_kernel)
                .addmm_(memory_state, self.memory_non_linear_kernel)
                .addmm_(self._non_linear_state, self.non_linear_kernel)
                .add_(self.bias)
            )
            .mul_(self._epsilon)
        )

        return self._non_linear_state

    def forward(self, xt: torch.Tensor, memory_state: torch.FloatTensor) -> torch.FloatTensor:
        """
        Forward pass for the non-linear cell using Euler integration.

        :param xt: Input tensor at time t.
        :param memory_state: Memory state at time t.
        :return: The non-linear state at time t.
        """

        return self._forward_function(xt, memory_state)

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        """
        Reset the non-linear state.

        :param batch_size: The batch size.
        :param device: The device to use.
        """

        self._non_linear_state = torch.zeros((batch_size, self.non_linear_kernel.shape[0]), dtype=torch.float32,
                                             device=device, requires_grad=False)


class RMNCell(torch.nn.Module):
    def __init__(self,
                 input_units: int,
                 non_linear_units: int,
                 memory_units: int,
                 *,
                 memory_scaling: float = 1.0,
                 non_linear_scaling: float = 1.0,
                 input_memory_scaling: float = 1.0,
                 input_non_linear_scaling: float = 1.0,
                 memory_non_linear_scaling: float = 1.0,
                 spectral_radius: float = 0.9,
                 leaky_rate: float = 0.5,
                 input_memory_connectivity: int = 1,
                 input_non_linear_connectivity: int = 1,
                 non_linear_connectivity: int = 1,
                 memory_non_linear_connectivity: int = 1,
                 bias: bool = True,
                 bias_scaling: float = None,
                 distribution: str = 'uniform',
                 signs_from: str | None = None,
                 fixed_input_kernel: bool = False,
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 circular_non_linear_kernel: bool = False,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3,
                 legendre: bool = False,
                 theta: float = 1.0,
                 just_memory: bool = False) -> None:

        super().__init__()

        validate_params(just_memory, input_units, memory_units, non_linear_units, leaky_rate,
                        memory_non_linear_connectivity, input_non_linear_connectivity, non_linear_connectivity,
                        input_memory_connectivity, distribution, non_linearity, signs_from)

        self.memory = MemoryCell(input_units, memory_units, input_memory_scaling=input_memory_scaling,
                                 memory_scaling=memory_scaling, input_memory_connectivity=input_memory_connectivity,
                                 theta=theta, legendre=legendre, distribution=distribution, signs_from=signs_from,
                                 fixed_input_kernel=fixed_input_kernel)

        if just_memory:
            self.non_linear = None
        else:
            self.non_linear = NonLinearCell(input_units, non_linear_units, memory_units,
                                            input_non_linear_scaling=input_non_linear_scaling,
                                            memory_non_linear_scaling=memory_non_linear_scaling,
                                            input_non_linear_connectivity=input_non_linear_connectivity,
                                            non_linear_connectivity=non_linear_connectivity,
                                            spectral_radius=spectral_radius,
                                            leaky_rate=leaky_rate, non_linear_scaling=non_linear_scaling,
                                            memory_non_linear_connectivity=memory_non_linear_connectivity, bias=bias,
                                            bias_scaling=bias_scaling, distribution=distribution, signs_from=signs_from,
                                            fixed_input_kernel=fixed_input_kernel, non_linearity=non_linearity,
                                            effective_rescaling=effective_rescaling,
                                            circular_non_linear_kernel=circular_non_linear_kernel,
                                            euler=euler, epsilon=epsilon, gamma=gamma)

    @torch.no_grad()
    def forward(self, xt: torch.Tensor) -> tuple[torch.FloatTensor | None, torch.FloatTensor]:

        memory_state = self.memory(xt)
        non_linear_state = None if self.non_linear is None else self.non_linear(xt, memory_state)

        return non_linear_state, memory_state

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        self.memory.reset_state(batch_size, device)
        if self.non_linear is not None:
            self.non_linear.reset_state(batch_size, device)


class ReservoirMemoryNetwork(torch.nn.Module):
    def __init__(self,
                 input_units: int,
                 non_linear_units: int,
                 memory_units: int,
                 *,
                 initial_transients: int = 0,
                 memory_scaling: float = 1.0,
                 non_linear_scaling: float = 1.0,
                 input_memory_scaling: float = 1.0,
                 input_non_linear_scaling: float = 1.0,
                 memory_non_linear_scaling: float = 1.0,
                 spectral_radius: float = 0.9,
                 leaky_rate: float = 0.5,
                 input_memory_connectivity: int = 1,
                 input_non_linear_connectivity: int = 1,
                 non_linear_connectivity: int = 1,
                 memory_non_linear_connectivity: int = 1,
                 bias: bool = True,
                 bias_scaling: float = None,
                 distribution: str = 'uniform',
                 signs_from: str | None = None,
                 fixed_input_kernels: bool = False,
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 circular_non_linear_kernel: bool = False,
                 euler: bool = False,
                 epsilon: float = 1e-3,
                 gamma: float = 1e-3,
                 legendre: bool = False,
                 theta: float = 1.0,
                 just_memory: bool = False) -> None:

        super().__init__()
        self._initial_transients = initial_transients
        self.net = RMNCell(input_units, non_linear_units, memory_units, memory_scaling=memory_scaling,
                           non_linear_scaling=non_linear_scaling, input_memory_scaling=input_memory_scaling,
                           input_non_linear_scaling=input_non_linear_scaling,
                           memory_non_linear_scaling=memory_non_linear_scaling, spectral_radius=spectral_radius,
                           leaky_rate=leaky_rate, input_memory_connectivity=input_memory_connectivity,
                           input_non_linear_connectivity=input_non_linear_connectivity,
                           non_linear_connectivity=non_linear_connectivity,
                           memory_non_linear_connectivity=memory_non_linear_connectivity, bias=bias,
                           bias_scaling=bias_scaling, distribution=distribution, signs_from=signs_from,
                           fixed_input_kernel=fixed_input_kernels, non_linearity=non_linearity,
                           effective_rescaling=effective_rescaling,
                           circular_non_linear_kernel=circular_non_linear_kernel, euler=euler, epsilon=epsilon,
                           gamma=gamma, legendre=legendre, theta=theta, just_memory=just_memory)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.FloatTensor | None, torch.FloatTensor]:

        is_dim_2 = x.dim() == 2
        seq_len = x.shape[1]

        if not self.just_memory:
            non_linear_states = torch.empty((x.shape[0], seq_len, self.net.non_linear.non_linear_kernel.shape[0]),
                                            dtype=torch.float32, device=x.device, requires_grad=False)
        memory_states = torch.empty((x.shape[0], seq_len, self.net.memory.memory_kernel.shape[0]), dtype=torch.float32,
                                    device=x.device, requires_grad=False)

        for t in range(seq_len):
            xt = x[:, t].unsqueeze(1) if is_dim_2 else x[:, t]
            if not self.just_memory:
                non_linear_state, memory_state = self.net(xt)
                non_linear_states[:, t].copy_(non_linear_state)
                memory_states[:, t].copy_(memory_state)
            else:
                _, memory_state = self.net(xt)
                memory_states[:, t].copy_(memory_state)

        if not self.just_memory:
            non_linear_states = non_linear_states[:, self._initial_transients:]
        memory_states = memory_states[:, self._initial_transients:]

        return (non_linear_states, memory_states) if not self.just_memory else (None, memory_states)
