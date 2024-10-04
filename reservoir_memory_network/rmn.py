import torch
import numpy as np

from tqdm import tqdm

from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.preprocessing import StandardScaler

from utils.initialization import init_memory_kernel, init_input_kernel, init_non_linear_kernel, init_bias


def validate_params(just_memory, input_units, memory_units, non_linear_units, spectral_radius, leaky_rate,
                    memory_non_linear_connectivity, input_non_linear_connectivity, non_linear_connectivity,
                    input_memory_connectivity, distribution, non_linearity):
    if input_units < 1:
        raise ValueError("Input units must be greater than 0.")
    if memory_units < 1:
        raise ValueError("Memory units must be greater than 0.")
    if not (1 <= input_memory_connectivity <= memory_units):
        raise ValueError("Input to memory connectivity must be in [1, memory_units].")
    if distribution not in ['uniform', 'normal']:
        raise ValueError("Distribution must be 'uniform', or 'normal'.")
    if non_linearity not in ['tanh', 'identity']:
        raise ValueError("Non-linearity must be 'tanh' or 'identity'.")

    if not just_memory:
        if non_linear_units < 1:
            raise ValueError("Non linear units must be greater than 0.")
        if not (0 <= spectral_radius <= 1):
            raise ValueError("Spectral radius must be in [0, 1].")
        if not (0 < leaky_rate <= 1):
            raise ValueError("Leaky rate must be in (0, 1].")
        if not (1 <= input_non_linear_connectivity <= non_linear_units):
            raise ValueError("Input to non linear connectivity must be in [1, non_linear_units].")
        if not (1 <= non_linear_connectivity <= non_linear_units):
            raise ValueError("Non linear connectivity must be in [1, non_linear_units].")
        if not (1 <= memory_non_linear_connectivity <= non_linear_units):
            raise ValueError("Memory to non linear connectivity must be in [1, non_linear_units].")


class MemoryCell(torch.nn.Module):
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
        super().__init__()

        # Input to memory reservoir kernel
        self.input_memory_kernel = init_input_kernel(
            input_units, memory_units, input_memory_connectivity,
            input_memory_scaling, 'fixed' if fixed_input_kernel else distribution,
            signs_from=signs_from
        )

        # Memory reservoir kernel
        self.memory_kernel = init_memory_kernel(memory_units, theta, legendre, memory_scaling)
        self._memory_state = None

    @torch.no_grad()
    def forward(self, xt: torch.Tensor) -> torch.FloatTensor:
        # m(t) =                    +             Vx * x(t)
        self._memory_state = torch.addmm(torch.matmul(xt, self.input_memory_kernel),
                                         self._memory_state, self.memory_kernel)  # Vm * m(t-1)

        return self._memory_state

    def reset_state(self, batch_size: int, device: torch.device) -> None:
        self._memory_state = torch.zeros((batch_size, self.memory_kernel.shape[0]), dtype=torch.float32,
                                         device=device, requires_grad=False)


class NonLinearCell(torch.nn.Module):
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
        super().__init__()

        self._leaky_rate = leaky_rate
        self._one_minus_leaky_rate = 1 - leaky_rate
        # Input to non-linear reservoir kernel
        self.input_non_linear_kernel = init_input_kernel(
            input_units, non_linear_units, input_non_linear_connectivity,
            input_non_linear_scaling, 'fixed' if circular_non_linear_kernel and fixed_input_kernel
            else distribution,
            signs_from=signs_from if circular_non_linear_kernel and fixed_input_kernel else None
        )
        # Non-linear reservoir kernel
        self.non_linear_kernel = init_non_linear_kernel(non_linear_units, non_linear_connectivity, distribution,
                                                        spectral_radius, leaky_rate, effective_rescaling,
                                                        circular_non_linear_kernel, euler, gamma,
                                                        non_linear_scaling)
        self._epsilon = epsilon
        # Memory to non-linear reservoir connectivity
        self.memory_non_linear_kernel = init_input_kernel(memory_units, non_linear_units,
                                                          memory_non_linear_connectivity, memory_non_linear_scaling,
                                                          distribution)
        self.bias = init_bias(bias, non_linear_units, input_non_linear_scaling, bias_scaling)

        self._non_linear_function: Callable = torch.tanh if non_linearity == 'tanh' else lambda x: x

        self._non_linear_state = None
        self._forward_function: Callable = self._forward_euler if euler else self._forward_leaky_integrator

    @torch.no_grad()
    def _forward_leaky_integrator(self, xt: torch.Tensor, memory_state: torch.FloatTensor) -> torch.FloatTensor:
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

    @torch.no_grad()
    def forward(self, xt: torch.Tensor, memory_state: torch.FloatTensor) -> torch.FloatTensor:
        return self._forward_function(xt, memory_state)

    def reset_state(self, batch_size: int, device: torch.device) -> None:
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

        validate_params(just_memory, input_units, memory_units, non_linear_units, spectral_radius, leaky_rate,
                        memory_non_linear_connectivity, input_non_linear_connectivity, non_linear_connectivity,
                        input_memory_connectivity, distribution, non_linearity)

        self.memory = MemoryCell(input_units, memory_units, input_memory_scaling=input_memory_scaling,
                                 memory_scaling=memory_scaling, input_memory_connectivity=input_memory_connectivity,
                                 theta=theta, legendre=legendre, distribution=distribution, signs_from=signs_from,
                                 fixed_input_kernel=fixed_input_kernel)

        if just_memory:
            self.non_linear = None
        else:
            self.non_linear = NonLinearCell(input_units, non_linear_units,
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
        if self.non_linear is not None:
            non_linear_state = self.non_linear(xt, memory_state)
            return non_linear_state, memory_state
        else:
            return None, memory_state

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
