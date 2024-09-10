import torch

from utils.initialization import (sparse_tensor_init, sparse_recurrent_tensor_init, spectral_norm_scaling,
                                  sparse_eye_init, fast_spectral_rescaling, circular_tensor_init)


class RMNCell(torch.nn.Module):
    def __init__(self,
                 input_units: int,
                 non_linear_units: int,
                 memory_units: int,
                 *,
                 input_memory_scaling: float = 1.0,
                 input_non_linear_scaling: float = 1.0,
                 memory_non_linear_scaling: float = 1.0,
                 spectral_radius: float = 0.99,
                 leaky_rate: float = 1.0,
                 input_memory_connectivity: int = 1,
                 input_non_linear_connectivity: int = 1,
                 non_linear_connectivity: int = 1,
                 memory_non_linear_connectivity: int = 1,
                 bias: bool = True,
                 distribution: str = 'uniform',
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 bias_scaling: float | None,
                 circular_non_linear_kernel: bool = True):

        super().__init__()

        if input_units < 1:
            raise ValueError("Input units must be greater than 0.")
        self.input_units = input_units
        if non_linear_units < 1:
            raise ValueError("Recurrent units must be greater than 0.")
        self.non_linear_units = non_linear_units
        if memory_units < 1:
            raise ValueError("Recurrent units must be greater than 0.")
        self.memory_units = memory_units
        self.input_memory_scaling = input_memory_scaling
        self.input_non_linear_scaling = input_non_linear_scaling
        if spectral_radius > 1 or spectral_radius < 0:
            raise ValueError("Spectral radius must be in [0, 1].")
        self.spectral_radius = spectral_radius
        if leaky_rate > 1 or leaky_rate < 0:
            raise ValueError("Leaky rate must be in [0, 1].")
        self.leaky_rate = leaky_rate
        self.one_minus_leaky_rate = 1 - leaky_rate
        if input_memory_connectivity > memory_units or input_memory_connectivity < 1:
            raise ValueError("Input to memory connectivity must be in [1, memory_units].")
        self.input_memory_connectivity = input_memory_connectivity
        if input_non_linear_connectivity > non_linear_units or input_non_linear_connectivity < 1:
            raise ValueError("Input to non linear connectivity must be in [1, non_linear_units].")
        self.input_non_linear_connectivity = input_non_linear_connectivity
        if non_linear_connectivity > non_linear_units or non_linear_connectivity < 1:
            raise ValueError("Non linear connectivity must be in [1, non_linear_units].")

        # Input to memory reservoir connectivity
        self.input_memory_kernel = sparse_tensor_init(input_units, memory_units,
                                                      C=input_memory_connectivity) * input_memory_scaling
        self.input_memory_kernel = torch.nn.Parameter(self.input_memory_kernel, requires_grad=False)

        # Input to non-linear reservoir connectivity
        self.input_non_linear_kernel = sparse_tensor_init(input_units, non_linear_units,
                                                          C=input_non_linear_connectivity) * input_non_linear_scaling
        self.input_non_linear_kernel = torch.nn.Parameter(self.input_non_linear_kernel, requires_grad=False)

        # Non-linear reservoir connectivity
        if circular_non_linear_kernel:
            W = circular_tensor_init(non_linear_units, distribution=distribution)
        else:
            W = sparse_recurrent_tensor_init(non_linear_units, C=non_linear_units, distribution=distribution)

        # re-scale the weight matrix to control the effective spectral radius of the linearized system
        if effective_rescaling and leaky_rate != 1:
            I = sparse_eye_init(non_linear_units)
            W = W * leaky_rate + (I * (1 - leaky_rate))
            W = spectral_norm_scaling(W, spectral_radius)
            self.non_linear_kernel = (W + I * (leaky_rate - 1)) * (1 / leaky_rate)
        else:
            if distribution == 'normal':
                W = spectral_radius * W  # NB: W was already rescaled to 1 (circular_non_linear law)
            elif distribution == 'uniform' and non_linear_connectivity == non_linear_units:  # fully connected uniform
                W = fast_spectral_rescaling(W, spectral_radius)
            else:  # sparse connections uniform
                W = spectral_norm_scaling(W, spectral_radius)
            self.non_linear_kernel = W
        self.non_linear_kernel = torch.nn.Parameter(self.non_linear_kernel, requires_grad=False)

        # Memory reservoir connectivity
        self.memory_kernel = circular_tensor_init(memory_units, distribution='fixed')
        self.memory_kernel = torch.nn.Parameter(self.memory_kernel, requires_grad=False)

        # Memory to non-linear reservoir connectivity
        self.memory_non_linear_kernel = sparse_tensor_init(memory_units, non_linear_units,
                                                           C=memory_non_linear_connectivity) * memory_non_linear_scaling
        self.memory_non_linear_kernel = torch.nn.Parameter(self.memory_non_linear_kernel, requires_grad=False)

        if bias:
            if bias_scaling is None:
                self.bias_scaling = input_non_linear_scaling
            else:
                self.bias_scaling = bias_scaling
            # uniform init in [-1, +1] times bias_scaling
            self.bias = (2 * torch.rand(self.non_linear_units) - 1) * self.bias_scaling
            self.bias = torch.nn.Parameter(self.bias, requires_grad=False)
        else:
            # zero bias
            self.bias = torch.zeros(self.non_linear_units)
            self.bias = torch.nn.Parameter(self.bias, requires_grad=False)

        self.non_linearity = non_linearity
        self.non_linearity_function: Callable = torch.tanh if non_linearity == 'tanh' else lambda x: x
        self.memory_state = None
        self.non_linear_state = None

    def forward(self, xt) -> torch.FloatTensor:

        if self.memory_state is None:
            self.memory_state = torch.zeros((xt.shape[0], self.memory_units), dtype=torch.float32, device=xt.device,
                                            requires_grad=False)
        if self.non_linear_state is None:
            self.non_linear_state = torch.zeros((xt.shape[0], self.non_linear_units), dtype=torch.float32,
                                                device=xt.device, requires_grad=False)

        # memory part
        input_memory_part = torch.matmul(xt, self.input_memory_kernel)  # Vx * x(t)
        torch.matmul(self.memory_state, self.memory_kernel, out=self.memory_state)  # Vm * m(t-1)
        self.memory_state.add_(input_memory_part)  # m(t) = Vx * x(t) + Vm * m(t-1)

        # non-linear part
        input_non_linear_part = torch.matmul(xt, self.input_non_linear_kernel)  # Wx * x(t)
        non_linear_part = torch.matmul(self.non_linear_state, self.non_linear_kernel)  # Wh * h(t-1)
        memory_non_linear_part = torch.matmul(self.memory_state, self.memory_non_linear_kernel)  # Wm * m(t)
        # Wx * x(t) + Wh * h(t-1) + Wm * m(t) + b
        combined_input = input_non_linear_part.add_(non_linear_part).add_(memory_non_linear_part).add_(self.bias)

        # h(t) = (1 - alpha) * h(t-1) + alpha * f(Wx * x(t) + Wh * h(t-1) + Wm * m(t) + b)
        self.non_linear_state.mul_(self.one_minus_leaky_rate).add_(self.leaky_rate *
                                                                   self.non_linearity_function(combined_input))

        return self.non_linear_state, self.memory_state


class ReservoirMemoryNetwork(torch.nn.Module):
    def __init__(self,
                 input_units: int,
                 non_linear_units: int,
                 memory_units: int,
                 *,
                 initial_transients: int = 0,
                 input_memory_scaling: float = 1.0,
                 input_non_linear_scaling: float = 1.0,
                 memory_non_linear_scaling: float = 1.0,
                 spectral_radius: float = 0.99,
                 leaky_rate: float = 1.0,
                 input_memory_connectivity: int = 1,
                 input_non_linear_connectivity: int = 1,
                 non_linear_connectivity: int = 1,
                 memory_non_linear_connectivity: int = 1,
                 bias: bool = True,
                 distribution: str = 'uniform',
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 bias_scaling: float | None,
                 circular_non_linear_kernel: bool = True):
        """ Shallow reservoir to be used as a Recurrent Neural Network layer.

        :param input_units: Number of input non_linear_units.
        :param non_linear_units: Number of recurrent neurons in the reservoir.
        :param leaky_rate:
        :param input_non_linear_connectivity:
        :param non_linear_connectivity:
        :param bias:
        :param distribution:
        :param non_linearity:
        :param input_units: number of input units
        :param non_linear_units: number of recurrent neurons in the reservoir
        :param input_non_linear_scaling: max abs value of a weight in the input-reservoir
            connections. Note that whis value also scales the unitary input bias
        :param spectral_radius: max abs eigenvalue of the recurrent matrix
        """
        super().__init__()
        self.initial_transients = initial_transients
        self.net = RMNCell(input_units,
                           non_linear_units,
                           memory_units,
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
                           circular_non_linear_kernel=circular_non_linear_kernel)

    import torch

    def forward(self, x) -> torch.Tensor:

        non_linear_states = torch.empty((x.shape[0], x.shape[1], self.net.non_linear_units), dtype=torch.float32,
                                        device=x.device, requires_grad=False)
        memory_states = torch.empty((x.shape[0], x.shape[1], self.net.memory_units), dtype=torch.float32,
                                    device=x.device, requires_grad=False)

        is_dim_2 = x.dim() == 2

        with torch.no_grad():
            for t in range(x.shape[1]):
                xt = x[:, t].unsqueeze(1) if is_dim_2 else x[:, t]
                non_linear_state, memory_state = self.net(xt)
                non_linear_states[:, t, :].copy_(non_linear_state)
                memory_states[:, t, :].copy_(memory_state)

        non_linear_states = non_linear_states[:, self.initial_transients:, :]
        memory_states = memory_states[:, self.initial_transients:, :]

        return non_linear_states, memory_states

    def reset_state(self):
        self.net.non_linear_state = None
        self.net.memory_state = None
