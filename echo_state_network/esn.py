import torch
import torch.nn
from numpy import dtype
import numpy as np

from .utils.initialization import (sparse_tensor_init, sparse_recurrent_tensor_init, spectral_norm_scaling,
                                   sparse_eye_init, fast_spectral_rescaling, circular_tensor_init)


class ReservoirCell(torch.nn.Module):
    def __init__(self,
                 input_units: int,
                 recurrent_units: int,
                 *,
                 input_scaling: float = 1.0,
                 spectral_radius: float = 0.99,
                 leaky_rate: float = 1.0,
                 input_connectivity: int = 1,
                 recurrent_connectivity: int = 1,
                 bias: bool = True,
                 distribution: str = 'uniform',
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 bias_scaling: float | None,
                 circular_recurrent_kernel: bool = True):

        """
        Shallow reservoir to be used as a cell of a Recurrent Neural Network.


        """

        super().__init__()

        if input_units < 1:
            raise ValueError("Input units must be greater than 0.")
        self.input_units = input_units
        if recurrent_units < 1:
            raise ValueError("Recurrent units must be greater than 0.")
        self.recurrent_units = recurrent_units
        self.input_scaling = input_scaling
        if spectral_radius > 1 or spectral_radius < 0:
            raise ValueError("Spectral radius must be in [0, 1].")
        self.spectral_radius = spectral_radius
        if leaky_rate > 1 or leaky_rate < 0:
            raise ValueError("Leaky rate must be in [0, 1].")
        self.leaky_rate = leaky_rate
        self.input_connectivity = input_connectivity
        if recurrent_connectivity > recurrent_units:
            raise ValueError("Recurrent connectivity must be in [0, recurrent_units].")
        self.recurrent_connectivity = recurrent_connectivity

        self.input_kernel = sparse_tensor_init(input_units, recurrent_units, C=input_connectivity) * input_scaling
        self.input_kernel = torch.nn.Parameter(self.input_kernel, requires_grad=False)

        if circular_recurrent_kernel:
            W = circular_tensor_init(recurrent_units, distribution=distribution)
        else:
            W = sparse_recurrent_tensor_init(recurrent_units, C=recurrent_connectivity, distribution=distribution)

        # re-scale the weight matrix to control the effective spectral radius of the linearized system
        if effective_rescaling and leaky_rate != 1:
            I = sparse_eye_init(recurrent_units)
            W = W * leaky_rate + (I * (1 - leaky_rate))
            W = spectral_norm_scaling(W, spectral_radius)
            self.recurrent_kernel = (W + I * (leaky_rate - 1)) * (1 / leaky_rate)
        else:
            if distribution == 'normal':
                W = spectral_radius * W  # NB: W was already rescaled to 1 (circular law)
            elif distribution == 'uniform' and recurrent_connectivity == recurrent_units:  # fully connected uniform
                W = fast_spectral_rescaling(W, spectral_radius)
            else:  # sparse connections uniform
                W = spectral_norm_scaling(W, spectral_radius)
            self.recurrent_kernel = W
        self.recurrent_kernel = torch.nn.Parameter(self.recurrent_kernel, requires_grad=False)

        if bias:
            if bias_scaling is None:
                self.bias_scaling = input_scaling
            else:
                self.bias_scaling = bias_scaling
            # uniform init in [-1, +1] times bias_scaling
            self.bias = (2 * torch.rand(self.recurrent_units) - 1) * self.bias_scaling
            self.bias = torch.nn.Parameter(self.bias, requires_grad=False)
        else:
            # zero bias
            self.bias = torch.zeros(self.recurrent_units)
            self.bias = torch.nn.Parameter(self.bias, requires_grad=False)

        self.non_linearity = non_linearity
        self.state = None

    def forward(self, xt) -> torch.FloatTensor:

        """ Computes the output of the cell given the input and previous state.

        :param xt: The input at time t.
        """

        if self.state is None:
            self.state = (torch.zeros((xt.shape[0], self.recurrent_units), dtype=torch.float32, requires_grad=False)
                          .to(xt.device))

        input_part = torch.matmul(xt, self.input_kernel)
        state_part = torch.matmul(self.state, self.recurrent_kernel)

        if self.non_linearity == 'identity':
            output = input_part + self.bias + state_part
        elif self.non_linearity == 'tanh':
            output = torch.tanh(input_part + self.bias + state_part)
        else:
            raise ValueError("Invalid non linearity <<" + self.non_linearity + ">>. Only tanh and identity allowed.")

        self.state = self.state * (1 - self.leaky_rate) + output * self.leaky_rate
        return self.state


class EchoStateNetwork(torch.nn.Module):
    def __init__(self,
                 input_units: int,
                 recurrent_units: int,
                 *,
                 initial_transients: int = 0,
                 input_scaling: float = 1.0,
                 spectral_radius: float = 0.99,
                 leaky_rate: float = 1.0,
                 input_connectivity: int = 1,
                 recurrent_connectivity: int = 1,
                 bias: bool = True,
                 distribution: str = 'uniform',
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 bias_scaling: float | None,
                 circular_recurrent_kernel: bool = True):

        """ Shallow reservoir to be used as a Recurrent Neural Network layer.

        :param input_units: Number of input recurrent_units.
        :param recurrent_units: Number of recurrent neurons in the reservoir.
        :param leaky_rate:
        :param input_connectivity:
        :param recurrent_connectivity:
        :param bias:
        :param distribution:
        :param non_linearity:
        :param input_units: number of input units
        :param recurrent_units: number of recurrent neurons in the reservoir
        :param input_scaling: max abs value of a weight in the input-reservoir
            connections. Note that whis value also scales the unitary input bias
        :param spectral_radius: max abs eigenvalue of the recurrent matrix
        """
        super().__init__()
        self.initial_transients = initial_transients
        self.net = ReservoirCell(input_units,
                                 recurrent_units,
                                 input_scaling=input_scaling,
                                 spectral_radius=spectral_radius,
                                 leaky_rate=leaky_rate,
                                 input_connectivity=input_connectivity,
                                 recurrent_connectivity=recurrent_connectivity,
                                 bias=bias,
                                 distribution=distribution,
                                 non_linearity=non_linearity,
                                 effective_rescaling=effective_rescaling,
                                 bias_scaling=bias_scaling,
                                 circular_recurrent_kernel=circular_recurrent_kernel)

    def forward(self, x) -> torch.Tensor:

        """
        Computes the output of the cell given the input and previous state.

        :param x: The input time series

        :return: Hidden states for each time step
        """

        states = torch.empty((x.shape[0], 0, self.net.recurrent_units), dtype=torch.float32).to(x.device)
        for t in range(x.shape[1]):
            xt = x[:, t].unsqueeze(1) if x.dim() == 2 else x[:, t]
            state = self.net(xt)
            state = state.unsqueeze(1)
            states = torch.cat((states, state), dim=1)
        states = states[:, self.initial_transients:, :]
        return states

    def reset_state(self):
        self.net.state = None
