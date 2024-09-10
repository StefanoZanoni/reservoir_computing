import torch

from echo_state_network import EchoStateNetwork


class DeepEchoStateNetwork(torch.nn.Module):
    def __init__(self,
                 input_units: int,
                 total_units: int,
                 *,
                 number_of_layers: int = 1,
                 initial_transients: int = 0,
                 input_scaling: float = 1.0,
                 inter_scaling: float = 1.0,
                 spectral_radius: float = 0.99,
                 leaky_rate: float = 1.0,
                 input_connectivity: int = 1,
                 recurrent_connectivity: int = 1,
                 inter_connectivity: int = 1,
                 bias: bool = True,
                 distribution: str = 'uniform',
                 non_linearity: str = 'tanh',
                 effective_rescaling: bool = True,
                 bias_scaling: float | None,
                 concatenate: bool = False,
                 circular_recurrent_kernel: bool = True):

        super().__init__()
        self.number_of_layers = number_of_layers
        self.total_units = total_units
        self.concatenate = concatenate
        self.batch_first = True  # DeepReservoir only supports batch_first

        # in case in which all the reservoir layers are concatenated, each level
        # contains units/layers neurons. This is done to keep the number of
        # state variables projected to the next layer fixed,
        # i.e., the number of trainable parameters does not depend on concatenate_non_linear
        if concatenate:
            self.recurrent_units = np.int(total_units / number_of_layers)
        else:
            self.recurrent_units = total_units

        # creates a list of reservoirs
        # the first:
        reservoir_layers = [
            EchoStateNetwork(
                input_units=input_units,
                recurrent_units=self.recurrent_units,
                initial_transients=initial_transients,
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
                circular_recurrent_kernel=circular_recurrent_kernel
            )
        ]

        # all the others:
        # last_h_size may be different for the first layer
        # because of the remainder if concatenate_non_linear=True
        last_h_size = self.recurrent_units + total_units % number_of_layers
        for _ in range(number_of_layers - 1):
            reservoir_layers.append(
                EchoStateNetwork(
                    last_h_size,
                    self.recurrent_units,
                    input_scaling=inter_scaling,
                    spectral_radius=spectral_radius,
                    leaky_rate=leaky_rate,
                    input_connectivity=inter_connectivity,
                    recurrent_connectivity=recurrent_connectivity,
                    bias=bias,
                    distribution=distribution,
                    non_linearity=non_linearity,
                    effective_rescaling=effective_rescaling,
                    bias_scaling=bias_scaling,
                    circular_recurrent_kernel=circular_recurrent_kernel
                )
            )
            last_h_size = self.recurrent_units
        self.reservoir = torch.nn.ModuleList(reservoir_layers)

    def forward(self, x: torch.Tensor):
        """ Compute the output of the deep reservoir.

        :param x: Input tensor

        :return: hidden states, last state
        """

        # list of all the states in all the layers
        states = []
        # List of the states in all the layers for the last time step.
        # states_last is a list because different layers may have different sizes.
        states_last = []

        layer_input = x.clone()

        for res_idx, reservoir_layer in enumerate(self.reservoir):
            state = reservoir_layer(layer_input)
            states.append(state)
            states_last.append(state[:, -1, :])
            layer_input = state

        if self.concatenate:
            states = torch.cat(states, dim=2)
        else:
            states = states[-1]

        return states, states_last

    def reset_state(self):
        for reservoir in self.reservoir:
            reservoir.reset_state()
