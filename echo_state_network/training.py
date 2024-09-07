import torch

from tqdm import tqdm

from echo_state_network.esn import EchoStateNetwork
from training_method import TrainingMethod


def train_esn(device: torch.device, dataset: torch.utils.data.DataLoader, training_method: TrainingMethod,
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
              bias_scaling: float | None) -> EchoStateNetwork:
    model = EchoStateNetwork(input_units, recurrent_units, bias_scaling=bias_scaling,
                             input_scaling=input_scaling,
                             spectral_radius=spectral_radius,
                             leaky_rate=leaky_rate,
                             input_connectivity=input_connectivity,
                             recurrent_connectivity=recurrent_connectivity,
                             bias=bias,
                             distribution=distribution,
                             non_linearity=non_linearity,
                             effective_rescaling=effective_rescaling).to(device)

    with torch.no_grad():
        states, ys = [], []
        for x, y in tqdm(dataset, desc="Training Progress"):
            x, y = x.to(device), y.to(device)
            state = model(x)
            states.append(state)
            ys.append(y)
        states = torch.cat(states, dim=0)
        ys = torch.cat(ys, dim=0)
        training_method.fit(states, ys)
        model.initialize_readout_weights(training_method.weights)

    return model
