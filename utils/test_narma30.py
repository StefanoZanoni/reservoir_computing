import numpy as np
import torch

from datasets import Narma30
from echo_state_network import DeepEchoStateNetwork
from reservoir_memory_network import DeepReservoirMemoryNetwork
from utils.save_results import save_results


def nrmse(y_pred: np.ndarray[np.float32], y_true: np.ndarray[np.float32]) -> float:
    # Normalized Root Mean Squared Error
    # where the normalization is done by the root-mean-square of the target trajectory
    return np.sqrt(np.mean((y_pred - y_true) ** 2)) / np.sqrt(np.mean(y_true ** 2))


def test_narma30(model: DeepEchoStateNetwork | DeepReservoirMemoryNetwork, use_last_state: bool, device: torch.device,
                 initial_transients: int) -> tuple[float, float]:
    # training
    training_data = Narma30(training=True)
    training_data.target = training_data.target[:, initial_transients:]
    training_dataloader = torch.utils.data.DataLoader(training_data,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      drop_last=False)
    model.fit(training_dataloader, device, standardize=False, use_last_state=use_last_state,
              disable_progress_bar=False)

    # validation
    validation_data = Narma30(validation=True)
    validation_data.target = validation_data.target[:, initial_transients:]
    validation_dataloader = torch.utils.data.DataLoader(validation_data,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        drop_last=False)
    validation_score = model.score(validation_dataloader, nrmse, device, standardize=False,
                                   use_last_state=use_last_state, disable_progress_bar=False)

    # test
    test_data = Narma30(test=True)
    test_data.target = test_data.target[:, initial_transients:]
    testing_dataloader = torch.utils.data.DataLoader(test_data,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     drop_last=False)
    test_score = model.score(testing_dataloader, nrmse, device, standardize=False, use_last_state=use_last_state,
                             disable_progress_bar=False)

    return validation_score, test_score
