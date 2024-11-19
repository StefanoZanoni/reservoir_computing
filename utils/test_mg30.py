import numpy as np
import torch

from datasets import MG30
from echo_state_network import DeepEchoStateNetwork
from reservoir_memory_network import DeepReservoirMemoryNetwork
from utils.save_results import save_results


def mse(y_pred: np.ndarray[np.float32], y_true: np.ndarray[np.float32]) -> float:
    return np.mean((y_pred.flatten() - y_true.flatten()) ** 2)


def test_mg30(model: DeepEchoStateNetwork | DeepReservoirMemoryNetwork, use_last_state: bool, device: torch.device,
              initial_transients: int) -> tuple[float, float]:

    # training
    training_data = MG30(training=True)
    training_data.target = training_data.target[:, initial_transients:]
    training_dataloader = torch.utils.data.DataLoader(training_data,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      drop_last=False)
    model.fit(training_dataloader, device, standardize=True, use_last_state=use_last_state,
              disable_progress_bar=False)

    # validation
    validation_data = MG30(validation=True)
    validation_data.target = validation_data.target[:, initial_transients:]
    validation_dataloader = torch.utils.data.DataLoader(validation_data,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        drop_last=False)
    validation_score = model.score(validation_dataloader, mse, device, standardize=True,
                                   use_last_state=use_last_state, disable_progress_bar=False)

    # test
    test_data = MG30(test=True)
    test_data.target = test_data.target[:, initial_transients:]
    testing_dataloader = torch.utils.data.DataLoader(test_data,
                                                     batch_size=1,
                                                     shuffle=False,
                                                     drop_last=False)
    test_score = model.score(testing_dataloader, mse, device, standardize=True, use_last_state=use_last_state,
                             disable_progress_bar=False)

    return validation_score, test_score
