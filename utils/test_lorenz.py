import torch
import numpy as np
import csv

from tqdm import tqdm
from utils.save_results import save_results
from datasets import Lorenz96
from echo_state_network import DeepEchoStateNetwork
from reservoir_memory_network import DeepReservoirMemoryNetwork


def nrmse(y_pred: np.ndarray[np.float32], y_true: np.ndarray[np.float32]) -> float:
    # Normalized Root Mean Squared Error
    # where the normalization is done by the root-mean-square of the target trajectory
    y_true_temp = y_true.squeeze()
    return np.sqrt(np.mean((y_pred - y_true_temp) ** 2)) / np.sqrt(np.mean(y_true_temp ** 2) + 1e-9)


def test_lorenz(N: int, F: float, dataset_size: int, lag: int, model: DeepEchoStateNetwork | DeepReservoirMemoryNetwork,
                use_last_state: bool, device: torch.device, initial_transients: int, training_batch_size: int,
                validation_batch_size: int, test_batch_size: int) -> tuple[float, float]:
    # training
    training_data = Lorenz96(N, F, dataset_size, lag)
    training_data.target = training_data.target[:, initial_transients:]
    training_dataloader = torch.utils.data.DataLoader(training_data,
                                                      batch_size=training_batch_size,
                                                      shuffle=True,
                                                      drop_last=True)
    model.fit(training_dataloader, device, standardize=True, use_last_state=use_last_state,
              disable_progress_bar=False)

    # validation
    validation_data = Lorenz96(N, F, dataset_size, lag)
    validation_data.target = validation_data.target[:, initial_transients:]
    validation_dataloader = torch.utils.data.DataLoader(validation_data,
                                                        batch_size=validation_batch_size,
                                                        shuffle=True,
                                                        drop_last=True)
    validation_score = model.score(validation_dataloader, nrmse, device, standardize=True,
                                   use_last_state=use_last_state, disable_progress_bar=False)

    # test
    test_data = Lorenz96(N, F, dataset_size, lag)
    test_data.target = test_data.target[:, initial_transients:]
    testing_dataloader = torch.utils.data.DataLoader(test_data,
                                                     batch_size=test_batch_size,
                                                     shuffle=True,
                                                     drop_last=True)
    test_score = model.score(testing_dataloader, nrmse, device, standardize=True, use_last_state=use_last_state,
                             disable_progress_bar=False)

    return validation_score, test_score
