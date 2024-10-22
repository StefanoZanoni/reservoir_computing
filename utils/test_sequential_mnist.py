import torch
import numpy as np

from datasets import SequentialMNIST
from torch.utils.data import random_split

from utils.save_results import save_results
from echo_state_network import DeepEchoStateNetwork
from reservoir_memory_network import DeepReservoirMemoryNetwork


def accuracy(y_pred: np.ndarray[np.float32], y_true: np.ndarray[np.float32]) -> float:
    return (y_pred.flatten() == y_true.flatten()).sum().item() / len(y_true) * 100


def test_sequential_mnist(runs: int, model: DeepEchoStateNetwork | DeepReservoirMemoryNetwork, results_path: str,
                          hyperparameters: dict, validation_ratio: float, training_batch_size: int,
                          validation_batch_size: int, testing_batch_size: int, use_last_state: bool,
                          device: torch.device):

    data = SequentialMNIST(training=True, normalize=True)
    total_size = len(data)
    val_size = int(validation_ratio * total_size)
    train_size = total_size - val_size
    training_dataset, validation_dataset = random_split(data, [train_size, val_size])
    training_dataloader = torch.utils.data.DataLoader(training_dataset,
                                                      batch_size=training_batch_size,
                                                      shuffle=True,
                                                      drop_last=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset,
                                                        batch_size=validation_batch_size,
                                                        shuffle=True,
                                                        drop_last=True)

    model.fit(training_dataloader, device, standardize=True, use_last_state=use_last_state,
              disable_progress_bar=False)
    validation_scores = []
    for _ in range(runs):
        validation_score = model.score(validation_dataloader, accuracy, device, standardize=True,
                                       use_last_state=use_last_state, disable_progress_bar=False)
        validation_scores.append(validation_score)

    data = SequentialMNIST(training=False, normalize=True)
    testing_dataset = torch.utils.data.DataLoader(data,
                                                  batch_size=testing_batch_size,
                                                  shuffle=True,
                                                  drop_last=True)

    test_scores = []
    for _ in range(runs):
        test_score = model.score(testing_dataset, accuracy, device, standardize=True, use_last_state=use_last_state,
                                 disable_progress_bar=False)
        test_scores.append(test_score)

    save_results(results_path, hyperparameters, np.mean(validation_score), np.std(validation_scores),
                 np.mean(test_score), np.std(test_scores), 'accuracy', 'greater')
