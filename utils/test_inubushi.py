import torch
import numpy as np
import csv

from tqdm import tqdm
from utils.save_results import save_results
from datasets import Inubushi
from echo_state_network import DeepEchoStateNetwork
from reservoir_memory_network import DeepReservoirMemoryNetwork


def nrmse(y_pred: np.ndarray[np.float32], y_true: np.ndarray[np.float32]) -> float:
    return np.sqrt(np.mean((y_pred - y_true) ** 2)) / (np.max(y_true) - np.min(y_true))


def test_inubushi(runs: int, v: float, results_path: str, hyperparameters: dict,
                  model: DeepEchoStateNetwork | DeepReservoirMemoryNetwork, max_delay: int, device: str,
                  use_last_state: bool, initial_transients: int) -> None:
    validation_scores = []
    test_scores = []
    for _ in range(runs):
        nrmse_validation = []
        nrmse_test = []
        for k in tqdm(range(max_delay), 'Delay', disable=False):
            k += 1  # k starts from 1
            # training
            training_data = Inubushi(k, v, training=True)
            training_data.target = training_data.target[:, initial_transients:]
            training_dataloader = torch.utils.data.DataLoader(training_data,
                                                              batch_size=1,
                                                              shuffle=False,
                                                              drop_last=False)
            model.fit(training_dataloader, device, standardize=True, use_last_state=use_last_state,
                      disable_progress_bar=True)

            # validation
            validation_data = Inubushi(k, v, training=False)
            validation_data.target = validation_data.target[:, initial_transients:]
            validation_dataloader = torch.utils.data.DataLoader(validation_data,
                                                                batch_size=1,
                                                                shuffle=False,
                                                                drop_last=False)
            validation_score = model.score(validation_dataloader, nrmse, device, standardize=True,
                                           use_last_state=use_last_state, disable_progress_bar=True)
            nrmse_validation.append(validation_score)

            # test
            test_data = Inubushi(k, v, training=False)
            test_data.target = test_data.target[:, initial_transients:]
            test_dataloader = torch.utils.data.DataLoader(test_data,
                                                          batch_size=1,
                                                          shuffle=False,
                                                          drop_last=False)
            test_score = model.score(test_dataloader, nrmse, device, standardize=True,
                                     use_last_state=use_last_state, disable_progress_bar=True)
            nrmse_test.append(test_score)
        validation_scores.append(nrmse_validation)
        test_scores.append(nrmse_test)

    mean_validation_scores = np.mean(validation_scores, axis=0)
    mean_test_scores = np.mean(test_scores, axis=0)
    std_validation_scores = np.std(validation_scores, axis=0)
    std_test_scores = np.std(test_scores, axis=0)

    save_results(results_path, hyperparameters, np.mean(mean_validation_scores), np.mean(std_validation_scores),
                 np.mean(mean_test_scores), np.mean(std_test_scores), 'nrmse', 'less')

    with open(f'{results_path}/nrmse_{v}_v.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['validation_mean_nrmse', ' validation_std_nrmse', 'test_mean_nrmse', 'test_std_nrmse'])
        for i in range(max_delay):
            writer.writerow([mean_validation_scores[i], std_validation_scores[i],
                             mean_test_scores[i], std_test_scores[i]])
