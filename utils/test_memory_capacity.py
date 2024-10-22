import torch
import numpy as np
import csv

from tqdm import tqdm
from utils.save_results import save_results
from datasets import MemoryCapacity
from echo_state_network import DeepEchoStateNetwork
from reservoir_memory_network import DeepReservoirMemoryNetwork


def determination_coefficient(y_pred: np.ndarray[np.float32], y_true: np.ndarray[np.float32]) -> float:
    y_true_mean = np.mean(y_true)
    numerator = np.sum((y_true - y_true_mean) * (y_pred - y_true_mean)) ** 2
    denominator = np.sum((y_true - y_true_mean) ** 2) * np.sum((y_pred - y_true_mean) ** 2)
    return numerator / denominator


def test_memory_capacity(runs: int, results_path, hyperparameters: dict, model: DeepEchoStateNetwork | DeepReservoirMemoryNetwork,
                         max_delay: int, device: str, use_last_state: bool, initial_transients: int) -> None:
    mcs_validation = []
    mcs_test = []
    validation_determination_coefficients = []
    test_determination_coefficients = []
    for _ in range(runs):
        mc_ks_validation = []
        mc_ks_test = []
        for k in tqdm(range(max_delay), 'Delay', disable=False):
            k += 1  # k starts from 1

            # training
            training_data = MemoryCapacity(k, training=True)
            training_data.target = training_data.target[:, initial_transients:]
            training_dataloader = torch.utils.data.DataLoader(training_data,
                                                              batch_size=1,
                                                              shuffle=False,
                                                              drop_last=False)
            model.fit(training_dataloader, device, standardize=True, use_last_state=use_last_state,
                      disable_progress_bar=True)

            # validation
            validation_data = MemoryCapacity(k, training=False)
            validation_data.target = validation_data.target[:, initial_transients:]
            validation_dataloader = torch.utils.data.DataLoader(validation_data,
                                                                batch_size=1,
                                                                shuffle=False,
                                                                drop_last=False)
            predictions = (
                model.predict(validation_dataloader, device, standardize=True,
                              use_last_state=use_last_state, disable_progress_bar=True)).reshape(-1)
            mc_k = determination_coefficient(predictions, validation_data.target.cpu().numpy())
            mc_ks_validation.append(mc_k)

            # test
            test_data = MemoryCapacity(k, training=False)
            test_data.target = test_data.target[:, initial_transients:]
            test_dataloader = torch.utils.data.DataLoader(test_data,
                                                          batch_size=1,
                                                          shuffle=False,
                                                          drop_last=False)
            predictions = (
                model.predict(test_dataloader, device, standardize=True, use_last_state=use_last_state,
                              disable_progress_bar=True)).reshape(-1)
            mc_k = determination_coefficient(predictions, test_data.target.cpu().numpy())
            mc_ks_test.append(mc_k)

        mcs_validation.append(float(sum(mc_ks_validation)))
        mcs_test.append(float(sum(mc_ks_test)))

        validation_determination_coefficients.append(mc_ks_validation)
        test_determination_coefficients.append(mc_ks_test)

        save_results(results_path, hyperparameters, np.mean(mcs_validation), np.std(mcs_validation), np.mean(mcs_test),
                     np.std(mcs_test), 'memory_capacity', 'greater')

        validation_determination_coefficients = np.mean(validation_determination_coefficients, axis=0)
        test_determination_coefficients = np.mean(test_determination_coefficients, axis=0)
        with open(f'{results_path}/determination_coefficients.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Validation Determination Coefficients', 'Test Determination Coefficients'])
            for val, test in zip(validation_determination_coefficients, test_determination_coefficients):
                writer.writerow([val, test])
