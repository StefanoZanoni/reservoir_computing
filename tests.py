import os
import json
import torch
import numpy as np

from argparse import ArgumentParser

from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sympy.physics.units import action
from tqdm import tqdm
from torch.utils.data import random_split

from echo_state_network import EchoStateNetwork
from datasets import SequentialMNIST


if __name__ == '__main__':

    # select device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    # parse arguments
    parser = ArgumentParser()

    parser.add_argument('--dataset', type=str, default='sequential_mnist', help='Dataset to use')
    parser.add_argument('--model', type=str, default='esn', help='Model to use')
    parser.add_argument('--validation_percentage', type=float, default=0.2, help='Validation percentage')
    parser.add_argument('--batch_training', type=int, default=1, help='Training batch size')
    parser.add_argument('--batch_validation', type=int, default=1, help='Validation batch size')
    parser.add_argument('--batch_testing', type=int, default=1, help='Testing batch size')
    parser.add_argument('--input_units', type=int, default=1, help='Number of input units')
    parser.add_argument('--recurrent_units', type=int, default=1, help='Number of recurrent units')
    parser.add_argument('--input_scaling', type=float, default=1.0, help='Input scaling')
    parser.add_argument('--spectral_radius', type=float, default=0.99, help='Spectral radius')
    parser.add_argument('--leaky_rate', type=float, default=1.0, help='Leaky rate')
    parser.add_argument('--input_connectivity', type=int, default=1, help='Input connectivity')
    parser.add_argument('--recurrent_connectivity', type=int, default=1, help='Recurrent connectivity')
    parser.add_argument('--bias', action='store_true', help='Whether to use bias or not')
    parser.add_argument('--distribution', type=str, default='uniform', help='Weights distribution to use')
    parser.add_argument('--non_linearity', type=str, default='tanh', help='Non-linearity to use')
    parser.add_argument('--effective_rescaling', type=bool, default=True,
                        help='Whether to use effective rescaling or not')
    parser.add_argument('--bias_scaling', type=float, default=None, help='Bias scaling')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha value for Ridge Regression')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations for Ridge Regression')
    parser.add_argument('--initial_transients', type=int, default=1, help='Number of initial transients')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='Tolerance for Ridge Regression')

    args = parser.parse_args()

    # set arguments
    dataset_name = args.dataset
    model_name = args.model
    validation_percentage = args.validation_percentage
    training_batch_size = args.batch_validation
    validation_batch_size = args.batch_validation
    testing_batch_size = args.batch_testing
    input_units = args.input_units
    recurrent_units = args.recurrent_units
    input_scaling = args.input_scaling
    spectral_radius = args.spectral_radius
    leaky_rate = args.leaky_rate
    input_connectivity = args.input_connectivity
    recurrent_connectivity = args.recurrent_connectivity
    bias = args.bias
    distribution = args.distribution
    non_linearity = args.non_linearity
    effective_rescaling = args.effective_rescaling
    bias_scaling = args.bias_scaling
    alpha = args.alpha
    max_iter = args.max_iter
    initial_transients = args.initial_transients
    tolerance = args.tolerance

    hyperparameters = {'validation_percentage': validation_percentage,
                       'training_batch_size': training_batch_size,
                       'validation_batch_size': validation_batch_size,
                       'testing_batch_size': testing_batch_size,
                       'input_units': input_units,
                       'recurrent_units': recurrent_units,
                       'input_scaling': input_scaling,
                       'spectral_radius': spectral_radius,
                       'leaky_rate': leaky_rate,
                       'input_connectivity': input_connectivity,
                       'recurrent_connectivity': recurrent_connectivity,
                       'bias': bias,
                       'distribution': distribution,
                       'non_linearity': non_linearity,
                       'effective_rescaling': effective_rescaling,
                       'bias_scaling': bias_scaling,
                       'alpha': alpha,
                       'max_iter': max_iter,
                       'initial_transients': initial_transients,
                       'tolerance': tolerance}

    trainer = RidgeClassifier(alpha=alpha, max_iter=max_iter, tol=tolerance)

    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists(f'./results/{model_name}'):
        os.makedirs(f'./results/{model_name}')
    if not os.path.exists(f'./results/{model_name}/{dataset_name}'):
        os.makedirs(f'./results/{model_name}/{dataset_name}')

    # choose model
    if model_name == 'esn':
        model = EchoStateNetwork(input_units, recurrent_units, initial_transients=initial_transients,
                                 input_scaling=input_scaling, spectral_radius=spectral_radius, leaky_rate=leaky_rate,
                                 input_connectivity=input_connectivity, recurrent_connectivity=recurrent_connectivity,
                                 bias=bias, distribution=distribution, non_linearity=non_linearity,
                                 effective_rescaling=effective_rescaling, bias_scaling=bias_scaling).to(device)

    # choose a task
    if dataset_name == 'sequential_mnist':
        data = SequentialMNIST(training=True, normalize=True)
        total_size = len(data)
        val_size = int(0.2 * total_size)
        train_size = total_size - val_size
        training_dataset, validation_dataset = random_split(data, [train_size, val_size])
        training_dataloader = torch.utils.data.DataLoader(training_dataset,
                                                          batch_size=training_batch_size,
                                                          shuffle=True,
                                                          drop_last=True)

        with torch.no_grad():
            states, ys = [], []
            for i, (x, y) in enumerate(tqdm(training_dataloader, desc="Training Progress")):
                x, y = x.to(device), y.to(device)
                state = model(x)[:, -400, :]
                states.append(state.cpu().numpy())
                ys.append(y.cpu().numpy())
            # Concatenate the states and targets along the batch dimension
            states = np.concatenate(states, axis=0)
            ys = np.concatenate(ys, axis=0)

            # Flatten the states tensor to combine time series length and batch dimensions
            states = states.reshape(-1, states.shape[-1])
            # Repeat the targets to match the number of time steps
            ys = np.repeat(ys, states.shape[0] // ys.shape[0])
            scaler = StandardScaler().fit(states)
            states = scaler.transform(states)

            trainer.fit(states, ys)

        validation_dataloader = torch.utils.data.DataLoader(validation_dataset,
                                                            batch_size=validation_batch_size,
                                                            shuffle=True,
                                                            drop_last=True)

        # Validation
        model.reset_state()
        with torch.no_grad():
            states, targets = [], []
            for i, (x, y) in enumerate(tqdm(validation_dataloader, desc="Validation Progress")):
                x, y = x.to(device), y.to(device)
                state = model(x)[:, -400, :]
                states.append(state.cpu().numpy())
                targets.append(y.cpu().numpy())
            states = np.concatenate(states, axis=0)
            targets = np.concatenate(targets, axis=0)

            # Flatten the states tensor to combine time series length and batch dimensions
            states = states.reshape(-1, states.shape[-1])
            # Repeat the targets to match the number of time steps
            ys = np.repeat(ys, states.shape[0] // ys.shape[0])

            states = scaler.transform(states)
            accuracy = trainer.score(states, targets) * 100

            try:
                with open(f'./results/{model_name}/{dataset_name}/score.json', 'r') as f:
                    score = json.load(f)
            except FileNotFoundError:
                score = {'validation_accuracy': 0.0}

            if accuracy > score['validation_accuracy']:
                # Save the hyperparameters and the accuracy
                score = {'validation_accuracy': accuracy}
                with open(f'./results/{model_name}/{dataset_name}/hyperparameters.json', 'w') as f:
                    json.dump(hyperparameters, f, indent=4)
                with open(f'./results/{model_name}/{dataset_name}/score.json', 'w') as f:
                    json.dump(score, f)

        data = SequentialMNIST(training=False, normalize=True)
        testing_dataset = torch.utils.data.DataLoader(data,
                                                      batch_size=testing_batch_size,
                                                      shuffle=True,
                                                      drop_last=True)

        # Testing
        # model.reset_state()
        # with torch.no_grad():
        #     states, targets = [], []
        #     for i, (x, y) in enumerate(tqdm(testing_dataset, desc="Testing Progress")):
        #         x, y = x.to(device), y.to(device)
        #         state = model(x)[:, -300, :]
        #         states.append(state.cpu().numpy())
        #         targets.append(y.cpu().numpy())
        #     states = np.concatenate(states, axis=0)
        #     targets = np.concatenate(targets, axis=0)
        #
        #     # Flatten the states tensor to combine time series length and batch dimensions
        #     states = states.reshape(-1, states.shape[-1])
        #     # Repeat the targets to match the number of time steps
        #     ys = np.repeat(ys, states.shape[0] // ys.shape[0])
        #
        #     states = scaler.transform(states)
        #     accuracy = trainer.score(states, targets) * 100
        #
        #     score['test_accuracy'] = accuracy
        #     with open(f'./results/{model_name}/{dataset_name}/score.json', 'w') as f:
        #         json.dump(score, f)
