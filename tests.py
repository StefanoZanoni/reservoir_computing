import os
import json
import random

import torch
import numpy as np

from argparse import ArgumentParser

from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import random_split

from tqdm import tqdm

from echo_state_network import DeepEchoStateNetwork
from reservoir_memory_network import DeepReservoirMemoryNetwork
from datasets import SequentialMNIST, MemoryCapacity

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

    # general arguments
    parser.add_argument('--cpu', action='store_true', help='Force to use the CPU')
    parser.add_argument('--dataset', type=str, default='sequential_mnist', help='Dataset to use')
    parser.add_argument('--model', type=str, default='esn', help='Model to use')

    # dataset arguments
    parser.add_argument('--validation_percentage', type=float, default=0.2, help='Validation percentage')
    parser.add_argument('--batch_training', type=int, default=1, help='Training batch size')
    parser.add_argument('--batch_validation', type=int, default=1, help='Validation batch size')
    parser.add_argument('--batch_testing', type=int, default=1, help='Testing batch size')

    # number of units
    parser.add_argument('--input_units', type=int, default=1, help='Number of input units')
    parser.add_argument('--non_linear_units', type=int, default=1, help='Number of non linear units')
    parser.add_argument('--memory_units', type=int, default=1, help='Number of memory units')

    # scaling
    parser.add_argument('--non_linear_scaling', type=float, default=1e-2,
                        help='non linear scaling for Euler non linear')
    parser.add_argument('--input_memory_scaling', type=float, default=1.0, help='Input memory scaling')
    parser.add_argument('--input_non_linear_scaling', type=float, default=1.0, help='Input non linear scaling')
    parser.add_argument('--memory_non_linear_scaling', type=float, default=1.0, help='Memory non linear scaling')
    parser.add_argument('--inter_non_linear_scaling', type=float, default=1.0, help='Inter non linear scaling')
    parser.add_argument('--inter_memory_scaling', type=float, default=1.0, help='Inter memory scaling')
    parser.add_argument('--bias_scaling', type=float, default=None, help='Bias scaling')

    # general parameters
    parser.add_argument('--spectral_radius', type=float, default=0.99, help='Spectral radius')
    parser.add_argument('--leaky_rate', type=float, default=1.0, help='Leaky rate')
    parser.add_argument('--bias', action='store_true', help='Whether to use bias or not')
    parser.add_argument('--distribution', type=str, default='uniform', help='Weights distribution to use')
    parser.add_argument('--non_linearity', type=str, default='tanh', help='Non-linearity to use')
    parser.add_argument('--effective_rescaling', action='store_true', help='Whether to use effective rescaling or not')
    parser.add_argument('--euler', action='store_true', help='Whether to use Euler non linear or not')
    parser.add_argument('--epsilon', type=float, default=1e-3, help='Epsilon value for Euler non linear')
    parser.add_argument('--gamma', type=float, default=1e-3, help='Gamma value for Euler non linear')
    parser.add_argument('--circular_non_linear', action='store_true', help='Whether to use ring topology or not')
    parser.add_argument('--legendre_memory', action='store_true', help='Whether to use Legendre memory or not')
    parser.add_argument('--theta', type=float, default=1.0, help='Theta value for Legendre memory')
    parser.add_argument('--use_last_state', action='store_true', help='Whether to use just the last state or not')
    parser.add_argument('--seed', type=int, default=None, help='Seed for the random number generator')

    # connectivity
    parser.add_argument('--input_memory_connectivity', type=int, default=1, help='Input memory connectivity')
    parser.add_argument('--input_non_linear_connectivity', type=int, default=1, help='Input non linear connectivity')
    parser.add_argument('--non_linear_connectivity', type=int, default=1, help='Non linear connectivity')
    parser.add_argument('--memory_non_linear_connectivity', type=int, default=1, help='Memory non linear connectivity')
    parser.add_argument('--inter_non_linear_connectivity', type=int, default=1, help='Inter non linear connectivity')
    parser.add_argument('--inter_memory_connectivity', type=int, default=1, help='Inter memory connectivity')

    # training
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha value for Ridge Regression')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations for Ridge Regression')
    parser.add_argument('--initial_transients', type=int, default=1, help='Number of initial transients')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='Tolerance for Ridge Regression')

    # deep reservoirs
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers for deep reservoirs')
    parser.add_argument('--concatenate_non_linear', action='store_true',
                        help='Whether to concatenate the non linear or not')
    parser.add_argument('--concatenate_memory', action='store_true',
                        help='Whether to concatenate the memory or not')

    args = parser.parse_args()

    # set arguments
    cpu = args.cpu
    if cpu:
        device = torch.device('cpu')
    dataset_name = args.dataset
    model_name = args.model
    validation_percentage = args.validation_percentage
    training_batch_size = args.batch_validation
    validation_batch_size = args.batch_validation
    testing_batch_size = args.batch_testing
    input_units = args.input_units
    non_linear_units = args.non_linear_units
    memory_units = args.memory_units
    input_non_linear_scaling = args.input_non_linear_scaling
    input_memory_scaling = args.input_memory_scaling
    memory_non_linear_scaling = args.memory_non_linear_scaling
    inter_non_linear_scaling = args.inter_non_linear_scaling
    inter_memory_scaling = args.inter_memory_scaling
    spectral_radius = args.spectral_radius
    leaky_rate = args.leaky_rate
    input_non_linear_connectivity = args.input_non_linear_connectivity
    non_linear_connectivity = args.non_linear_connectivity
    memory_non_linear_connectivity = args.memory_non_linear_connectivity
    input_memory_connectivity = args.input_memory_connectivity
    inter_non_linear_connectivity = args.inter_non_linear_connectivity
    inter_memory_connectivity = args.inter_memory_connectivity
    bias = args.bias
    distribution = args.distribution
    non_linearity = args.non_linearity
    effective_rescaling = args.effective_rescaling
    bias_scaling = args.bias_scaling
    alpha = args.alpha
    max_iter = args.max_iter
    initial_transients = args.initial_transients
    tolerance = args.tolerance
    number_of_layers = args.num_layers
    concatenate_non_linear = args.concatenate_non_linear
    concatenate_memory = args.concatenate_memory
    circular_non_linear = args.circular_non_linear
    euler = args.euler
    epsilon = args.epsilon
    gamma = args.gamma
    non_linear_scaling = args.non_linear_scaling
    legendre_memory = args.legendre_memory
    theta = args.theta
    use_last_state = args.use_last_state
    seed = args.seed

    if seed:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    if dataset_name == 'sequential_mnist':
        task = 'classification'
    elif dataset_name == 'memory_capacity':
        task = 'regression'

    trainer = RidgeClassifier(alpha=alpha, max_iter=max_iter, tol=tolerance, solver='svd')

    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists(f'./results/{model_name}'):
        os.makedirs(f'./results/{model_name}')
    if not os.path.exists(f'./results/{model_name}/{dataset_name}'):
        os.makedirs(f'./results/{model_name}/{dataset_name}')

    # choose model
    if model_name == 'esn':
        hyperparameters = {'validation_percentage': validation_percentage,
                           'training_batch_size': training_batch_size,
                           'validation_batch_size': validation_batch_size,
                           'testing_batch_size': testing_batch_size,
                           'input_units': input_units,
                           'non_linear_units': non_linear_units,
                           'input_non_linear_scaling': input_non_linear_scaling,
                           'inter_non_linear_scaling': inter_non_linear_scaling,
                           'spectral_radius': spectral_radius,
                           'leaky_rate': leaky_rate,
                           'input_non_linear_connectivity': input_non_linear_connectivity,
                           'non_linear_connectivity': non_linear_connectivity,
                           'inter_non_linear_connectivity': inter_non_linear_connectivity,
                           'bias': bias,
                           'distribution': distribution,
                           'non_linearity': non_linearity,
                           'effective_rescaling': effective_rescaling,
                           'bias_scaling': bias_scaling,
                           'alpha': alpha,
                           'max_iter': max_iter,
                           'initial_transients': initial_transients,
                           'tolerance': tolerance,
                           'number_of_layers': number_of_layers,
                           'concatenate_non_linear': concatenate_non_linear,
                           'circular_non_linear': circular_non_linear,
                           'euler': euler,
                           'epsilon': epsilon,
                           'gamma': gamma,
                           'non_linear_scaling': non_linear_scaling}

        model = DeepEchoStateNetwork(task,
                                     input_units,
                                     non_linear_units,
                                     number_of_layers=number_of_layers,
                                     initial_transients=initial_transients,
                                     input_scaling=input_non_linear_scaling,
                                     inter_scaling=inter_non_linear_scaling,
                                     spectral_radius=spectral_radius,
                                     leaky_rate=leaky_rate,
                                     input_connectivity=input_non_linear_connectivity,
                                     recurrent_connectivity=non_linear_connectivity,
                                     inter_connectivity=inter_non_linear_connectivity,
                                     bias=bias,
                                     distribution=distribution,
                                     non_linearity=non_linearity,
                                     effective_rescaling=effective_rescaling,
                                     bias_scaling=bias_scaling,
                                     concatenate=concatenate_non_linear,
                                     circular_recurrent_kernel=circular_non_linear,
                                     euler=euler,
                                     epsilon=epsilon,
                                     gamma=gamma,
                                     recurrent_scaling=non_linear_scaling,
                                     alpha=alpha,
                                     max_iter=max_iter,
                                     tolerance=tolerance).to(device)
    elif model_name == 'rmn':
        hyperparameters = {'validation_percentage': validation_percentage,
                           'training_batch_size': training_batch_size,
                           'validation_batch_size': validation_batch_size,
                           'testing_batch_size': testing_batch_size,
                           'input_units': input_units,
                           'non_linear_units': non_linear_units,
                           'memory_units': memory_units,
                           'input_memory_scaling': input_memory_scaling,
                           'input_non_linear_scaling': input_non_linear_scaling,
                           'memory_non_linear_scaling': memory_non_linear_scaling,
                           'spectral_radius': spectral_radius,
                           'leaky_rate': leaky_rate,
                           'input_memory_connectivity': input_memory_connectivity,
                           'input_non_linear_connectivity': input_non_linear_connectivity,
                           'non_linear_connectivity': non_linear_connectivity,
                           'memory_non_linear_connectivity': memory_non_linear_connectivity,
                           'bias': bias,
                           'distribution': distribution,
                           'non_linearity': non_linearity,
                           'effective_rescaling': effective_rescaling,
                           'bias_scaling': bias_scaling,
                           'alpha': alpha,
                           'max_iter': max_iter,
                           'initial_transients': initial_transients,
                           'tolerance': tolerance,
                           'number_of_layers': number_of_layers,
                           'concatenate_non_linear': concatenate_non_linear,
                           'concatenate_memory': concatenate_memory,
                           'circular_non_linear': circular_non_linear,
                           'euler': euler,
                           'epsilon': epsilon,
                           'gamma': gamma,
                           'non_linear_scaling': non_linear_scaling,
                           'legendre_memory': legendre_memory,
                           'theta': theta}
        model = DeepReservoirMemoryNetwork(task,
                                           input_units,
                                           non_linear_units,
                                           memory_units,
                                           number_of_layers=number_of_layers,
                                           initial_transients=initial_transients,
                                           input_memory_scaling=input_memory_scaling,
                                           input_non_linear_scaling=input_non_linear_scaling,
                                           memory_non_linear_scaling=memory_non_linear_scaling,
                                           spectral_radius=spectral_radius,
                                           leaky_rate=leaky_rate,
                                           input_memory_connectivity=input_memory_connectivity,
                                           input_non_linear_connectivity=input_non_linear_connectivity,
                                           non_linear_connectivity=non_linear_connectivity,
                                           memory_non_linear_connectivity=memory_non_linear_connectivity,
                                           bias=bias,
                                           distribution=distribution,
                                           non_linearity=non_linearity,
                                           effective_rescaling=effective_rescaling,
                                           bias_scaling=bias_scaling,
                                           concatenate_non_linear=concatenate_non_linear,
                                           concatenate_memory=concatenate_memory,
                                           circular_non_linear_kernel=circular_non_linear,
                                           alpha=alpha,
                                           max_iter=max_iter,
                                           tolerance=tolerance,
                                           legendre=legendre_memory,
                                           theta=theta).to(device)

    # choose a task
    if dataset_name == 'sequential_mnist':
        data = SequentialMNIST(training=True)
        total_size = len(data)
        val_size = int(0.2 * total_size)
        train_size = total_size - val_size
        training_dataset, validation_dataset = random_split(data, [train_size, val_size])
        training_dataloader = torch.utils.data.DataLoader(training_dataset,
                                                          batch_size=training_batch_size,
                                                          shuffle=True,
                                                          drop_last=True)

        # Training
        model.fit(training_dataloader, device, standardize=True, use_last_state=use_last_state)

        # Validation
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset,
                                                            batch_size=validation_batch_size,
                                                            shuffle=True,
                                                            drop_last=True)
        model.reset_state()
        accuracy = model.score(validation_dataloader, device, standardize=True, use_last_state=use_last_state) * 100

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
    elif dataset_name == 'memory_capacity':
        if model_name == 'esn':
            max_delay = non_linear_units * 2
        if model_name == 'rmn':
            max_delay = memory_units * 2

        mcs = []
        for run in range(10):
            mc_ks = []
            for k in tqdm(range(max_delay), 'Delay'):
                k += 1  # k starts from 1
                training_data = MemoryCapacity(k, training=True)
                test_data = MemoryCapacity(k, training=False)
                training_dataloader = torch.utils.data.DataLoader(training_data,
                                                                  batch_size=1,
                                                                  shuffle=False,
                                                                  drop_last=False)
                test_dataloader = torch.utils.data.DataLoader(test_data,
                                                              batch_size=1,
                                                              shuffle=False,
                                                              drop_last=False)
                model.fit(training_dataloader, device, standardize=True, use_last_state=use_last_state,
                          show_progress_bar=False)
                model.reset_state()
                predictions = (model.predict(test_dataloader, device, standardize=True, use_last_state=use_last_state,
                                             show_progress_bar=False)).reshape(-1)
                target_mean = test_data.target.mean().cpu().numpy()
                predictions_mean = predictions.mean()
                numerator = np.sum((test_data.target.cpu().numpy() - target_mean) * (predictions - predictions_mean)) ** 2
                denominator_target_t = np.sum((test_data.target.cpu().numpy() - target_mean) ** 2)
                denominator_prediction_t = np.sum((predictions - predictions_mean) ** 2)
                mc_k = numerator / (denominator_target_t * denominator_prediction_t)
                mc_ks.append(mc_k)

            mcs.append(float(sum(mc_ks)))

        score = {'mean_memory_capacity': float(np.mean(mcs)),
                 'std_memory_capacity': float(np.std(mcs))}
        with open(f'./results/{model_name}/{dataset_name}/hyperparameters.json', 'w') as f:
            json.dump(hyperparameters, f, indent=4)
        with open(f'./results/{model_name}/{dataset_name}/score.json', 'w') as f:
            json.dump(score, f, indent=4)
