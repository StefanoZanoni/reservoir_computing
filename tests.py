import os
import json
import random
import csv
import time
from textwrap import indent

import torch
import numpy as np

from argparse import ArgumentParser

from tqdm import tqdm

from echo_state_network import DeepEchoStateNetwork
from reservoir_memory_network import DeepReservoirMemoryNetwork
from datasets import SequentialMNIST, MemoryCapacity, MG17
from utils.test_sequential_mnist import test_sequential_mnist
from utils.test_mg17 import test_mg17
from utils.test_memory_capacity import test_memory_capacity

import warnings
import sklearn

# Disable all warnings from sklearn
warnings.filterwarnings("ignore", module="sklearn")

torch.set_num_threads(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())


def generate_results_path(model_name, dataset_name, number_of_layers, non_linear_units, memory_units, euler,
                          legendre_memory, chebyshev_memory, just_memory):
    base_path = f'./results/{model_name}/{dataset_name}/depth_{number_of_layers}/'
    if model_name == 'esn':
        return base_path + f'{non_linear_units}_euler/' if euler else base_path + f'{non_linear_units}/'
    elif model_name == 'rmn':
        memory_type = '_legendre' if legendre_memory else '_chebyshev' if chebyshev_memory else ''
        euler_suffix = '_euler' if euler else ''
        if just_memory:
            return base_path + f'{memory_units}m{memory_type}/'
        else:
            return base_path + f'{memory_units}m{memory_type}_{non_linear_units}nl{euler_suffix}/'
    return base_path


if __name__ == '__main__':

    # parse arguments
    parser = ArgumentParser()

    # general arguments
    parser.add_argument('--cpu', action='store_true', help='Force to use the CPU')
    parser.add_argument('--dataset', type=str, default='sequential_mnist', help='Dataset to use')
    parser.add_argument('--model', type=str, default='esn', help='Model to use')

    # dataset arguments
    parser.add_argument('--validation_ratio', type=float, default=0.2, help='Validation ratio')
    parser.add_argument('--batch_training', type=int, default=1, help='Training batch size')
    parser.add_argument('--batch_validation', type=int, default=1, help='Validation batch size')
    parser.add_argument('--batch_testing', type=int, default=1, help='Testing batch size')

    # number of units
    parser.add_argument('--input_units', type=int, default=1, help='Number of input units')
    parser.add_argument('--non_linear_units', type=int, default=1, help='Number of non linear units')
    parser.add_argument('--memory_units', type=int, default=1, help='Number of memory units')

    # scaling
    parser.add_argument('--memory_scaling', type=float, default=1.0, help='scaling factor for memory kernel')
    parser.add_argument('--non_linear_scaling', type=float, default=1.0,
                        help='scaling factor for non linear kernel')
    parser.add_argument('--input_memory_scaling', type=float, default=1.0, help='Input memory scaling')
    parser.add_argument('--input_non_linear_scaling', type=float, default=1.0, help='Input non linear scaling')
    parser.add_argument('--memory_non_linear_scaling', type=float, default=1.0, help='Memory non linear scaling')
    parser.add_argument('--inter_non_linear_scaling', type=float, default=1.0, help='Inter non linear scaling')
    parser.add_argument('--inter_memory_scaling', type=float, default=1.0, help='Inter memory scaling')
    parser.add_argument('--bias_scaling', type=float, default=None, help='Bias scaling')

    # general parameters
    parser.add_argument('--spectral_radius', type=float, default=0.9, help='Spectral radius')
    parser.add_argument('--leaky_rate', type=float, default=0.5, help='Leaky rate')
    parser.add_argument('--bias', action='store_true', help='Whether to use bias or not')
    parser.add_argument('--distribution', type=str, default='uniform', help='Weights distribution to use')
    parser.add_argument('--signs_from', type=str, default=None, help='Signs source to use')
    parser.add_argument('--fixed_input_kernels', action='store_true', help='Whether to use fixed input kernels or not')
    parser.add_argument('--non_linearity', type=str, default='tanh', help='Non-linearity to use')
    parser.add_argument('--effective_rescaling', action='store_true', help='Whether to use effective rescaling or not')
    parser.add_argument('--euler', action='store_true', help='Whether to use Euler non linear or not')
    parser.add_argument('--epsilon', type=float, default=1e-3, help='Epsilon value for Euler non linear')
    parser.add_argument('--gamma', type=float, default=1e-3, help='Gamma value for Euler non linear')
    parser.add_argument('--circular_non_linear', action='store_true', help='Whether to use ring topology or not')
    parser.add_argument('--legendre_memory', action='store_true', help='Whether to use Legendre memory or not')
    parser.add_argument('--chebyshev_memory', action='store_true', help='Whether to use Chebyshev memory or not')
    parser.add_argument('--theta', type=float, default=1.0, help='Theta value for Legendre memory')
    parser.add_argument('--use_last_state', action='store_true', help='Whether to use just the last state or not')
    parser.add_argument('--seed', type=int, default=None, help='Seed for the random number generator')
    parser.add_argument('--just_memory', action='store_true', help='Whether to use just the memory or not')
    parser.add_argument('--input_to_all_non_linear', action='store_true',
                        help='Whether to pass the input to all non linear layers or not')
    parser.add_argument('--input_to_all_memory', action='store_true',
                        help='Whether to pass the input to all memory layers or not')

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
    parser.add_argument('--initial_transients', type=int, default=0, help='Number of initial transients')
    parser.add_argument('--tolerance', type=float, default=1e-4, help='Tolerance for Ridge Regression')

    # deep reservoirs
    parser.add_argument('--number_of_non_linear_layers', type=int, default=1,
                        help='Number of non linear layers for deep reservoirs')
    parser.add_argument('--number_of_memory_layers', type=int, default=1,
                        help='Number of memory layers for deep reservoirs')
    parser.add_argument('--concatenate_non_linear', action='store_true',
                        help='Whether to concatenate the non linear or not')
    parser.add_argument('--concatenate_memory', action='store_true',
                        help='Whether to concatenate the memory or not')

    args = parser.parse_args()

    # select device
    cpu = args.cpu
    if cpu:
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')

    # set arguments
    dataset_name = args.dataset
    model_name = args.model

    validation_ratio = args.validation_ratio
    training_batch_size = args.batch_validation
    validation_batch_size = args.batch_validation
    testing_batch_size = args.batch_testing

    number_of_non_linear_layers = args.number_of_non_linear_layers
    number_of_memory_layers = args.number_of_memory_layers
    concatenate_non_linear = args.concatenate_non_linear
    concatenate_memory = args.concatenate_memory

    input_units = args.input_units
    non_linear_units = args.non_linear_units
    memory_units = args.memory_units

    non_linear_scaling = args.non_linear_scaling
    memory_scaling = args.memory_scaling
    input_non_linear_scaling = args.input_non_linear_scaling
    input_memory_scaling = args.input_memory_scaling
    memory_non_linear_scaling = args.memory_non_linear_scaling
    inter_non_linear_scaling = args.inter_non_linear_scaling
    inter_memory_scaling = args.inter_memory_scaling

    spectral_radius = args.spectral_radius
    leaky_rate = args.leaky_rate
    effective_rescaling = args.effective_rescaling

    input_non_linear_connectivity = args.input_non_linear_connectivity
    non_linear_connectivity = args.non_linear_connectivity
    memory_non_linear_connectivity = args.memory_non_linear_connectivity
    input_memory_connectivity = args.input_memory_connectivity
    inter_non_linear_connectivity = args.inter_non_linear_connectivity
    inter_memory_connectivity = args.inter_memory_connectivity

    bias = args.bias
    bias_scaling = args.bias_scaling

    distribution = args.distribution
    signs_from = args.signs_from
    fixed_input_kernel = args.fixed_input_kernels
    non_linearity = args.non_linearity
    circular_non_linear = args.circular_non_linear
    input_to_all_non_linear = args.input_to_all_non_linear
    input_to_all_memory = args.input_to_all_memory

    alpha = args.alpha
    max_iter = args.max_iter
    initial_transients = args.initial_transients
    tolerance = args.tolerance

    euler = args.euler
    epsilon = args.epsilon
    gamma = args.gamma

    legendre_memory = args.legendre_memory
    chebyshev_memory = args.chebyshev_memory
    theta = args.theta

    use_last_state = args.use_last_state
    seed = args.seed
    just_memory = args.just_memory

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
    elif dataset_name == 'memory_capacity' or dataset_name == 'mg17':
        task = 'regression'

    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists(f'./results/{model_name}'):
        os.makedirs(f'./results/{model_name}')
    if not os.path.exists(f'./results/{model_name}/{dataset_name}'):
        os.makedirs(f'./results/{model_name}/{dataset_name}')

    results_path = generate_results_path(model_name, dataset_name, number_of_non_linear_layers, non_linear_units,
                                         memory_units, euler, legendre_memory, chebyshev_memory, just_memory)

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # choose model
    if model_name == 'esn':
        hyperparameters = {'validation_ratio': validation_ratio,
                           'training_batch_size': training_batch_size,
                           'validation_batch_size': validation_batch_size,
                           'testing_batch_size': testing_batch_size,

                           'number_of_non_linear_layers': number_of_non_linear_layers,

                           'input_units': input_units,
                           'non_linear_units': non_linear_units,

                           'non_linear_scaling': non_linear_scaling,
                           'input_non_linear_scaling': input_non_linear_scaling,

                           'input_non_linear_connectivity': input_non_linear_connectivity,
                           'non_linear_connectivity': non_linear_connectivity,

                           'spectral_radius': spectral_radius,
                           'leaky_rate': leaky_rate,

                           'distribution': distribution,
                           'non_linearity': non_linearity,

                           'alpha': alpha,
                           'max_iter': max_iter,
                           'tolerance': tolerance,
                           'initial_transients': initial_transients,
                           }
        if concatenate_non_linear:
            hyperparameters['concatenate_non_linear'] = True
        if input_to_all_non_linear:
            hyperparameters['input_to_all_non_linear'] = True
        if effective_rescaling:
            hyperparameters['effective_rescaling'] = True
        if bias:
            hyperparameters['bias'] = True
            hyperparameters['bias_scaling'] = bias_scaling
        if fixed_input_kernel:
            hyperparameters['fixed_input_kernel'] = True
            hyperparameters['signs_from'] = signs_from
        if circular_non_linear:
            hyperparameters['circular_non_linear'] = True
        if euler:
            hyperparameters['euler'] = True
            hyperparameters['epsilon'] = epsilon
            hyperparameters['gamma'] = gamma
        if number_of_non_linear_layers > 1:
            hyperparameters['inter_non_linear_scaling'] = inter_non_linear_scaling
            hyperparameters['inter_non_linear_connectivity'] = inter_non_linear_connectivity

        model = DeepEchoStateNetwork(task,
                                     input_units,
                                     non_linear_units,

                                     number_of_layers=number_of_non_linear_layers,
                                     concatenate=concatenate_non_linear,

                                     input_scaling=input_non_linear_scaling,
                                     recurrent_scaling=non_linear_scaling,
                                     inter_scaling=inter_non_linear_scaling,

                                     spectral_radius=spectral_radius,
                                     leaky_rate=leaky_rate,
                                     effective_rescaling=effective_rescaling,

                                     input_connectivity=input_non_linear_connectivity,
                                     recurrent_connectivity=non_linear_connectivity,
                                     inter_connectivity=inter_non_linear_connectivity,

                                     bias=bias,
                                     bias_scaling=bias_scaling,

                                     distribution=distribution,
                                     signs_from=signs_from,
                                     fixed_input_kernel=fixed_input_kernel,
                                     non_linearity=non_linearity,
                                     circular_recurrent_kernel=circular_non_linear,

                                     euler=euler,
                                     epsilon=epsilon,
                                     gamma=gamma,

                                     alpha=alpha,
                                     max_iter=max_iter,
                                     tolerance=tolerance,
                                     initial_transients=initial_transients,
                                     input_to_all=input_to_all_non_linear,
                                     ).to(device)
    elif model_name == 'rmn':
        hyperparameters = {'validation_ratio': validation_ratio,
                           'training_batch_size': training_batch_size,
                           'validation_batch_size': validation_batch_size,
                           'testing_batch_size': testing_batch_size,

                           'number_of_non_linear_layers': number_of_non_linear_layers,
                           'number_of_memory_layers': number_of_memory_layers,

                           'input_units': input_units,
                           'non_linear_units': non_linear_units,
                           'memory_units': memory_units,

                           'memory_scaling': memory_scaling,
                           'non_linear_scaling': non_linear_scaling,
                           'input_memory_scaling': input_memory_scaling,
                           'input_non_linear_scaling': input_non_linear_scaling,
                           'memory_non_linear_scaling': memory_non_linear_scaling,

                           'input_memory_connectivity': input_memory_connectivity,
                           'input_non_linear_connectivity': input_non_linear_connectivity,
                           'non_linear_connectivity': non_linear_connectivity,
                           'memory_non_linear_connectivity': memory_non_linear_connectivity,

                           'spectral_radius': spectral_radius,
                           'leaky_rate': leaky_rate,

                           'distribution': distribution,
                           'non_linearity': non_linearity,

                           'alpha': alpha,
                           'max_iter': max_iter,
                           'tolerance': tolerance,
                           'initial_transients': initial_transients,
                           }
        if concatenate_non_linear:
            hyperparameters['concatenate_non_linear'] = True
        if input_to_all_non_linear:
            hyperparameters['input_to_all_non_linear'] = True
        if effective_rescaling:
            hyperparameters['effective_rescaling'] = True
        if bias:
            hyperparameters['bias'] = True
            hyperparameters['bias_scaling'] = bias_scaling
        if fixed_input_kernel:
            hyperparameters['fixed_input_kernel'] = True
            hyperparameters['signs_from'] = signs_from
        if circular_non_linear:
            hyperparameters['circular_non_linear'] = True
        if euler:
            hyperparameters['euler'] = True
            hyperparameters['epsilon'] = epsilon
            hyperparameters['gamma'] = gamma

        if concatenate_memory:
            hyperparameters['concatenate_memory'] = True
        if input_to_all_memory:
            hyperparameters['input_to_all_memory'] = True
        if legendre_memory:
            hyperparameters['legendre_memory'] = True
            hyperparameters['theta'] = theta
        if just_memory:
            hyperparameters['just_memory'] = True
        if number_of_non_linear_layers > 1:
            hyperparameters['inter_non_linear_scaling'] = inter_non_linear_scaling
            hyperparameters['inter_non_linear_connectivity'] = inter_non_linear_connectivity
        if number_of_memory_layers > 1:
            hyperparameters['inter_memory_scaling'] = inter_memory_scaling
            hyperparameters['inter_memory_connectivity'] = inter_memory_connectivity

        model = DeepReservoirMemoryNetwork(task, input_units, non_linear_units, memory_units,
                                           number_of_non_linear_layers=number_of_non_linear_layers,
                                           number_of_memory_layers=number_of_memory_layers,
                                           initial_transients=initial_transients,
                                           memory_scaling=memory_scaling, non_linear_scaling=non_linear_scaling,
                                           input_memory_scaling=input_memory_scaling,
                                           input_non_linear_scaling=input_non_linear_scaling,
                                           memory_non_linear_scaling=memory_non_linear_scaling,
                                           inter_non_linear_scaling=inter_non_linear_scaling,
                                           inter_memory_scaling=inter_memory_scaling, spectral_radius=spectral_radius,
                                           leaky_rate=leaky_rate, input_memory_connectivity=input_memory_connectivity,
                                           input_non_linear_connectivity=input_non_linear_connectivity,
                                           non_linear_connectivity=non_linear_connectivity,
                                           memory_non_linear_connectivity=memory_non_linear_connectivity,
                                           inter_non_linear_connectivity=inter_non_linear_connectivity,
                                           inter_memory_connectivity=inter_memory_connectivity, bias=bias,
                                           bias_scaling=bias_scaling, distribution=distribution, signs_from=signs_from,
                                           fixed_input_kernel=fixed_input_kernel,
                                           non_linearity=non_linearity, effective_rescaling=effective_rescaling,
                                           concatenate_non_linear=concatenate_non_linear,
                                           concatenate_memory=concatenate_memory,
                                           circular_non_linear_kernel=circular_non_linear, euler=euler, epsilon=epsilon,
                                           gamma=gamma, alpha=alpha, max_iter=max_iter, tolerance=tolerance,
                                           legendre=legendre_memory, theta=theta, just_memory=just_memory,
                                           input_to_all_non_linear=input_to_all_non_linear,
                                           input_to_all_memory=input_to_all_memory).to(device)

    # choose a task
    if dataset_name == 'sequential_mnist':

        test_sequential_mnist(model, results_path, hyperparameters, validation_ratio, training_batch_size,
                              validation_batch_size, testing_batch_size, use_last_state, device)

    elif dataset_name == 'memory_capacity':
        if model_name == 'esn':
            if concatenate_non_linear:
                max_delay = non_linear_units * 2
            else:
                max_delay = non_linear_units * number_of_non_linear_layers * 2
        elif model_name == 'rmn' and not just_memory:
            if concatenate_non_linear and not concatenate_memory:
                max_delay = (memory_units * number_of_memory_layers + non_linear_units) * 2
            elif concatenate_memory and not concatenate_non_linear:
                max_delay = (memory_units + non_linear_units * number_of_non_linear_layers) * 2
            else:
                max_delay = ((memory_units * number_of_memory_layers + non_linear_units * number_of_non_linear_layers)
                             * 2)
        else:
            if concatenate_memory:
                max_delay = memory_units * 2
            else:
                max_delay = memory_units * number_of_layers * 2

        test_memory_capacity(results_path, hyperparameters, model, max_delay, device, use_last_state,
                             initial_transients)

    elif dataset_name == 'mg17':
        test_mg17(model, results_path, hyperparameters, use_last_state, device, initial_transients)
