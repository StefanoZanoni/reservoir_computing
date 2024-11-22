import os
import random
import subprocess

import torch
import numpy as np

from argparse import ArgumentParser

from echo_state_network import DeepEchoStateNetwork
from reservoir_memory_network import DeepReservoirMemoryNetwork
from datasets import SequentialMNIST, MemoryCapacity, MG17
from utils.test_sequential_mnist import test_sequential_mnist
from utils.test_mg17 import test_mg17
from utils.test_mg30 import test_mg30
from utils.test_memory_capacity import test_memory_capacity
from utils.test_inubushi import test_inubushi
from utils.test_lorenz import test_lorenz
from utils.save_results import save_results

import warnings
import sklearn

import torch._inductor.config as config

config.cpp_wrapper = True

# Disable all warnings from sklearn
warnings.filterwarnings("ignore", module="sklearn")

torch.set_num_threads(os.cpu_count())
os.environ['OMP_NUM_THREADS'] = str(os.cpu_count())
os.environ['MKL_NUM_THREADS'] = str(os.cpu_count())


def get_library_path(library_name):
    try:
        result = subprocess.run(['ldconfig', '-p'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        for line in result.stdout.splitlines():
            if library_name in line:
                return line.split()[-1]
    except Exception as e:
        print(f"Error finding {library_name}: {e}")
    return None


jemalloc_path = get_library_path('jemalloc.so')
tcmalloc_path = get_library_path('libtcmalloc.so')

if jemalloc_path and tcmalloc_path:
    os.environ['LD_PRELOAD'] = f'{jemalloc_path}:{tcmalloc_path}:' + os.environ.get('LD_PRELOAD', '')
elif jemalloc_path:
    os.environ['LD_PRELOAD'] = f'{jemalloc_path}:' + os.environ.get('LD_PRELOAD', '')
elif tcmalloc_path:
    os.environ['LD_PRELOAD'] = f'{tcmalloc_path}:' + os.environ.get('LD_PRELOAD', '')


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


def create_model(args, device):
    if args.dataset == 'sequential_mnist':
        task = 'classification'
    elif (args.dataset == 'memory_capacity' or args.dataset == 'mg17' or args.dataset == 'mg30'
          or args.dataset == 'inubushi' or args.dataset == 'lorenz96'):
        task = 'regression'

    if args.model == 'esn':
        model = DeepEchoStateNetwork(
            task,
            args.input_units,
            args.non_linear_units,
            number_of_layers=args.number_of_non_linear_layers,
            concatenate=args.concatenate_non_linear,
            input_scaling=args.input_non_linear_scaling,
            recurrent_scaling=args.non_linear_scaling,
            inter_scaling=args.inter_non_linear_scaling,
            spectral_radius=args.spectral_radius,
            leaky_rate=args.leaky_rate,
            effective_rescaling=args.effective_rescaling,
            input_connectivity=args.input_non_linear_connectivity,
            recurrent_connectivity=args.non_linear_connectivity,
            inter_connectivity=args.inter_non_linear_connectivity,
            bias=args.bias,
            bias_scaling=args.bias_scaling,
            distribution=args.distribution,
            signs_from=args.signs_from,
            fixed_input_kernel=args.fixed_input_kernels,
            non_linearity=args.non_linearity,
            circular_recurrent_kernel=args.circular_non_linear,
            euler=args.euler,
            epsilon=args.epsilon,
            gamma=args.gamma,
            alpha=args.alpha,
            max_iter=args.max_iter,
            tolerance=args.tolerance,
            initial_transients=args.initial_transients,
            input_to_all=args.input_to_all_non_linear,
        ).to(device)

        hyperparameters = {'validation_ratio': args.validation_ratio,
                           'training_batch_size': args.batch_training,
                           'validation_batch_size': args.batch_validation,
                           'testing_batch_size': args.batch_testing,

                           'number_of_non_linear_layers': args.number_of_non_linear_layers,

                           'input_units': args.input_units,
                           'non_linear_units': args.non_linear_units,

                           'input_non_linear_connectivity': args.input_non_linear_connectivity,
                           'non_linear_connectivity': args.non_linear_connectivity,

                           'distribution': args.distribution,
                           'non_linearity': args.non_linearity,

                           'alpha': args.alpha,
                           'max_iter': args.max_iter,
                           'tolerance': args.tolerance,
                           'initial_transients': args.initial_transients,
                           }
        if args.distribution == 'uniform':
            hyperparameters['input_non_linear_scaling'] = args.input_non_linear_scaling
        if not args.euler:
            hyperparameters['spectral_radius'] = args.spectral_radius
            hyperparameters['leaky_rate'] = args.leaky_rate
        if args.concatenate_non_linear:
            hyperparameters['concatenate_non_linear'] = True
        if args.input_to_all_non_linear:
            hyperparameters['input_to_all_non_linear'] = True
        if args.effective_rescaling:
            hyperparameters['effective_rescaling'] = True
        if args.bias:
            hyperparameters['bias'] = True
            if args.bias_scaling is None:
                hyperparameters['bias_scaling'] = args.input_non_linear_scaling
            else:
                hyperparameters['bias_scaling'] = args.bias_scaling
        if args.fixed_input_kernels:
            hyperparameters['fixed_input_kernel'] = True
            hyperparameters['signs_from'] = args.signs_from
        if args.circular_non_linear:
            hyperparameters['circular_non_linear'] = True
        if args.euler:
            hyperparameters['euler'] = True
            hyperparameters['epsilon'] = args.epsilon
            hyperparameters['gamma'] = args.gamma
            hyperparameters['non_linear_scaling'] = args.non_linear_scaling
        if args.number_of_non_linear_layers > 1:
            hyperparameters['inter_non_linear_scaling'] = args.inter_non_linear_scaling
            hyperparameters['inter_non_linear_connectivity'] = args.inter_non_linear_connectivity
    elif args.model == 'rmn':
        model = DeepReservoirMemoryNetwork(
            task,
            args.input_units,
            args.non_linear_units,
            args.memory_units,
            number_of_non_linear_layers=args.number_of_non_linear_layers,
            number_of_memory_layers=args.number_of_memory_layers,
            initial_transients=args.initial_transients,
            memory_scaling=args.memory_scaling,
            non_linear_scaling=args.non_linear_scaling,
            input_memory_scaling=args.input_memory_scaling,
            input_non_linear_scaling=args.input_non_linear_scaling,
            memory_non_linear_scaling=args.memory_non_linear_scaling,
            inter_non_linear_scaling=args.inter_non_linear_scaling,
            inter_memory_scaling=args.inter_memory_scaling,
            spectral_radius=args.spectral_radius,
            leaky_rate=args.leaky_rate,
            input_memory_connectivity=args.input_memory_connectivity,
            input_non_linear_connectivity=args.input_non_linear_connectivity,
            non_linear_connectivity=args.non_linear_connectivity,
            memory_non_linear_connectivity=args.memory_non_linear_connectivity,
            inter_non_linear_connectivity=args.inter_non_linear_connectivity,
            inter_memory_connectivity=args.inter_memory_connectivity,
            bias=args.bias,
            bias_scaling=args.bias_scaling,
            distribution=args.distribution,
            signs_from=args.signs_from,
            fixed_input_kernel=args.fixed_input_kernels,
            non_linearity=args.non_linearity,
            effective_rescaling=args.effective_rescaling,
            concatenate_non_linear=args.concatenate_non_linear,
            concatenate_memory=args.concatenate_memory,
            circular_non_linear_kernel=args.circular_non_linear,
            euler=args.euler,
            epsilon=args.epsilon,
            gamma=args.gamma,
            alpha=args.alpha,
            max_iter=args.max_iter,
            tolerance=args.tolerance,
            legendre=args.legendre_memory,
            legendre_input=args.legendre_input,
            theta=args.theta,
            just_memory=args.just_memory,
            input_to_all_non_linear=args.input_to_all_non_linear,
            input_to_all_memory=args.input_to_all_memory,
        ).to(device)

        hyperparameters = {'validation_ratio': args.validation_ratio,
                           'training_batch_size': args.batch_training,
                           'validation_batch_size': args.batch_validation,
                           'testing_batch_size': args.batch_testing,

                           'number_of_non_linear_layers': args.number_of_non_linear_layers,
                           'number_of_memory_layers': args.number_of_memory_layers,

                           'input_units': args.input_units,
                           'non_linear_units': args.non_linear_units,
                           'memory_units': args.memory_units,

                           'input_memory_connectivity': args.input_memory_connectivity,
                           'input_non_linear_connectivity': args.input_non_linear_connectivity,
                           'non_linear_connectivity': args.non_linear_connectivity,
                           'memory_non_linear_connectivity': args.memory_non_linear_connectivity,

                           'distribution': args.distribution,
                           'non_linearity': args.non_linearity,

                           'alpha': args.alpha,
                           'max_iter': args.max_iter,
                           'tolerance': args.tolerance,
                           'initial_transients': args.initial_transients,
                           }
        if not args.legendre_memory:
            hyperparameters['memory_scaling'] = args.memory_scaling
        if args.distribution == 'uniform' and not args.legendre_input:
            hyperparameters['input_memory_scaling'] = args.input_memory_scaling
        if args.distribution == 'uniform':
            hyperparameters['input_non_linear_scaling'] = args.input_non_linear_scaling
            hyperparameters['memory_non_linear_scaling'] = args.memory_non_linear_scaling
        if not args.euler or not args.just_memory:
            hyperparameters['spectral_radius'] = args.spectral_radius
            hyperparameters['leaky_rate'] = args.leaky_rate
        if args.concatenate_non_linear and not args.just_memory:
            hyperparameters['concatenate_non_linear'] = True
        if args.input_to_all_non_linear and not args.just_memory:
            hyperparameters['input_to_all_non_linear'] = True
        if args.effective_rescaling and not args.just_memory:
            hyperparameters['effective_rescaling'] = True
        if args.bias and not args.just_memory:
            hyperparameters['bias'] = True
            if args.bias_scaling is None:
                hyperparameters['bias_scaling'] = args.input_non_linear_scaling
            else:
                hyperparameters['bias_scaling'] = args.bias_scaling
        if args.fixed_input_kernels:
            hyperparameters['fixed_input_kernel'] = True
            hyperparameters['signs_from'] = args.signs_from
        if args.legendre_input:
            hyperparameters['legendre_input'] = True
            hyperparameters['theta'] = args.theta
        if args.circular_non_linear:
            hyperparameters['circular_non_linear'] = True
        if args.euler:
            hyperparameters['euler'] = True
            hyperparameters['epsilon'] = args.epsilon
            hyperparameters['gamma'] = args.gamma
            hyperparameters['non_linear_scaling'] = args.non_linear_scaling

        if args.concatenate_memory:
            hyperparameters['concatenate_memory'] = True
        if args.input_to_all_memory:
            hyperparameters['input_to_all_memory'] = True
        if args.legendre_memory:
            hyperparameters['legendre_memory'] = True
            hyperparameters['theta'] = args.theta
        if args.just_memory:
            hyperparameters['just_memory'] = True
        if args.number_of_non_linear_layers > 1 and not args.just_memory:
            hyperparameters['inter_non_linear_scaling'] = args.inter_non_linear_scaling
            hyperparameters['inter_non_linear_connectivity'] = args.inter_non_linear_connectivity
        if args.number_of_memory_layers > 1:
            hyperparameters['inter_memory_scaling'] = args.inter_memory_scaling
            hyperparameters['inter_memory_connectivity'] = args.inter_memory_connectivity
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    return model, hyperparameters


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
    parser.add_argument('--legendre_input', action='store_true', help='Whether to use Legendre input kernel or not')
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
    parser.add_argument('--runs', type=int, default=1, help='Number of runs')
    parser.add_argument('--v', type=float, default=1, help='v value for Inubushi dataset')
    parser.add_argument('--N', type=int, default=4, help='Number of variables in Lorenz96 model')
    parser.add_argument('--F', type=float, default=8, help='Forcing term in Lorenz96 model')
    parser.add_argument('--dataset_size', type=int, default=1, help='Number of examples in the dataset')
    parser.add_argument('--lag', type=int, default=1, help='Number of time steps to predict into the future')

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
    training_batch_size = args.batch_training
    validation_batch_size = args.batch_validation
    testing_batch_size = args.batch_testing

    number_of_non_linear_layers = args.number_of_non_linear_layers

    non_linear_units = args.non_linear_units
    memory_units = args.memory_units

    use_last_state = args.use_last_state
    seed = args.seed
    just_memory = args.just_memory

    euler = args.euler
    legendre_memory = args.legendre_memory
    chebyshev_memory = args.chebyshev_memory

    runs = args.runs

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

    if not os.path.exists('./results'):
        os.makedirs('./results')
    if not os.path.exists(f'./results/{model_name}'):
        os.makedirs(f'./results/{model_name}')
    if not os.path.exists(f'./results/{model_name}/{dataset_name}'):
        os.makedirs(f'./results/{model_name}/{dataset_name}')

    results_path = generate_results_path(model_name, dataset_name, number_of_non_linear_layers, non_linear_units,
                                         memory_units, euler, legendre_memory, chebyshev_memory, just_memory)
    if dataset_name == 'inubushi':
        results_path = f'{results_path[:-1]}_v_{round(args.v, 2)}'

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    # choose a task
    if dataset_name == 'sequential_mnist':

        test_scores, validation_scores = [], []
        for run in range(runs):
            model, hyperparameters = create_model(args, device)
            validation_score, test_score = test_sequential_mnist(model, validation_ratio, training_batch_size,
                                                                 validation_batch_size,
                                                                 testing_batch_size, use_last_state, device)
            test_scores.append(test_score)
            validation_scores.append(validation_score)

        save_results(results_path, hyperparameters, np.mean(validation_scores), np.std(validation_scores),
                     np.mean(test_scores), np.std(test_scores), 'accuracy', 'greater')

    elif dataset_name == 'memory_capacity':
        if model_name == 'esn':
            max_delay = non_linear_units * 2
        elif model_name == 'rmn' and not just_memory:
            max_delay = non_linear_units * 2
        else:
            max_delay = memory_units * 2

        model, hyperparameters = create_model(args, device)

        test_memory_capacity(runs, results_path, hyperparameters, model, max_delay, device, use_last_state,
                             args.initial_transients)

    elif dataset_name == 'mg17':

        test_scores, validation_scores = [], []
        for _ in range(runs):
            model, hyperparameters = create_model(args, device)
            validation_score, test_score = test_mg17(model, use_last_state, device, args.initial_transients)
            test_scores.append(test_score)
            validation_scores.append(validation_score)

        save_results(results_path, hyperparameters, np.mean(validation_scores), np.std(validation_scores),
                     np.mean(test_scores), np.std(test_scores), 'mse', 'less')

    elif dataset_name == 'mg30':

        test_scores, validation_scores = [], []
        for _ in range(runs):
            model, hyperparameters = create_model(args, device)
            validation_score, test_score = test_mg30(model, use_last_state, device, args.initial_transients)
            test_scores.append(test_score)
            validation_scores.append(validation_score)

        save_results(results_path, hyperparameters, np.mean(validation_scores), np.std(validation_scores),
                     np.mean(test_scores), np.std(test_scores), 'mse', 'less')

    elif dataset_name == 'inubushi':
        model, hyperparameters = create_model(args, device)
        test_inubushi(runs, args.v, results_path, hyperparameters, model, 20, device, use_last_state,
                      args.initial_transients)

    elif dataset_name == 'lorenz96':
        test_scores, validation_scores = [], []
        for _ in range(runs):
            model, hyperparameters = create_model(args, device)
            validation_score, test_score = test_lorenz(args.N, args.F, args.dataset_size, args.lag, model,
                                                       use_last_state, device, args.initial_transients,
                                                       args.batch_training, args.batch_validation, args.batch_testing)
            test_scores.append(test_score)
            validation_scores.append(validation_score)

        save_results(results_path, hyperparameters, np.mean(validation_scores), np.std(validation_scores),
                     np.mean(test_scores), np.std(test_scores), 'nrmse', 'less')
