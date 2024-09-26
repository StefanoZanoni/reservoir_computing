import os
import subprocess

import numpy as np
from numpy.ma.core import concatenate

from tqdm import tqdm


def test_rmn():
    spectral_radius = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
    leaky_rates = np.arange(0.1, 1.1, 0.1)
    distributions = ['uniform', 'normal', 'fixed']
    alphas = [1e-3, 1e-2, 1e-1, 1]
    effective_rescaling = [True, False]
    bias = [True, False]
    bias_scaling = np.arange(0.1, 3.1, 0.1)
    epsilons = [1e-2, 1e-3, 1e-4]
    gammas = [1e-2, 1e-3, 1e-4]
    non_linear_scaling = np.arange(0.1, 3.1, 0.1)
    memory_scaling = np.arange(0.1, 1.1, 0.1)
    input_non_linear_scaling = np.arange(0.1, 3.1, 0.1)
    input_memory_scaling = np.arange(0.1, 3.1, 0.1)
    memory_non_linear_scaling = np.arange(0.1, 3.1, 0.1)

    units = [1, 2, 4, 8, 16]
    for neurons in tqdm(units, desc=f'Testing RMN memory capacity', disable=True):

        thetas = np.arange(neurons, neurons * 8, 1)

        input_memory_connectivity = np.arange(1, neurons + 1, 1)
        input_non_linear_connectivity = np.arange(1, neurons + 1, 1)
        non_linear_connectivity = np.arange(1, neurons + 1, 1)
        memory_non_linear_connectivity = np.arange(1, neurons + 1, 1)

        for _ in range(10):

            sr = np.random.choice(spectral_radius)
            lr = np.random.choice(leaky_rates)
            dist = np.random.choice(distributions)
            alpha = np.random.choice(alphas)
            er = np.random.choice(effective_rescaling)
            b = np.random.choice(bias)

            input_nl_connectivity = np.random.choice(input_non_linear_connectivity)
            input_m_connectivity = np.random.choice(input_memory_connectivity)
            nl_connectivity = np.random.choice(non_linear_connectivity)
            m_nl_connectivity = np.random.choice(memory_non_linear_connectivity)

            epsilon = np.random.choice(epsilons)
            gamma = np.random.choice(gammas)

            nls = np.random.choice(non_linear_scaling)
            bs = np.random.choice(bias_scaling)
            ms = np.random.choice(memory_scaling)
            inms = np.random.choice(input_memory_scaling)
            mns = np.random.choice(memory_non_linear_scaling)
            inls = np.random.choice(input_non_linear_scaling)

            theta = np.random.choice(thetas)

            command = [
                'python', 'tests.py',
                '--cpu',
                '--dataset', 'memory_capacity',
                '--model', 'rmn',
                '--num_layers', '1',
                '--seed', '5',
                '--batch_training', '1',
                '--batch_validation', '1',
                '--batch_testing', '1',
                '--input_units', '1',
                '--non_linear_units', str(neurons),
                '--memory_units', str(neurons),

                '--input_non_linear_scaling', str(inls),
                '--input_memory_scaling', str(inms),
                '--memory_non_linear_scaling', str(mns),
                '--non_linear_scaling', str(nls),
                '--memory_scaling', '1',
                '--bias_scaling', str(bs),

                '--input_memory_connectivity', str(input_m_connectivity),
                '--input_non_linear_connectivity', str(input_nl_connectivity),
                '--non_linear_connectivity', str(nl_connectivity),
                '--memory_non_linear_connectivity', str(m_nl_connectivity),

                '--spectral_radius', str(sr),
                '--leaky_rate', str(lr),
                '--distribution', dist,
                '--non_linearity', 'identity',
                '--alpha', str(alpha),
                '--max_iter', '2000',
                '--tolerance', '1e-6',
                '--initial_transients', str(100),
                '--epsilon', str(epsilon),
                '--gamma', str(gamma),
            ]

            if er:
                command.append('--effective_rescaling')
            if b:
                command.append('--bias')

            subprocess.run(command)

            command_legendre = [
                'python', 'tests.py',
                '--cpu',
                '--dataset', 'memory_capacity',
                '--model', 'rmn',
                '--num_layers', '1',
                '--seed', '5',
                '--batch_training', '1',
                '--batch_validation', '1',
                '--batch_testing', '1',
                '--input_units', '1',
                '--non_linear_units', str(neurons),
                '--memory_units', str(neurons),

                '--input_non_linear_scaling', str(inls),
                '--input_memory_scaling', str(inms),
                '--memory_non_linear_scaling', str(mns),
                '--non_linear_scaling', str(nls),
                '--memory_scaling', '1',
                '--bias_scaling', str(bs),

                '--input_memory_connectivity', str(input_m_connectivity),
                '--input_non_linear_connectivity', str(input_nl_connectivity),
                '--non_linear_connectivity', str(nl_connectivity),
                '--memory_non_linear_connectivity', str(m_nl_connectivity),

                '--spectral_radius', str(sr),
                '--leaky_rate', str(lr),
                '--distribution', dist,
                '--non_linearity', 'identity',
                '--alpha', str(alpha),
                '--max_iter', '2000',
                '--tolerance', '1e-6',
                '--initial_transients', str(100),
                '--epsilon', str(epsilon),
                '--gamma', str(gamma),
                '--legendre_memory',
                '--theta', str(theta),
            ]

            if er:
                command_legendre.append('--effective_rescaling')
            if b:
                command_legendre.append('--bias')

            subprocess.run(command_legendre)

            command_just_memory = [
                'python', 'tests.py',
                '--cpu',
                '--dataset', 'memory_capacity',
                '--model', 'rmn',
                '--num_layers', '1',
                '--seed', '5',
                '--batch_training', '1',
                '--batch_validation', '1',
                '--batch_testing', '1',
                '--input_units', '1',
                '--non_linear_units', str(neurons),
                '--memory_units', str(neurons),

                '--input_non_linear_scaling', str(inls),
                '--input_memory_scaling', str(inms),
                '--memory_non_linear_scaling', str(mns),
                '--non_linear_scaling', str(nls),
                '--memory_scaling', '1',
                '--bias_scaling', str(bs),

                '--input_memory_connectivity', str(input_m_connectivity),
                '--input_non_linear_connectivity', str(input_nl_connectivity),
                '--non_linear_connectivity', str(nl_connectivity),
                '--memory_non_linear_connectivity', str(m_nl_connectivity),

                '--spectral_radius', str(sr),
                '--leaky_rate', str(lr),
                '--distribution', dist,
                '--non_linearity', 'identity',
                '--alpha', str(alpha),
                '--max_iter', '2000',
                '--tolerance', '1e-6',
                '--initial_transients', str(100),
                '--epsilon', str(epsilon),
                '--gamma', str(gamma),
                '--just_memory',
            ]

            if er:
                command_just_memory.append('--effective_rescaling')
            if b:
                command_just_memory.append('--bias')

            subprocess.run(command_just_memory)

            command_legendre_just_memory = [
                'python', 'tests.py',
                '--cpu',
                '--dataset', 'memory_capacity',
                '--model', 'rmn',
                '--num_layers', '1',
                '--seed', '5',
                '--batch_training', '1',
                '--batch_validation', '1',
                '--batch_testing', '1',
                '--input_units', '1',
                '--non_linear_units', str(neurons),
                '--memory_units', str(neurons),

                '--input_non_linear_scaling', str(inls),
                '--input_memory_scaling', str(inms),
                '--memory_non_linear_scaling', str(mns),
                '--non_linear_scaling', str(nls),
                '--memory_scaling', '1',
                '--bias_scaling', str(bs),

                '--input_memory_connectivity', str(input_m_connectivity),
                '--input_non_linear_connectivity', str(input_nl_connectivity),
                '--non_linear_connectivity', str(nl_connectivity),
                '--memory_non_linear_connectivity', str(m_nl_connectivity),

                '--spectral_radius', str(sr),
                '--leaky_rate', str(lr),
                '--distribution', dist,
                '--non_linearity', 'identity',
                '--alpha', str(alpha),
                '--max_iter', '2000',
                '--tolerance', '1e-6',
                '--initial_transients', str(100),
                '--epsilon', str(epsilon),
                '--gamma', str(gamma),
                '--legendre_memory',
                '--just_memory',
                '--theta', str(theta)
            ]

            if er:
                command_legendre_just_memory.append('--effective_rescaling')
            if b:
                command_legendre_just_memory.append('--bias')

            subprocess.run(command_legendre_just_memory)


def test_esn():
    spectral_radius = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
    leaky_rates = np.arange(0.1, 1.1, 0.1)
    distributions = ['uniform', 'normal', 'fixed']
    alphas = [1e-3, 1e-2, 1e-1, 1]
    effective_rescaling = [True, False]
    bias = [True, False]
    bias_scaling = np.arange(0.1, 3.1, 0.1)
    epsilons = [1e-2, 1e-3, 1e-4]
    gammas = [1e-2, 1e-3, 1e-4]
    non_linear_scaling = np.arange(0.1, 3.1, 0.1)
    input_non_linear_scaling = np.arange(0.1, 3.1, 0.1)
    inter_non_linear_scaling = np.arange(0.1, 3.1, 0.1)

    units = [1, 2, 4, 8, 16]
    for neurons in tqdm(units, desc=f'Testing ESN memory capacity'):

        input_non_linear_connectivity = np.arange(1, neurons + 1, 1)
        non_linear_connectivity = np.arange(1, neurons + 1, 1)
        inter_non_linear_connectivity = np.arange(1, neurons + 1, 1)

        for _ in range(10):
            sr = np.random.choice(spectral_radius)
            lr = np.random.choice(leaky_rates)
            dist = np.random.choice(distributions)
            alpha = np.random.choice(alphas)
            er = np.random.choice(effective_rescaling)
            b = np.random.choice(bias)
            input_connectivity = np.random.choice(input_non_linear_connectivity)
            recurrent_connectivity = np.random.choice(non_linear_connectivity)
            inter_connectivity = np.random.choice(inter_non_linear_connectivity)
            epsilon = np.random.choice(epsilons)
            gamma = np.random.choice(gammas)
            rs = np.random.choice(non_linear_scaling)
            inls = np.random.choice(input_non_linear_scaling)
            ins = np.random.choice(inter_non_linear_scaling)
            bs = np.random.choice(bias_scaling)

            command = [
                'python', 'tests.py',
                '--cpu',
                '--dataset', 'memory_capacity',
                '--model', 'esn',
                '--num_layers', '2',
                '--seed', '5',
                '--batch_training', '1',
                '--batch_validation', '1',
                '--batch_testing', '1',
                '--input_units', '1',
                '--non_linear_units', str(neurons),

                '--input_non_linear_scaling', str(inls),
                '--inter_non_linear_scaling', str(ins),
                '--non_linear_scaling', str(rs),
                '--bias_scaling', str(bs),

                '--input_non_linear_connectivity', str(input_connectivity),
                '--inter_non_linear_connectivity', str(inter_connectivity),
                '--non_linear_connectivity', str(recurrent_connectivity),

                '--spectral_radius', str(sr),
                '--leaky_rate', str(lr),
                '--distribution', dist,
                '--non_linearity', 'identity',
                '--alpha', str(alpha),
                '--max_iter', '2000',
                '--tolerance', '1e-6',
                '--initial_transients', str(100),
                '--epsilon', str(epsilon),
                '--gamma', str(gamma),
            ]

            if er:
                command.append('--effective_rescaling')
            if b:
                command.append('--bias')

            subprocess.run(command)


if __name__ == '__main__':
    #test_esn()
    test_rmn()
