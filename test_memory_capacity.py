import os
import subprocess

import numpy as np
from numpy.ma.core import concatenate

from tqdm import tqdm


def test_rmn():
    spectral_radius = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    leaky_rates = np.arange(0.4, 1.1, 0.1)
    distributions = ['uniform', 'normal']
    alphas = [1e-2, 1e-1, 1]
    initial_transients = [0, 50, 100]
    effective_rescaling = [True, False]
    bias = [True, False]
    epsilons = [1e-2, 1e-3, 1e-4]
    gammas = [1e-2, 1e-3, 1e-4]
    non_linear_scaling = [1e-1, 1e-2, 1e-3]
    circular_non_linear = [True, False]

    memory_units = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    non_linear_units = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    for memory_neurons in tqdm(memory_units, desc=f'Testing RMN memory capacity'):
        input_memory_connectivity = np.arange(int(max(1, memory_neurons / 2)), int(memory_neurons + 1), 1)
        inter_memory_connectivity = np.arange(int(max(1, memory_neurons / 2)), int(memory_neurons + 1), 1)

        for non_linear_neurons in non_linear_units:
            input_non_linear_connectivity = np.arange(int(max(1, non_linear_neurons / 2)), int(non_linear_neurons + 1),
                                                      1)
            non_linear_connectivity = np.arange(int(max(1, non_linear_neurons / 2)), int(non_linear_neurons + 1), 1)
            inter_non_linear_connectivity = np.arange(int(max(1, non_linear_neurons / 2)), int(non_linear_neurons + 1),
                                                      1)
            memory_non_linear_connectivity = np.arange(int(max(1, non_linear_neurons / 2)), int(non_linear_neurons + 1),
                                                       1)
            for _ in range(10):
                sr = np.random.choice(spectral_radius)
                lr = np.random.choice(leaky_rates)
                dist = np.random.choice(distributions)
                alpha = np.random.choice(alphas)
                it = np.random.choice(initial_transients)
                er = np.random.choice(effective_rescaling)
                b = np.random.choice(bias)
                input_nl_connectivity = np.random.choice(input_non_linear_connectivity)
                input_m_connectivity = np.random.choice(input_memory_connectivity)
                non_linear_connectivity = np.random.choice(non_linear_connectivity)
                m_nl_connectivity = np.random.choice(memory_non_linear_connectivity)
                inter_nl_connectivity = np.random.choice(inter_non_linear_connectivity)
                inter_m_connectivity = np.random.choice(inter_memory_connectivity)
                epsilon = np.random.choice(epsilons)
                gamma = np.random.choice(gammas)
                nls = np.random.choice(non_linear_scaling)
                circular = np.random.choice(circular_non_linear)

                command = [
                    'python', 'tests.py', '--dataset', 'memory_capacity', '--model', 'rmn',
                    '--num_layers', '1', '--seed', '5',
                    '--batch_training', '1', '--batch_validation', '1', '--batch_testing', '1',
                    '--input_units', '1', '--non_linear_units', str(neurons), '--memory_units', str(neurons),
                    '--input_non_linear_scaling', '1', '--inter_non_linear_scaling', '1', '--input_memory_scaling', '1',
                    '--memory_non_linear_scaling', '1', '--inter_memory_scaling', '1',
                    '--input_memory_connectivity', '1',
                    '--input_non_linear_connectivity', str(input_nl_connectivity),
                    '--input_memory_connectivity', str(input_m_connectivity),
                    '--inter_non_linear_connectivity', str(inter_nl_connectivity),
                    '--non_linear_connectivity', str(non_linear_connectivity),
                    '--memory_non_linear_connectivity', str(m_nl_connectivity),
                    '--inter_memory_connectivity', str(inter_m_connectivity),
                    '--spectral_radius', str(sr), '--leaky_rate', str(lr),
                    '--distribution', dist, '--non_linearity', 'identity',
                    '--alpha', str(alpha), '--max_iter', '1000', '--tolerance', '1e-6',
                    '--initial_transients', str(it), '--epsilon', str(epsilon),
                    '--gamma', str(gamma), '--non_linear_scaling', str(nls)
                ]

                if er:
                    command.append('--effective_rescaling')
                if b:
                    command.append('--bias')
                if circular:
                    command.append('--circular_non_linear')

                subprocess.run(command)

                command_euler = [
                    'python', 'tests.py', '--dataset', 'memory_capacity', '--model', 'rmn',
                    '--num_layers', '1', '--seed', '5',
                    '--batch_training', '1', '--batch_validation', '1', '--batch_testing', '1',
                    '--input_units', '1', '--non_linear_units', str(neurons), '--memory_units', str(neurons),
                    '--input_non_linear_scaling', '1', '--inter_non_linear_scaling', '1', '--input_memory_scaling', '1',
                    '--memory_non_linear_scaling', '1', '--inter_memory_scaling', '1',
                    '--input_memory_connectivity', '1',
                    '--input_non_linear_connectivity', str(input_nl_connectivity),
                    '--input_memory_connectivity', str(input_m_connectivity),
                    '--inter_non_linear_connectivity', str(inter_nl_connectivity),
                    '--non_linear_connectivity', str(non_linear_connectivity),
                    '--memory_non_linear_connectivity', str(m_nl_connectivity),
                    '--inter_memory_connectivity', str(inter_m_connectivity),
                    '--spectral_radius', str(sr), '--leaky_rate', str(lr),
                    '--distribution', dist, '--non_linearity', 'identity',
                    '--alpha', str(alpha), '--max_iter', '1000', '--tolerance', '1e-6',
                    '--initial_transients', str(it), '--epsilon', str(epsilon),
                    '--gamma', str(gamma), '--non_linear_scaling', str(nls),
                    '--euler'
                ]

                if er:
                    command_euler.append('--effective_rescaling')
                if b:
                    command_euler.append('--bias')
                if circular:
                    command_euler.append('--circular_non_linear')

                subprocess.run(command_euler)

                command_legendre = [
                    'python', 'tests.py', '--dataset', 'memory_capacity', '--model', 'rmn',
                    '--num_layers', '1', '--seed', '5',
                    '--batch_training', '1', '--batch_validation', '1', '--batch_testing', '1',
                    '--input_units', '1', '--non_linear_units', str(neurons), '--memory_units', str(neurons),
                    '--input_non_linear_scaling', '1', '--inter_non_linear_scaling', '1', '--input_memory_scaling', '1',
                    '--memory_non_linear_scaling', '1', '--inter_memory_scaling', '1',
                    '--input_memory_connectivity', '1',
                    '--input_non_linear_connectivity', str(input_nl_connectivity),
                    '--input_memory_connectivity', str(input_m_connectivity),
                    '--inter_non_linear_connectivity', str(inter_nl_connectivity),
                    '--non_linear_connectivity', str(non_linear_connectivity),
                    '--memory_non_linear_connectivity', str(m_nl_connectivity),
                    '--inter_memory_connectivity', str(inter_m_connectivity),
                    '--spectral_radius', str(sr), '--leaky_rate', str(lr),
                    '--distribution', dist, '--non_linearity', 'identity',
                    '--alpha', str(alpha), '--max_iter', '1000', '--tolerance', '1e-6',
                    '--initial_transients', str(it), '--epsilon', str(epsilon),
                    '--gamma', str(gamma), '--non_linear_scaling', str(nls),
                    '--legendre_memory'
                ]

                if er:
                    command_legendre.append('--effective_rescaling')
                if b:
                    command_legendre.append('--bias')
                if circular:
                    command_legendre.append('--circular_non_linear')

                subprocess.run(command_legendre)

                command_legendre_euler = [
                    'python', 'tests.py', '--dataset', 'memory_capacity', '--model', 'rmn',
                    '--num_layers', '1', '--seed', '5',
                    '--batch_training', '1', '--batch_validation', '1', '--batch_testing', '1',
                    '--input_units', '1', '--non_linear_units', str(neurons), '--memory_units', str(neurons),
                    '--input_non_linear_scaling', '1', '--inter_non_linear_scaling', '1', '--input_memory_scaling', '1',
                    '--memory_non_linear_scaling', '1', '--inter_memory_scaling', '1',
                    '--input_memory_connectivity', '1',
                    '--input_non_linear_connectivity', str(input_nl_connectivity),
                    '--input_memory_connectivity', str(input_m_connectivity),
                    '--inter_non_linear_connectivity', str(inter_nl_connectivity),
                    '--non_linear_connectivity', str(non_linear_connectivity),
                    '--memory_non_linear_connectivity', str(m_nl_connectivity),
                    '--inter_memory_connectivity', str(inter_m_connectivity),
                    '--spectral_radius', str(sr), '--leaky_rate', str(lr),
                    '--distribution', dist, '--non_linearity', 'identity',
                    '--alpha', str(alpha), '--max_iter', '1000', '--tolerance', '1e-6',
                    '--initial_transients', str(it), '--epsilon', str(epsilon),
                    '--gamma', str(gamma), '--non_linear_scaling', str(nls),
                    '--legendre_memory', '--euler'
                ]

                if er:
                    command_legendre_euler.append('--effective_rescaling')
                if b:
                    command_legendre_euler.append('--bias')
                if circular:
                    command_legendre_euler.append('--circular_non_linear')

                subprocess.run(command_legendre_euler)

                command_just_memory = [
                    'python', 'tests.py', '--dataset', 'memory_capacity', '--model', 'rmn',
                    '--num_layers', '1', '--seed', '5',
                    '--batch_training', '1', '--batch_validation', '1', '--batch_testing', '1',
                    '--input_units', '1', '--non_linear_units', str(neurons), '--memory_units', str(neurons),
                    '--input_non_linear_scaling', '1', '--inter_non_linear_scaling', '1', '--input_memory_scaling', '1',
                    '--memory_non_linear_scaling', '1', '--inter_memory_scaling', '1',
                    '--input_memory_connectivity', '1',
                    '--input_non_linear_connectivity', str(input_nl_connectivity),
                    '--input_memory_connectivity', str(input_m_connectivity),
                    '--inter_non_linear_connectivity', str(inter_nl_connectivity),
                    '--non_linear_connectivity', str(non_linear_connectivity),
                    '--memory_non_linear_connectivity', str(m_nl_connectivity),
                    '--inter_memory_connectivity', str(inter_m_connectivity),
                    '--spectral_radius', str(sr), '--leaky_rate', str(lr),
                    '--distribution', dist, '--non_linearity', 'identity',
                    '--alpha', str(alpha), '--max_iter', '1000', '--tolerance', '1e-6',
                    '--initial_transients', str(it), '--epsilon', str(epsilon),
                    '--gamma', str(gamma), '--non_linear_scaling', str(nls),
                    '--just_memory'
                ]

                if er:
                    command_just_memory.append('--effective_rescaling')
                if b:
                    command_just_memory.append('--bias')
                if circular:
                    command_just_memory.append('--circular_non_linear')

                subprocess.run(command_just_memory)

                command_legendre_just_memory = [
                    'python', 'tests.py', '--dataset', 'memory_capacity', '--model', 'rmn',
                    '--num_layers', '1', '--seed', '5',
                    '--batch_training', '1', '--batch_validation', '1', '--batch_testing', '1',
                    '--input_units', '1', '--non_linear_units', str(neurons), '--memory_units', str(neurons),
                    '--input_non_linear_scaling', '1', '--inter_non_linear_scaling', '1', '--input_memory_scaling', '1',
                    '--memory_non_linear_scaling', '1', '--inter_memory_scaling', '1',
                    '--input_memory_connectivity', '1',
                    '--input_non_linear_connectivity', str(input_nl_connectivity),
                    '--input_memory_connectivity', str(input_m_connectivity),
                    '--inter_non_linear_connectivity', str(inter_nl_connectivity),
                    '--non_linear_connectivity', str(non_linear_connectivity),
                    '--memory_non_linear_connectivity', str(m_nl_connectivity),
                    '--inter_memory_connectivity', str(inter_m_connectivity),
                    '--spectral_radius', str(sr), '--leaky_rate', str(lr),
                    '--distribution', dist, '--non_linearity', 'identity',
                    '--alpha', str(alpha), '--max_iter', '1000', '--tolerance', '1e-6',
                    '--initial_transients', str(it), '--epsilon', str(epsilon),
                    '--gamma', str(gamma), '--non_linear_scaling', str(nls),
                    '--legendre_memory', '--just_memory'
                ]

                if er:
                    command_legendre_just_memory.append('--effective_rescaling')
                if b:
                    command_legendre_just_memory.append('--bias')
                if circular:
                    command_legendre_just_memory.append('--circular_non_linear')

                subprocess.run(command_legendre_just_memory)


def test_esn():
    spectral_radius = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    leaky_rates = np.arange(0.4, 1.1, 0.1)
    distributions = ['uniform', 'normal']
    alphas = [1e-2, 1e-1, 1]
    initial_transients = [0, 50, 100]
    effective_rescaling = [True, False]
    bias = [True, False]
    epsilons = [1e-2, 1e-3, 1e-4]
    gammas = [1e-2, 1e-3, 1e-4]
    recurrent_scaling = [1e-1, 1e-2, 1e-3]

    units = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    for neurons in tqdm(units, desc=f'Testing ESN memory capacity'):
        input_non_linear_connectivity = np.arange(int(max(1, neurons / 2)), int(neurons + 1), 1)
        non_linear_connectivity = np.arange(int(max(1, neurons / 2)), int(neurons + 1), 1)
        inter_non_linear_connectivity = np.arange(int(max(1, neurons / 2)), int(neurons + 1), 1)
        for _ in range(10):
            sr = np.random.choice(spectral_radius)
            lr = np.random.choice(leaky_rates)
            dist = np.random.choice(distributions)
            alpha = np.random.choice(alphas)
            it = np.random.choice(initial_transients)
            er = np.random.choice(effective_rescaling)
            b = np.random.choice(bias)
            input_connectivity = np.random.choice(input_non_linear_connectivity)
            recurrent_connectivity = np.random.choice(non_linear_connectivity)
            inter_connectivity = np.random.choice(inter_non_linear_connectivity)
            epsilon = np.random.choice(epsilons)
            gamma = np.random.choice(gammas)
            rs = np.random.choice(recurrent_scaling)

            command = [
                'python', 'tests.py', '--dataset', 'memory_capacity', '--model', 'esn',
                '--num_layers', '2', '--circular_non_linear', '--seed', '5',
                '--batch_training', '1', '--batch_validation', '1', '--batch_testing', '1',
                '--input_units', '1', '--non_linear_units', str(neurons),
                '--input_non_linear_scaling', '1', '--inter_non_linear_scaling', '1',
                '--input_non_linear_connectivity', str(input_connectivity),
                '--inter_non_linear_connectivity', str(inter_connectivity),
                '--non_linear_connectivity', str(recurrent_connectivity),
                '--spectral_radius', str(sr), '--leaky_rate', str(lr),
                '--distribution', dist, '--non_linearity', 'identity',
                '--alpha', str(alpha), '--max_iter', '1000', '--tolerance', '1e-6',
                '--initial_transients', str(it), '--epsilon', str(epsilon),
                '--gamma', str(gamma), '--non_linear_scaling', str(rs)
            ]

            if er:
                command.append('--effective_rescaling')
            if b:
                command.append('--bias')

            subprocess.run(command)

            command_euler = [
                'python', 'tests.py', '--dataset', 'memory_capacity', '--model', 'esn',
                '--num_layers', '2', '--circular_non_linear', '--seed', '5',
                '--batch_training', '1', '--batch_validation', '1', '--batch_testing', '1',
                '--input_units', '1', '--non_linear_units', str(neurons),
                '--input_non_linear_scaling', '1', '--inter_non_linear_scaling', '1',
                '--input_non_linear_connectivity', str(input_connectivity),
                '--inter_non_linear_connectivity', str(inter_connectivity),
                '--non_linear_connectivity', str(recurrent_connectivity),
                '--spectral_radius', str(sr), '--leaky_rate', str(lr),
                '--distribution', dist, '--non_linearity', 'identity',
                '--alpha', str(alpha), '--max_iter', '1000', '--tolerance', '1e-6',
                '--initial_transients', str(it), '--epsilon', str(epsilon),
                '--gamma', str(gamma), '--non_linear_scaling', str(rs),
                '--euler'
            ]

            if er:
                command_euler.append('--effective_rescaling')
            if b:
                command_euler.append('--bias')

            subprocess.run(command_euler)


if __name__ == '__main__':
    test_esn()
    test_rmn()
