import os

import numpy as np

from tqdm import tqdm


def test_esn():
    spectral_radius = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    leaky_rates = np.arange(0.4, 1.1, 0.1)
    distributions = ['uniform', 'normal']
    alphas = [1e-2, 1e-1, 1]
    initial_transients = [0, 50, 100]
    effective_rescaling = [True, False]
    bias = [True, False]
    euler = [True, False]
    epsilons = [1e-2, 1e-3, 1e-4]
    gammas = [1e-2, 1e-3, 1e-4]
    recurrent_scaling = [1e-1, 1e-2, 1e-3]

    units = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    for neurons in tqdm(units, desc=f'Testing memory capacity'):
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
            e = np.random.choice(euler)
            epsilon = np.random.choice(epsilons)
            gamma = np.random.choice(gammas)
            rs = np.random.choice(recurrent_scaling)

            if b:
                bias_flag = '--bias'
            else:
                bias_flag = ''
            if er:
                er_flag = '--effective_rescaling'
            else:
                er_flag = ''
            if e:
                euler_flag = '--euler'
            else:
                euler_flag = ''

            os.system(f'python tests.py --dataset memory_capacity --model esn'
                      f' --num_layers 2'
                      f' --circular_non_linear'
                      f' --seed 5'
                      f' --batch_training 1 --batch_validation 1 --batch_testing 1'
                      f' --input_units 1 --non_linear_units {neurons}'
                      f' --input_non_linear_scaling 1'
                      f' --inter_non_linear_scaling 1'
                      f' --input_non_linear_connectivity {input_connectivity}'
                      f' --inter_non_linear_connectivity {inter_connectivity}'
                      f' --non_linear_connectivity {recurrent_connectivity}'
                      f' --spectral_radius {sr} --leaky_rate {lr}'
                      f' --distribution {dist}'
                      f' --non_linearity identity'
                      f' --alpha {alpha} --max_iter 1000 --tolerance 1e-6'
                      f' --initial_transients {it}'
                      f' {er_flag}'
                      f' {bias_flag}'
                      f' {euler_flag}'
                      f' --epsilon {epsilon}'
                      f' --gamma {gamma}'
                      f' --non_linear_scaling {rs}'
                      )


if __name__ == '__main__':
    test_esn()
