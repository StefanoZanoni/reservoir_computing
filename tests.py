import torch
import numpy as np

from argparse import ArgumentParser

from networkx.utils import argmap
from sklearn.linear_model import RidgeClassifier
from tqdm import tqdm
from triton.runtime import driver

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
    parser.add_argument('--batch', type=int, default=1, help='batch size')
    parser.add_argument('--input_units', type=int, default=1, help='Number of input units')
    parser.add_argument('--recurrent_units', type=int, default=1, help='Number of recurrent units')
    parser.add_argument('--input_scaling', type=float, default=1.0, help='Input scaling')
    parser.add_argument('--spectral_radius', type=float, default=0.99, help='Spectral radius')
    parser.add_argument('--leaky_rate', type=float, default=1.0, help='Leaky rate')
    parser.add_argument('--input_connectivity', type=int, default=1, help='Input connectivity')
    parser.add_argument('--recurrent_connectivity', type=int, default=1, help='Recurrent connectivity')
    parser.add_argument('--bias', type=bool, default=True, help='Whether to use bias or not')
    parser.add_argument('--distribution', type=str, default='uniform', help='Weights distribution to use')
    parser.add_argument('--non_linearity', type=str, default='tanh', help='Non-linearity to use')
    parser.add_argument('--effective_rescaling', type=bool, default=True,
                        help='Whether to use effective rescaling or not')
    parser.add_argument('--bias_scaling', type=float, default=None, help='Bias scaling')
    parser.add_argument('--alpha', type=float, default=1.0, help='Alpha value for Ridge Regression')
    parser.add_argument('--max_iter', type=int, default=1000, help='Maximum number of iterations for Ridge Regression')
    parser.add_argument('--initial_transients', type=int, default=1, help='Number of initial transients')

    args = parser.parse_args()

    # set arguments
    dataset_name = args.dataset
    model_name = args.model
    batch_size = args.batch
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

    trainer = RidgeClassifier(alpha=alpha, max_iter=max_iter)

    # choose model
    if model_name == 'esn':
        model = EchoStateNetwork(initial_transients, input_units, recurrent_units, bias_scaling=bias_scaling,
                                 input_scaling=input_scaling,
                                 spectral_radius=spectral_radius,
                                 leaky_rate=leaky_rate,
                                 input_connectivity=input_connectivity,
                                 recurrent_connectivity=recurrent_connectivity,
                                 bias=bias,
                                 distribution=distribution,
                                 non_linearity=non_linearity,
                                 effective_rescaling=effective_rescaling).to(device)

    # choose a task
    if dataset_name == 'sequential_mnist':
        data = SequentialMNIST()
        training_dataset = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True)

        with torch.no_grad():
            states, ys = [], []
            for i, (x, y) in enumerate(tqdm(training_dataset, desc="Training Progress")):
                x, y = x.to(device), y.to(device)
                state = model(x)
                states.append(state.cpu().numpy())
                ys.append(y.cpu().numpy())
            # Concatenate the states and targets along the batch dimension
            states = np.concatenate(states, axis=0)
            ys = np.concatenate(ys, axis=0)

            # Flatten the states tensor to combine time series length and batch dimensions
            states = states.reshape(-1, states.shape[-1])
            # Repeat the targets to match the number of time steps
            ys = np.repeat(ys, states.shape[0] // ys.shape[0])

            trainer.fit(states, ys)
            model.initialize_readout_weights(torch.tensor(trainer.coef_, dtype=torch.float32, device=device).T)

        data = SequentialMNIST(training=False)
        testing_dataset = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, drop_last=True)

        with torch.no_grad():
            predictions, targets = [], []
            for i, (x, y) in enumerate(tqdm(testing_dataset, desc="Testing Progress")):
                x, y = x.to(device), y.to(device)
                prediction = model.predict(x)
                predictions.append(prediction.cpu().numpy())
                targets.append(y.cpu().numpy())
            predictions = np.concatenate(predictions, axis=0)
            targets = np.concatenate(targets, axis=0)
            accuracy = (np.argmax(predictions, axis=1) == targets).mean() * 100
            print(f"Accuracy: {accuracy}%")
