import torch

from argparse import ArgumentParser

from echo_state_network import train_esn, EchoStateNetwork

from datasets import SequentialMNIST

from training_method import RidgeRegression

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
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--training_method', type=str, default='ridge', help='Training method to use')
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

    args = parser.parse_args()

    # set arguments
    dataset_name = args.dataset
    model_name = args.model
    batch_size = args.batch
    training_method_name = args.training_method
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

    if dataset_name == 'sequential_mnist':
        data = SequentialMNIST()
        dataset = torch.utils.data.DataLoader(data, batch_size=1, shuffle=True)
    if training_method_name == 'ridge':
        training_method = RidgeRegression(device)
    if model_name == 'esn':
        model = train_esn(device, dataset, training_method, input_units, recurrent_units, input_scaling=input_scaling,
                          spectral_radius=spectral_radius, leaky_rate=leaky_rate, input_connectivity=input_connectivity,
                          recurrent_connectivity=recurrent_connectivity, bias=bias, distribution=distribution,
                          non_linearity=non_linearity, effective_rescaling=effective_rescaling,
                          bias_scaling=bias_scaling)
