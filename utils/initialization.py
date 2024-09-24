import torch
import numpy as np


def eye_init(M: int) -> torch.FloatTensor:

    """
    Generates an identity matrix of size M x M.

    :param M: Number of units.

    :return: Identity matrix.
    """

    if M <= 0:
        raise ValueError(f"Invalid number of units <<{M}>>. Must be positive.")

    return torch.eye(M, dtype=torch.float32)


def sparse_tensor_init(M: int, N: int, C: int = 1, distribution: str = 'uniform', scaling: float = 1.0)\
        -> torch.FloatTensor:

    """
    Generates an M x N matrix to be used as sparse kernel.
    For each row only C elements are non-zero (i.e., each input dimension is projected only to C neurons).
    The non-zero elements are generated randomly from a uniform distribution in [-1,1], from a normal distribution, or
    they are generated deterministically all equals to 1 if distribution is 'fixed'.

    :param M: Number of rows.
    :param N: Number of columns.
    :param C: Number of nonzero elements.
    :param distribution: Initialisation strategy.
    It can be 'uniform', 'normal' or 'fixed'.
    In the case of normal, the N(0,1) is used, scaled to a variance equal to 1/C.
    :param scaling: Scaling factor for fixed and uniform distribution.

    :return: M x N matrix.
    """

    if C > N:
        raise ValueError(f"Invalid number of non-zero elements <<{C}>>. Must be less than or equal to N.")
    if M <= 0 or N <= 0:
        raise ValueError(f"Invalid number of units <<{M}>> and <<{N}>>. Must be positive.")
    if C <= 0:
        raise ValueError(f"Invalid number of non-zero elements <<{C}>>. Must be positive.")
    if distribution not in ['uniform', 'normal', 'fixed']:
        raise ValueError(f"Invalid distribution <<{distribution}>>. Only uniform, normal and fixed allowed.")

    if distribution == 'fixed':
        values = torch.ones(M * C, dtype=torch.float32) * scaling
    elif distribution == 'uniform':
        values = (torch.rand(M * C, dtype=torch.float32) * 2 - 1) * scaling
    elif distribution == 'normal':
        values = torch.randn(M * C, dtype=torch.float32) / np.sqrt(C)

    indices = torch.zeros((M * C, 2), dtype=torch.long)
    for i in range(M):
        idx = torch.randperm(N)[:C]
        indices[i * C:(i + 1) * C, 0] = i
        indices[i * C:(i + 1) * C, 1] = idx

    return torch.sparse_coo_tensor(indices.T, values, (M, N), dtype=torch.float32).to_dense()


def spectral_norm_scaling(W: torch.FloatTensor, rho_desired: float) -> torch.FloatTensor:

    """
    Rescales W to have rho(W) = rho_desired

    :param W: Weight matrix.
    :param rho_desired: Desired spectral radius.

    :return: W rescaled to have spectral radius equal to rho_desired.
    """

    e = torch.linalg.eigvals(W)
    rho_curr = torch.max(torch.abs(e)).item()
    return W * (rho_desired / rho_curr)


def fast_spectral_rescaling(W: torch.FloatTensor, rho_desired: float) -> torch.FloatTensor:

    """
    Rescales a W uniformly sampled in (-1,1) to have rho(W) = rho_desired.
    This method is fast since we don't need to compute the spectrum of W, which is very slow.

    NB: this method works only if W is uniformly sampled in (-1,1).
    In particular, W must be fully connected!

    :param W: There must be a square matrix uniformly sampled in (-1,1), fully connected.
    :param rho_desired: Desired spectral radius.

    :return: W rescaled to have spectral radius equal to rho_desired.
    """

    units = W.shape[0]
    value = (rho_desired / np.sqrt(units)) * (6 / np.sqrt(12))
    return W * value


def circular_tensor_init(M: int, distribution: str = 'uniform', scaling: float = 1.0) -> torch.FloatTensor:

    """
    Generates an M x M matrix with ring topology.
    Each neuron receives input only from one neuron and propagates its activation only to one other neuron.
    The non-zero elements are generated randomly from a uniform distribution in [-1,1], from a normal distribution or
    are fixed to 1.

    :param M: Number of hidden units
    :param distribution: Initialisation strategies.
    It can be 'uniform', 'normal' or 'fixed'.
    In the case of normal, the N(0,1) is used.
    :param scaling: Scaling factor for fixed and uniform distribution.

    :return: M x M sparse matrix.
    """

    if M <= 0:
        raise ValueError(f"Invalid number of units <<{M}>>. Must be positive.")
    if distribution not in ['uniform', 'normal', 'fixed']:
        raise ValueError(f"Invalid distribution <<{distribution}>>. Only uniform, normal and fixed allowed.")

    dense_shape = (M, M)
    indices = torch.cat([torch.stack([torch.arange(1, M), torch.arange(M - 1)], dim=1), torch.tensor([[0, M - 1]])],
                        dim=0)

    if distribution == 'fixed':
        values = torch.ones(M, dtype=torch.float32) * scaling
    elif distribution == 'uniform':
        values = (torch.rand(M, dtype=torch.float32) * 2 - 1) * scaling
    elif distribution == 'normal':
        values = torch.randn(M, dtype=torch.float32)

    return torch.sparse_coo_tensor(indices.T, values, dense_shape, dtype=torch.float32).to_dense()


def skewsymmetric(M: int, scaling: float = 1.0) -> torch.FloatTensor:

    """
    Generate a skewsymmetric matrix.

    :param M: Number of hidden units.
    :param scaling: Scaling factor.
    """

    if M <= 0:
        raise ValueError(f"Invalid number of units <<{M}>>. Must be positive.")

    W = scaling * (2 * torch.rand(M, M, dtype=torch.float32) - 1)
    W = W - W.T
    return W.to_dense()


def legendre_tensor_init(M: int, theta: float) -> torch.FloatTensor:
    """
    Generate a dense matrix leveraging the Legendre polynomials.

    :param M: Number of hidden units.
    :param theta: Scaling factor.

    :return: M x M matrix.
    """

    indices = torch.cartesian_prod(torch.arange(M), torch.arange(M))
    values = torch.where(indices[:, 0] < indices[:, 1],
                         -(2 * indices[:, 0] + 1) / theta,
                         (2 * indices[:, 0] + 1) / theta * (-1) ** (indices[:, 0] - indices[:, 1] + 1))

    return torch.sparse_coo_tensor(indices.T, values, (M, M), dtype=torch.float32).to_dense()


def init_bias(bias, non_linear_units, input_scaling, bias_scaling):
    if bias:
        bias_scaling = input_scaling if bias_scaling is None else bias_scaling
        tensor = (2 * torch.rand(non_linear_units) - 1) * bias_scaling
        tensor = tensor.to_dense()
    else:
        tensor = torch.zeros(non_linear_units, layout=torch.sparse_coo)
    return torch.nn.Parameter(tensor, requires_grad=False)


def rescale_kernel(W, spectral_radius, leaky_rate, effective_rescaling, distribution, non_linear_connectivity,
                   non_linear_units):
    if effective_rescaling and leaky_rate != 1:
        I = eye_init(non_linear_units)
        W = W * leaky_rate + (I * (1 - leaky_rate))
        W = spectral_norm_scaling(W, spectral_radius)
        return (W + I * (leaky_rate - 1)) * (1 / leaky_rate)
    if distribution == 'normal' and non_linear_units != 1:
        return spectral_radius * W
    if distribution == 'uniform' and non_linear_connectivity == non_linear_units and non_linear_units != 1:
        return fast_spectral_rescaling(W, spectral_radius)
    return spectral_norm_scaling(W, spectral_radius)


def init_non_linear_kernel(non_linear_units, non_linear_connectivity, distribution, spectral_radius, leaky_rate,
                           effective_rescaling, circular_non_linear_kernel, euler, gamma, non_linear_scaling):
    if euler:
        W = skewsymmetric(non_linear_units, non_linear_scaling)
        kernel = W - gamma * eye_init(non_linear_units)
    else:
        if circular_non_linear_kernel:
            W = circular_tensor_init(non_linear_units, distribution='fixed', scaling=non_linear_scaling)
            kernel = W
        else:
            W = sparse_tensor_init(non_linear_units, non_linear_units, C=non_linear_connectivity,
                                   distribution=distribution, scaling=non_linear_scaling)
            kernel = rescale_kernel(W, spectral_radius, leaky_rate, effective_rescaling, distribution,
                                    non_linear_connectivity, non_linear_units)
    return torch.nn.Parameter(kernel, requires_grad=False)


def init_input_kernel(input_units, units, input_connectivity, input_scaling, distribution):
    kernel = sparse_tensor_init(input_units, units, C=input_connectivity, scaling=input_scaling,
                                distribution=distribution)
    return torch.nn.Parameter(kernel, requires_grad=False)


def init_memory_kernel(memory_units, theta, legendre, scaling):
    if legendre:
        M = legendre_tensor_init(memory_units, theta)
        return torch.nn.Parameter(torch.matrix_exp(M), requires_grad=False)
    kernel = circular_tensor_init(memory_units, distribution='fixed', scaling=scaling)
    return torch.nn.Parameter(kernel, requires_grad=False)
