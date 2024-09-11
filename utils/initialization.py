import torch
import numpy as np
from numpy.core.numeric import indices
from triton.language import dtype


def sparse_eye_init(M: int) -> torch.FloatTensor:
    """
    Generates an M x M matrix to be used as sparse identity matrix for the
    re-scaling of the sparse recurrent input_kernel in presence of non-zero leakage.
    The neurons are connected according to a ring topology, where each neuron
    receives input only from one neuron and propagates its activation only to
    one other neuron. All the non-zero elements are set to 1.

    :param M: Number of hidden units.

    :return: Sparse identity matrix.
    """

    indices = torch.arange(M, dtype=torch.long).repeat(2, 1).T
    values = torch.ones(M, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices.T, values, (M, M), dtype=torch.float32).to_dense()


def sparse_tensor_init(M: int, N: int, C: int = 1) -> torch.FloatTensor:
    """
    Generates an M x N matrix to be used as sparse input kernel.
    For each row only C elements are non-zero (i.e., each input dimension is projected only to C neurons).
    The non-zero elements are generated randomly from a uniform distribution in [-1,1].

    :param M: Number of rows.
    :param N: Number of columns.
    :param C: Number of nonzero elements.

    :return: M x N matrix.
    """

    assert N >= C
    indices = torch.zeros((M * C, 2), dtype=torch.long)
    values = torch.empty(M * C, dtype=torch.float32)

    for i in range(M):
        idx = np.random.choice(N, size=C, replace=False)
        indices[i * C:(i + 1) * C, 0] = i
        indices[i * C:(i + 1) * C, 1] = torch.from_numpy(idx)
        values[i * C:(i + 1) * C] = torch.from_numpy(2 * np.random.rand(C).astype('f') - 1)

    return torch.sparse_coo_tensor(indices.T, values, (M, N), dtype=torch.float32).to_dense()


def sparse_recurrent_tensor_init(M: int, C: int = 1, distribution: str = 'uniform') -> torch.FloatTensor:
    """
    Generates an M x M matrix to be used as sparse recurrent input_kernel.
    For each column only C elements are non-zero (i.e., each recurrent neuron
    takes input from C other recurrent neurons). The non-zero elements are
    generated randomly from a uniform distribution in [-1,1] or from a normal distribution.

    :param M: Number of hidden units
    :param C: Number of nonzero elements per column
    :param distribution: Initialisation strategy. It can be 'uniform' or 'normal'

    :return: M x M matrix
    """

    assert M >= C
    indices = torch.zeros((M * C, 2), dtype=torch.long)
    values = np.random.uniform(-1, 1, M * C).astype('f') if distribution == 'uniform' \
        else np.random.randn(M * C).astype('f') / np.sqrt(C)

    for i in range(M):
        idx = np.random.choice(M, size=C, replace=False)
        indices[i * C:(i + 1) * C, 0] = torch.from_numpy(idx)
        indices[i * C:(i + 1) * C, 1] = i

    values = torch.from_numpy(values)
    return torch.sparse_coo_tensor(indices.T, values, (M, M), dtype=torch.float32).to_dense()


def spectral_norm_scaling(W: torch.FloatTensor, rho_desired: float) -> torch.FloatTensor:
    """ Rescales W to have rho(W) = rho_desired

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

    :param W: There Must be a square matrix uniformly sampled in (-1,1), fully connected.
    :param rho_desired: Desired spectral radius.

    :return: W rescaled to have spectral radius equal to rho_desired.
    """

    units = W.shape[0]
    value = rho_desired * (6 / (np.sqrt(12 * units)))
    return W * value


def circular_tensor_init(M: int, distribution: str = 'uniform') -> torch.FloatTensor:
    """
    Generates an M x M matrix with ring topology.
    Each neuron receives input only from one neuron and propagates its activation only to one other neuron.
    The non-zero elements are generated randomly from a uniform distribution in [-1,1], from a normal distribution or
    are fixed to 1.

    :param M: Number of hidden units
    :param distribution: Initialisation strategies.
    It can be 'uniform', 'normal' or 'fixed'.

    :return: M x M sparse matrix.
    """

    dense_shape = (M, M)  # the shape of the dense version of the matrix
    indices = torch.stack([torch.arange(M), torch.arange(1, M + 1) % M], dim=1)

    if distribution == 'fixed':
        values = torch.ones(M, dtype=torch.float32)
    elif distribution == 'uniform':
        values = torch.rand(M, dtype=torch.float32) * 2 - 1
    elif distribution == 'normal':
        values = torch.randn(M, dtype=torch.float32) / np.sqrt(1)  # circular law (rescaling)
    else:
        raise ValueError(f"Invalid distribution <<{distribution}>>. Only uniform, normal and fixed allowed.")

    return torch.sparse_coo_tensor(indices.T, values, dense_shape, dtype=torch.float32).to_dense()


def skewsymmetric(units: int, scaling: float) -> torch.FloatTensor:
    """
    Generate a skewsymmetric matrix.
    """

    W = scaling * (2 * torch.rand(units, units, dtype=torch.float32) - 1)  # uniform in (-recur_scaling, recur_scaling)
    W = W - W.T
    return W


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
