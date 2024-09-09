import torch
import numpy as np
from numpy.core.numeric import indices


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

    dense_shape = torch.Size([M, M])

    # gives the shape of a ring matrix:
    indices = torch.zeros((M, 2), dtype=torch.long)
    for i in range(M):
        indices[i, :] = i
    values = torch.ones(M)

    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def sparse_tensor_init(M: int, N: int, C: int = 1) -> torch.FloatTensor:
    """
    Generates an M x N matrix to be used as sparse input kernel.
    For each row only C elements are non-zero (i.e., each input dimension is projected only to C neurons).
    The non-zero elements are generated randomly from a uniform distribution in [-1,1].

    :param M: Number of rows.
    :param N: Number of columns.
    :param C: Number of nonzero elements.

    :return: M x N dense matrix.
    """

    assert N >= C
    dense_shape = torch.Size([M, N])  # shape of the dense version of the matrix
    indices = torch.zeros((M * C, 2), dtype=torch.long)
    k = 0
    for i in range(M):
        # the indices of non-zero elements in the i-th row of the matrix
        idx = np.random.choice(N, size=C, replace=False)
        for j in range(C):
            indices[k, 0] = i
            indices[k, 1] = idx[j]
            k += 1
    values = 2 * np.random.rand(M * C).astype('f') - 1
    values = torch.from_numpy(values)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def sparse_recurrent_tensor_init(M: int, C: int = 1, distribution: str = 'uniform') -> torch.FloatTensor:
    """ Generates an M x M matrix to be used as sparse recurrent input_kernel.
    For each column only C elements are non-zero (i.e., each recurrent neuron
    takes input from C other recurrent neurons). The non-zero elements are
    generated randomly from a uniform distribution in [-1,1] or from a normal distribution.

    :param M: Number of hidden units
    :param C: Number of nonzero elements per column
    :param distribution: Initialisation strategy. It can be 'uniform' or 'normal'
    :return: MxM dense matrix
    """

    assert M >= C
    dense_shape = torch.Size([M, M])  # the shape of the dense version of the matrix
    indices = torch.zeros((M * C, 2), dtype=torch.long)
    k = 0
    for i in range(M):
        # the indices of non-zero elements in the i-th column of the matrix
        idx = np.random.choice(M, size=C, replace=False)
        for j in range(C):
            indices[k, 0] = idx[j]
            indices[k, 1] = i
            k += 1

    if distribution == 'uniform':
        values = 2 * np.random.rand(M * C).astype('f') - 1
    elif distribution == 'normal':
        values = np.random.randn(M * C).astype('f') / np.sqrt(C)  # circular law (rescaling)
    else:
        raise ValueError("Invalid distribution <<" + distribution + ">>. Only uniform and normal allowed.")

    values = torch.from_numpy(values)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()


def spectral_norm_scaling(W: torch.FloatTensor, rho_desired: float) -> torch.FloatTensor:
    """ Rescales W to have rho(W) = rho_desired

    :param W:
    :param rho_desired:
    :return:
    """
    e, _ = np.linalg.eig(W.cpu())
    rho_curr = max(abs(e))
    return W * (rho_desired / rho_curr)


def fast_spectral_rescaling(W: torch.FloatTensor, rho_desired: float) -> torch.FloatTensor:
    """ Rescales a W uniformly sampled in (-1,1) to have rho(W) = rho_desired.
    This method is fast since we don't need to compute the spectrum of W, which is very slow.

    NB: this method works only if W is uniformly sampled in (-1,1).
    In particular, W must be fully connected!

    :param W: must be a square matrix uniformly sampled in (-1,1), fully connected.
    :param rho_desired:
    :return:
    """
    units = W.shape[0]
    value = (rho_desired / np.sqrt(units)) * (6 / np.sqrt(12))
    W = value * W
    return W


def circular_tensor_init(M: int, distribution: str = 'uniform') -> torch.FloatTensor:
    """
    Generates an M x M matrix with ring topology.
    Each neuron receives input only from one neuron and propagates its activation only to one other neuron.
    The non-zero elements are generated randomly from a uniform distribution in [-1,1] or from a normal distribution.

    :param M: Number of hidden units
    :param distribution: Initialisation strategies.
    It can be 'uniform' or 'normal'

    :return: MxM sparse matrix.
    """

    dense_shape = torch.Size([M, M])  # the shape of the dense version of the matrix
    indices = torch.zeros((M, 2), dtype=torch.long)
    for i in range(M):
        indices[i, 0] = i
        indices[i, 1] = (i + 1) % M

    if distribution == 'uniform':
        values = 2 * np.random.rand(M).astype('f') - 1
    elif distribution == 'normal':
        values = np.random.randn(M).astype('f') / np.sqrt(1)  # circular law (rescaling)
    else:
        raise ValueError("Invalid distribution <<" + distribution + ">>. Only uniform and normal allowed.")

    values = torch.from_numpy(values)
    return torch.sparse_coo_tensor(indices.T, values, dense_shape).to_dense().float()
