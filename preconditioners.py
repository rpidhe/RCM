"""Contains implementations of pre-conditioners for PCG method."""

from typing import Callable
import numpy as np
from scipy.sparse import diags,spdiags
import scipy.sparse as sp
def get_preconditioner(name: str) -> Callable:
    """Return pre-conditioner based on name."""
    if name == 'jacobi':
        return jacobi
    raise KeyError('No pre-conditioner for provided name = {}'.format(name))


def jacobi(a_matrix: sp.csr_matrix, residual: np.array) -> np.array:
    """Return vector(np.matrix) obtained by multiplication of inverted a_matrix argument diagonal by residual vector."""
    _to_return = {}  # type: dict
    if 'inverted' not in _to_return:
        # we want to calculate matrix inversion only once...
        _to_return['inverted'] = 1.0 / (a_matrix.diagonal(0) + 1e-6)
    return _to_return['inverted'] * residual
