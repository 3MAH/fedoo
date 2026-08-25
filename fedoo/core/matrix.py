"""Small matrix-normalization helpers shared by problem implementations."""

import numpy as np
from scipy import sparse


def as_global_csr(matrix, size, copy=True):
    """Return a square CSR matrix sized for the global problem DOFs.

    Scalar assembly placeholders are interpreted as a zero matrix, matching
    the convention used by Fedoo assemblies before a contribution exists.
    """
    if np.isscalar(matrix):
        return sparse.csr_matrix((size, size))
    if sparse.issparse(matrix):
        result = matrix.tocsr(copy=copy)
    else:
        result = sparse.csr_matrix(np.asarray(matrix))
    if result.shape != (size, size):
        if not copy:
            result = result.copy()
        result.resize((size, size))
    return result
