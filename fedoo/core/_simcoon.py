"""Simcoon batch calls shared by the constitutive laws and the weak forms.

Every call runs on ``fedoo.core.base.N_THREADS`` threads (``fedoo.get_config()["N_THREADS"]``,
0 = all the cores).
"""

import numpy as np
import simcoon as sim

from fedoo.core import base


def rotate_voigt(values, DR, kind, active=True):
    """Rotate Voigt vectors ``(6, N)`` or 6x6 tensors ``(6, 6, N)`` by the rotation
    matrices ``DR`` ``(3, 3, N)``, or by one ``(3, 3)`` matrix for all of them.

    ``kind`` is the simcoon rotation rule: "stress", "strain", "stiffness", ...
    """
    # one parallel C++ loop, without building a scipy Rotation (from_matrix
    # is serial and costs about 1 us per point)
    return sim.rotate_voigt(values, DR, kind, active, base.N_THREADS)


def objective_rate(corate, F0, F1, dtime, return_de):
    """``simcoon.objective_rate`` on ``N_THREADS`` threads."""
    return sim.objective_rate(corate, F0, F1, dtime, return_de, base.N_THREADS)


def lt_convert(Lt, F, stress, converter_key):
    """``simcoon.Lt_convert`` on ``N_THREADS`` threads."""
    return sim.Lt_convert(Lt, F, stress, converter_key, base.N_THREADS)


def log_strain(F):
    """Logarithmic strain ``ln V`` in Voigt form ``(6, N)`` of ``F`` ``(3, 3, N)``."""
    return sim.Log_strain(F, True, False, base.N_THREADS)


def deformation_gradient(grad):
    """``F = I + grad`` and ``J = det F`` from the displacement gradient.

    ``grad``: ``(3, 3, N)`` array or 3x3 nested lists of ``(N,)`` arrays. Returns
    ``F`` Fortran ordered ``(3, 3, N)`` and ``J`` ``(N,)``.
    """
    return sim.G_UdX_to_F(grad, return_J=True, n_threads=base.N_THREADS)


def rotvec_to_matrix(rotvec):
    """Rotation matrices ``(N, 3, 3)`` of the rotation vectors ``(N, 3)``."""
    rotvec = np.asarray(rotvec, dtype=float)
    return sim.rotvec_to_matrix(rotvec.T, base.N_THREADS).transpose(2, 0, 1)


def matrix_to_rotvec(matrix):
    """Rotation vectors ``(N, 3)`` (angle in [0, pi]) of the rotation matrices
    ``(N, 3, 3)``."""
    matrix = np.asarray(matrix, dtype=float)
    return sim.matrix_to_rotvec(matrix.transpose(1, 2, 0), base.N_THREADS).T


def dexp_inv(rotvec):
    """Inverse left Jacobian of SO(3) of rotation vectors ``(..., 3)``, shape ``(..., 3, 3)``.

    ``dR R^T = [dexp(w) dw]x``: maps a spatial spin to the rotation vector increment.
    """
    rotvec = np.asarray(rotvec, dtype=float)
    out = sim.dexp_inv(rotvec.reshape(-1, 3).T, base.N_THREADS)  # (3, 3, M)
    return out.transpose(2, 0, 1).reshape(rotvec.shape + (3,))
