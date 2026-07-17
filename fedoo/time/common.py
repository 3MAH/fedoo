from dataclasses import dataclass

import numpy as np
from scipy import sparse


def scatter_dense_block(block, dof_indices, shape):
    """Scatter a dense ``(k, k)`` block onto a global sparse matrix.

    ``dof_indices`` (length ``k``) gives the global row/column position of each
    local DOF: entry ``block[i, j]`` lands at
    ``(dof_indices[i], dof_indices[j])``. Returns a
    :class:`scipy.sparse.csr_matrix` of the requested ``shape``.
    """
    idx = np.asarray(dof_indices, dtype=int)
    k = len(idx)
    rows = np.repeat(idx, k)
    cols = np.tile(idx, k)
    return sparse.csr_matrix(
        (np.asarray(block, dtype=float).ravel(), (rows, cols)), shape=shape
    )


@dataclass(frozen=True)
class RayleighDamping:
    """Rayleigh damping descriptor: ``C = alpha*M + beta*K``."""

    alpha: float = 0.0
    beta: float = 0.0


def newmark_acceleration_velocity(beta, gamma, dt, delta_disp, v_n, a_n):
    """Newmark update of acceleration and velocity from a displacement increment.

    Given the displacement increment ``delta_disp = u_{n+1} - u_n`` and the
    start-of-step velocity ``v_n`` and acceleration ``a_n``, return the
    end-of-step ``(acceleration, velocity)`` predicted by the Newmark relations::

        a_{n+1} = 1/(beta*dt**2) * (delta_disp - dt*v_n) + (1 - 0.5/beta) * a_n
        v_{n+1} = v_n + dt * ((1 - gamma) * a_n + gamma * a_{n+1})

    This single definition is shared by every generalized-alpha term (storage,
    stiffness, dissipation) so the recurrence constants live in exactly one
    place.
    """
    if delta_disp is None or (np.isscalar(delta_disp) and delta_disp == 0):
        delta_disp = np.zeros_like(v_n)
    else:
        delta_disp = np.asarray(delta_disp)
    if delta_disp.shape != np.shape(v_n):
        if delta_disp.size != np.size(v_n):
            raise ValueError(
                "delta_disp and velocity must contain the same number of DOFs."
            )
        # Assemblies can map the increment to the global flat DOF layout while
        # storing dynamic state variables in their node-shaped representation.
        delta_disp = delta_disp.reshape(np.shape(v_n))

    a0 = 1.0 / (beta * dt**2)
    acc = a0 * (delta_disp - dt * v_n) + (1.0 - 0.5 / beta) * a_n
    vel = v_n + dt * ((1.0 - gamma) * a_n + gamma * acc)
    return acc, vel
