"""The 2Daxi IPC contact tangent must scale linearly with 2*pi*r.

The weight is applied once (residual W*g, tangent 0.5*(W*H + H*W)). A
two-sided W*H*W would square it and make the contact stiffness grow
quadratically with the radius. Doubling the radius (geometry otherwise
fixed) must therefore double the assembled contact matrix, not quadruple
it.
"""

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import norm as spnorm

import fedoo as fd

pytest.importorskip("ipctk")


class _FakeProblem:
    # initialize() only reads these two with fixed kappa / no friction / no ccd
    n_global_dof = 0

    def get_disp(self):
        return 0  # rest configuration


def _contact_matrix_norm(r0, d=0.01, length=1.0, dhat=0.05):
    fd.ModelingSpace("2Daxi")
    fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1.0, 0.3))

    nodes = np.array([[r0, 0.0], [r0, length], [r0 + d, -0.5], [r0 + d, length + 0.5]])
    mesh = fd.Mesh(nodes, np.array([[0, 1], [2, 3]]), "lin2")

    contact = fd.constraint.IPCContact(
        mesh,
        surface_mesh=mesh,
        dhat=dhat,
        dhat_is_relative=False,
        barrier_stiffness=1.0,
        adaptive_barrier_stiffness=False,
        use_ccd=False,
    )
    contact.initialize(_FakeProblem())
    assert len(contact._collisions) > 0
    return spnorm(sp.csr_matrix(contact.global_matrix))


def test_axi_contact_stiffness_scales_linearly_with_radius():
    ratio = _contact_matrix_norm(r0=2.0) / _contact_matrix_norm(r0=1.0)
    assert ratio == pytest.approx(2.0, rel=0.05)  # W*H*W would give ~4
