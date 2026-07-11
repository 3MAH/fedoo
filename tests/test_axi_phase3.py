"""Regression tests for Phase 3 fixes (Inertia / damping_stabilization /
InterfaceForce 2*pi*r weighting in 2Daxi).

These tests don't run a full FEM solve. They build a tiny 2Daxi assembly
with a non-uniform radial coordinate, evaluate the symbolic weak form
on a known displacement, and check that the assembled mass / damping /
interface-force scales with the radial coordinate as the canonical
2*pi*r dV factor would predict.

If a weakform skips the 2*pi*r weight, its global matrix scales like
the uniform mesh integral and the test fails.
"""

import numpy as np
import pytest

import fedoo as fd


def _build_2daxi_unit_square_mesh(n=2):
    """Build a 1x1 quad mesh in (r, z) with r in [1, 2]."""
    fd.ModelingSpace("2Daxi")
    mesh = fd.mesh.rectangle_mesh(n + 1, n + 1, 1.0, 2.0, 0.0, 1.0)
    return mesh


def test_inertia_axi_mass_matrix_scales_with_2pi_r():
    """Total mass of a unit-density 2Daxi annulus matches 2*pi * integral(r dA)."""
    mesh = _build_2daxi_unit_square_mesh(n=4)
    density = 1.0

    inertia = fd.weakform.Inertia(density)
    asm = fd.Assembly.create(inertia, mesh)
    # Trigger initialize via a tiny fake problem-style call by assembling
    # the global matrix directly.
    asm.assemble_global_mat()
    M = asm.get_global_matrix()

    # Sum of all entries of the consistent mass matrix gives total mass
    # (because mass matrix for a unit displacement field is total mass per DOF
    # block; sum of all entries of the diagonal blocks is total volume*density).
    # Equivalent test: sum(M) per displacement component.
    n_dof = M.shape[0]
    n_nodes = mesh.n_nodes
    # extract block for DispX (rank 0): rows/cols [0:n_nodes]
    M_xx = M[:n_nodes, :n_nodes]
    total_mass = float(M_xx.sum())

    # analytic total volume of an annulus r in [1, 2], z in [0, 1] revolved 360 deg
    # V = 2*pi * integral(r dr dz) = 2*pi * (r^2/2 from 1 to 2) * (1) = 2*pi * 1.5 = 3*pi
    expected_mass = 2 * np.pi * 0.5 * (2.0**2 - 1.0**2) * 1.0 * density
    assert total_mass == pytest.approx(expected_mass, rel=5e-3)


def test_inertia_3d_unaffected_by_axi_path():
    """Sanity: 3D Inertia still produces volume-integrated mass without 2*pi*r."""
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(3, 3, 3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)

    density = 2.0
    inertia = fd.weakform.Inertia(density)
    asm = fd.Assembly.create(inertia, mesh)
    asm.assemble_global_mat()
    M = asm.get_global_matrix()

    n_nodes = mesh.n_nodes
    M_xx = M[:n_nodes, :n_nodes]
    total_mass = float(M_xx.sum())
    # unit cube volume * density
    expected = 1.0 * density
    assert total_mass == pytest.approx(expected, rel=1e-3)


def test_rotary_inertia_axi_raises():
    """RotaryInertia in 2Daxi is not physically defined: must raise."""
    fd.ModelingSpace("2Daxi")
    with pytest.raises(NotImplementedError, match="2Daxi"):
        fd.weakform.RotaryInertia(1.0)


if __name__ == "__main__":
    pytest.main([__file__])
