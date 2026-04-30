"""Structural tests for the simcoon API on axi-block tensors.

These tests are independent of fedoo's problem assembly: they feed
a deformation gradient with the block structure expected for an
axisymmetric kinematics into simcoon's lower-level routines and
check that the *structural* zeros (slots 4 and 5 of the 6-vector,
columns/rows 2 of any 3x3 returned tensor) are preserved.

The 6-vector slot ordering used here is the one documented on
``fedoo.core.mechanical3d.Mechanical3D``::

    (rr, zz, theta-theta, gamma_rz, 0, 0)

In the 3x3 layout, the hoop direction lives on row/col 2; the
meridional (r, z) plane occupies rows/cols (0, 1).

If any of these tests fail, simcoon's polar / objective-rate /
rotation routines do *not* preserve the axi block structure
exactly, and fedoo's UL+axi pipeline cannot be relied upon at
finite strain even after the F[theta-theta] hoop-formula fix.
"""

import numpy as np
import pytest

simcoon = pytest.importorskip("simcoon")
sim = simcoon
from simcoon import Rotation


def _axi_block_F(F_rr=1.2, F_zz=0.95, F_rz=0.1, F_zr=0.05, F_tt=1.5):
    """Build a single 3x3 deformation gradient with axi block structure."""
    F = np.eye(3, dtype=float)
    F[0, 0] = F_rr
    F[1, 1] = F_zz
    F[0, 1] = F_rz
    F[1, 0] = F_zr
    F[2, 2] = F_tt
    # off-block-diagonal entries (hoop coupling) must stay exactly zero
    return F


def test_objective_rate_log_preserves_axi_block_DR():
    """sim.objective_rate('log', I, F_axi, dt) returns DR with axi block."""
    F1 = np.asfortranarray(_axi_block_F().reshape(3, 3, 1))
    F0 = np.asfortranarray(np.eye(3).reshape(3, 3, 1))

    DStrain, D, DR, Omega = sim.objective_rate("log", F0, F1, 1.0, True)

    DR0 = DR[:, :, 0]
    # off-block-diagonal hoop entries must be exactly 0
    assert DR0[0, 2] == 0.0
    assert DR0[1, 2] == 0.0
    assert DR0[2, 0] == 0.0
    assert DR0[2, 1] == 0.0
    # hoop direction does not rotate in axisymmetric kinematics
    assert DR0[2, 2] == pytest.approx(1.0, abs=1e-14)


def test_objective_rate_log_preserves_axi_block_DStrain():
    """The 6-vector strain rate has zero shear in slots 4 and 5."""
    F1 = np.asfortranarray(_axi_block_F().reshape(3, 3, 1))
    F0 = np.asfortranarray(np.eye(3).reshape(3, 3, 1))

    DStrain, D, DR, Omega = sim.objective_rate("log", F0, F1, 1.0, True)

    # DStrain is the 6-vector strain increment: slots 4, 5 = gamma_xz,
    # gamma_yz must remain identically zero
    assert DStrain[4, 0] == 0.0
    assert DStrain[5, 0] == 0.0
    # D is the 3x3 rate-of-deformation tensor: hoop row/col must be zero
    # off-block-diagonal (only the diagonal D[2,2] is allowed to be nonzero)
    D0 = D[:, :, 0]
    assert D0[0, 2] == 0.0 and D0[1, 2] == 0.0
    assert D0[2, 0] == 0.0 and D0[2, 1] == 0.0


def test_objective_rate_log_preserves_axi_block_Omega():
    """The spin tensor has zero entries in the hoop row / column."""
    F1 = np.asfortranarray(_axi_block_F().reshape(3, 3, 1))
    F0 = np.asfortranarray(np.eye(3).reshape(3, 3, 1))

    _, _, _, Omega = sim.objective_rate("log", F0, F1, 1.0, True)

    Omega0 = Omega[:, :, 0]
    assert Omega0[0, 2] == 0.0
    assert Omega0[1, 2] == 0.0
    assert Omega0[2, 0] == 0.0
    assert Omega0[2, 1] == 0.0
    # hoop axis does not spin in axisymmetric motion
    assert Omega0[2, 2] == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("corate", ["log", "log_R", "jaumann", "green_naghdi"])
def test_all_objective_rates_preserve_axi_block(corate):
    """Every objective rate fedoo exposes should preserve the axi block."""
    F1 = np.asfortranarray(_axi_block_F().reshape(3, 3, 1))
    F0 = np.asfortranarray(np.eye(3).reshape(3, 3, 1))

    DStrain, D, DR, Omega = sim.objective_rate(corate, F0, F1, 1.0, True)

    DR0 = DR[:, :, 0]
    assert DR0[0, 2] == 0.0 and DR0[1, 2] == 0.0
    assert DR0[2, 0] == 0.0 and DR0[2, 1] == 0.0
    assert DR0[2, 2] == pytest.approx(1.0, abs=1e-14)
    assert DStrain[4, 0] == 0.0 and DStrain[5, 0] == 0.0


def test_log_strain_voigt_preserves_axi_zeros():
    """sim.Log_strain in Voigt form has zero gamma_xz / gamma_yz."""
    F = _axi_block_F()
    eps = sim.Log_strain(F, True, False)  # voigt_form=True
    # eps has shape (6, 1) — flatten for easier indexing
    eps = np.asarray(eps).ravel()
    assert eps[4] == 0.0
    assert eps[5] == 0.0


def test_log_strain_hoop_component_equals_ln_lambda_theta():
    """Slot 2 of the Voigt log strain is ln(F_theta-theta)."""
    lam_theta = 1.5
    F = _axi_block_F(F_tt=lam_theta)
    eps = np.asarray(sim.Log_strain(F, True, False)).ravel()
    assert eps[2] == pytest.approx(np.log(lam_theta), rel=1e-12)


def test_log_strain_hoop_independent_of_meridional_block():
    """Hoop log strain does not couple to r-z deformation."""
    lam_theta = 1.3
    eps_a = np.asarray(
        sim.Log_strain(_axi_block_F(F_tt=lam_theta), True, False)
    ).ravel()
    eps_b = np.asarray(
        sim.Log_strain(
            _axi_block_F(F_rr=2.0, F_zz=0.5, F_rz=0.3, F_zr=0.2, F_tt=lam_theta),
            True,
            False,
        )
    ).ravel()
    # slot 2 = ln(lam_theta) regardless of meridional block
    assert eps_a[2] == pytest.approx(eps_b[2], rel=1e-12)


def test_rotation_apply_strain_preserves_axi_zeros():
    """A meridional rotation applied to an axi strain keeps slots 4, 5 = 0."""
    # build an axi-block DR (rotation by 30 deg in r-z plane)
    angle = np.pi / 6
    DR_full = np.eye(3)
    DR_full[0, 0] = np.cos(angle)
    DR_full[0, 1] = -np.sin(angle)
    DR_full[1, 0] = np.sin(angle)
    DR_full[1, 1] = np.cos(angle)

    rot = Rotation.from_matrix(DR_full[np.newaxis, :, :])

    # axi strain: (rr, zz, theta-theta, gamma_rz, 0, 0)
    strain_axi = np.array([[0.05], [-0.02], [0.10], [0.03], [0.0], [0.0]], order="F")
    rotated = rot.apply_strain(strain_axi)

    rotated = np.asarray(rotated)
    # off-meridional shears must remain identically zero
    assert rotated[4, 0] == pytest.approx(0.0, abs=1e-14)
    assert rotated[5, 0] == pytest.approx(0.0, abs=1e-14)


def test_rotation_apply_strain_preserves_hoop_invariance():
    """A meridional rotation does not change the hoop strain component."""
    angle = np.pi / 4
    DR_full = np.eye(3)
    DR_full[0, 0] = np.cos(angle)
    DR_full[0, 1] = -np.sin(angle)
    DR_full[1, 0] = np.sin(angle)
    DR_full[1, 1] = np.cos(angle)

    rot = Rotation.from_matrix(DR_full[np.newaxis, :, :])

    hoop = 0.07
    strain_axi = np.array([[0.10], [-0.03], [hoop], [0.04], [0.0], [0.0]], order="F")
    rotated = np.asarray(rot.apply_strain(strain_axi))
    # slot 2 (hoop) is invariant under rotation about the hoop axis
    assert rotated[2, 0] == pytest.approx(hoop, rel=1e-12)


def test_rotation_apply_stress_preserves_axi_zeros():
    """Same structural property for stress rotation."""
    angle = -np.pi / 3
    DR_full = np.eye(3)
    DR_full[0, 0] = np.cos(angle)
    DR_full[0, 1] = -np.sin(angle)
    DR_full[1, 0] = np.sin(angle)
    DR_full[1, 1] = np.cos(angle)

    rot = Rotation.from_matrix(DR_full[np.newaxis, :, :])

    # axi stress: (sigma_rr, sigma_zz, sigma_theta, sigma_rz, 0, 0)
    stress_axi = np.array([[100.0], [50.0], [80.0], [20.0], [0.0], [0.0]], order="F")
    rotated = np.asarray(rot.apply_stress(stress_axi))
    assert rotated[4, 0] == pytest.approx(0.0, abs=1e-12)
    assert rotated[5, 0] == pytest.approx(0.0, abs=1e-12)
    # hoop stress is invariant under in-plane rotation
    assert rotated[2, 0] == pytest.approx(80.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Phase 6d: internal-variable rotation through DR derived from objective_rate.
# This mirrors fedoo's set_start codepath
# (see stress_equilibrium.StressEquilibrium.set_start at "rot.apply_strain"):
# the DR returned by sim.objective_rate is fed to Rotation.from_matrix and
# applied to the saved strain history. We verify that, given an axi-block F1,
# this round-trip preserves the slot zeros for the strain history.
# ---------------------------------------------------------------------------


def test_set_start_rotation_axi_block_preserves_slot_zeros():
    """DR from objective_rate(I, F_axi) rotates an axi strain to an axi strain."""
    F1 = np.asfortranarray(_axi_block_F().reshape(3, 3, 1))
    F0 = np.asfortranarray(np.eye(3).reshape(3, 3, 1))
    _, _, DR, _ = sim.objective_rate("log", F0, F1, 1.0, True)

    rot = Rotation.from_matrix(DR.transpose(2, 0, 1))

    strain_axi = np.array([[0.05], [-0.02], [0.10], [0.03], [0.0], [0.0]], order="F")
    rotated = np.asarray(rot.apply_strain(strain_axi))
    assert rotated[4, 0] == pytest.approx(0.0, abs=1e-12)
    assert rotated[5, 0] == pytest.approx(0.0, abs=1e-12)
    # hoop strain unchanged because DR has DR[2,2] = 1 and zero hoop coupling
    assert rotated[2, 0] == pytest.approx(0.10, rel=1e-12)


def test_set_start_rotation_preserves_axi_zeros_for_pure_meridional_F():
    """A purely meridional rotation of F (no hoop stretch) -> hoop component
    of any axi strain remains untouched."""
    F1 = np.eye(3)
    angle = np.pi / 8
    F1[0, 0] = np.cos(angle)
    F1[0, 1] = -np.sin(angle)
    F1[1, 0] = np.sin(angle)
    F1[1, 1] = np.cos(angle)
    F1 = np.asfortranarray(F1.reshape(3, 3, 1))
    F0 = np.asfortranarray(np.eye(3).reshape(3, 3, 1))
    _, _, DR, _ = sim.objective_rate("log", F0, F1, 1.0, True)

    DR0 = DR[:, :, 0]
    # DR must remain axi-block
    assert DR0[0, 2] == 0.0 and DR0[1, 2] == 0.0
    assert DR0[2, 0] == 0.0 and DR0[2, 1] == 0.0
    assert DR0[2, 2] == pytest.approx(1.0, abs=1e-14)
