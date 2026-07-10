"""Tests for the cylindrical->Cartesian rotation of vector / tensor fields
when revolving 2Daxi data into a full 3D representation.

These tests pin down the analytical rotation rules for the helper
``_revolve_axi_field``:

* Scalar fields are tiled along theta unchanged.
* 2-vector fields ``(dr, dz)`` map to 3D Cartesian
  ``(dr*cos, dr*sin, dz)`` at each theta.
* 6-tensor fields in fedoo's slot ordering ``(rr, zz, tt, rz, 0, 0)``
  rotate about the symmetry axis Z by theta to produce the Cartesian
  Voigt 6-vector.

If a future change accidentally drops the rotation, these tests fail.
"""

import numpy as np
import pytest

from fedoo.post_processing.axi_to_3d import _revolve_axi_field


def _theta(n=4):
    return np.linspace(0, 2 * np.pi, n, endpoint=False)


def test_scalar_field_is_tiled():
    n_nodes = 3
    n_theta = 5
    field = np.array([1.0, 2.0, 3.0])
    theta = _theta(n_theta)
    out = _revolve_axi_field(field, theta, n_nodes)
    assert out.shape == (1, n_nodes * n_theta)
    # node-major layout: each 2D node value repeated for every theta.
    # Reshape to (n_nodes_2d, n_theta); each row should be constant.
    grid = out.reshape(1, n_nodes, n_theta)
    for i in range(n_nodes):
        assert np.allclose(grid[0, i], field[i])


def test_vector_field_disp():
    n_nodes = 2
    # node 0: (dr, dz) = (1, 0); node 1: (dr, dz) = (0, 5)
    field = np.array([[1.0, 0.0], [0.0, 5.0]])
    theta = _theta(4)  # 0, pi/2, pi, 3pi/2
    out = _revolve_axi_field(field, theta, n_nodes)

    assert out.shape == (3, n_nodes * 4)

    # node-major layout: index [node_i, theta_k] after reshape (n_nodes, n_theta).
    out_X = out[0].reshape(n_nodes, 4)
    out_Y = out[1].reshape(n_nodes, 4)
    out_Z = out[2].reshape(n_nodes, 4)

    # node 0 at theta = 0: (1*cos0, 1*sin0, 0) = (1, 0, 0)
    assert out_X[0, 0] == pytest.approx(1.0)
    assert out_Y[0, 0] == pytest.approx(0.0)
    assert out_Z[0, 0] == pytest.approx(0.0)

    # node 0 at theta = pi/2: (cos, sin, 0) = (0, 1, 0)
    assert out_X[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert out_Y[0, 1] == pytest.approx(1.0)

    # node 1 at theta = pi: (0*cos, 0*sin, 5) = (0, 0, 5) regardless
    assert out_X[1, 2] == pytest.approx(0.0)
    assert out_Y[1, 2] == pytest.approx(0.0, abs=1e-12)
    assert out_Z[1, 2] == pytest.approx(5.0)


def test_tensor_pure_radial_stress():
    """Pure sigma_rr at theta=0 -> sigma_xx in Cartesian; at theta=pi/2 -> sigma_yy."""
    n_nodes = 1
    # (s_rr, s_zz, s_tt, s_rz, 0, 0) = (1, 0, 0, 0, 0, 0)
    field = np.array([[1.0], [0.0], [0.0], [0.0], [0.0], [0.0]])
    theta = np.array([0.0, np.pi / 2])
    out = _revolve_axi_field(field, theta, n_nodes)
    assert out.shape == (6, 2)

    # at theta = 0: sigma_xx = 1, sigma_yy = 0, sigma_zz = 0, off-diag = 0
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(0.0, abs=1e-12)
    assert out[3, 0] == pytest.approx(0.0, abs=1e-12)

    # at theta = pi/2: sigma_xx = 0, sigma_yy = 1
    assert out[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert out[1, 1] == pytest.approx(1.0)
    assert out[3, 1] == pytest.approx(0.0, abs=1e-12)


def test_tensor_pure_hoop_stress():
    """Pure sigma_tt -> rotates to sigma_yy at theta=0 and sigma_xx at theta=pi/2."""
    n_nodes = 1
    field = np.array([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]])
    theta = np.array([0.0, np.pi / 2])
    out = _revolve_axi_field(field, theta, n_nodes)

    # at theta=0: e_theta = (0, 1, 0), so sigma_tt -> sigma_yy
    assert out[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert out[1, 0] == pytest.approx(1.0)
    # at theta=pi/2: e_theta = (-1, 0, 0), so sigma_tt -> sigma_xx
    assert out[0, 1] == pytest.approx(1.0)
    assert out[1, 1] == pytest.approx(0.0, abs=1e-12)


def test_tensor_pure_axial_stress_invariant():
    """sigma_zz_cyl maps directly to sigma_zz_cartesian at every theta."""
    n_nodes = 1
    field = np.array([[0.0], [3.0], [0.0], [0.0], [0.0], [0.0]])
    theta = _theta(8)
    out = _revolve_axi_field(field, theta, n_nodes)
    # slot 2 of the Cartesian 6-vector should be 3.0 at every theta
    assert np.allclose(out[2], 3.0)
    # all other slots should be zero
    assert np.allclose(out[0], 0.0)
    assert np.allclose(out[1], 0.0)
    assert np.allclose(out[3], 0.0, atol=1e-12)
    assert np.allclose(out[4], 0.0, atol=1e-12)
    assert np.allclose(out[5], 0.0, atol=1e-12)


def test_tensor_rz_shear_rotates_to_xz_yz():
    """sigma_rz at theta=0 -> sigma_xz, at theta=pi/2 -> sigma_yz."""
    n_nodes = 1
    field = np.array([[0.0], [0.0], [0.0], [2.0], [0.0], [0.0]])
    theta = np.array([0.0, np.pi / 2, np.pi])
    out = _revolve_axi_field(field, theta, n_nodes)

    # theta = 0: e_r = (1, 0, 0), so sigma_rz -> sigma_xz (slot 4)
    assert out[4, 0] == pytest.approx(2.0)
    assert out[5, 0] == pytest.approx(0.0, abs=1e-12)
    # theta = pi/2: e_r = (0, 1, 0), so sigma_rz -> sigma_yz (slot 5)
    assert out[4, 1] == pytest.approx(0.0, abs=1e-12)
    assert out[5, 1] == pytest.approx(2.0)
    # theta = pi: e_r = (-1, 0, 0), so sigma_rz -> -sigma_xz
    assert out[4, 2] == pytest.approx(-2.0)
    assert out[5, 2] == pytest.approx(0.0, abs=1e-12)


def test_tensor_isotropic_pressure_invariant():
    """An isotropic cylindrical state -p*I rotates to -p*I in Cartesian."""
    n_nodes = 1
    p = 5.0
    field = np.array([[-p], [-p], [-p], [0.0], [0.0], [0.0]])
    theta = _theta(8)
    out = _revolve_axi_field(field, theta, n_nodes)
    # diagonals all -p, off-diagonals all 0
    assert np.allclose(out[0], -p)
    assert np.allclose(out[1], -p)
    assert np.allclose(out[2], -p)
    assert np.allclose(out[3], 0.0, atol=1e-12)
    assert np.allclose(out[4], 0.0, atol=1e-12)
    assert np.allclose(out[5], 0.0, atol=1e-12)


def test_strain_field_uses_engineering_shear_for_xy():
    """For a Strain field, slot 3 of the output is gamma_xy = 2 * eps_xy.

    A pure radial strain eps_rr at theta=pi/4 in Cartesian gives
        eps_xy_tensor = sc * (eps_rr - 0) = (1/2) * eps_rr
        gamma_xy_engineering = 2 * eps_xy_tensor = eps_rr.
    """
    n_nodes = 1
    eps_rr = 0.04
    field = np.array([[eps_rr], [0.0], [0.0], [0.0], [0.0], [0.0]])
    theta = np.array([np.pi / 4])
    out = _revolve_axi_field(field, theta, n_nodes, field_name="Strain")
    assert out[3, 0] == pytest.approx(eps_rr, rel=1e-12)


def test_stress_field_uses_tensor_xy_no_factor_of_2():
    """For a stress (default), slot 3 = sigma_xy with no engineering factor."""
    n_nodes = 1
    sigma_rr = 100.0
    field = np.array([[sigma_rr], [0.0], [0.0], [0.0], [0.0], [0.0]])
    theta = np.array([np.pi / 4])
    out = _revolve_axi_field(field, theta, n_nodes, field_name="Stress")
    # sigma_xy = sin*cos*(sigma_rr - 0) = (1/2)*sigma_rr at theta = pi/4
    assert out[3, 0] == pytest.approx(0.5 * sigma_rr, rel=1e-12)


def test_explicit_kind_overrides_field_name_heuristic():
    """When kind is explicit, it wins over any name-based detection."""
    n_nodes = 1
    val = 0.04
    field = np.array([[val], [0.0], [0.0], [0.0], [0.0], [0.0]])
    theta = np.array([np.pi / 4])

    # name says "Strain" but kind="stress" -> no factor of 2
    out_stress = _revolve_axi_field(
        field, theta, n_nodes, kind="stress", field_name="MyStrainLikeField"
    )
    assert out_stress[3, 0] == pytest.approx(0.5 * val, rel=1e-12)

    # name says "Stress" but kind="strain" -> factor of 2
    out_strain = _revolve_axi_field(
        field, theta, n_nodes, kind="strain", field_name="MyStressLikeField"
    )
    assert out_strain[3, 0] == pytest.approx(val, rel=1e-12)


def test_invalid_kind_raises():
    """Unknown kind must raise."""
    n_nodes = 1
    field = np.zeros((6, 1))
    theta = np.array([0.0])
    with pytest.raises(ValueError, match="kind"):
        _revolve_axi_field(field, theta, n_nodes, kind="bogus")


if __name__ == "__main__":
    pytest.main([__file__])
