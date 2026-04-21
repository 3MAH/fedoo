"""Analytical validation of rigid body free fall under gravity."""

import numpy as np
import pytest

import fedoo as fd


G = 9.81


def _run_freefall(dt, t_end, mass=1.0, z0=1.0, radius=0.1):
    try:
        import pyvista as pv
    except ImportError:  # pragma: no cover
        pytest.skip("pyvista not available")

    space = fd.ModelingSpace("3D")
    space.new_variable("DispX")
    space.new_variable("DispY")
    space.new_variable("DispZ")
    space.new_vector("Disp", ("DispX", "DispY", "DispZ"))

    mesh = fd.Mesh.from_pyvista(
        pv.Sphere(
            radius=radius, center=(0, 0, z0), theta_resolution=6, phi_resolution=6
        )
    )
    body = fd.constraint.RigidBody(
        mesh,
        mass=mass,
        inertia_tensor=0.004 * np.eye(3),
        center_of_mass=np.array([0.0, 0.0, z0]),
    )
    body.set_force([0.0, 0.0, -mass * G])

    pb = body.solve(dt=dt, tmax=t_end, print_info=0)
    dof = pb.get_dof_solution()
    idx = body.assembly._dof_indices
    return z0 + dof[idx[2]]


@pytest.mark.parametrize("dt", [1e-3, 5e-4])
def test_freefall_matches_analytical_solution(dt):
    t_end = 0.3
    z_sim = _run_freefall(dt=dt, t_end=t_end)
    z_exact = 1.0 - 0.5 * G * t_end**2
    # Newmark (β=0.25, γ=0.5) is energy-conserving and exact on quadratic
    # trajectories up to round-off; 1e-4 m is ample margin.
    assert abs(z_sim - z_exact) < 1e-4


def test_freefall_no_horizontal_drift():
    t_end = 0.3
    try:
        import pyvista as pv
    except ImportError:  # pragma: no cover
        pytest.skip("pyvista not available")

    space = fd.ModelingSpace("3D")
    space.new_variable("DispX")
    space.new_variable("DispY")
    space.new_variable("DispZ")
    space.new_vector("Disp", ("DispX", "DispY", "DispZ"))

    mesh = fd.Mesh.from_pyvista(
        pv.Sphere(radius=0.1, center=(0, 0, 1.0), theta_resolution=6, phi_resolution=6)
    )
    body = fd.constraint.RigidBody(
        mesh,
        mass=1.0,
        inertia_tensor=0.004 * np.eye(3),
        center_of_mass=np.array([0.0, 0.0, 1.0]),
    )
    body.set_force([0.0, 0.0, -G])

    pb = body.solve(dt=1e-3, tmax=t_end, print_info=0)
    dof = pb.get_dof_solution()
    idx = body.assembly._dof_indices
    dx, dy = dof[idx[0]], dof[idx[1]]
    assert abs(dx) < 1e-10
    assert abs(dy) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])
