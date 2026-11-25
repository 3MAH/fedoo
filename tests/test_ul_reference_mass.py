"""Inertia on the updated Lagrangian mesh integrates the reference mass."""

import numpy as np
import pytest

import fedoo as fd

DENSITY = 2.0
STRETCH = (1.3, 0.9, 1.2)


def _system(space, nlgeom, dynamic, stretched=True):
    fd.Assembly.delete_memory()
    fd.ModelingSpace(space)
    if space == "3D":
        mesh = fd.mesh.box_mesh(nx=3, ny=3, nz=3, elm_type="hex8")
    else:
        mesh = fd.mesh.rectangle_mesh(nx=3, ny=3, x_min=1, x_max=2, elm_type="quad4")
    law = fd.constitutivelaw.ElasticIsotrop(100, 0.2)
    law.set_density(DENSITY)
    assembly = fd.Assembly.create(
        fd.weakform.StressEquilibrium(law, nlgeom=nlgeom), mesh
    )
    pb = fd.problem.NonLinear(assembly)
    pb.set_solver("direct_scipy")
    if dynamic:
        pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())
    pb.initialize()
    pb.dtime = 0.2
    pb.set_start()
    trial = np.zeros(pb.n_dof)
    n = mesh.n_nodes
    if stretched:
        for i, stretch in enumerate(STRETCH[: mesh.ndim]):
            trial[i * n : (i + 1) * n] = (stretch - 1) * mesh.nodes[:, i]
    pb._dU = trial
    pb.update()
    return assembly.current.get_global_matrix().toarray()[:n, :n]


def _mass(space, nlgeom, stretched=True):
    """Total mass in the Newmark system (beta = 1/4): M / (beta dt^2)."""
    stiffness = _system(space, nlgeom, False, stretched)
    dynamic = _system(space, nlgeom, True, stretched)
    return (dynamic - stiffness).sum() * 0.25 * 0.2**2


@pytest.mark.parametrize("space, reference_mass", [("3D", 2.0), ("2Dplane", 2.0)])
@pytest.mark.parametrize("nlgeom", ["TL", "UL"])
def test_mass_does_not_follow_the_volume(space, reference_mass, nlgeom):
    np.testing.assert_allclose(_mass(space, nlgeom), reference_mass, rtol=1e-12)


def test_axisymmetric_mass_does_not_follow_the_volume():
    # ring 1 <= r <= 2 of unit height: rho * pi * (2^2 - 1^2)
    reference_mass = DENSITY * np.pi * 3
    np.testing.assert_allclose(
        _mass("2Daxi", "UL", stretched=False), reference_mass, rtol=1e-12
    )
    np.testing.assert_allclose(_mass("2Daxi", "UL"), reference_mass, rtol=1e-12)
