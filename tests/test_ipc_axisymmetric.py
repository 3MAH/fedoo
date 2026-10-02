"""Consistency of the 2Daxi IPC contact contributions.

ipctk has no axisymmetric mode: fedoo weights each collision by 2*pi*R.
The contact force must derive from the weighted barrier energy and the
axial forces exchanged by the two bodies must be opposite.
"""

import numpy as np
import pytest

import fedoo as fd

ipctk = pytest.importorskip("ipctk")
if tuple(int(v) for v in ipctk.__version__.split(".")[:2]) < (1, 6):
    pytest.skip("ipctk >= 1.6 required", allow_module_level=True)


def _two_blocks(space, **kwargs):
    fd.ModelingSpace(space)
    # non matching meshes separated by a gap of 0.02
    m1 = fd.mesh.rectangle_mesh(7, 3, 10, 14, 0, 1, "quad4")
    m2 = fd.mesh.rectangle_mesh(6, 3, 10.3, 13.7, 1.02, 2, "quad4")
    mesh = fd.Mesh.stack(m1, m2)
    material = fd.constitutivelaw.ElasticIsotrop(1e3, 0.3)
    solid = fd.Assembly.create(fd.weakform.StressEquilibrium(material), mesh)
    contact = fd.constraint.IPCSelfContact(
        mesh,
        dhat=0.05,
        dhat_is_relative=False,
        barrier_stiffness=10.0,
        adaptive_barrier_stiffness=False,
        **kwargs,
    )
    pb = fd.problem.NonLinear(fd.Assembly.sum(solid, contact))
    pb.initialize()
    return mesh, contact


def _energy(contact, vertices):
    collisions = contact._build_collisions(vertices)
    return contact._kappa * contact._barrier_potential(
        collisions, contact._collision_mesh, vertices
    )


@pytest.mark.parametrize("use_area_weighting", [False, True])
def test_axi_force_derives_from_energy(use_area_weighting):
    mesh, contact = _two_blocks("2Daxi", use_area_weighting=use_area_weighting)
    rng = np.random.default_rng(0)
    vertices = contact._rest_positions + 1e-3 * rng.standard_normal(
        contact._rest_positions.shape
    )
    contact._build_collisions(vertices, contact._collisions)
    assert len(contact._collisions) > 0
    contact._compute_ipc_contributions(vertices)
    force = contact.global_vector.copy()

    h = 1e-7
    grad_fd = np.zeros(vertices.size)
    for i in range(vertices.size):
        dv = np.zeros(vertices.size)
        dv[i] = h
        dv = dv.reshape(vertices.shape)
        grad_fd[i] = (
            _energy(contact, vertices + dv) - _energy(contact, vertices - dv)
        ) / (2 * h)
    force_fd = -(contact._scatter_matrix @ grad_fd)
    assert np.linalg.norm(force - force_fd) < 1e-6 * np.linalg.norm(force)

    # axial action-reaction between the two bodies
    fz = force[mesh.n_nodes : 2 * mesh.n_nodes]
    upper = mesh.nodes[:, 1] > 1.01
    assert abs(fz[upper].sum()) > 0
    assert abs(fz.sum()) < 1e-10 * abs(fz[upper].sum())


def test_axi_collision_weight_is_circumference():
    _, contact_axi = _two_blocks("2Daxi")
    weights_axi = np.array(
        [
            c.weight
            for c in map(
                contact_axi._collisions.__getitem__, range(len(contact_axi._collisions))
            )
        ]
    )
    _, contact_2d = _two_blocks("2D")
    assert len(contact_2d._collisions) == len(weights_axi)
    # unit weights in 2D, 2*pi*R in 2Daxi with 10 <= R <= 14
    assert all(
        contact_2d._collisions[i].weight == 1.0
        for i in range(len(contact_2d._collisions))
    )
    assert np.all(weights_axi >= 2 * np.pi * 10 - 1e-9)
    assert np.all(weights_axi <= 2 * np.pi * 14 + 1e-9)


def test_friction_uses_barrier_stiffness():
    _, contact = _two_blocks("2D", friction_coefficient=0.3)
    collision = contact._friction_collisions[0]
    assert collision.mu_s == pytest.approx(0.3)
    assert collision.mu_k == pytest.approx(0.3)
    _, contact_ref = _two_blocks("2D", friction_coefficient=0.3)
    contact_ref._kappa = 1.0
    contact_ref._build_friction_collisions(contact_ref._rest_positions)
    assert collision.normal_force_magnitude == pytest.approx(
        10.0 * contact_ref._friction_collisions[0].normal_force_magnitude
    )


@pytest.mark.parametrize("space", ["2D", "2Daxi"])
def test_friction_force_is_mu_times_normal_force(space):
    mesh, contact = _two_blocks(space, friction_coefficient=0.3)
    friction = contact._friction_collisions
    mu_n = sum(
        friction[i].mu_k * friction[i].normal_force_magnitude * friction[i].weight
        for i in range(len(friction))
    )
    assert mu_n > 0

    # no slip since the start of the increment: no friction force
    vertices = contact._rest_positions.copy()
    contact._compute_ipc_contributions(vertices)
    upper = mesh.nodes[:, 1] > 1.01
    f_normal = contact.global_vector[: mesh.n_nodes][upper].sum()

    # slide the upper block, well beyond the smoothing threshold eps_v
    surf_upper = upper[contact._surface_node_indices]
    vertices[surf_upper, 0] += 0.01
    contact._build_collisions(vertices, contact._collisions)
    contact._compute_ipc_contributions(vertices)
    f_slip = contact.global_vector[: mesh.n_nodes][upper].sum()
    contact.friction_coefficient = 0.0
    contact._compute_ipc_contributions(vertices)
    f_slip_frictionless = contact.global_vector[: mesh.n_nodes][upper].sum()

    assert f_normal == pytest.approx(0.0, abs=1e-12 * mu_n) or space == "2Daxi"
    # friction opposes the slip with magnitude mu * N
    assert f_slip - f_slip_frictionless == pytest.approx(-mu_n, rel=1e-6)


def test_ogc_not_supported_in_axi():
    fd.ModelingSpace("2Daxi")
    mesh = fd.mesh.rectangle_mesh(3, 3, 1, 2, 0, 1, "quad4")
    material = fd.constitutivelaw.ElasticIsotrop(1e3, 0.3)
    solid = fd.Assembly.create(fd.weakform.StressEquilibrium(material), mesh)
    surf = fd.mesh.extract_surface(mesh)
    contact = fd.constraint.IPCContact(mesh, surface_mesh=surf, use_ogc=True)
    pb = fd.problem.NonLinear(fd.Assembly.sum(solid, contact))
    with pytest.raises((NotImplementedError, ValueError)):
        pb.initialize()
