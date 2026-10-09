"""IPC thickness offsets: activation, derivatives, CCD and rollback."""

import numpy as np
import pytest

import fedoo as fd
from fedoo.core.base import InvalidKinematicStateError

ipctk = pytest.importorskip("ipctk")
if tuple(int(v) for v in ipctk.__version__.split(".")[:2]) < (1, 6):
    pytest.skip("ipctk >= 1.6 required", allow_module_level=True)


def _contact(space="2D", gap=0.5, dmin=0.4, **kwargs):
    modeling = fd.ModelingSpace(space)
    for variable in ["DispX", "DispY"] + (["DispZ"] if space == "3D" else []):
        modeling.new_variable(variable)
    modeling.new_vector("Disp", list(modeling.list_variables()))
    if space == "3D":
        nodes = np.array(
            [
                [0, 0, 0],
                [2, 0, 0],
                [0, 2, 0],
                [0.2, 0.2, gap],
                [0.8, 0.2, gap],
                [0.2, 0.8, gap],
            ]
        )
        mesh = fd.Mesh(nodes, np.array([[0, 1, 2], [3, 4, 5]]), "tri3")
    else:
        nodes = np.array([[10, 0], [12, 0], [10.5, gap], [11.5, gap]])
        mesh = fd.Mesh(nodes, np.array([[0, 1], [2, 3]]), "lin2")
    contact = fd.constraint.IPCContact(
        mesh,
        surface_mesh=mesh,
        dhat=0.2,
        dhat_is_relative=False,
        dmin=dmin,
        barrier_stiffness=10,
        adaptive_barrier_stiffness=False,
        line_search_energy=False,
        **kwargs,
    )
    pb = fd.problem.NonLinear(contact)
    pb.initialize()
    return pb, contact


@pytest.mark.parametrize("space", ["2D", "3D", "2Daxi"])
@pytest.mark.parametrize("use_area_weighting", [False, True])
def test_offset_activation_and_energy_gradient(space, use_area_weighting):
    _, contact = _contact(space, use_area_weighting=use_area_weighting)
    assert len(contact._collisions) > 0
    assert all(
        contact._collisions[i].dmin == pytest.approx(0.4)
        for i in range(len(contact._collisions))
    )
    vertices = contact._rest_positions.copy()
    force = contact.global_vector.copy()
    gradient = np.zeros(vertices.size)
    h = 1e-7
    for i in range(vertices.size):
        delta = np.zeros(vertices.size)
        delta[i] = h
        delta = delta.reshape(vertices.shape)
        energies = []
        for trial in [vertices + delta, vertices - delta]:
            collisions = contact._build_collisions(trial)
            energies.append(
                contact._kappa
                * contact._barrier_potential(collisions, contact._collision_mesh, trial)
            )
        gradient[i] = (energies[0] - energies[1]) / (2 * h)
    np.testing.assert_allclose(
        force, -(contact._scatter_matrix @ gradient), rtol=1e-6, atol=1e-8
    )

    # Outside dmin+dhat the barrier and its force vanish.
    vertices[len(vertices) // 2 :, -1] += 0.11
    collisions = contact._build_collisions(vertices)
    assert len(collisions) == 0


@pytest.mark.parametrize("gap", [0.5, 1.0])
def test_ccd_preserves_offset_for_active_and_new_contacts(gap):
    pb, contact = _contact(gap=gap)
    correction = np.zeros(pb.n_dof)
    correction[pb.mesh.n_nodes + np.array([2, 3])] = -1.0
    alpha = contact._ccd_line_search(pb, correction)
    assert 0 < alpha < gap - 0.4
    for fraction in np.linspace(0, 1, 21):
        assert gap - fraction * alpha > 0.4


def test_ccd_fallback_never_drops_physical_offset(monkeypatch):
    pb, contact = _contact(gap=1.0)
    calls = []

    def fake_ccd(*args, min_distance, **kwargs):
        calls.append(min_distance)
        return 0 if len(calls) == 1 else 0.5

    monkeypatch.setattr(ipctk, "compute_collision_free_stepsize", fake_ccd)
    alpha = contact._ccd_line_search(pb, np.zeros(pb.n_dof))
    assert calls == pytest.approx([0.42, 0.4])
    assert alpha == pytest.approx(0.45)


def test_offset_survives_collision_rebuild_and_friction_rollback():
    pb, contact = _contact(friction_coefficient=0.3)
    reference_force = contact.global_vector.copy()
    pb._dU = np.zeros(pb.n_dof)
    pb._dU[2:4] = 0.02  # tangential slip of the upper edge
    contact.update(pb)
    assert np.linalg.norm(contact.global_vector - reference_force) > 0
    pb._dU = 0
    contact.to_start(pb)
    np.testing.assert_allclose(contact.global_vector, reference_force)
    assert all(
        contact._collisions[i].dmin == pytest.approx(0.4)
        for i in range(len(contact._collisions))
    )


@pytest.mark.parametrize("dmin", [-1, np.nan, np.inf])
@pytest.mark.parametrize(
    "kind", [fd.constraint.IPCContact, fd.constraint.IPCSelfContact]
)
def test_invalid_offset_is_rejected(kind, dmin):
    with pytest.raises(ValueError, match="dmin"):
        kind(None, dmin=dmin)


def test_infeasible_initial_gap_is_rejected():
    with pytest.raises(InvalidKinematicStateError, match="dmin"):
        _contact(gap=0.3)


def test_ogc_rejects_offset():
    with pytest.raises(NotImplementedError, match="dmin"):
        fd.constraint.IPCContact(None, dmin=0.4, use_ogc=True)


def test_constructor_name_and_space_are_last():
    space = fd.ModelingSpace("2D")
    contact = fd.constraint.IPCContact(
        None,
        None,
        False,
        1e-3,
        True,
        None,
        0,
        1e-3,
        "lbvh",
        True,
        False,
        None,
        False,
        False,
        0.0,
        1,
        True,
        "LegacyContact",
        space,
    )
    self_contact = fd.constraint.IPCSelfContact(
        None,
        1e-3,
        True,
        None,
        0,
        1e-3,
        "lbvh",
        True,
        False,
        None,
        False,
        False,
        0.0,
        1,
        True,
        "LegacySelfContact",
        space,
    )
    for assembly, name in [
        (contact, "LegacyContact"),
        (self_contact, "LegacySelfContact"),
    ]:
        assert assembly.name == name
        assert assembly.space is space
        assert assembly._dmin == 0
        assert not assembly._use_area_weighting


def test_proximity_safeguard_uses_gap_above_offset():
    pb, contact = _contact(gap=0.401)
    pb._nr_min_subiter = 0
    contact._n_collisions_at_start = len(contact._collisions)
    contact.update(pb)
    assert pb._nr_min_subiter == 3


def test_barrier_stiffness_tuning_receives_offset(monkeypatch):
    _, contact = _contact()
    calls = {}
    initial = ipctk.initial_barrier_stiffness
    update = ipctk.update_barrier_stiffness

    def initial_with_offset(*args, **kwargs):
        calls["initial"] = kwargs["dmin"]
        return initial(*args, **kwargs)

    def update_with_offset(*args, **kwargs):
        calls["update"] = kwargs["dmin"]
        return update(*args, **kwargs)

    monkeypatch.setattr(ipctk, "initial_barrier_stiffness", initial_with_offset)
    monkeypatch.setattr(ipctk, "update_barrier_stiffness", update_with_offset)
    contact._initialize_kappa(contact._rest_positions)
    contact._update_kappa_adaptive(contact._rest_positions)
    assert calls == {"initial": 0.4, "update": 0.4}


def test_self_contact_and_relative_dhat_keep_absolute_offset():
    fd.ModelingSpace("2D")
    material = fd.constitutivelaw.ElasticIsotrop(1e3, 0.3)
    lower = fd.mesh.rectangle_mesh(3, 3, 0, 2, 0, 1, "quad4")
    upper = fd.mesh.rectangle_mesh(3, 3, 0.5, 1.5, 1.5, 2.5, "quad4")
    mesh = fd.Mesh.stack(lower, upper)
    solid = fd.Assembly.create(fd.weakform.StressEquilibrium(material), mesh)
    contact = fd.constraint.IPCSelfContact(
        mesh,
        dhat=0.1,
        dhat_is_relative=True,
        dmin=0.4,
        barrier_stiffness=10,
    )
    pb = fd.problem.NonLinear(fd.Assembly.sum(solid, contact))
    pb.initialize()
    assert contact._actual_dhat == pytest.approx(0.1 * np.linalg.norm([2, 2.5]))
    assert len(contact._collisions) > 0
    assert all(
        contact._collisions[i].dmin == pytest.approx(0.4)
        for i in range(len(contact._collisions))
    )
