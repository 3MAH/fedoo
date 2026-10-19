"""IPC contact between 3D beam centerlines (``lin2`` curves in 3D).

The centerlines are codimensional collision edges; with ``dmin`` equal to
the fiber diameter the barrier models contact between equal-radius rods.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

import fedoo as fd
from fedoo.core.base import InvalidKinematicStateError

ipctk = pytest.importorskip("ipctk")


@pytest.fixture(autouse=True)
def fresh_memory():
    fd.Assembly.delete_memory()
    yield
    fd.Assembly.delete_memory()


def _disp_space():
    space = fd.ModelingSpace("3D")
    for variable in ["DispX", "DispY", "DispZ"]:
        space.new_variable(variable)
    space.new_vector("Disp", ["DispX", "DispY", "DispZ"])
    return space


def _crossed_rods(gap, n_nodes=10, half_length=1.0):
    """Rod along X at z=0 and rod along Y at z=gap, crossing mid-element."""
    s = np.linspace(-half_length, half_length, n_nodes)
    zero = np.zeros(n_nodes)
    nodes = np.vstack([np.c_[s, zero, zero], np.c_[zero, s, zero + gap]])
    elements = np.vstack(
        [
            np.c_[np.arange(n_nodes - 1), np.arange(1, n_nodes)],
            np.c_[np.arange(n_nodes - 1), np.arange(1, n_nodes)] + n_nodes,
        ]
    )
    return fd.Mesh(nodes, elements, "lin2")


def _initialized_contact(contact):
    pb = fd.problem.NonLinear(contact)
    pb.initialize()
    return contact


def _contact(mesh, dmin, dhat, self_contact=False, **kwargs):
    options = dict(
        dhat=dhat,
        dhat_is_relative=False,
        dmin=dmin,
        barrier_stiffness=10,
        adaptive_barrier_stiffness=False,
        line_search_energy=False,
        **kwargs,
    )
    if self_contact:
        return _initialized_contact(fd.constraint.IPCSelfContact(mesh, **options))
    return _initialized_contact(
        fd.constraint.IPCContact(mesh, surface_mesh=mesh, **options)
    )


@pytest.mark.parametrize("use_area_weighting", [False, True])
@pytest.mark.parametrize("self_contact", [False, True])
def test_lin2_mesh_in_3d_is_a_collision_curve(self_contact, use_area_weighting):
    _disp_space()
    gap, dmin, dhat = 0.15, 0.1, 0.1
    mesh = _crossed_rods(gap)
    contact = _contact(
        mesh,
        dmin,
        dhat,
        self_contact=self_contact,
        use_area_weighting=use_area_weighting,
    )
    if use_area_weighting:
        # ipctk weights the codimensional edges by their length
        lengths = np.linalg.norm(
            np.diff(mesh.nodes[mesh.elements], axis=1)[:, 0], axis=1
        )
        edge_areas = contact._collision_mesh.edge_areas
        if callable(edge_areas):  # method in ipctk 1.6, property after #273
            edge_areas = edge_areas()
        assert np.allclose(edge_areas, lengths)

    collision_mesh = contact._collision_mesh
    assert collision_mesh.num_faces == 0
    assert collision_mesh.num_edges == mesh.n_elements
    if self_contact:
        assert contact._surface_mesh is mesh

    # The crossing rods are gap apart, inside dmin + dhat: active contact.
    assert len(contact._collisions) > 0
    assert np.isclose(contact._minimum_distance(contact._rest_positions), gap)


def test_lin2_rods_closer_than_diameter_are_rejected():
    _disp_space()
    with pytest.raises(InvalidKinematicStateError, match="dmin"):
        _contact(_crossed_rods(gap=0.08), dmin=0.1, dhat=0.05)


def test_excluded_rings_on_fine_rod():
    # Straight rod with edge length 0.1, below dmin: the vertex pairs 2
    # edges apart are 0.2 < dmin and must not collide.
    _disp_space()
    n_nodes, h, dmin, dhat = 12, 0.1, 0.25, 0.02
    nodes = np.c_[np.arange(n_nodes) * h, np.zeros(n_nodes), np.zeros(n_nodes)]
    elements = np.c_[np.arange(n_nodes - 1), np.arange(1, n_nodes)]
    mesh = fd.Mesh(nodes, elements, "lin2")

    # Edges (0, 1) and (3, 4) are 0.2 apart; with 3 rings the vertex
    # pair (0, 4) is allowed, so the pair collides below dmin.
    with pytest.raises(InvalidKinematicStateError, match="dmin"):
        _contact(mesh, dmin, dhat, excluded_rings=3)
    # With 4 rings, the nearest allowed pair, edges (0, 1) and (4, 5), is
    # 0.3 apart, above dmin + dhat.
    contact = _contact(mesh, dmin, dhat, excluded_rings=4)
    assert contact._minimum_distance(contact._rest_positions) > dmin + dhat
    can_collide = contact._collision_mesh.can_collide
    assert can_collide(0, 5) and not can_collide(0, 4)


def test_crossed_beams_contact():
    # A pinned beam along Y is pushed onto a clamped beam along X. Without
    # contact the final centerline gap would be gap - push = 0.05 < dmin.
    fd.ModelingSpace("3D")
    radius, gap, push, dhat = 0.05, 0.2, 0.15, 0.02
    dmin = 2 * radius
    n_nodes = 10  # crossing at mid-element; edge length 0.22 > dmin + dhat
    mesh = _crossed_rods(gap, n_nodes=n_nodes)

    material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
    section = fd.constitutivelaw.BeamCircular(material, radius)
    beams = fd.Assembly.create(fd.weakform.BeamEquilibrium(section), mesh)
    contact = fd.constraint.IPCContact(
        mesh,
        surface_mesh=mesh,
        dhat=dhat,
        dhat_is_relative=False,
        dmin=dmin,
        use_ccd=True,
    )
    pb = fd.problem.NonLinear(fd.Assembly.sum(beams, contact))

    lower_ends = [0, n_nodes - 1]
    upper_ends = [n_nodes, 2 * n_nodes - 1]
    pb.bc.add("Dirichlet", lower_ends, "Disp", 0)
    pb.bc.add("Dirichlet", lower_ends, "Rot", 0)
    pb.bc.add("Dirichlet", upper_ends, "DispX", 0)
    pb.bc.add("Dirichlet", upper_ends, "DispY", 0)
    pb.bc.add("Dirichlet", upper_ends, "DispZ", -push)
    pb.bc.add("Dirichlet", upper_ends, "RotY", 0)  # no rigid spin about Y
    pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=15)
    pb.nlsolve(dt=0.1, tmax=1.0, update_dt=True, print_info=0)

    vertices = contact._get_current_vertices(pb)
    assert len(contact._collisions) > 0
    min_d = contact._minimum_distance(vertices)
    assert dmin * (1 - 1e-6) <= min_d < dmin + dhat

    # The contact force bends both beams: the lower one goes down, the
    # upper one stays above its prescribed end displacement.
    disp_z = pb.get_disp()[2]
    assert disp_z[:n_nodes].min() < -0.005
    assert disp_z[n_nodes:].max() > -push + 0.005


def test_energy_line_search_accounts_for_external_forces():
    # The upper beam is pushed onto the lower one by a dead force, not by a
    # prescribed displacement. The energy backtracking used to ignore the
    # external work: under force control the elastic energy increases along
    # every admissible step, alpha fell to 2**-12 and the increments failed.
    fd.ModelingSpace("3D")
    radius, gap, dhat, n_nodes = 0.05, 0.2, 0.02, 10
    mesh = _crossed_rods(gap, n_nodes=n_nodes)
    section = fd.constitutivelaw.BeamCircular(
        fd.constitutivelaw.ElasticIsotrop(1e5, 0.3), radius, k=0.9
    )
    wf = fd.weakform.BeamEquilibrium(section, nlgeom=True, kinematics="exact")
    contact = fd.constraint.IPCContact(
        mesh,
        surface_mesh=mesh,
        dhat=dhat,
        dhat_is_relative=False,
        dmin=2 * radius,
        use_ccd=True,
        ccd_method="additive",
    )
    pb = fd.problem.NonLinear(fd.Assembly.sum(fd.Assembly.create(wf, mesh), contact))
    lower_ends = [0, n_nodes - 1]
    upper_ends = [n_nodes, 2 * n_nodes - 1]
    pb.bc.add("Dirichlet", lower_ends, "Disp", 0)
    pb.bc.add("Dirichlet", lower_ends, "Rot", 0)
    pb.bc.add("Dirichlet", upper_ends, "Disp", 0)
    pb.bc.add("Dirichlet", upper_ends, "RotY", 0)
    pb.bc.add("Neumann", [n_nodes + n_nodes // 2], "DispZ", -20.0)
    pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=15)
    increments = []
    pb.nlsolve(
        dt=0.1,
        tmax=1.0,
        update_dt=True,
        dt_max=0.1,
        print_info=0,
        callback=lambda problem: increments.append(problem.time),
    )
    # no increment was cut (14 increments with 11 failures before the fix)
    assert len(increments) == 10
    vertices = contact._get_current_vertices(pb)
    assert len(contact._collisions) > 0
    assert contact._minimum_distance(vertices) >= 2 * radius * (1 - 1e-6)
