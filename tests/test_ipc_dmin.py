"""Tests for the ``dmin`` thickness offset of ``IPCContact``.

Two 2D elastic blocks are pushed against each other.  With a minimum
distance ``dmin`` larger than the final geometric gap, the barrier must
keep the surfaces ``dmin`` apart and transmit the contact force to the
lower block.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

import fedoo as fd
from fedoo.core.base import InvalidKinematicStateError

ipctk = pytest.importorskip("ipctk")


@pytest.fixture
def fresh_2d_space():
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dstress")
    yield
    fd.Assembly.delete_memory()


def _two_blocks(gap):
    mesh_bot = fd.mesh.rectangle_mesh(
        nx=8, ny=4, x_min=0.0, x_max=1.0, y_min=0.0, y_max=0.5
    )
    mesh_top = fd.mesh.rectangle_mesh(
        nx=6, ny=3, x_min=0.2, x_max=0.8, y_min=0.5 + gap, y_max=0.8 + gap
    )
    return fd.Mesh.stack(mesh_bot, mesh_top)


def test_dmin_keeps_surfaces_apart(fresh_2d_space):
    # Surface edge length is 0.1 (top block): dmin + dhat must stay below.
    gap, dmin, dhat, push = 0.3, 0.06, 0.02, 0.27
    mesh = _two_blocks(gap)

    mat = fd.constitutivelaw.ElasticIsotrop(1e4, 0.3)
    wf = fd.weakform.StressEquilibrium(mat, nlgeom=True)
    solid = fd.Assembly.create(wf, mesh)

    surf = fd.mesh.extract_surface(mesh)
    ipc = fd.constraint.IPCContact(
        mesh,
        surface_mesh=surf,
        dhat=dhat,
        dhat_is_relative=False,
        dmin=dmin,
        use_ccd=True,
    )
    assembly = fd.Assembly.sum(solid, ipc)

    pb = fd.problem.NonLinear(assembly)
    pb.bc.add("Dirichlet", mesh.find_nodes("Y", 0.0), "Disp", 0)
    top = mesh.find_nodes("Y", 0.8 + gap)
    pb.bc.add("Dirichlet", top, "DispX", 0)
    pb.bc.add("Dirichlet", top, "DispY", -push)
    pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=15)
    pb.nlsolve(dt=0.1, tmax=1.0, update_dt=True, print_info=0)

    # Without the offset the final gap (0.03) would exceed dhat and no
    # contact would occur.  With dmin the surfaces must stay dmin apart
    # and the barrier must be active.  ipctk returns squared distances.
    vertices = ipc._get_current_vertices(pb)
    assert len(ipc._collisions) > 0
    min_d = np.sqrt(
        ipc._collisions.compute_minimum_distance(ipc._collision_mesh, vertices)
    )
    assert min_d >= dmin * (1 - 1e-6)
    assert min_d < dmin + dhat

    # The contact force is transmitted: the top face of the lower block
    # is pushed down.
    disp = pb.get_disp()
    contact_face = mesh.find_nodes("Y", 0.5)
    assert disp[1, contact_face].min() < -0.005


def _strip_mesh(n_cols=8, h=0.1):
    """Flat tri3 strip, two rows of nodes spaced by ``h``."""
    x = np.arange(n_cols) * h
    nodes = np.vstack(
        [
            np.c_[x, np.zeros(n_cols), np.zeros(n_cols)],
            np.c_[x, np.full(n_cols, h), np.zeros(n_cols)],
        ]
    )
    elements = []
    for i in range(n_cols - 1):
        elements += [[i, i + 1, n_cols + i + 1], [i, n_cols + i + 1, n_cols + i]]
    return fd.Mesh(nodes, np.array(elements), "tri3")


def _shell_contact(mesh, dmin, dhat, **kwargs):
    space = fd.ModelingSpace("3D")
    for variable in ["DispX", "DispY", "DispZ"]:
        space.new_variable(variable)
    space.new_vector("Disp", ["DispX", "DispY", "DispZ"])
    contact = fd.constraint.IPCContact(
        mesh,
        surface_mesh=mesh,
        dhat=dhat,
        dhat_is_relative=False,
        dmin=dmin,
        barrier_stiffness=10,
        adaptive_barrier_stiffness=False,
        line_search_energy=False,
        **kwargs,
    )
    pb = fd.problem.NonLinear(contact)
    pb.initialize()
    return contact


def test_excluded_rings_allows_offset_above_edge_length():
    # Second-ring vertices of the strip are 0.2 apart, below dmin: the
    # default one-ring exclusion leaves an infeasible initial state.
    dmin, dhat = 0.25, 0.02
    with pytest.raises(InvalidKinematicStateError, match="dmin"):
        _shell_contact(_strip_mesh(), dmin, dhat)
    # A primitive pair is skipped only when all its vertex pairs lie within
    # the excluded rings, so the nearest allowed pair is about
    # (excluded_rings - 1) edge lengths away: five rings leave nothing
    # closer than dmin + dhat (the nearest allowed edge pair is 0.316 apart).
    contact = _shell_contact(_strip_mesh(), dmin, dhat, excluded_rings=5)
    assert contact._minimum_distance(contact._rest_positions) > dmin + dhat
    can_collide = contact._collision_mesh.can_collide
    assert can_collide(0, 6) and not can_collide(0, 5)


@pytest.mark.parametrize("rings", [0, 1.5, -2])
def test_invalid_excluded_rings(rings):
    with pytest.raises(ValueError, match="excluded_rings"):
        fd.constraint.IPCContact(None, excluded_rings=rings)
