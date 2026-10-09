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
