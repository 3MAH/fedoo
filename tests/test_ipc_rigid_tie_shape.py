"""Regression test: IPCContact must not double-count global DOFs.

Before this fix, ``_compute_ipc_contributions`` padded ``global_matrix``
and ``global_vector`` by ``self._n_global_dof`` whenever there were extra
global DOFs and no rigid bodies registered via ``add_rigid_body``. The
scatter matrix already maps to the full ``n_dof = nvar*n_nodes +
n_global_dof``, so that padding inflated the result to ``n_dof +
n_global_dof`` rows, mismatching the FEM ``Assembly`` sibling and
raising ``ValueError: inconsistent shapes`` inside ``assembly_sum``.

The tests below exercise:
 * RigidTie (12 global DOFs) + IPCContact across a deformable bottom +
   two rigid-tied blocks.
 * PeriodicBC + IPCContact on a single periodic box.

For each, we run ``nlsolve`` for one increment and assert no
``ValueError`` and that the IPC and elastic blocks have the same
matrix/vector shape.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

import fedoo as fd

ipctk = pytest.importorskip("ipctk")


def _reset_space():
    """Ensure each test starts with a fresh ModelingSpace."""
    fd.Assembly.delete_memory()


@pytest.fixture
def fresh_3d_space():
    _reset_space()
    fd.ModelingSpace("3D")
    yield
    _reset_space()


def test_rigid_tie_plus_ipc_assembles_with_consistent_shape(fresh_3d_space):
    """Two RigidTie + IPCContact: matrix shapes must match across all
    assemblies in the sum."""
    # Deformable bottom plate
    mesh_bot = fd.mesh.box_mesh(
        nx=4,
        ny=4,
        nz=3,
        x_min=0,
        x_max=1,
        y_min=0,
        y_max=1,
        z_min=0.0,
        z_max=0.5,
    )
    # Two rigid blocks above with an initial gap
    mesh_A = fd.mesh.box_mesh(
        nx=3,
        ny=3,
        nz=2,
        x_min=0.05,
        x_max=0.40,
        y_min=0.30,
        y_max=0.70,
        z_min=0.70,
        z_max=0.85,
    )
    mesh_B = fd.mesh.box_mesh(
        nx=3,
        ny=3,
        nz=2,
        x_min=0.60,
        x_max=0.95,
        y_min=0.30,
        y_max=0.70,
        z_min=0.70,
        z_max=0.85,
    )

    mesh = fd.Mesh.stack(fd.Mesh.stack(mesh_bot, mesh_A), mesh_B)
    n_bot = mesh_bot.n_nodes
    nodes_A = np.arange(n_bot, n_bot + mesh_A.n_nodes)
    nodes_B = np.arange(n_bot + mesh_A.n_nodes, mesh.n_nodes)

    mat = fd.constitutivelaw.ElasticIsotrop(1e4, 0.3)
    wf = fd.weakform.StressEquilibrium(mat, nlgeom=True)
    solid = fd.Assembly.create(wf, mesh)

    surf = fd.mesh.extract_surface(mesh, quad2tri=True)
    ipc = fd.constraint.IPCContact(
        mesh,
        surface_mesh=surf,
        dhat=0.03,
        dhat_is_relative=False,
        use_ccd=False,
    )
    assembly = fd.Assembly.sum(solid, ipc)

    pb = fd.problem.NonLinear(assembly)
    # Two RigidTie constraints contribute 12 global DOFs total
    pb.bc.add(fd.constraint.RigidTie(nodes_A, name="tieA"))
    pb.bc.add(fd.constraint.RigidTie(nodes_B, name="tieB"))

    pb.bc.add("Dirichlet", mesh.find_nodes("Z", 0.0), "Disp", 0)
    # Push rigid blocks down — initial gap 0.20, dhat=0.03, push by 0.19 so
    # we cross into the barrier zone and exercise the padded-matrix code path.
    pb.bc.add("Dirichlet", "RigidDispZ", -0.19)
    pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=15)

    expected_n_dof = pb.n_node_dof + pb.n_global_dof
    assert pb.n_global_dof == 12

    # Drive nlsolve through one increment. Before the fix this raised
    # ``ValueError: inconsistent shapes`` from ``assembly_sum`` the moment
    # IPC detected its first collision.
    pb.nlsolve(dt=0.2, tmax=0.2, update_dt=True, print_info=0)

    assert ipc.global_matrix.shape == (expected_n_dof, expected_n_dof)
    assert ipc.global_vector.shape == (expected_n_dof,)
    assert solid.global_matrix.shape == ipc.global_matrix.shape
    assert solid.global_vector.shape == ipc.global_vector.shape


def test_periodic_bc_plus_ipc_assembles_with_consistent_shape(fresh_3d_space):
    """PeriodicBC + IPCContact: even with no contact, the global-DOF
    padding path must produce matrices of size ``n_dof``, not
    ``n_dof + n_global_dof``."""
    mesh = fd.mesh.box_mesh(
        nx=4,
        ny=4,
        nz=4,
        x_min=0,
        x_max=1,
        y_min=0,
        y_max=1,
        z_min=0,
        z_max=1,
    )

    mat = fd.constitutivelaw.ElasticIsotrop(1e4, 0.3)
    wf = fd.weakform.StressEquilibrium(mat, nlgeom=False)
    solid = fd.Assembly.create(wf, mesh)

    surf = fd.mesh.extract_surface(mesh, quad2tri=True)
    # No initial contact; we only need IPC's assemble path to be exercised
    # alongside the PeriodicBC-driven global DOFs.
    ipc = fd.constraint.IPCContact(
        mesh,
        surface_mesh=surf,
        dhat=0.02,
        dhat_is_relative=False,
        use_ccd=False,
    )
    assembly = fd.Assembly.sum(solid, ipc)

    pb = fd.problem.NonLinear(assembly)
    pb.bc.add(fd.constraint.PeriodicBC(periodicity_type="small_strain", dim=3))
    # Pin one corner against rigid body translation
    corner = mesh.nearest_node([0.0, 0.0, 0.0])
    pb.bc.add("Dirichlet", corner, "Disp", 0)
    # Drive a small mean strain — small_strain BC exposes E_xx as a global DOF
    pb.bc.add("Dirichlet", "E_xx", 0.005)
    for v in ("E_yy", "E_zz", "E_xy", "E_xz", "E_yz"):
        pb.bc.add("Dirichlet", v, 0.0)

    pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=10)

    expected_n_dof = pb.n_node_dof + pb.n_global_dof
    assert pb.n_global_dof > 0

    pb.nlsolve(dt=1.0, tmax=1.0, update_dt=True, print_info=0)

    assert ipc.global_matrix.shape == (expected_n_dof, expected_n_dof)
    assert ipc.global_vector.shape == (expected_n_dof,)
    assert solid.global_matrix.shape == ipc.global_matrix.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
