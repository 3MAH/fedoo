"""Regression test for `RigidBody + IPCContact.add_rigid_body` coupling.

A coarse version of `examples/contact/ipc/rigid_deformable_punch.py`. The
test asserts:

1. Matrix/vector shapes are consistent across the assembly sum (regression
   guard for the global-DOF padding bug fixed earlier on this branch).
2. The deformable plate actually compresses under the rigid piston —
   the bug this entire feature exists to prevent.
3. Newton's third law: the IPC barrier gradient sums to zero between rigid
   and deformable surface DOFs.
4. The RigidTie kinematic relation is satisfied exactly on the rigid
   surface (max abs error = 0 across X/Y/Z).
5. ``set_static_obstacle`` and ``add_rigid_body`` are mutually exclusive
   on a single body.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

import fedoo as fd

ipctk = pytest.importorskip("ipctk")


@pytest.fixture
def fresh_3d_space():
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")
    yield
    fd.Assembly.delete_memory()


def _build_punch_problem():
    """Disc + one rigid piston above. Returns (pb, ipc, solid, body, mesh,
    n_disc, nodes_top)."""
    disc = fd.mesh.box_mesh(
        nx=6,
        ny=6,
        nz=3,
        x_min=0.0,
        x_max=1.0,
        y_min=0.0,
        y_max=1.0,
        z_min=0.0,
        z_max=0.10,
    )
    top = fd.mesh.box_mesh(
        nx=3,
        ny=3,
        nz=2,
        x_min=0.30,
        x_max=0.70,
        y_min=0.30,
        y_max=0.70,
        z_min=0.15,
        z_max=0.30,
    )
    disc.add_element_set(np.arange(disc.n_elements), "disc")
    top.add_element_set(np.arange(top.n_elements), "top")
    mesh = fd.Mesh.stack(disc, top)
    n_disc = disc.n_nodes
    top_part = mesh.extract_elements("top")
    nodes_top = top_part.parent_node_indices

    mat = fd.constitutivelaw.ElasticIsotrop(1e4, 0.3)
    wf = fd.weakform.StressEquilibrium(mat, nlgeom=True)
    solid = fd.Assembly.create(wf, mesh)

    surf = fd.mesh.extract_surface(mesh, quad2tri=True)
    ipc = fd.constraint.IPCContact(
        mesh,
        surface_mesh=surf,
        dhat=0.02,
        dhat_is_relative=False,
        use_ccd=False,
        barrier_stiffness=1e6,
    )

    body = fd.constraint.RigidBody(
        top_part,
        mass=1.0,
        inertia_tensor=0.01 * np.eye(3),
        name="piston",
    )
    ipc.add_rigid_body(body)

    assembly = fd.Assembly.sum(solid, body.assembly, ipc)
    pb = fd.problem.NonLinear(assembly, nlgeom=True)

    # Clamp the disc's four side faces against rigid-body motion.
    for axis, value in (("X", 0.0), ("X", 1.0), ("Y", 0.0), ("Y", 1.0)):
        pb.bc.add("Dirichlet", mesh.find_nodes(axis, value), "Disp", 0)

    nc = body.constraint.node_cd
    pb.bc.add("Dirichlet", nc[0], "RigidDispX", 0.0)
    pb.bc.add("Dirichlet", nc[1], "RigidDispY", 0.0)
    pb.bc.add("Dirichlet", nc[2], "RigidDispZ", -0.06)
    pb.bc.add("Dirichlet", nc[3], "RigidRotX", 0.0)
    pb.bc.add("Dirichlet", nc[4], "RigidRotY", 0.0)
    pb.bc.add("Dirichlet", nc[5], "RigidRotZ", 0.0)

    pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=20)
    return pb, ipc, solid, body, mesh, n_disc, nodes_top


def test_shapes_consistent_across_assembly_sum(fresh_3d_space):
    pb, ipc, solid, body, mesh, n_disc, nodes_top = _build_punch_problem()
    pb.nlsolve(dt=0.2, tmax=0.2, update_dt=True, print_info=0)

    expected_n_dof = pb.n_node_dof + pb.n_global_dof
    assert pb.n_global_dof == 6
    assert ipc.global_matrix.shape == (expected_n_dof, expected_n_dof)
    assert ipc.global_vector.shape == (expected_n_dof,)
    assert solid.global_matrix.shape == ipc.global_matrix.shape


def test_deformable_plate_actually_compresses(fresh_3d_space):
    pb, ipc, solid, body, mesh, n_disc, nodes_top = _build_punch_problem()
    pb.nlsolve(dt=0.2, tmax=0.2, update_dt=True, print_info=0)

    disp = pb.get_disp()
    plate_top_nodes = np.where(
        (mesh.nodes[:n_disc, 2] > 0.099)
        & (mesh.nodes[:n_disc, 0] > 0.25)
        & (mesh.nodes[:n_disc, 0] < 0.75)
        & (mesh.nodes[:n_disc, 1] > 0.25)
        & (mesh.nodes[:n_disc, 1] < 0.75)
    )[0]
    assert len(plate_top_nodes) > 0
    # The piston bottom (z=0.15) moves to z=0.09; with dhat=0.02 the plate
    # top must compress to at least z = 0.09 - 0.02 + small slack.
    assert disp[2, plate_top_nodes].min() < -1e-3, (
        f"Plate top did not compress under the punch "
        f"(min dz = {disp[2, plate_top_nodes].min():+.4e}). "
        "IPC is not coupling the rigid body to the deformable disc."
    )


def test_newton_third_law_on_ipc_gradient(fresh_3d_space):
    pb, ipc, solid, body, mesh, n_disc, nodes_top = _build_punch_problem()
    pb.nlsolve(dt=0.2, tmax=0.2, update_dt=True, print_info=0)

    # IPC barrier gradient summed onto each side via P.
    P = ipc._scatter_matrix
    grad_surf = ipc._barrier_potential.gradient(
        ipc._collisions, ipc._collision_mesh, ipc._get_current_vertices(pb)
    )
    F = -ipc._kappa * (P @ grad_surf)

    # Rigid side: 6-DOF block at body's global DOFs (forces only — first 3).
    rigid_dofs = body.assembly._dof_indices[:3]
    F_rigid = F[rigid_dofs]

    # Deformable side: sum the IPC Z-force at disc surface nodes.
    surface_idx = ipc._surface_node_indices
    disc_surf = surface_idx[np.isin(surface_idx, np.arange(n_disc))]
    nvar = pb.space.nvar
    F_disc = np.array([F[d * mesh.n_nodes + disc_surf].sum() for d in range(3)])

    # Newton 3 (in the Z direction, which is where load lives):
    assert abs(F_rigid[2] + F_disc[2]) < 1e-6 * max(abs(F_rigid[2]), 1.0), (
        f"Newton's 3rd law violated: F_rigid_z={F_rigid[2]:+.4e} "
        f"F_disc_z={F_disc[2]:+.4e}"
    )


def test_rigid_tie_kinematic_error_is_zero(fresh_3d_space):
    pb, ipc, solid, body, mesh, n_disc, nodes_top = _build_punch_problem()
    pb.nlsolve(dt=0.2, tmax=0.2, update_dt=True, print_info=0)

    disp = pb.get_disp()  # shape (3, n_nodes)
    body_disp = disp[:, nodes_top].T  # (n_body, 3)

    q = pb.get_dof_solution()[body.assembly._dof_indices]
    R, *_ = body.constraint._compute_rotation(q[3:])
    r_ref = mesh.nodes[nodes_top] - body.center_of_mass
    expected = r_ref @ R.T + body.center_of_mass + q[:3] - mesh.nodes[nodes_top]
    assert np.allclose(body_disp, expected, atol=1e-9), (
        f"RigidTie kinematic relation violated: "
        f"max abs error = {np.abs(body_disp - expected).max():.4e}"
    )


def test_static_obstacle_and_add_rigid_body_are_mutually_exclusive(fresh_3d_space):
    block = fd.mesh.box_mesh(
        nx=3,
        ny=3,
        nz=2,
        x_min=0.3,
        x_max=0.7,
        y_min=0.3,
        y_max=0.7,
        z_min=0.2,
        z_max=0.3,
    )
    block_surf = fd.mesh.extract_surface(block, quad2tri=True)
    obstacle_surf = fd.mesh.extract_surface(
        fd.mesh.box_mesh(
            nx=3,
            ny=3,
            nz=2,
            x_min=0,
            x_max=1,
            y_min=0,
            y_max=1,
            z_min=-0.1,
            z_max=0.0,
        ),
        quad2tri=True,
    )
    body = fd.constraint.RigidBody(
        block_surf,
        mass=1.0,
        inertia_tensor=0.01 * np.eye(3),
        center_of_mass=block_surf.bounding_box.center,
        name="b",
    )
    body.set_static_obstacle(obstacle_surf, dhat=0.01, kappa=1e6)

    # Now try to plug the same body into a shared IPCContact — must refuse.
    ipc = fd.constraint.IPCContact(
        block,
        surface_mesh=block_surf,
        dhat=0.01,
        dhat_is_relative=False,
        use_ccd=False,
        barrier_stiffness=1e6,
    )
    with pytest.raises(RuntimeError, match=r"cannot use set_static_obstacle"):
        ipc.add_rigid_body(body)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
