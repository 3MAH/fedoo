"""Problem.apply_boundary_conditions: free / eliminated dof and constraint
matrix MatCB, kept as the same object while the constraints do not change."""

import numpy as np
import pytest
import scipy.sparse as sp

import fedoo as fd


def _problem():
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=7, ny=4, x_max=3.0, elm_type="quad4")
    material = fd.constitutivelaw.ElasticIsotrop(200e3, 0.3)
    pb = fd.problem.Linear(
        fd.Assembly.create(fd.weakform.StressEquilibrium(material), mesh)
    )
    return pb, mesh


def _reference_sets(pb):
    """Free and eliminated dof computed with python sets."""
    blocked, slave = set(), set()
    for e in pb.bc.generate(pb, 1):
        if e.bc_type == "Dirichlet":
            blocked.update(np.asarray(e._dof_index).tolist())
        elif e.bc_type == "MPC":
            slave.update(np.asarray(e._dof_index[0]).tolist())
    slave |= blocked
    free = sorted(set(range(pb.n_dof)) - slave)
    return sorted(blocked), sorted(slave), free


def test_free_and_eliminated_dof_with_mpc():
    pb, mesh = _problem()
    left = mesh.find_nodes("X", 0.0)
    right = mesh.find_nodes("X", 3.0)
    pb.bc.add("Dirichlet", left, "Disp", 0)
    pb.bc.add("Dirichlet", right[:2], "DispX", 0.1)
    # DispY of the right nodes tied to the last one
    n = len(right) - 1
    pb.bc.mpc(
        [right[:-1], np.full(n, right[-1])],
        ["DispY", "DispY"],
        np.array([np.ones(n), -np.ones(n)]),
    )
    pb.apply_boundary_conditions()
    blocked, slave, free = _reference_sets(pb)
    assert np.array_equal(pb._dof_free, free)
    assert np.array_equal(np.sort(pb._dof_slave), slave)
    assert np.array_equal(np.sort(pb._dof_blocked), blocked)
    mat_cb = pb._MatCB.toarray()
    assert mat_cb.shape == (pb.n_dof, len(free))
    assert np.array_equal(mat_cb[free], np.eye(len(free)))
    # reactions with the mpc dof
    pb.solve()
    forces = pb.get_ext_forces()
    # reference: identity on the free and blocked dof given in another order
    M = pb._MFext
    dof_idt = list(pb._dof_free[::-1]) + list(set(blocked))
    ref = (
        sp.coo_matrix(
            (
                np.hstack((M.data, np.ones(len(dof_idt)))),
                (np.hstack((M.row, dof_idt)), np.hstack((M.col, dof_idt))),
            ),
            shape=M.shape,
        )
        .tocsr()
        .T
        @ pb._get_force_vector()
    )
    assert np.array_equal(forces, ref)


def test_identity_constraint_matrix_matches_coo_conversion():
    pb, mesh = _problem()
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0.0), "Disp", 0)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 3.0), "DispY", -0.1)
    pb.apply_boundary_conditions()
    free = pb._dof_free
    blocked, slave, ref_free = _reference_sets(pb)
    assert np.array_equal(free, ref_free)
    assert pb._MFext is None
    ref = sp.coo_matrix(
        (np.ones(len(free)), (free, np.arange(len(free)))),
        shape=(pb.n_dof, len(free)),
    ).tocsc()
    mat_cb = pb._MatCB
    assert mat_cb.format == "csc"
    assert mat_cb.indices.dtype == ref.indices.dtype
    for name in ("indptr", "indices", "data"):
        assert np.array_equal(getattr(mat_cb, name), getattr(ref, name))


def test_constraint_matrix_object_kept_while_unchanged():
    pb, mesh = _problem()
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0.0), "Disp", 0)
    right = pb.bc.add("Dirichlet", mesh.find_nodes("X", 3.0), "DispY", 0.1)
    pb.solve()
    mat_cb = pb._MatCB
    reduced = pb._reduced_system_matrix()
    pb.apply_boundary_conditions(0.5)  # other values, same dof
    assert pb._MatCB is mat_cb
    assert pb._reduced_system_matrix() is reduced
    right.node_set = right.node_set[1:]  # other dof: new matrix
    pb.apply_boundary_conditions()
    assert pb._MatCB is not mat_cb
    assert pb._MatCB.shape[1] == mat_cb.shape[1] + 1


def test_factorization_kept_with_unchanged_constraints():
    pb, mesh = _problem()
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0.0), "Disp", 0)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 3.0), "DispY", 0.1)
    try:
        pb.set_reuse_factorization(True)
    except RuntimeError:
        pytest.skip("no direct solver backend")
    pb.solve()
    assert pb._factor_valid
    pb.apply_boundary_conditions()
    assert pb._factor_valid
    pb.bc.add("Dirichlet", mesh.find_nodes("Y", 1.0), "DispX", 0)
    pb.apply_boundary_conditions()
    assert not pb._factor_valid


def test_solve_with_scalar_rhs():
    """B and D given as scalars (no apply_boundary_conditions after set_B)
    with homogeneous conditions: the right-hand side is still an array."""
    from fedoo.core.problem import Problem

    pb, mesh = _problem()
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0.0), "Disp", 0)
    pb.solve()
    pb.set_B(0)
    pb.set_D(0)
    Problem.solve(pb)
    assert np.array_equal(pb.get_X(), np.zeros(pb.n_dof))
