"""PETSc solvers (petsc4py is optional, installed by the user)."""

import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import fedoo as fd
from fedoo.core import base

requires_petsc = pytest.mark.skipif(not base.HAS_PETSC, reason="petsc4py not installed")


def _petsc_has_mumps():
    # a PETSc built from pip has no direct solver package
    return base.HAS_PETSC and base._petsc().Sys.hasExternalPackage("mumps")


requires_petsc_mumps = pytest.mark.skipif(
    not _petsc_has_mumps(), reason="PETSc built without MUMPS"
)


def _box_problem(ndim=3, nx=8, bc=True):
    if ndim == 3:
        fd.ModelingSpace("3D")
        mesh = fd.mesh.box_mesh(
            nx=nx + 1, ny=nx // 2 + 1, nz=nx // 2 + 1, x_max=2.0, elm_type="hex8"
        )
    else:
        fd.ModelingSpace("2Dstress")
        mesh = fd.mesh.rectangle_mesh(nx=nx + 1, ny=nx // 2 + 1, x_max=2.0)
    material = fd.constitutivelaw.ElasticIsotrop(200e3, 0.3)
    pb = fd.problem.Linear(
        fd.Assembly.create(fd.weakform.StressEquilibrium(material), mesh)
    )
    if bc:
        pb.bc.add("Dirichlet", mesh.find_nodes("X", 0.0), "Disp", 0)
        pb.bc.add("Dirichlet", mesh.find_nodes("X", 2.0), "DispY", 0.1)
    return pb


def _solve(**solver):
    pb = _box_problem()
    pb.set_solver(**solver)
    pb.solve()
    return pb


@pytest.mark.parametrize("ndim", [2, 3])
def test_rigid_body_modes_are_in_the_kernel(ndim):
    pb = _box_problem(ndim, bc=False)
    pb.update()  # assembly
    pb.apply_boundary_conditions()
    A = pb._reduced_system_matrix()
    modes = pb._rigid_body_modes()
    assert modes.shape == (A.shape[0], 3 if ndim == 2 else 6)
    assert np.linalg.matrix_rank(modes) == modes.shape[1]
    scale = abs(A).max() * abs(modes).max()
    assert abs(A @ modes).max() < 1e-10 * scale


@requires_petsc
@pytest.mark.parametrize(
    "solver",
    [
        pytest.param(
            dict(
                solver="petsc",
                solver_type="preonly",
                pc_type="lu",
                pc_factor_mat_solver_type="mumps",
            ),
            marks=requires_petsc_mumps,
        ),
        pytest.param(
            dict(
                solver="petsc",
                solver_type="preonly",
                pc_type="cholesky",
                pc_factor_mat_solver_type="mumps",
            ),
            marks=requires_petsc_mumps,
        ),
        dict(solver="petsc", solver_type="cg", pc_type="gamg", rtol=1e-12),
        dict(
            solver="petsc",
            solver_type="cg",
            pc_type="gamg",
            rtol=1e-12,
            reuse_preconditioner=True,
        ),
    ],
)
def test_petsc_matches_direct(solver):
    ref = _solve(solver="direct_scipy").get_disp()
    u = _solve(**solver).get_disp()
    assert np.allclose(u, ref, rtol=1e-8, atol=1e-8 * abs(ref).max())


@requires_petsc_mumps
def test_petsc_solver_kept_and_matrix_updated_in_place():
    from scipy.sparse.linalg import spsolve

    pb = _box_problem()
    pb.set_solver(
        "petsc", solver_type="preonly", pc_type="lu", pc_factor_mat_solver_type="mumps"
    )
    pb.solve()
    solver = pb._kept_solver
    mat = solver._mat
    A = pb._reduced_system_matrix()
    b = np.arange(A.shape[0], dtype=float)
    # same pattern, new values: the same PETSc matrix, updated in place
    A3 = A.copy()
    A3.data *= 3.0
    x = pb._solve(A3, b)
    assert pb._kept_solver is solver and solver._mat is mat
    assert np.allclose(x, spsolve(A3.tocsc(), b), rtol=1e-10, atol=1e-12)
    # new pattern: new matrix
    x = pb._solve(A3[:-1, :-1].tocsr(), b[:-1])
    assert solver._mat is not mat
    assert np.allclose(x, spsolve(A3[:-1, :-1].tocsc(), b[:-1]), rtol=1e-10, atol=1e-12)


@requires_petsc
def test_gamg_uses_rigid_body_modes():
    its = {}
    for nns in (False, True):
        pb = _box_problem(nx=16)
        pb.set_solver(
            "petsc", solver_type="cg", pc_type="gamg", rtol=1e-8, near_null_space=nns
        )
        pb.solve()
        its[nns] = pb._kept_solver.ksp.getIterationNumber()
    assert its[True] < its[False]


@requires_petsc_mumps
def test_config_selects_petsc_as_direct_solver():
    config = fd.get_config()
    saved = {
        k: config[k] for k in ("USE_PYPARDISO", "USE_MUMPS", "USE_PETSC", "USE_UMFPACK")
    }
    try:
        fd.get_config()["USE_PETSC"] = True
        assert fd.get_config()["USE_PETSC"]
        assert not (fd.get_config()["USE_MUMPS"] or fd.get_config()["USE_PYPARDISO"])
        for symmetric in (False, True):
            pb = _box_problem()
            pb.set_solver("direct", symmetric=symmetric)
            assert pb.solver is base._solver_petsc
            pb.solve()
            ref = _solve(solver="direct_scipy").get_disp()
            assert np.allclose(
                pb.get_disp(), ref, rtol=1e-10, atol=1e-10 * abs(ref).max()
            )
    finally:
        for k, v in saved.items():
            setattr(base, k, v)


_SCRIPT = textwrap.dedent(
    """
    import sys
    sys.path.insert(0, {root!r})
    sys.path.insert(0, {tests!r})
    from test_petsc import _box_problem
    pb = _box_problem()
    pb.set_solver("petsc")
    pb.solve()
    ksp = pb._kept_solver.ksp
    print("RESULT", ksp.getType(), ksp.getPC().getType())
    from fedoo.core import base
    if base.USE_MUMPS:  # python-mumps on one process (no deadlock under mpiexec)
        pb.set_solver("mumps")
        pb.solve()
        print("MUMPS OK")
    """
)


def _run_script(tmp_path, args, mpiexec=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(fd.__file__)))
    script = tmp_path / "script.py"
    script.write_text(
        _SCRIPT.format(root=root, tests=os.path.dirname(os.path.abspath(__file__)))
    )
    cmd = [sys.executable, str(script), *args]
    if mpiexec is not None:
        cmd = [mpiexec, "-n", "2", *cmd]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout + out.stderr


@requires_petsc
def test_command_line_petsc_options(tmp_path):
    out = _run_script(
        tmp_path, ["-ksp_type", "cg", "-pc_type", "jacobi", "--script-arg"]
    )
    assert "RESULT cg jacobi" in out
    assert "unused" not in out.lower()  # the script's own arguments


@requires_petsc
def test_mpi_linear_solver_server(tmp_path):
    mpiexec = shutil.which("mpiexec", path=os.path.dirname(sys.executable))
    if mpiexec is None:
        pytest.skip("no mpiexec next to the python executable")
    out = _run_script(tmp_path, ["-mpi_linear_solver_server"], mpiexec=mpiexec)
    # rank 1 serves the solves of rank 0: the script runs once
    assert out.count("RESULT") == 1
    if base.USE_MUMPS:
        assert out.count("MUMPS OK") == 1
