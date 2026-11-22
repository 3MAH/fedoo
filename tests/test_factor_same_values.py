"""The direct solvers kept by a problem skip the factorization of a matrix
whose values are those of the matrix already factorized."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

import fedoo as fd
from fedoo.core import base
from fedoo.core.base import USE_MUMPS, HAS_PETSC, _MumpsFactor


def _petsc_has_mumps():
    return HAS_PETSC and base._petsc().Sys.hasExternalPackage("mumps")


def _matrix(n=50, seed=0):
    A = sp.random(n, n, density=0.15, random_state=seed, format="csr")
    A = (A + A.T + 10 * sp.eye(n)).tocsr()
    A.sort_indices()
    return A


def _count_factorizations(factor):
    calls = []
    ctx_factor = factor._ctx.factor

    def counted(*args, **kargs):
        calls.append(kargs.get("reuse_analysis"))
        return ctx_factor(*args, **kargs)

    factor._ctx.factor = counted
    return calls


@pytest.mark.skipif(not USE_MUMPS, reason="python-mumps not installed")
@pytest.mark.parametrize("symmetric", [False, True])
def test_mumps_skips_factorization_of_same_values(symmetric):
    A = _matrix()
    b = np.arange(A.shape[0], dtype=float)
    factor = _MumpsFactor(symmetric)
    calls = _count_factorizations(factor)
    factor.factor(A)
    x = factor.solve(b)
    factor.factor(A.copy())  # other object, same values: no factorization
    assert np.array_equal(factor.solve(b), x)
    M = A.copy()
    M.data[3] *= 2.0
    if symmetric:
        M = ((M + M.T) * 0.5).tocsr()
        M.sort_indices()
    factor.factor(M)
    factor.factor(M)
    assert calls == [False, True]
    from scipy.sparse.linalg import spsolve

    ref = spsolve(M.tocsc(), b)
    assert np.allclose(factor.solve(b), ref, rtol=1e-10, atol=1e-12 * abs(ref).max())


def test_mumps_same_values_without_context_arrays(monkeypatch):
    """When the context does not expose the arrays it was given, the values
    are compared to a copy; a failed factorization is redone."""
    calls = []

    class FakeMumpsContext:
        fail = False

        def set_matrix(self, A, symmetric=False):
            pass

        def factor(self, A=None, reuse_analysis=False):
            calls.append(reuse_analysis)
            if self.fail:
                raise RuntimeError("singular matrix")

        def solve(self, B):
            return B

    monkeypatch.setitem(sys.modules, "mumps", SimpleNamespace(Context=FakeMumpsContext))
    A = sp.csr_matrix([[2.0, 1.0], [1.0, -1.0]])
    factor = _MumpsFactor()
    factor.factor(A)
    factor.factor(A.copy())
    assert calls == [False]
    factor.factor(2.0 * A)
    assert calls == [False, True]
    factor._ctx.fail = True
    with pytest.raises(RuntimeError):
        factor.factor(3.0 * A)
    factor._ctx.fail = False
    factor.factor(3.0 * A)  # the failed factorization is not reused
    assert calls == [False, True, True, True]


@pytest.mark.skipif(not USE_MUMPS, reason="python-mumps not installed")
def test_mumps_refactorizes_values_modified_in_place():
    A = _matrix()
    b = np.arange(A.shape[0], dtype=float)
    factor = _MumpsFactor()
    calls = _count_factorizations(factor)
    factor.factor(A)
    A.data[3] *= 2.0  # same object, other values
    factor.factor(A)
    assert calls == [False, True]
    from scipy.sparse.linalg import spsolve

    ref = spsolve(A.tocsc(), b)
    assert np.allclose(factor.solve(b), ref, rtol=1e-8, atol=1e-10 * abs(ref).max())


@pytest.mark.skipif(not USE_MUMPS, reason="python-mumps not installed")
def test_mumps_failed_factorization_is_not_reused():
    """A factorization that raises leaves no pattern nor values to reuse."""
    A, other = _matrix(), _matrix(n=30, seed=1)
    b = np.arange(other.shape[0], dtype=float)
    factor = _MumpsFactor()
    factor.factor(A)
    ctx_factor = factor._ctx.factor

    def failing(*args, **kargs):
        raise RuntimeError("factorization failed")

    factor._ctx.factor = failing
    with pytest.raises(RuntimeError):
        factor.factor(other)  # new pattern
    calls = []
    factor._ctx.factor = lambda *args, **kargs: (
        calls.append(kargs.get("reuse_analysis")),
        ctx_factor(*args, **kargs),
    )[1]
    factor.factor(other)
    assert calls == [False]  # analysis redone, nothing skipped
    from scipy.sparse.linalg import spsolve

    ref = spsolve(other.tocsc(), b)
    assert np.allclose(factor.solve(b), ref, rtol=1e-8, atol=1e-10 * abs(ref).max())


@pytest.mark.skipif(not USE_MUMPS, reason="python-mumps not installed")
@pytest.mark.parametrize("symmetric", [False, True])
def test_nonlinear_prediction_reuses_last_factorization(symmetric):
    """The elastic prediction of an increment solves with the tangent of the
    last iteration of the previous one: it is not factorized again."""
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=11, ny=5, x_max=4.0, elm_type="quad4")
    material = fd.constitutivelaw.ElasticIsotrop(200e3, 0.3)
    pb = fd.problem.NonLinear(
        fd.Assembly.create(fd.weakform.StressEquilibrium(material), mesh)
    )
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0.0), "Disp", 0)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 4.0), "DispY", 0.1)
    pb.set_nr_criterion("Displacement", tol=1e-10)
    # an explicit choice: the automatic one replaces the solver at the first solve
    pb.set_solver("direct", symmetric=symmetric)
    calls = _count_factorizations(pb._kept_solver)
    pb.nlsolve(dt=0.25, update_dt=False, print_info=0)
    # one factorization for each of the two patterns of the reduced matrix
    # (plain product, then fixed pattern), then only back-substitutions
    assert calls == [False, False]
    assert pb.n_iter == 4


@pytest.mark.skipif(not _petsc_has_mumps(), reason="PETSc built without MUMPS")
def test_petsc_skips_setup_of_same_values():
    from scipy.sparse.linalg import spsolve

    fd.ModelingSpace("2Dstress")
    pb = fd.problem.Linear(
        fd.Assembly.create(
            fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1.0, 0.3)),
            fd.mesh.rectangle_mesh(nx=5, ny=3),
        )
    )
    pb.set_solver(
        "petsc", solver_type="preonly", pc_type="lu", pc_factor_mat_solver_type="mumps"
    )
    solver = pb._kept_solver
    A = _matrix()
    b = np.arange(A.shape[0], dtype=float)
    setups = []
    ksp_setup = solver._ksp.setUp
    solver._ksp = SimpleNamespace(
        setUp=lambda: setups.append(1) or ksp_setup(),
        **{
            name: getattr(solver.ksp, name)
            for name in (
                "reset",
                "setOperators",
                "getPC",
                "solve",
                "getConvergedReason",
                "getIterationNumber",
            )
        },
    )
    x = pb._solve(A, b)
    assert np.array_equal(pb._solve(A.copy(), b), x)
    assert len(setups) == 1
    x = pb._solve(2.0 * A, b)
    assert len(setups) == 2
    assert np.allclose(x, spsolve(2.0 * A.tocsc(), b), rtol=1e-10, atol=1e-12)


@pytest.mark.skipif(not HAS_PETSC, reason="petsc4py not installed")
def test_petsc_reset_preconditioner_with_same_values():
    """reset_preconditioner rebuilds the kept preconditioner at the next
    factor, even for a matrix with the same values."""
    pytest.importorskip("petsc4py")
    from fedoo.core.base import _PetscSolver

    solver = _PetscSolver(
        solver_type="cg", pc_type="gamg", reuse_preconditioner=True, rtol=1e-12
    )
    A = _matrix()
    b = np.arange(A.shape[0], dtype=float)
    reuse_at_setup = []
    reuse = [False]  # last flag given to the PC (petsc4py has no getter)
    ksp, pc = solver._ksp, solver._ksp.getPC()

    class PC:
        def __getattr__(self, name):
            return getattr(pc, name)

        def setReusePreconditioner(self, flag):
            reuse[0] = flag
            pc.setReusePreconditioner(flag)

    solver._ksp = SimpleNamespace(
        setUp=lambda: reuse_at_setup.append(reuse[0]) or ksp.setUp(),
        getPC=lambda: PC(),
        **{
            name: getattr(ksp, name)
            for name in (
                "reset",
                "setOperators",
                "solve",
                "getConvergedReason",
                "getIterationNumber",
            )
        },
    )
    solver.factor(A)
    solver.factor(2.0 * A)  # preconditioner of A kept
    solver.factor(2.0 * A)  # same values: nothing to do
    assert reuse_at_setup == [False, True]
    solver.reset_preconditioner()
    solver.factor(2.0 * A)  # same values, but a new preconditioner is asked
    assert reuse_at_setup == [False, True, False]
    solver.factor(2.0 * A)
    assert len(reuse_at_setup) == 3
    from scipy.sparse.linalg import spsolve

    ref = spsolve(2.0 * A.tocsc(), b)
    assert np.allclose(solver.solve(b), ref, rtol=1e-8, atol=1e-10 * abs(ref).max())
