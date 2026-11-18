"""Tests for the factorization reuse mechanism."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

import fedoo as fd
import fedoo.core.base as base
from fedoo.core.base import (
    USE_MUMPS,
    USE_PETSC,
    USE_PYPARDISO,
    _MumpsFactor,
    _PypardisoFactor,
    _solver_mumps,
    _solver_pardiso,
)


HAS_DIRECT_BACKEND = USE_PYPARDISO or USE_MUMPS or USE_PETSC


def _build_plate_problem():
    fd.ModelingSpace("2Dstress")
    fd.mesh.hole_plate_mesh(
        nr=11,
        nt=11,
        length=100,
        height=100,
        radius=20,
        elm_type="quad4",
        name="Domain",
    )
    fd.constitutivelaw.ElasticIsotrop(2e5, 0.3, name="ElasticLaw")
    fd.weakform.StressEquilibrium("ElasticLaw", name="WeakForm")
    fd.Assembly.create("WeakForm", "Domain", name="Assembly", mesh_change=True)
    pb = fd.problem.Linear("Assembly")

    mesh = fd.Mesh["Domain"]
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    right = mesh.find_nodes("X", mesh.bounding_box.xmax)
    bottom = mesh.find_nodes("Y", mesh.bounding_box.ymin)

    pb.bc.add("Dirichlet", left, "DispX", 0)
    pb.bc.add("Dirichlet", bottom, "DispY", 0)
    pb.bc.add("Dirichlet", right, "DispX", 0.1)
    pb.apply_boundary_conditions()
    return pb


@pytest.mark.skipif(
    not HAS_DIRECT_BACKEND,
    reason="No direct backend (pypardiso, python-mumps or petsc4py) installed",
)
def test_reuse_factorization_matches_baseline():
    """Solving with factor reuse must give the same result as without."""
    # Baseline: standard solve
    pb_ref = _build_plate_problem()
    pb_ref.solve()
    U_ref = pb_ref.get_disp().copy()

    # With factor reuse
    pb = _build_plate_problem()
    pb.set_reuse_factorization(True)
    pb.solve()
    U = pb.get_disp()

    assert np.allclose(U, U_ref, atol=1e-12)


@pytest.mark.skipif(
    not HAS_DIRECT_BACKEND,
    reason="No direct backend (pypardiso, python-mumps or petsc4py) installed",
)
def test_reuse_factorization_repeated_solves():
    """Multiple solves with same A and reuse must all match the baseline."""
    pb_ref = _build_plate_problem()
    pb_ref.solve()
    U_ref = pb_ref.get_disp().copy()

    pb = _build_plate_problem()
    pb.set_reuse_factorization(True)
    for _ in range(3):
        pb.solve()
        assert np.allclose(pb.get_disp(), U_ref, atol=1e-12)
    pb.set_reuse_factorization(False)


def test_set_reuse_factorization_no_backend():
    """Without a direct backend, set_reuse_factorization must raise."""
    if HAS_DIRECT_BACKEND:
        pytest.skip("a direct backend is available, can't test the no-backend path")

    pb = _build_plate_problem()
    with pytest.raises(RuntimeError, match="pypardiso|python-mumps|petsc4py"):
        pb.set_reuse_factorization(True)


def test_invalidate_factorization_via_set_A():
    """set_A must invalidate the cached factorization."""
    if not HAS_DIRECT_BACKEND:
        pytest.skip("requires a direct backend")

    pb = _build_plate_problem()
    pb.set_reuse_factorization(True)
    pb.solve()  # factorizes
    assert pb._factor_valid is True

    # touching A must invalidate
    pb.set_A(pb.get_A())
    assert pb._factor_valid is False


def test_discarded_pardiso_contexts_release_native_memory(monkeypatch):
    pb = _build_plate_problem()
    released = []

    class FakePardisoSolver:
        def __init__(self, mtype):
            self.factored = False

        def factorize(self, A):
            self.factored = True

        def solve(self, A, B):
            assert self.factored
            return B

        def free_memory(self, everything=False):
            assert everything and self.factored
            self.factored = False
            released.append(self)

    monkeypatch.setattr(base, "USE_PYPARDISO", True)
    monkeypatch.setitem(
        sys.modules, "pypardiso", SimpleNamespace(PyPardisoSolver=FakePardisoSolver)
    )
    A, B = sp.eye(2, format="csr"), np.ones(2)
    pb.set_reuse_factorization(True)
    pb.set_reuse_factorization(True)  # An unused context has no native allocation.
    pb._solve(A, B)
    pb._solve(A, B)
    assert released == []
    pb.set_reuse_factorization(True)
    assert len(released) == 1
    pb._solve(A, B)
    pb.set_reuse_factorization(False)
    pb.set_reuse_factorization(False)
    assert len(released) == 2


def test_solver_mumps_default_path():
    """``_solver_mumps`` must use the python-mumps Context API.

    Regression test for a latent bug where the function called
    ``mumps.spsolve`` which does not exist in the python-mumps package.
    The bug went unnoticed in CI because CI installs pypardiso (x86),
    so the mumps default path was never exercised. Triggers on any
    arm64 environment where python-mumps is the auto-selected backend.
    """
    try:
        import mumps  # noqa: F401
    except ImportError:
        pytest.skip("python-mumps not installed")

    # Small symmetric positive-definite tridiagonal system: A x = b
    n = 8
    main = 2.0 * np.ones(n)
    off = -1.0 * np.ones(n - 1)
    A = sp.diags([off, main, off], offsets=[-1, 0, 1], format="csr")
    b = np.arange(1.0, n + 1.0)

    x = _solver_mumps(A, b)
    assert np.allclose(A @ x, b, atol=1e-10)


@pytest.mark.parametrize("symmetric, expected_mtype", [(False, 11), (True, -2)])
def test_pardiso_symmetry_mode(monkeypatch, symmetric, expected_mtype):
    """Pardiso receives the requested general or symmetric matrix type."""
    calls = []
    matrices = []

    class FakePardisoSolver:
        def __init__(self, mtype):
            calls.append(mtype)

        def solve(self, A, B):
            matrices.append(A)
            return B

        def factorize(self, A):
            matrices.append(A)

    monkeypatch.setitem(
        sys.modules,
        "pypardiso",
        SimpleNamespace(PyPardisoSolver=FakePardisoSolver),
    )

    A = sp.csr_matrix([[2.0, 1.0], [1.0, -1.0]])
    B = np.ones(2)
    assert np.array_equal(_solver_pardiso(A, B, symmetric=symmetric), B)
    factor = _PypardisoFactor(symmetric=symmetric)
    factor.factor(A)
    assert calls == [expected_mtype, expected_mtype]
    expected_A = sp.triu(A, format="csr") if symmetric else A
    assert all((matrix != expected_A).nnz == 0 for matrix in matrices)


@pytest.mark.parametrize("symmetric, expected_sym", [(False, None), (True, 2)])
def test_mumps_symmetry_mode(monkeypatch, symmetric, expected_sym):
    """Standalone MUMPS receives the requested general or symmetric mode.

    ``sym`` is only passed when a symmetric factorization is requested, so
    that the general path keeps working with python-mumps releases whose
    ``Context`` has no ``sym`` argument.
    """
    calls = []
    matrices = []

    class FakeMumpsContext:
        def __init__(self, sym=None):
            calls.append(sym)

        def set_matrix(self, A, symmetric=False):
            matrices.append(A)

        def factor(self, A=None, reuse_analysis=False):
            if A is not None:
                matrices.append(A)

        def solve(self, B):
            return B

    monkeypatch.setitem(sys.modules, "mumps", SimpleNamespace(Context=FakeMumpsContext))

    A = sp.csr_matrix([[2.0, 1.0], [1.0, -1.0]])
    B = np.ones(2)
    assert np.array_equal(_solver_mumps(A, B, symmetric=symmetric), B)
    factor = _MumpsFactor(symmetric=symmetric)
    factor.factor(A)
    assert calls == [expected_sym, expected_sym]
    expected_A = sp.tril(A, format="csr") if symmetric else A
    assert all((matrix != expected_A).nnz == 0 for matrix in matrices)


def test_mumps_threads_follow_the_user_setting(monkeypatch):
    """fedoo caps the MUMPS threads at 8 unless OMP_NUM_THREADS is set."""
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    assert base._mumps_threads() is None
    monkeypatch.delenv("OMP_NUM_THREADS")
    assert 1 <= base._mumps_threads() <= 8


def test_mumps_symmetric_legacy_context(monkeypatch):
    """python-mumps 0.0.6 has no ``sym`` argument: the symmetry is given
    with the matrix to ``set_matrix``."""
    calls = []

    class LegacyMumpsContext:
        def __init__(self):
            pass

        def set_matrix(self, A, symmetric=False):
            calls.append(("set_matrix", symmetric, A.nnz))

        def factor(self, A=None, reuse_analysis=False):
            calls.append(("factor", A, reuse_analysis))

        def solve(self, B):
            return B

    monkeypatch.setitem(
        sys.modules, "mumps", SimpleNamespace(Context=LegacyMumpsContext)
    )

    A = sp.csr_matrix([[2.0, 1.0], [1.0, -1.0]])
    B = np.ones(2)
    assert np.array_equal(_solver_mumps(A, B, symmetric=True), B)
    # full matrix given, set_matrix keeps its upper triangle
    assert calls == [("set_matrix", True, 4), ("factor", None, False)]


def test_mumps_factor_reuses_analysis_on_same_pattern(monkeypatch):
    """The analysis is reused only while the sparsity pattern is unchanged,
    explicit zeros included."""
    reuse = []

    class FakeMumpsContext:
        def set_matrix(self, A, symmetric=False):
            pass

        def factor(self, A=None, reuse_analysis=False):
            reuse.append(reuse_analysis)

        def solve(self, B):
            return B

    monkeypatch.setitem(sys.modules, "mumps", SimpleNamespace(Context=FakeMumpsContext))

    A = sp.csr_matrix([[2.0, 1.0], [1.0, -1.0]])
    A_zero = A.copy()
    A_zero.data[1] = 0.0  # explicit zero: same pattern
    A_other = sp.csr_matrix([[2.0, 0.0], [1.0, -1.0]])
    factor = _MumpsFactor()
    for M in (A, 3.0 * A, A_zero, A_other, A_other):
        factor.factor(M)
    assert reuse == [False, True, True, False, True]


@pytest.mark.skipif(not USE_MUMPS, reason="python-mumps not installed")
@pytest.mark.parametrize("symmetric", [False, True])
def test_mumps_values_updated_in_place(symmetric):
    """With an unchanged pattern, the new values are written into the arrays
    held by the python-mumps context (no set_matrix): same solutions."""
    from scipy.sparse.linalg import spsolve

    n = 60
    A = sp.random(n, n, density=0.15, random_state=3, format="csr")
    A = (A + A.T + 10 * sp.eye(n)).tocsr()
    A.sort_indices()
    b = np.arange(n, dtype=float)
    factor = _MumpsFactor(symmetric)
    factor.factor(A)
    assert factor._entries is not None
    for scale in (2.0, 0.5):
        M = A.copy()
        M.data = M.data * (1 + 0.3 * np.cos(scale * np.arange(M.nnz)))
        M = ((M + M.T) * 0.5).tocsr()
        M.sort_indices()
        assert np.array_equal(M.indices, A.indices)
        factor.factor(M)
        ref = spsolve(M.tocsc(), b)
        assert np.allclose(
            factor.solve(b), ref, rtol=1e-10, atol=1e-12 * abs(ref).max()
        )


@pytest.mark.parametrize("seed", [0, 1])
def test_reduction_operator_matches_product(seed):
    """The fixed-pattern reduction gives MatCB.T A MatCB, with a pattern that
    does not depend on the values of A."""
    from fedoo.core.problem import _reduction_operator

    rng = np.random.default_rng(seed)
    n = 40
    A = sp.random(n, n, density=0.2, random_state=seed, format="csr")
    A = (A + A.T + sp.eye(n)).tocsr()
    # free dof (identity), blocked dof (empty rows) and MPC rows
    free = np.sort(rng.choice(n, 25, replace=False))
    slave = np.setdiff1d(np.arange(n), free)[:8]
    rows = list(free) + list(np.repeat(slave, 2))
    cols = list(range(25)) + list(rng.integers(0, 25, 2 * len(slave)))
    vals = [1.0] * 25 + list(rng.normal(size=2 * len(slave)))
    mat_cb = sp.coo_matrix((vals, (rows, cols)), shape=(n, 25)).tocsc()

    op, indices, indptr = _reduction_operator(A, mat_cb)
    for data in (A.data, rng.normal(size=A.nnz), np.zeros(A.nnz)):
        Ad = sp.csr_matrix((data, A.indices, A.indptr), shape=A.shape)
        R = sp.csr_matrix((op @ data, indices, indptr), shape=(25, 25))
        ref = (mat_cb.T @ Ad @ mat_cb).toarray()
        assert np.allclose(R.toarray(), ref, rtol=1e-12, atol=1e-12)
        assert R.nnz == len(indices)  # zeros kept: fixed pattern


def test_reduced_matrix_pattern_is_fixed():
    """Repeated reductions with the same pattern of A keep the same pattern."""
    pb = _build_plate_problem()
    pb.solve()
    A = pb.get_A()
    assert A.format == "csr"
    patterns = []
    for scale in (1.0, 2.0, 0.5):
        A2 = A.copy()
        A2.data *= scale
        A2.data[::7] = 0.0  # values that would be dropped by the product
        pb.set_A(A2)
        R = pb._reduced_system_matrix()
        ref = pb._MatCB.T @ A2 @ pb._MatCB
        assert abs(R - ref).max() <= 1e-12 * abs(ref).max()
        patterns.append((R.indptr.copy(), R.indices.copy()))
    # first call: plain product; the operator is built the second time
    assert np.array_equal(patterns[1][0], patterns[2][0])
    assert np.array_equal(patterns[1][1], patterns[2][1])


def test_set_solver_symmetric_option_and_factor_reuse():
    """The direct solver and its reuse context share the symmetry choice."""
    pb = _build_plate_problem()
    pb.set_solver("direct", symmetric=True)

    assert pb._solver_symmetric is True
    if USE_PYPARDISO:
        assert pb._ProblemBase__solver[1] is _solver_pardiso
        pb.set_reuse_factorization(True)
        assert pb._factor_context._solver.mtype == -2
    elif USE_MUMPS:
        assert pb._ProblemBase__solver[1] is _solver_mumps


def test_set_solver_symmetric_requires_bool():
    pb = _build_plate_problem()
    with pytest.raises(TypeError, match="bool expected"):
        pb.set_solver("direct", symmetric=1)


if __name__ == "__main__":
    pytest.main([__file__])
