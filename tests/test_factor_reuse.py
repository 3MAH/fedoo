"""Tests for the factorization reuse mechanism."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

import fedoo as fd
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

        def factor(self, A):
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
