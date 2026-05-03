"""Tests for the factorization reuse mechanism."""

import numpy as np
import pytest

import fedoo as fd
from fedoo.core.base import USE_MUMPS, USE_PETSC, USE_PYPARDISO


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
    fd.Assembly.create("WeakForm", "Domain", name="Assembly", MeshChange=True)
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
