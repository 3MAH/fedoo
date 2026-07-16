"""Tests for the KDTree-based periodicity helpers in
fedoo.util.test_periodicity."""

import numpy as np
import pytest

import fedoo as fd
from fedoo.util.test_periodicity import (
    is_periodic,
    match_opposing_faces,
    pair_node_sets,
)


def _box_nodes(nx=3, ny=3, nz=3):
    """Return the (nx*ny*nz, 3) coordinate array of a structured box [0,1]^3."""
    x, y, z = np.meshgrid(
        np.linspace(0.0, 1.0, nx),
        np.linspace(0.0, 1.0, ny),
        np.linspace(0.0, 1.0, nz),
        indexing="ij",
    )
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def test_clean_box_periodic_at_tight_tol():
    crd = _box_nodes()
    assert is_periodic(crd, tol=1e-12, dim=3) is True


def test_match_opposing_faces_returns_paired_indices():
    crd = _box_nodes(nx=4, ny=3, nz=3)
    minus, plus = match_opposing_faces(crd, axis=0, tol=1e-12)
    assert len(minus) == len(plus) == 9  # 3 * 3 face nodes
    # Pairs must match on (y, z) exactly
    np.testing.assert_allclose(crd[minus][:, 1:], crd[plus][:, 1:])
    # And differ only on x
    assert np.all(np.isclose(crd[minus][:, 0], 0.0))
    assert np.all(np.isclose(crd[plus][:, 0], 1.0))


def test_displaced_face_raises_clear_error():
    crd = _box_nodes()
    # Shift one node on the +x face by 0.1 in y; pair distance becomes 0.1.
    on_xplus = np.where(np.isclose(crd[:, 0], 1.0))[0]
    crd[on_xplus[0], 1] += 0.1
    with pytest.raises(ValueError, match="max pairing distance"):
        match_opposing_faces(crd, axis=0, tol=1e-3)


def test_unequal_face_counts_raises_clear_error():
    crd = _box_nodes()
    # Add an extra node near the -x face, slightly inside.
    crd = np.vstack([crd, [[1e-9, 0.5, 0.5]]])
    with pytest.raises(ValueError, match=r"\d+ vs \d+ nodes"):
        match_opposing_faces(crd, axis=0, tol=1e-6)


def test_is_periodic_returns_false_on_displaced_face():
    crd = _box_nodes()
    on_xplus = np.where(np.isclose(crd[:, 0], 1.0))[0]
    crd[on_xplus[0], 1] += 0.1
    assert is_periodic(crd, tol=1e-3, dim=3) is False


def test_is_periodic_handles_floating_point_noise():
    """Mesh with sub-tol noise on boundary positions still pairs OK
    via KDTree (the old lexsort approach failed at tight tol)."""
    crd = _box_nodes()
    # Add up to ~1e-13 random noise on every coord
    rng = np.random.default_rng(0)
    crd = crd + rng.uniform(-1e-13, 1e-13, size=crd.shape)
    assert is_periodic(crd, tol=1e-9, dim=3) is True


def test_pair_node_sets_reorders_plus():
    crd = _box_nodes(nx=3, ny=2, nz=2)
    on_xminus = np.where(np.isclose(crd[:, 0], 0.0))[0]
    on_xplus = np.where(np.isclose(crd[:, 0], 1.0))[0]
    # Shuffle plus side so it is not pre-paired with minus.
    rng = np.random.default_rng(1)
    on_xplus_shuffled = on_xplus[rng.permutation(len(on_xplus))]
    minus, plus = pair_node_sets(crd, on_xminus, on_xplus_shuffled, [1, 2], tol=1e-9)
    # After pairing, (y, z) must match index-for-index
    np.testing.assert_allclose(crd[minus][:, 1:], crd[plus][:, 1:])


def test_periodic_bc_works_on_clean_box_at_default_tol():
    """End-to-end: PeriodicBC with default tol must initialize on a
    clean structured box without errors."""
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(3, 3, 3)
    fd.constitutivelaw.ElasticIsotrop(2e5, 0.3, name="Elastic")
    wf = fd.weakform.StressEquilibrium("Elastic", name="WF")
    asm = fd.Assembly.create(wf, mesh)
    pb = fd.problem.Linear(asm)
    bc = fd.constraint.PeriodicBC("small_strain")  # default tol=1e-8
    pb.bc.add(bc)
    pb.apply_boundary_conditions()  # initializes the BC list


if __name__ == "__main__":
    pytest.main([__file__])
