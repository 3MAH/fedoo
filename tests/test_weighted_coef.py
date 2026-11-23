"""Gauss-weighted assembly coefficients computed when read or in the kernel."""

import numpy as np
import pytest
from scipy import sparse

import fedoo as fd
import fedoo.core._sparsematrix as sm

numba_only = pytest.mark.skipif(
    sm._gauss_sum_atb_jobs_numba is None, reason="numba is not installed"
)


def test_weighted_coef_values_and_products():
    rng = np.random.default_rng(0)
    w = rng.random(12)
    a, b = rng.random(12), rng.random((12, 3))[:, 1]  # b: strided view
    c = sm._WeightedCoef([a, 2.0, b], w)
    ref = ((a + 2.0) + b) * w
    assert np.array_equal(c.values(), ref)
    assert c.values() is c.values()  # computed once
    assert np.array_equal(np.float64(-1.5) * c, np.float64(-1.5) * ref)
    assert np.array_equal(3 * c, 3 * ref)
    assert np.array_equal(c * 3, ref * 3)
    assert np.array_equal(sm._coef_values(c), ref)
    assert sm._coef_values(4.0) == 4.0


def _bloc_matrix(coefs, a_list, b, lazy, mat_lumping=False):
    """Element matrices of one bloc from the jobs kernel (or numpy)."""
    n_gp, n_elm, n = b.shape

    # operators: csr matrices with n entries per row (one row per gauss point)
    def csr(values):
        cols = (np.arange(n_gp * n_elm)[:, None] % n_elm) * n + np.arange(n)
        return sparse.csr_matrix(
            (
                values.reshape(-1),
                cols.reshape(-1),
                np.arange(0, n * n_gp * n_elm + 1, n),
            ),
            shape=(n_gp * n_elm, n * n_elm),
        )

    MM = sm._BlocSparse(1, 1, n_gp)
    A = [csr(a) for a in a_list]
    B = csr(b)
    coefs = coefs if lazy else [sm._coef_values(c) for c in coefs]
    MM.addToBlocATB(A, B, coefs, 0, 0, mat_lumping)
    MM._flush()
    return MM.data[0][0]


def _coefficients(rng, size, w):
    tangent = rng.normal(size=(size, 6, 6))
    parts_list = [
        [tangent[:, 0, 1]],  # strided view
        [tangent[:, 2, 2], tangent[:, 3, 1], 0.5],  # sum of parts
        [2],  # constant
        [np.array([1.25])],  # one value array
        [rng.normal(size=size)],
        [tangent[::-1, 1, 1]],  # negative stride: weighted before the kernel
        [rng.integers(0, 5, size)],  # not float64: weighted before the kernel
    ]
    return lambda: [sm._WeightedCoef(p, w) for p in parts_list]


def _reference(coefs, a_list, b):
    """Element matrices from the weighted arrays, in numpy."""
    n_gp = b.shape[0]
    s = sum(c.values().reshape(n_gp, -1, 1) * a for c, a in zip(coefs, a_list))
    return np.einsum("gei,gej->eij", s, b)


@pytest.mark.parametrize("kernel", [True, False])
@pytest.mark.parametrize("mat_lumping", [False, True])
def test_lazy_weighting_matches_weighted_arrays(monkeypatch, mat_lumping, kernel):
    """Parts read in place by the kernel (strided, constants, sums) give the
    element matrices of the weighted arrays; also with the kernel disabled."""
    if not kernel:
        monkeypatch.setattr(sm, "_gauss_sum_atb_jobs_numba", None)
    rng = np.random.default_rng(2)
    n_gp, n_elm, n = 4, 50, 8
    w = rng.random(n_gp * n_elm)
    new_coefs = _coefficients(rng, n_gp * n_elm, w)
    a_list = [rng.normal(size=(n_gp, n_elm, n)) for _ in new_coefs()]
    b = rng.normal(size=(n_gp, n_elm, n))
    lazy = _bloc_matrix(new_coefs(), a_list, b, True, mat_lumping)
    eager = _bloc_matrix(new_coefs(), a_list, b, False, mat_lumping)
    assert np.array_equal(lazy, eager)
    ref = _reference(new_coefs(), a_list, b)
    if mat_lumping:  # row sums on the diagonal
        ref = np.einsum("eij,ik->eik", ref, np.eye(n))
    np.testing.assert_allclose(lazy, ref, rtol=1e-12, atol=1e-12)


@numba_only
@pytest.mark.parametrize("weights", ["float32", "strided", "short"])
def test_weights_not_readable_in_place(weights):
    rng = np.random.default_rng(6)
    n_gp, n_elm, n = 2, 30, 4
    size = n_gp * n_elm
    w = {
        "float32": rng.random(size).astype(np.float32),
        "strided": rng.random(2 * size)[::2],
        "short": rng.random(1),
    }[weights]
    parts = [rng.normal(size=size), 0.5]
    a = [rng.normal(size=(n_gp, n_elm, n))]
    b = rng.normal(size=(n_gp, n_elm, n))
    coef = sm._WeightedCoef(parts, w)
    lazy = _bloc_matrix([coef], a, b, True)
    assert coef._values is not None  # weighted before the kernel
    assert np.array_equal(lazy, _bloc_matrix([sm._WeightedCoef(parts, w)], a, b, False))


@numba_only
def test_kernel_weighting_leaves_the_coefficient_unweighted():
    rng = np.random.default_rng(3)
    n_gp, n_elm, n = 4, 20, 8
    w = rng.random(n_gp * n_elm)
    coef = sm._WeightedCoef([rng.normal(size=n_gp * n_elm)], w)
    a = rng.normal(size=(n_gp, n_elm, n))
    _bloc_matrix([coef], [a], a, True)
    assert coef._values is None  # read in place: no weighted array


def test_single_operator_and_scaled_coefficient():
    rng = np.random.default_rng(5)
    n_gp, n_elm, n = 3, 10, 4
    size = n_gp * n_elm
    w = rng.random(size)
    part = rng.normal(size=size)
    a = rng.normal(size=(n_gp, n_elm, n))
    b = rng.normal(size=(n_gp, n_elm, n))

    def csr(values):
        cols = (np.arange(size)[:, None] % n_elm) * n + np.arange(n)
        return sparse.csr_matrix(
            (values.reshape(-1), cols.reshape(-1), np.arange(0, n * size + 1, n)),
            shape=(size, n * n_elm),
        )

    res = []
    for coef in (sm._WeightedCoef([part], w), part * w):
        MM = sm._BlocSparse(1, 1, n_gp)
        MM.addToBlocATB(csr(a), csr(b), coef, 0, 0)
        MM.addToBlocATB(csr(b), csr(a), -0.5 * coef, 0, 0)
        MM._flush()
        res.append(MM.data[0][0])
    assert np.array_equal(res[0], res[1])


class _Weighted(np.ndarray):
    """Coefficient weighted beforehand, as the assembly did."""

    def values(self):
        return np.asarray(self)


@pytest.mark.parametrize(
    "elm_type, assume_sym, nlgeom",
    [("hex8", True, False), ("hex8", False, "UL"), ("hex20", True, "TL")],
)
def test_assembly_matches_weighted_arrays(monkeypatch, elm_type, assume_sym, nlgeom):
    """Same global matrix and vector, on one state, as with the coefficients
    weighted before the element products."""
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=4, ny=3, nz=3, elm_type=elm_type)
    wf = fd.weakform.StressEquilibrium(
        fd.constitutivelaw.ElasticIsotrop(200e3, 0.3), nlgeom=nlgeom
    )
    assembly = fd.Assembly.create(wf, mesh)
    assembly.assume_sym = assume_sym
    pb = fd.problem.NonLinear(assembly)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", mesh.bounding_box.xmin), "Disp", 0)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", mesh.bounding_box.xmax), "DispY", 0.05)
    pb.nlsolve(dt=0.5, tmax=1, print_info=0)

    assembly.assemble_global_mat("all")
    K, F = assembly.global_matrix.copy(), np.array(assembly.global_vector)
    created = []

    def recorded(parts, weights):
        created.append(sm._WeightedCoef(parts, weights))
        return created[-1]

    monkeypatch.setattr("fedoo.core.assembly._WeightedCoef", recorded)
    assembly.assemble_global_mat("matrix")
    assert np.array_equal(assembly.global_matrix.data, K.data)
    if sm._gauss_sum_atb_jobs_numba is not None:
        assert any(c._values is None for c in created)  # read in place

    monkeypatch.setattr(
        "fedoo.core.assembly._WeightedCoef",
        lambda parts, weights: (
            sm._WeightedCoef(parts, weights).values().view(_Weighted)
        ),
    )
    assembly.assemble_global_mat("all")
    K0, F0 = assembly.global_matrix, assembly.global_vector

    assert np.array_equal(K.indptr, K0.indptr) and np.array_equal(K.indices, K0.indices)
    assert np.array_equal(K.data, K0.data)
    assert np.array_equal(F, F0)
    assert np.abs(F).max() > 0 and K.nnz > 0
