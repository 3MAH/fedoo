"""The numba assembly kernel gives the numpy element matrices.

To rounding only: numpy sums in an order that depends on its BLAS library and
on the size of the blocs.
"""

import numpy as np
import pytest

import fedoo as fd
import fedoo.core._sparsematrix as sm

pytestmark = pytest.mark.skipif(
    sm._gauss_sum_atb_numba is None, reason="numba not installed"
)


def _numpy(coefs, a_list, b, out=None):
    kernel = sm._gauss_sum_atb_numba
    sm._gauss_sum_atb_numba = None
    try:
        return sm._gauss_sum_atb(coefs, a_list, b, out)
    finally:
        sm._gauss_sum_atb_numba = kernel


def _same(x, y):
    return np.allclose(x, y, rtol=1e-14, atol=1e-14 * np.abs(y).max())


@pytest.mark.parametrize("n_a, n_b", [(8, 8), (4, 4), (8, 3)])
@pytest.mark.parametrize("n_terms", [1, 3])
def test_kernel_matches_numpy(n_a, n_b, n_terms):
    rng = np.random.default_rng(n_a * 100 + n_b + n_terms)
    n_gp, n_elm = 8, 1500
    a_list = [rng.normal(size=(n_gp, n_elm, n_a)) for _ in range(n_terms)]
    coefs = [rng.normal(size=(n_gp, n_elm)), 2.5, 1][:n_terms]
    b = rng.normal(size=(n_gp, n_elm, n_b))

    ref = _numpy(coefs, a_list, b)
    assert _same(sm._gauss_sum_atb(coefs, a_list, b), ref)

    # accumulation in place in an existing bloc
    out_ref, out = ref.copy(), ref.copy()
    _numpy(coefs, a_list, b, out_ref)
    res = sm._gauss_sum_atb(coefs, a_list, b, out)
    assert res is out
    assert _same(out, out_ref)


def _assembled_matrix(elm_type, assume_sym, nlgeom=False):
    import fedoo as fd

    if elm_type in ("hex8", "hex20"):
        fd.ModelingSpace("3D")
        mesh = fd.mesh.box_mesh(nx=5, ny=4, nz=3, elm_type=elm_type)
    else:
        fd.ModelingSpace("2Dstress")
        mesh = fd.mesh.rectangle_mesh(nx=6, ny=5, elm_type=elm_type)
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
    return assembly.global_matrix, assembly.global_vector


@pytest.mark.parametrize(
    "elm_type, assume_sym, nlgeom",
    [
        ("hex8", True, False),
        ("hex8", False, False),
        ("hex20", True, False),
        ("hex8", True, "UL"),
        ("quad4", True, False),
        ("tri3", False, False),
    ],
)
def test_global_csr_matches_numpy(elm_type, assume_sym, nlgeom):
    """Matrix summed by the gather kernel on the cached global pattern, and
    vector by the transposed operators, against the scipy path."""
    kernels = sm._gather_sum_numba, sm._csr_matvec_numba
    K, F = _assembled_matrix(elm_type, assume_sym, nlgeom)
    sm._gather_sum_numba = sm._csr_matvec_numba = None
    try:
        K0, F0 = _assembled_matrix(elm_type, assume_sym, nlgeom)
    finally:
        sm._gather_sum_numba, sm._csr_matvec_numba = kernels
    assert K.has_sorted_indices and K.nnz == K0.nnz
    assert np.array_equal(K.indptr, K0.indptr) and np.array_equal(K.indices, K0.indices)
    assert np.allclose(K.data, K0.data, rtol=1e-13, atol=1e-13 * abs(K0.data).max())
    assert np.allclose(F, F0, rtol=1e-13, atol=1e-13 * abs(F0).max())


def test_csr_matvec_and_transposed():
    from scipy import sparse

    rng = np.random.default_rng(3)
    M = sparse.random(300, 120, density=0.05, format="csr", random_state=rng)
    x, y = rng.normal(size=120), rng.normal(size=300)
    assert np.allclose(sm._csr_matvec(M, x), M @ x, rtol=1e-14, atol=1e-14)
    assert np.allclose(sm._csr_rmatvec(M, y), M.T @ y, rtol=1e-14, atol=1e-14)
    assert np.allclose(sm._csr_rmatvec(M, y), M.T @ y, rtol=1e-14, atol=1e-14)  # cached


@pytest.mark.parametrize(
    "elm_type, nlgeom", [("hex8", False), ("hex20", "UL"), ("quad4", False)]
)
def test_fused_products_match_single_calls(elm_type, nlgeom):
    """The element products computed together (one pass over the elements)
    give the matrix of the products computed one call at a time. Not bit for
    bit: two assemblies of the same updated lagrangian problem in one process
    already differ in the last bits."""
    K, F = _assembled_matrix(elm_type, True, nlgeom)
    fused = sm._gauss_sum_atb_jobs_numba
    sm._gauss_sum_atb_jobs_numba = None
    try:
        K0, F0 = _assembled_matrix(elm_type, True, nlgeom)
    finally:
        sm._gauss_sum_atb_jobs_numba = fused
    assert np.array_equal(K.indices, K0.indices)
    assert np.allclose(K.data, K0.data, rtol=1e-13, atol=1e-13 * abs(K0.data).max())
    assert np.allclose(F, F0, rtol=1e-13, atol=1e-13 * abs(F0).max())


@pytest.mark.skipif(
    sm._elementary_operator_values_numba is None, reason="numba not installed"
)
@pytest.mark.parametrize(
    "space, elm_type",
    [
        ("3D", "hex8"),
        ("3D", "hex20"),
        ("2Dstress", "tri6"),
        ("2Dstress", "quad4"),
        ("2Dstress", "tri3"),
        ("2Dstress", "quad8"),
    ],
)
def test_elementary_operators_numba_match_general_path(space, elm_type):
    """The elementary operators computed in csr order by the numba kernel are
    those of the general path (coo values converted to csr)."""
    fd.ModelingSpace(space)
    if space == "3D":
        mesh = fd.mesh.box_mesh(nx=4, ny=3, nz=3, elm_type=elm_type)
    else:
        mesh = fd.mesh.rectangle_mesh(nx=5, ny=4, elm_type=elm_type)
    # distorted geometry: non constant jacobians
    mesh.nodes = mesh.nodes + 0.05 * np.sin(3.0 * mesh.nodes[:, ::-1])
    material = fd.constitutivelaw.ElasticIsotrop(200e3, 0.3)
    assembly = fd.Assembly.create(fd.weakform.StressEquilibrium(material), mesh)

    def operators():
        fd.Assembly.delete_memory()
        assembly.compute_elementary_operators()
        key = next(k for k in fd.Assembly._saved_elementary_operators if k[0] is mesh)
        return fd.Assembly._saved_elementary_operators[key]

    fast = operators()
    kernel = sm._elementary_operator_values_numba
    sm._elementary_operator_values_numba = None
    try:
        ref = operators()
    finally:
        sm._elementary_operator_values_numba = kernel
    assert fast.keys() == ref.keys()
    for k in ref:
        for A, B in zip(fast[k], ref[k]):
            assert np.array_equal(A.indptr, B.indptr) and np.array_equal(
                A.indices, B.indices
            )
            assert np.allclose(
                A.data, B.data, rtol=1e-13, atol=1e-13 * abs(B.data).max()
            )


@pytest.mark.parametrize("shape", [(40, 8, 3, 3), (40, 4, 2, 2), (5, 3, 3)])
def test_small_det_inv_numba_match_numpy(shape):
    import fedoo.lib_elements.element_base as eb

    rng = np.random.default_rng(0)
    d = shape[-1]
    a = rng.random(shape) + np.eye(d)
    a_strided = np.swapaxes(a, 0, 1) if a.ndim == 4 else a[::-1]
    for x in (a, a_strided):
        det, inv = eb._small_det(x), eb._small_inv(x)
        kernels = list(eb._kernels())
        eb._KERNELS[:] = [None, None]
        try:
            assert np.array_equal(det, eb._small_det(x))
            assert np.array_equal(inv, eb._small_inv(x))
        finally:
            eb._KERNELS[:] = kernels

    a[1] = 0.0  # singular: same error as numpy
    with pytest.raises(np.linalg.LinAlgError):
        eb._small_inv(a)
