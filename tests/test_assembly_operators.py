"""Assembly paths that must give the results of their simpler form bit for
bit: thread count of the numba kernels, elementary operators applied to
several vectors at once, cached indices of the change of basis.
"""

import numpy as np
import pytest
from scipy import sparse

import fedoo as fd
import fedoo.core._sparsematrix as sm

numba_only = pytest.mark.skipif(
    sm._gauss_sum_atb_jobs_numba is None, reason="numba not installed"
)


@numba_only
def test_kernel_results_do_not_depend_on_threads():
    rng = np.random.default_rng(4)
    M = sparse.random(5000, 800, density=0.01, format="csr", random_state=rng)
    x = rng.normal(size=800)
    grain = sm._THREAD_GRAIN
    try:
        sm._THREAD_GRAIN = 10**12  # one thread
        one = sm._csr_matvec(M, x)
        sm._THREAD_GRAIN = 1  # all the threads
        many = sm._csr_matvec(M, x)
    finally:
        sm._THREAD_GRAIN = grain
    assert np.array_equal(one, many)


def test_mesh_coordinate_index():
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2)
    assembly = fd.Assembly.create(
        fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)), mesh
    )
    space = assembly.space
    for name in ("X", "Y", "Z"):
        rank = space.coordinate_rank(name)
        assert assembly._mesh_coordinate_index(rank) == mesh.crd_name.index(name)
    assert assembly._mesh_coordinate_index(99) is None


@numba_only
def test_elementary_operators_data_not_copied():
    """The operators computed by the numba kernel keep their value arrays
    (scipy copies a row of a larger 2d array)."""
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=3, ny=3, nz=3)
    assembly = fd.Assembly.create(
        fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)), mesh
    )
    fd.Assembly.delete_memory()
    assembly.compute_elementary_operators()
    key = next(k for k in fd.Assembly._saved_elementary_operators if k[0] is mesh)
    for ops in fd.Assembly._saved_elementary_operators[key].values():
        data = ops[0].data
        assert data.base is None or data.base.size == data.size


@numba_only
def test_transposed_operator_from_pattern():
    rng = np.random.default_rng(7)
    row = np.repeat(np.arange(50), 4)
    col = np.concatenate([rng.choice(30, 4, replace=False) for _ in range(50)])
    data = rng.normal(size=200)
    M = sm._coo_to_csr(data, row, col, (50, 30))
    y = rng.normal(size=50)
    ref = M.T.tocsr()
    ref.sort_indices()
    assert np.array_equal(sm._csr_rmatvec(M, y), sm._csr_matvec(ref, y))


def test_change_of_basis_indices_reused():
    """The change of basis matrix is rebuilt (new local frames) on the cached
    csr conversion of its coo indices."""
    fd.ModelingSpace("3D")
    mesh = fd.mesh.extract_edges(fd.mesh.box_mesh(3, 3, 3))
    props = fd.constitutivelaw.BeamCircular(
        fd.constitutivelaw.ElasticIsotrop(1e5, 0.3), 1
    )
    assembly = fd.Assembly.create(fd.weakform.BeamEquilibrium(props), mesh)
    first = assembly.get_change_of_basis_mat()
    fd.Assembly._saved_change_of_basis_mat.clear()
    second = assembly.get_change_of_basis_mat()
    assert second is not first
    assert second._fedoo_pattern is first._fedoo_pattern
    assert np.array_equal(second.data, first.data)
    assert np.array_equal(second.indices, first.indices)
    ref = sparse.coo_matrix(first).tocsr()
    assert np.array_equal(ref.indptr, first.indptr)


@numba_only
def test_multi_products_are_single_products():
    rng = np.random.default_rng(8)
    M = sparse.random(400, 150, density=0.05, format="csr", random_state=rng)
    xs = [rng.normal(size=150) for _ in range(3)]
    ys = [rng.normal(size=400) for _ in range(3)]
    for res, x in zip(sm._csr_matvec_multi(M, xs), xs):
        assert np.array_equal(res, sm._csr_matvec(M, x))
    for res, y in zip(sm._csr_rmatvec_multi(M, ys), ys):
        assert np.array_equal(res, sm._csr_rmatvec(M, y))
    assert sm._csr_matvec_multi(M, [rng.normal(size=149)]) is None
    assert sm._csr_matvec_multi(M.tocsc(), xs) is None


@pytest.mark.parametrize("location", ["GaussPoint", "Node", "Element"])
def test_grad_disp_by_operator(location):
    """The gradient computed operator by operator (shared elementary
    operators applied together) is get_gp_results of each component."""
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=4, ny=3, nz=3, elm_type="hex8")
    assembly = fd.Assembly.create(
        fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)), mesh
    )
    fd.problem.Linear(assembly)
    U = np.random.default_rng(9).normal(size=3 * mesh.n_nodes)
    grad = assembly.get_grad_disp(U, location)
    for i, line in enumerate(assembly.space.op_grad_u()):
        for j, op in enumerate(line):
            if location == "GaussPoint":
                ref = assembly.get_gp_results(op, U)
            elif location == "Node":
                ref = assembly.get_node_results(op, U)
            else:
                ref = assembly.get_element_results(op, U)
            assert np.array_equal(grad[i][j], ref)


def test_change_of_basis_indices_follow_the_element_table():
    """An element table modified in place gives new change of basis indices."""
    fd.ModelingSpace("3D")
    mesh = fd.mesh.extract_edges(fd.mesh.box_mesh(3, 3, 3))
    props = fd.constitutivelaw.BeamCircular(
        fd.constitutivelaw.ElasticIsotrop(1e5, 0.3), 1
    )
    assembly = fd.Assembly.create(fd.weakform.BeamEquilibrium(props), mesh)
    assembly.get_change_of_basis_mat()
    mesh.elements[0] = mesh.elements[1]  # in place
    fd.Assembly._saved_change_of_basis_mat.clear()
    modified = assembly.get_change_of_basis_mat()
    del mesh._saved_change_of_basis_indices  # nothing saved: built anew
    fd.Assembly._saved_change_of_basis_mat.clear()
    ref = assembly.get_change_of_basis_mat()
    assert np.array_equal(modified.indptr, ref.indptr)
    assert np.array_equal(modified.indices, ref.indices)
    assert np.array_equal(modified.data, ref.data)


@numba_only
def test_numba_thread_count_restored():
    import numba

    rng = np.random.default_rng(10)
    M = sparse.random(3000, 800, density=0.02, format="csr", random_state=rng)
    previous = numba.get_num_threads()
    n = max(1, min(3, numba.config.NUMBA_NUM_THREADS))
    try:
        numba.set_num_threads(n)
        sm._csr_matvec(M, rng.normal(size=800))
        assert numba.get_num_threads() == n
        with pytest.raises(RuntimeError):
            with sm._numba_threads(10**9):
                raise RuntimeError
        assert numba.get_num_threads() == n
    finally:
        numba.set_num_threads(previous)


def test_grad_disp_components_are_independent_arrays():
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=3, ny=3, nz=3)
    assembly = fd.Assembly.create(
        fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)), mesh
    )
    fd.problem.Linear(assembly)
    U = np.random.default_rng(11).normal(size=3 * mesh.n_nodes)
    grad = assembly.get_grad_disp(U, "GaussPoint")
    for line in grad:
        for value in line:
            assert value.base is None


def test_csr_matvec_wrong_length_raises():
    M = sparse.random(40, 30, density=0.2, format="csr", random_state=1)
    with pytest.raises(ValueError):
        sm._csr_matvec(M, np.ones(29))


@numba_only
@pytest.mark.parametrize("nlgeom", [False, "UL"])
def test_vector_grouped_products(monkeypatch, nlgeom):
    """The residual with the transposed products grouped by operator is the
    residual computed term by term."""
    import fedoo.core.assembly as assembly_module

    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=4, ny=3, nz=3)
    wf = fd.weakform.StressEquilibrium(
        fd.constitutivelaw.ElasticIsotrop(200e3, 0.3), nlgeom=nlgeom
    )
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", mesh.bounding_box.xmin), "Disp", 0)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", mesh.bounding_box.xmax), "DispY", 0.05)
    pb.nlsolve(dt=0.5, tmax=1, print_info=0)
    assembly.assemble_global_mat("vector")
    grouped = np.array(assembly.global_vector)
    assert np.any(grouped != 0)
    monkeypatch.setattr(assembly_module, "_csr_rmatvec_multi", lambda M, x: None)
    assembly.assemble_global_mat("vector")
    assert np.array_equal(assembly.global_vector, grouped)
