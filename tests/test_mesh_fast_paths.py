"""The vectorized mesh routines give the results of the original
(loop based) implementations, bit for bit."""

import numpy as np
import pytest
from scipy import sparse

import fedoo as fd
from fedoo.core.mesh import (
    _gausspoint_node_matrices,
    _sequential_sum,
)
from fedoo.lib_elements.element_list import get_element
from fedoo.mesh.functions import _unique_rows


def _reference_interpolation_matrices(
    elm_geom, n_nodes, gp2node, shape_function_gp, count
):
    n_elements, n = elm_geom.shape
    n_elm_gp = shape_function_gp.shape[0]
    row = np.empty((n_elements, n_elm_gp, n))
    col = np.empty((n_elements, n_elm_gp, n))
    row[:] = (
        np.arange(n_elements).reshape((-1, 1, 1))
        + np.arange(n_elm_gp).reshape(1, -1, 1) * n_elements
    )
    col[:] = elm_geom.reshape((n_elements, 1, n))
    data = gp2node.T.reshape((1, n_elm_gp, n)) / count[elm_geom].reshape(
        (n_elements, 1, n)
    )
    gp2nd = sparse.coo_matrix(
        (data.reshape(-1), (col.reshape(-1), row.reshape(-1))),
        shape=(n_nodes, n_elements * n_elm_gp),
    ).tocsr()
    data = np.empty((n_elements, n_elm_gp, n))
    data[:] = shape_function_gp.reshape((1, n_elm_gp, n))
    nd2gp = sparse.coo_matrix(
        (data.reshape(-1), (row.reshape(-1), col.reshape(-1))),
        shape=(n_elements * n_elm_gp, n_nodes),
    ).tocsr()
    return gp2nd, nd2gp


def _assert_same_csr(a, b):
    assert a.shape == b.shape
    for attr in ("data", "indices", "indptr"):
        x, y = getattr(a, attr), getattr(b, attr)
        assert x.dtype == y.dtype, attr
        assert np.array_equal(x, y), attr


@pytest.mark.parametrize(
    "mesh",
    [
        fd.mesh.box_mesh(5, 4, 3),
        fd.mesh.box_mesh(4, 3, 3, elm_type="hex20"),
        fd.mesh.rectangle_mesh(6, 5, elm_type="tri6"),
        fd.mesh.rectangle_mesh(7, 5, elm_type="quad9"),
    ],
)
def test_interpolation_matrices_match_coo_conversion(mesh):
    rng = np.random.default_rng(2)
    perm = rng.permutation(mesh.n_nodes)  # unsorted element nodes
    elements = perm[mesh.elements]
    elm = get_element(mesh.elm_type)()
    n_gp = len(elm.xi_pg)
    gp2node = np.linalg.pinv(elm.shape_function_gp)
    count = np.bincount(elements.reshape(-1))
    res = _gausspoint_node_matrices(
        elements, mesh.n_nodes, gp2node, np.asarray(elm.shape_function_gp), count
    )
    ref = _reference_interpolation_matrices(
        elements, mesh.n_nodes, gp2node, np.asarray(elm.shape_function_gp), count
    )
    assert n_gp == res[1].shape[0] // mesh.n_elements
    _assert_same_csr(res[0], ref[0])
    _assert_same_csr(res[1], ref[1])


def test_interpolation_with_repeated_nodes_uses_coo_conversion():
    # a hex8 collapsed into a wedge: nodes 2 and 5 repeated
    nodes = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1]], float
    )
    elements = np.array([[0, 1, 2, 2, 3, 4, 5, 5]])
    elm = get_element("hex8")()
    gp2node = fd.lib_elements.element_base.gausspoint_extrapolation_matrix(
        elm.xi_pg, elm.xi_nd, elm.shape_function_gp
    )
    count = np.bincount(elements.reshape(-1))
    sf = np.asarray(elm.shape_function_gp)
    assert _gausspoint_node_matrices(elements, 6, gp2node, sf, count) is None
    mesh = fd.Mesh(nodes, elements, "hex8")
    mesh.init_interpolation(8)
    ref = _reference_interpolation_matrices(elements, 6, gp2node, sf, count)
    _assert_same_csr(mesh._saved_gausspoint2node_mat[8], ref[0])
    _assert_same_csr(mesh._saved_node2gausspoint_mat[8], ref[1])


def _merge_nodes_reference(nodes, elements, node_sets, node_couples):
    # original implementation of Mesh.merge_nodes
    n_nodes = len(nodes)
    nds_del = node_couples[:, 1]
    nds_kept = node_couples[:, 0]
    unique_nodes, ordre = np.unique(nds_del, return_index=True)
    j = 0
    new_num = np.zeros(n_nodes, dtype="int")
    for nd in range(n_nodes):
        if j < len(nds_del) and nd == nds_del[ordre[j]]:
            deleted_nodes = np.where(nds_kept == nds_del[ordre[j]])[0]
            nds_kept[deleted_nodes] = nds_kept[ordre[j]]
            j += 1
        else:
            new_num[nd] = nd - j
    new_num[nds_del] = new_num[node_couples[:, 0]]
    list_nd_new = [nd for nd in range(n_nodes) if not (nd in nds_del)]
    node_sets = {key: new_num[value] for key, value in node_sets.items()}
    return nodes[list_nd_new], new_num[elements], node_sets


@pytest.mark.parametrize("seed", range(8))
def test_merge_nodes_matches_reference(seed):
    rng = np.random.default_rng(seed)
    mesh = fd.mesh.box_mesh(6, 5, 4)
    mesh.node_sets["s"] = np.arange(0, mesh.n_nodes, 3)
    n = mesh.n_nodes
    deleted = rng.choice(n, size=25, replace=False)
    kept = rng.choice(np.setdiff1d(np.arange(n), deleted), size=25)
    if seed % 2:  # chains: kept nodes that are deleted later or earlier
        k = rng.choice(25, size=10, replace=False)
        kept[k] = deleted[rng.permutation(25)[:10]]
        kept[kept == deleted] = kept[kept == deleted] + 1  # no self pair
    couples = np.c_[kept, deleted]
    ref_couples = couples.copy()
    ref = _merge_nodes_reference(
        mesh.nodes.copy(), mesh.elements.copy(), dict(mesh.node_sets), ref_couples
    )
    mesh.merge_nodes(couples)
    assert np.array_equal(mesh.nodes, ref[0])
    assert np.array_equal(mesh.elements, ref[1])
    assert np.array_equal(mesh.node_sets["s"], ref[2]["s"])
    assert np.array_equal(couples, ref_couples)  # same update of node_couples


def test_remove_and_isolated_nodes():
    mesh = fd.mesh.box_mesh(5, 4, 3)
    sub = mesh.extract_elements([0, 4, 7, 20])
    used = np.unique(sub.elements)
    expected = np.setdiff1d(np.arange(sub.n_nodes), used)
    assert np.array_equal(sub.find_isolated_nodes(), expected)
    n_nodes = sub.n_nodes
    assert sub.remove_isolated_nodes() == len(expected)
    assert sub.n_nodes == n_nodes - len(expected)
    assert np.array_equal(np.unique(sub.elements), np.arange(sub.n_nodes))
    mesh = fd.mesh.box_mesh(5, 4, 3)
    new_num = mesh.remove_nodes([3, 7, 7, 1000])
    kept = [nd for nd in range(60) if nd not in (3, 7)]
    assert np.array_equal(new_num[kept], np.arange(58))
    assert mesh.n_nodes == 58


def test_get_elements_from_nodes():
    mesh = fd.mesh.box_mesh(6, 5, 4)
    node_set = mesh.find_nodes("X", 0.0)
    s = set(node_set.tolist())
    mesh.node_sets["left"] = node_set
    for all_nodes in (True, False):
        test = all if all_nodes else any
        expected = [
            i
            for i, element in enumerate(mesh.elements)
            if test([n in s for n in element])
        ]
        for nset in (
            node_set,
            list(node_set),
            set(s),
            iter(node_set),
            node_set.astype(float),
            "left",
        ):
            res = mesh.get_elements_from_nodes(nset, all_nodes)
            assert isinstance(res, list) and res == expected
            assert all(type(i) is int for i in res)
    assert mesh.get_elements_from_nodes([]) == []


def test_unique_rows_matches_numpy():
    rng = np.random.default_rng(3)
    for n_cols in (1, 2, 3, 4, 6):
        rows = rng.integers(0, 7, size=(500, n_cols))
        index, counts = _unique_rows(rows)
        _, ref_index, ref_counts = np.unique(
            rows, axis=0, return_index=True, return_counts=True
        )
        assert np.array_equal(index, ref_index)
        assert np.array_equal(counts, ref_counts)


@pytest.mark.parametrize("elm_type", ["hex8", "hex20", "tet4", "tet10"])
def test_extract_surface_normals_outward(elm_type):
    if elm_type.startswith("hex"):
        mesh = fd.mesh.box_mesh(4, 4, 3, elm_type=elm_type)
    else:
        import pyvista as pv

        mesh = fd.Mesh.from_pyvista(pv.ImageData(dimensions=(4, 4, 3)).triangulate())
        if elm_type == "tet10":
            mesh = fd.mesh.change_elm_type(mesh, "tet10")
    perm = np.random.default_rng(4).permutation(mesh.n_nodes)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    mesh = fd.Mesh(mesh.nodes[perm], inv[mesh.elements], elm_type)
    surf = fd.mesh.extract_surface(mesh)
    crd = surf.nodes[surf.elements[:, :3]]
    normal = np.cross(crd[:, 1] - crd[:, 0], crd[:, 2] - crd[:, 0])
    center = mesh.nodes.mean(axis=0)
    assert np.all(np.sum(normal * (crd.mean(axis=1) - center), axis=1) > 0)


def test_sequential_sum_is_python_sum():
    rng = np.random.default_rng(5)
    x = rng.standard_normal(1001) * 10.0 ** rng.integers(-8, 8, 1001)
    assert _sequential_sum(x) == sum(x)
    y = rng.standard_normal((300, 4))
    assert np.array_equal(_sequential_sum(y), sum(y))
    assert _sequential_sum(np.array([-0.0, -0.0])) == sum(np.array([-0.0, -0.0]))
    assert _sequential_sum(np.zeros(0)) == 0
    narrow = np.array([200, 100], dtype=np.uint8)
    with np.errstate(over="ignore"):
        expected = sum(narrow)
        res = _sequential_sum(narrow)
    assert res == expected and res.dtype == expected.dtype


def _spr_reference(mesh, data, n_elm_gp):
    # original implementation (np.add.at)
    data = data.reshape(-1, 1) if data.ndim == 1 else data
    gp_coordinates = mesh.gausspoint_coordinates(n_elm_gp)
    element_ids = np.arange(mesh.n_elements).reshape(-1, 1, 1)
    gp_ids = np.arange(n_elm_gp).reshape(1, 1, -1)
    shape = (mesh.n_elements, mesh.n_elm_nodes, n_elm_gp)
    patch_node_ids = np.broadcast_to(mesh.elements[:, :, None], shape).reshape(-1)
    patch_gp_ids = np.broadcast_to(
        element_ids + gp_ids * mesh.n_elements, shape
    ).reshape(-1)
    shifted = gp_coordinates[patch_gp_ids] - mesh.nodes[patch_node_ids, : mesh.ndim]
    patch_scale = np.zeros(mesh.n_nodes)
    np.maximum.at(patch_scale, patch_node_ids, np.linalg.norm(shifted, axis=1))
    nonzero = patch_scale[patch_node_ids] > 0
    shifted[nonzero] /= patch_scale[patch_node_ids][nonzero].reshape(-1, 1)
    design = np.column_stack((np.ones(len(patch_node_ids)), shifted))
    n_terms = design.shape[1]
    normal = np.zeros((mesh.n_nodes, n_terms, n_terms))
    rhs = np.zeros((mesh.n_nodes, n_terms, data.shape[1]))
    np.add.at(normal, patch_node_ids, design[:, :, None] * design[:, None, :])
    np.add.at(rhs, patch_node_ids, design[:, :, None] * data[patch_gp_ids, None, :])
    node_data = (np.linalg.pinv(normal, rcond=1e-12) @ rhs)[:, 0, :]
    counts = np.bincount(patch_node_ids, minlength=mesh.n_nodes).reshape(-1, 1)
    patch_sum = np.zeros((mesh.n_nodes, data.shape[1]))
    np.add.at(patch_sum, patch_node_ids, data[patch_gp_ids])
    low_rank = np.linalg.matrix_rank(normal) < 2
    has_patch = counts[:, 0] > 0
    node_data[low_rank & has_patch] = (
        patch_sum[low_rank & has_patch] / counts[low_rank & has_patch]
    )
    return node_data


@pytest.mark.parametrize("elm_type", ["hex8", "quad4", "tri6"])
def test_spr_matches_add_at(elm_type):
    if elm_type == "hex8":
        mesh = fd.mesh.box_mesh(5, 4, 4)
    else:
        mesh = fd.mesh.rectangle_mesh(7, 5, elm_type=elm_type)
    # an isolated node (no patch)
    mesh = fd.Mesh(
        np.vstack([mesh.nodes, mesh.nodes[:1] + 10]), mesh.elements, elm_type
    )
    n_gp = fd.lib_elements.get_default_n_gp(elm_type)
    data = np.sin(np.arange(3 * mesh.n_elements * n_gp) * 0.7).reshape(-1, 3)
    res = mesh._gausspoint_to_node_spr(data, n_gp)
    assert np.array_equal(res, _spr_reference(mesh, data, n_gp))
    res1 = mesh._gausspoint_to_node_spr(data[:, 0], n_gp)
    assert np.array_equal(res1, _spr_reference(mesh, data[:, 0], n_gp)[:, 0])


@pytest.mark.parametrize("bad", ["too_large", "negative"])
def test_interpolation_with_invalid_node_index_raises(bad):
    # same error as the coo conversion: no csr built with out of range indices
    mesh = fd.mesh.box_mesh(3, 3, 3)
    elements = mesh.elements.copy()
    if bad == "too_large":
        nodes = mesh.nodes[:-1]
    else:
        nodes = mesh.nodes
        elements[0, 0] = -1
    elm = get_element("hex8")()
    gp2node = np.linalg.pinv(elm.shape_function_gp)
    sf = np.asarray(elm.shape_function_gp)
    count = np.ones(len(mesh.nodes))
    assert _gausspoint_node_matrices(elements, len(nodes), gp2node, sf, count) is None
    with pytest.raises(ValueError):
        fd.Mesh(nodes, elements, "hex8").init_interpolation(8)


@pytest.mark.parametrize("nx, ny", [(2, 2), (5, 4), (4, 7)])
def test_rectangle_connectivity_matches_loops(nx, ny):
    quad4 = [
        [nx * j + i, nx * j + i + 1, nx * (j + 1) + i + 1, nx * (j + 1) + i]
        for j in range(ny - 1)
        for i in range(nx - 1)
    ]
    tri3 = []
    for j in range(ny - 1):
        tri3 += [[nx * j + i, nx * j + i + 1, nx * (j + 1) + i] for i in range(nx - 1)]
        tri3 += [
            [nx * j + i + 1, nx * (j + 1) + i + 1, nx * (j + 1) + i]
            for i in range(nx - 1)
        ]
    for elm_type, expected in [("quad4", quad4), ("tri3", tri3)]:
        mesh = fd.mesh.rectangle_mesh(nx, ny, elm_type=elm_type)
        assert np.array_equal(mesh.elements, expected)


@pytest.mark.parametrize("nx, ny, nz", [(2, 2, 2), (3, 4, 5), (5, 3, 4)])
def test_box_connectivity_matches_loops(nx, ny, nz):
    n = nx * ny
    expected = [
        [
            nx * j + i + k * n,
            nx * j + i + 1 + k * n,
            nx * (j + 1) + i + 1 + k * n,
            nx * (j + 1) + i + k * n,
            nx * j + i + k * n + n,
            nx * j + i + 1 + k * n + n,
            nx * (j + 1) + i + 1 + k * n + n,
            nx * (j + 1) + i + k * n + n,
        ]
        for k in range(nz - 1)
        for j in range(ny - 1)
        for i in range(nx - 1)
    ]
    mesh = fd.mesh.box_mesh(nx, ny, nz, elm_type="hex8")
    assert np.array_equal(mesh.elements, expected)
