"""The global matrix of an assembly with a change of basis (beam, plate,
interface elements), computed by rotating the element matrices one by one
(numba), is mat_cb.T @ A @ mat_cb up to rounding."""

import numpy as np
import pytest

import fedoo as fd
import fedoo.core._sparsematrix as sm

pytestmark = pytest.mark.skipif(
    sm._change_of_basis_numba is None, reason="numba not installed"
)


def _matrices(assembly):
    """Global matrix by the kernel and by the sparse products."""
    method = sm._BlocSparse._tocsr_change_of_basis
    used = []

    def spy(self, *args):
        res = method(self, *args)
        used.append(res is not None)
        return res

    try:
        sm._BlocSparse._tocsr_change_of_basis = spy
        assembly.assemble_global_mat("matrix")
        fast = assembly.global_matrix.tocsr()
        assert used == [True]  # the kernel path, not its fallback
        sm._BlocSparse._tocsr_change_of_basis = lambda self, *args: None
        assembly.assemble_global_mat("matrix")
        ref = assembly.global_matrix.tocsr()
    finally:
        sm._BlocSparse._tocsr_change_of_basis = method
    return fast, ref


def _check(fast, ref):
    assert fast.shape == ref.shape
    assert fast.has_sorted_indices
    scale = abs(ref).max()
    assert abs(fast - ref).max() <= 1e-14 * scale
    # the pattern of the product, explicit zeros apart
    nonzero = fast.copy()
    nonzero.data[np.abs(nonzero.data) <= 1e-14 * scale] = 0
    nonzero.eliminate_zeros()
    ref = ref.copy()
    ref.data[np.abs(ref.data) <= 1e-14 * scale] = 0
    ref.eliminate_zeros()
    assert (abs(nonzero) > 0).sum() == (abs(ref) > 0).sum()


@pytest.mark.parametrize("elm_type", ["pquad4", "ptri3", "pquad8"])
def test_plate(elm_type):
    fd.ModelingSpace("3D")
    geom = {"pquad4": "quad4", "ptri3": "tri3", "pquad8": "quad8"}[elm_type]
    mesh = fd.mesh.rectangle_mesh(5, 4, 0, 10, 0, 8, elm_type=geom, ndim=3)
    # a curved, rotated surface: non trivial local frames
    x, y = mesh.nodes[:, 0].copy(), mesh.nodes[:, 1].copy()
    mesh.nodes[:, 2] = 0.05 * x**2 + 0.02 * x * y
    shell = fd.constitutivelaw.ShellHomogeneous(
        fd.constitutivelaw.ElasticIsotrop(2e5, 0.3), 0.5
    )
    assembly = fd.Assembly.create(fd.weakform.PlateEquilibrium(shell), mesh, elm_type)
    fd.problem.Linear(assembly)
    _check(*_matrices(assembly))


@pytest.mark.parametrize("assume_sym", [False, True])
@pytest.mark.parametrize("nlgeom", [False, True])
def test_beam(nlgeom, assume_sym):
    fd.ModelingSpace("3D")
    mesh = fd.mesh.extract_edges(fd.mesh.box_mesh(3, 3, 3, x_max=10, y_max=8, z_max=6))
    props = fd.constitutivelaw.BeamCircular(
        fd.constitutivelaw.ElasticIsotrop(1e5, 0.3), 1, k=0.9
    )
    wf = fd.weakform.BeamEquilibrium(props, consistent_tangent=True)
    # forced on a tangent that is not symmetric: same matrix as the products
    wf.assembly_options["assume_sym"] = assume_sym
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly, nlgeom=nlgeom)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0), ["Disp", "Rot"], 0)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 10), "DispY", -1)
    pb.nlsolve(dt=0.5, tmax=1, print_info=0)
    _check(*_matrices(assembly.current))


def test_axis_aligned_beams_drop_exact_zeros():
    """Frames along the axes: most of the structural entries are zero."""
    fd.ModelingSpace("3D")
    mesh = fd.mesh.extract_edges(fd.mesh.box_mesh(4, 4, 4))
    props = fd.constitutivelaw.BeamCircular(
        fd.constitutivelaw.ElasticIsotrop(1e5, 0.3), 0.1
    )
    assembly = fd.Assembly.create(fd.weakform.BeamEquilibrium(props), mesh)
    fd.problem.Linear(assembly)
    fast, ref = _matrices(assembly)
    _check(fast, ref)
    assert np.all(fast.data != 0)
    assert fast.nnz <= ref.nnz
    again, _ = _matrices(assembly)  # the cached pattern is left untouched
    assert (fast != again).nnz == 0


def test_interface():
    fd.ModelingSpace("2D")
    nodes = np.array([[0.0, 0.0], [1.0, 0.3], [0.0, 0.0], [1.0, 0.3]])
    mesh = fd.Mesh(nodes, np.array([[0, 1, 2, 3]]), "lin2interface")
    law = fd.constitutivelaw.CohesiveLaw(axis=1, tangent_mode="consistent")
    assembly = fd.Assembly.create(fd.weakform.InterfaceForce(law), mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.bc.add("Dirichlet", [0, 1], "Disp", 0.0)
    pb.bc.add("Dirichlet", [2, 3], "DispY", 0.004)
    pb.nlsolve(dt=1.0, tmax=1.0, print_info=0)
    _check(*_matrices(assembly))


def test_not_applicable_returns_none():
    """A local dof related to two nodes: the sparse products are used."""
    from scipy import sparse

    fd.ModelingSpace("3D")
    mesh = fd.mesh.extract_edges(fd.mesh.box_mesh(2, 2, 2))
    props = fd.constitutivelaw.BeamCircular(
        fd.constitutivelaw.ElasticIsotrop(1e5, 0.3), 1
    )
    assembly = fd.Assembly.create(fd.weakform.BeamEquilibrium(props), mesh)
    fd.problem.Linear(assembly)
    mat_cb = assembly.get_change_of_basis_mat().tocsr()
    MM = sm._BlocSparse(assembly.space.nvar, assembly.space.nvar, assembly.n_elm_gp)
    assert MM._tocsr_change_of_basis(mat_cb, mesh.n_nodes) is None  # empty
    assembly.assemble_global_mat("matrix")
    bloc = sm._BlocSparse(
        assembly.space.nvar,
        assembly.space.nvar,
        assembly.n_elm_gp,
        assembly._saved_bloc_structure,
    )
    bloc.data = [[None] * assembly.space.nvar for _ in range(assembly.space.nvar)]
    n_elm = mesh.n_elements
    n = len(bloc.row) // n_elm
    bloc.data[0][0] = np.ones((n_elm, int(np.sqrt(n)), int(np.sqrt(n))))
    bloc.isempty = False
    assert bloc._tocsr_change_of_basis(mat_cb, mesh.n_nodes) is not None
    mixed = mat_cb.tolil()
    mixed[0, 1] = 0.5  # local dof 0 also related to the node 1
    mixed[0, mesh.n_nodes - 1] = 0.5
    assert bloc._tocsr_change_of_basis(sparse.csr_matrix(mixed), mesh.n_nodes) is None
