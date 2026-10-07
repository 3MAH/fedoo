"""Check UL beam Jacobians against the actual assembled force residual."""

from copy import deepcopy

import numpy as np
import pytest

import fedoo as fd


def _beam(dim, shear=0, consistent=True, multiple=False):
    space = fd.ModelingSpace("2D" if dim == 2 else "3D")
    nodes = np.array([[0.0, 0.0, 0.0], [2.0, 0.3, 0.2], [3.0, 1.0, 0.5]])[:, :dim]
    elements = np.array([[0, 1], [1, 2]]) if multiple else np.array([[0, 1]])
    if not multiple:
        nodes = nodes[:2]
    mesh = fd.Mesh(nodes, elements, "lin2")
    material = fd.constitutivelaw.ElasticIsotrop(1200.0, 0.25)
    wf = fd.weakform.BeamEquilibrium(
        material,
        A=0.4,
        Jx=0.03,
        Iyy=0.02,
        Izz=0.05,
        k=shear,
        nlgeom=True,
        space=space,
        consistent_tangent=consistent,
    )
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    assembly.initialize(pb)
    u = np.zeros((space.nvar, mesh.n_nodes))
    u[space.variable_rank("DispX"), 1:] = 0.12
    u[space.variable_rank("DispY"), 1:] = 0.65
    if dim == 2:
        u[space.variable_rank("RotZ")] = np.linspace(0.25, 0.65, mesh.n_nodes)
    else:
        u[space.variable_rank("DispZ"), 1:] = -0.25
        for name, values in zip(
            ["RotX", "RotY", "RotZ"], [[0.2, 0.45], [-0.3, 0.15], [0.3, 0.7]]
        ):
            u[space.variable_rank(name)] = np.linspace(*values, mesh.n_nodes)
    pb._dU = u.ravel()
    pb.set_X(pb._dU.copy())
    assembly.update(pb)
    # Linearization is with respect to the next spatial increment.
    pb.set_X(np.zeros_like(pb._dU))
    assembly.update(pb)
    return wf, assembly, pb


# Cover each dimension/shear branch, with shared-node assembly in two cases,
# rather than the full product of dimension, shear and mesh configurations.
@pytest.mark.parametrize(
    "dim,shear,multiple", [(2, 0, False), (2, 0.8, True), (3, 0, True), (3, 0.8, False)]
)
def test_tangent_matches_assembled_internal_force(dim, shear, multiple):
    wf, assembly, pb = _beam(dim, shear, multiple=multiple)
    if dim == 3 and shear == 0:
        # Vary section rigidity without the old batch-size implementation check.
        wf.properties.A = np.tile([0.3, 0.5], assembly.n_elm_gp)
        assembly.update(pb)
    tangent = assembly.current.global_matrix.toarray()
    state = deepcopy(assembly.sv)
    base = pb._dU.copy()
    frame = assembly.current._element_local_frame.copy()
    pb._line_search_update = True

    def residual(delta):
        assembly.sv = assembly.current.sv = deepcopy(state)
        assembly.current._element_local_frame = frame.copy()
        pb._dU = base + delta
        pb.set_X(delta)
        assembly.update(pb, compute="vector")
        return -assembly.current.global_vector.copy()

    h = 2e-6
    numerical = np.empty_like(tangent)
    for col in range(len(base)):
        delta = np.zeros_like(base)
        delta[col] = h
        numerical[:, col] = (residual(delta) - residual(-delta)) / (2 * h)
    np.testing.assert_allclose(tangent, numerical, rtol=2e-6, atol=2e-6)
    # A translation of both nodes must not change internal forces.
    for rank in wf.space.get_rank_vector("Disp"):
        translation = np.zeros_like(base)
        translation[
            rank * assembly.mesh.n_nodes : (rank + 1) * assembly.mesh.n_nodes
        ] = 1
        np.testing.assert_allclose(tangent @ translation, 0, atol=1e-10)


def test_opt_out_retains_legacy_tangent():
    wf, assembly, _ = _beam(3)
    consistent = assembly.current.global_matrix.toarray()
    wf.consistent_tangent = False
    assembly.current.assemble_global_mat("matrix")
    legacy = assembly.current.global_matrix.toarray()
    np.testing.assert_allclose(legacy, legacy.T, atol=1e-10)
    assert not np.allclose(legacy, consistent)


@pytest.mark.parametrize("dim", [2, 3])
def test_rollback_restores_force_and_tangent(dim):
    _, assembly, pb = _beam(dim)
    assembly.set_start(pb)
    tangent = assembly.current.global_matrix.toarray().copy()
    force = assembly.current.global_vector.copy()
    base = pb._dU.copy()
    delta = np.full_like(base, 0.025)
    pb._dU = base + delta
    pb.set_X(delta)
    assembly.update(pb)
    pb._dU = base
    pb.set_X(np.zeros_like(base))
    assembly.to_start(pb)
    np.testing.assert_allclose(
        assembly.current.global_matrix.toarray(), tangent, atol=1e-10
    )
    np.testing.assert_allclose(assembly.current.global_vector, force, atol=1e-10)


def test_explicitly_disabled_geometric_stiffness_uses_material_tangent():
    wf, assembly, _ = _beam(3)
    wf.geometric_stiffness = False
    assembly.current.assemble_global_mat()
    assert not wf._use_consistent_tangent(assembly)
    matrix = assembly.current.global_matrix.toarray()
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)


def test_matrix_only_does_not_change_residual_and_correction_is_not_accumulated():
    _, assembly, _ = _beam(3)
    tangent = assembly.current.global_matrix.toarray()
    force = assembly.current.global_vector.copy()
    assembly.current.assemble_global_mat("matrix")
    np.testing.assert_allclose(assembly.current.global_matrix.toarray(), tangent)
    np.testing.assert_array_equal(assembly.current.global_vector, force)
    assembly.current.assemble_global_mat("all")
    np.testing.assert_allclose(assembly.current.global_matrix.toarray(), tangent)


@pytest.mark.parametrize("dim,shear", [(2, 0.8), (3, 0)])
def test_large_deflection_cantilever_converges(dim, shear):
    space = fd.ModelingSpace("2D" if dim == 2 else "3D")
    nodes = np.zeros((5, dim))
    nodes[:, 0] = np.linspace(0, 4, 5)
    mesh = fd.Mesh(nodes, np.column_stack([np.arange(4), np.arange(1, 5)]), "lin2")
    wf = fd.weakform.BeamEquilibrium(
        fd.constitutivelaw.ElasticIsotrop(1200, 0.25),
        A=0.4,
        Jx=0.03,
        Iyy=0.02,
        Izz=0.05,
        k=shear,
        nlgeom=True,
        space=space,
    )
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.set_solver("direct_scipy")
    pb.bc.add("Dirichlet", [0], "Disp", 0)
    pb.bc.add("Dirichlet", [0], "Rot", 0)
    pb.bc.add("Dirichlet", [4], "DispY", 1.5)
    if dim == 3:
        pb.bc.add("Dirichlet", [4], "DispZ", -0.5)
    pb.set_nr_criterion("Displacement", tol=1e-9, max_subiter=15)
    pb.nlsolve(dt=0.25, update_dt=False, print_info=0)
    assert pb.get_disp()[1, -1] == pytest.approx(1.5)
    assert np.isfinite(pb.get_dof_solution()).all()
    # Check equilibrium on the unconstrained DOFs, not only prescribed values.
    residual = assembly.current.global_vector
    np.testing.assert_allclose(residual[pb._dof_free], 0, atol=2e-7)
