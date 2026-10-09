"""Check the optional shell tangent against the evaluated force residual."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import fedoo as fd
from fedoo.lib_elements.element_list import get_element


def _shell(
    element="pquad4mitc",
    drilling=True,
    consistent=True,
    section=None,
    n_elements=1,
    extra_variable=False,
    damping=False,
):
    geometry = get_element(element).geometry_elm
    coordinates = geometry(1).xi_nd
    nodes = np.column_stack([coordinates, np.zeros(len(coordinates))]).astype(float)
    # Distorted reference geometry and oblique frame exercise coordinate maps.
    nodes[:, 0] += 0.12 * nodes[:, 0] * nodes[:, 1]
    nodes = nodes @ Rotation.from_rotvec([0.2, -0.3, 0.1]).as_matrix().T
    if n_elements > 1:
        nodes = np.vstack(
            [
                nodes @ Rotation.from_rotvec([0.1 * i, 0, 0]).as_matrix().T
                + [3 * i, 0, 0]
                for i in range(n_elements)
            ]
        )
        coordinates = np.tile(coordinates, (n_elements, 1))
    space = fd.ModelingSpace("3D")
    if extra_variable:
        space.new_variable("Uncoupled")
    mesh = fd.Mesh(nodes, np.arange(len(nodes)).reshape(n_elements, -1), geometry.name)
    material = fd.constitutivelaw.ElasticIsotrop(1200, 0.25)
    law = (
        section
        if section is not None
        else fd.constitutivelaw.ShellHomogeneous(material, 0.15)
    )
    wf = fd.weakform.PlateEquilibrium(
        law,
        nlgeom=True,
        true_drilling_rotation=drilling,
        consistent_tangent=consistent,
        space=space,
    )
    equation = wf
    if damping:
        equation += fd.weakform.ArtificialDamping(
            c_stab=0.01, variables=["Rot"], space=space
        )
    assembly = fd.Assembly.create(equation, mesh, elm_type=element)
    pb = fd.problem.NonLinear(assembly)
    assembly.initialize(pb)
    u = np.zeros((space.nvar, len(nodes)))
    u[space.get_rank_vector("Disp")] = (
        np.array([[0.08, 0.02, 0], [-0.04, 0.06, 0], [0.03, -0.05, 0]]) @ nodes.T
    )
    u[space.variable_rank("DispZ")] += 0.06 * coordinates[:, 0] * coordinates[:, 1]
    u[space.get_rank_vector("Rot")] = (
        np.array([0.15, -0.2, 0.12])[:, None] + 0.04 * nodes.T
    )
    pb._dU = u.ravel()
    pb.set_X(pb._dU.copy())
    assembly.update(pb)
    pb.set_X(np.zeros_like(pb._dU))
    assembly.update(pb)
    return wf, assembly, pb


@pytest.mark.parametrize(
    "element,drilling",
    [
        ("pquad4mitc", True),
        ("ptri3mitc", True),
        ("pquad4mitc", False),
        ("pquad4", True),
        ("pquad4sri", True),
        ("pquad8mitc", True),
    ],
)
def test_shell_tangent_matches_force_derivative(element, drilling):
    _, assembly, pb = _shell(element, drilling)
    _assert_force_derivative(assembly, pb, full=element == "pquad4mitc" and drilling)


def _assert_force_derivative(assembly, pb, full=False):
    matrix = assembly.current.global_matrix.copy()
    state = deepcopy(assembly.sv)
    base = pb._dU.copy()
    frame = assembly.current._element_local_frame.copy()
    pb._line_search_update = True
    law = assembly.weakform.constitutivelaw
    material_assemblies = getattr(law, "_material_assemblies", None) or []
    if getattr(law, "_material_assembly", None) is not None:
        material_assemblies = [law._material_assembly]
    material_states = [
        (deepcopy(a.sv), deepcopy(a.sv_start)) for a in material_assemblies
    ]

    def force(delta):
        assembly.sv = assembly.current.sv = deepcopy(state)
        assembly.current._element_local_frame = frame.copy()
        for material_assembly, (sv, sv_start) in zip(
            material_assemblies, material_states
        ):
            material_assembly.sv = deepcopy(sv)
            material_assembly.sv_start = deepcopy(sv_start)
        pb._dU = base + delta
        pb.set_X(delta)
        assembly.update(pb, compute="vector")
        return -assembly.current.global_vector.copy()

    # Full Jacobian for the default MITC4; a few mixed directions suffice for
    # the distinct interpolation/frame branches without a configuration sweep.
    directions = (
        np.eye(len(base))
        if full
        else np.random.default_rng(17).normal(size=(3, len(base)))
    )
    for direction in directions:
        h = 1e-6
        numerical = (force(h * direction) - force(-h * direction)) / (2 * h)
        np.testing.assert_allclose(matrix @ direction, numerical, rtol=3e-6, atol=3e-6)


@pytest.mark.parametrize("laminate", [False, True])
def test_shell_thickness_integrated_material_tangent(laminate):
    material = fd.constitutivelaw.ElasticIsotrop(1200, 0.25)
    if laminate:
        law = fd.constitutivelaw.ShellLaminateNonLinear(
            [material, fd.constitutivelaw.ElasticIsotrop(600, 0.2)],
            [0.05, 0.1],
            n_thickness_points=2,
        )
    else:
        law = fd.constitutivelaw.ShellHomogeneousNonLinear(
            material, 0.15, n_thickness_points=2
        )
    _, assembly, pb = _shell(section=law)
    _assert_force_derivative(assembly, pb)


def test_shell_tangent_with_varying_geometry_and_gauss_point_section():
    # Different element frames and pointwise stiffness exercise both local-node
    # extraction and Gauss-major coefficient ordering during contraction.
    law = fd.constitutivelaw.ShellHomogeneous(
        fd.constitutivelaw.ElasticIsotrop(np.linspace(900, 1500, 8), 0.25), 0.15
    )
    _, assembly, pb = _shell(section=law, n_elements=2)
    _assert_force_derivative(assembly, pb)


def test_shell_tangent_with_additional_variable_and_shifted_dof_ranks():
    _, assembly, pb = _shell(extra_variable=True)
    _assert_force_derivative(assembly, pb)


def test_shell_historical_tangent_remains_default():
    law = fd.constitutivelaw.ShellHomogeneous(
        fd.constitutivelaw.ElasticIsotrop(1200, 0.25), 0.15
    )
    wf = fd.weakform.PlateEquilibrium(law, space=fd.ModelingSpace("3D"))
    assert wf.consistent_tangent is False
    _, assembly, _ = _shell(consistent=False)
    assert assembly.elm_type == "pquad4mitc"
    matrix = assembly.current.global_matrix.toarray()
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)


@pytest.mark.parametrize("damping", [False, True])
def test_shell_tangent_fields_are_prepared_once_per_update(monkeypatch, damping):
    wf, assembly, pb = _shell(damping=damping)
    current = assembly.current
    force = current.global_vector.copy()
    matrix = current.global_matrix.toarray().copy()

    prepare = wf._prepare_consistent_tangent
    prepared = []

    def counted_prepare(assembly):
        tangent = prepare(assembly)
        prepared.append(tangent)
        return tangent

    monkeypatch.setattr(wf, "_prepare_consistent_tangent", counted_prepare)
    assembly.update(pb, compute="vector")
    np.testing.assert_allclose(current.global_vector, force, atol=1e-10)
    np.testing.assert_array_equal(current.global_matrix.toarray(), matrix)
    current.assemble_global_mat("matrix")
    current.assemble_global_mat("vector")
    wf.get_weak_equation(current, pb)
    assert len(prepared) == 1
    assert current.sv["_ShellTangentOperators"] is prepared[0]
    assert not hasattr(current, "_assembly_compute")
    np.testing.assert_allclose(current.global_matrix.toarray(), matrix, atol=1e-9)
    assembly.update(pb, compute="vector")
    assert len(prepared) == 2
    assert prepared[0] is not prepared[1]


def test_shell_trial_and_rollback_restore_frame_force_and_tangent():
    _, assembly, pb = _shell()
    assembly.set_start(pb)
    force = assembly.current.global_vector.copy()
    matrix = assembly.current.global_matrix.toarray().copy()
    frame = assembly.current.get_element_local_frame().copy()
    base = pb._dU.copy()
    delta = np.linspace(-0.01, 0.02, len(base))
    pb._line_search_update = True
    pb._dU = base + delta
    pb.set_X(delta)
    assembly.update(pb)
    assert not np.allclose(assembly.current.get_element_local_frame(), frame)
    pb._line_search_update = False
    pb._dU = base
    pb.set_X(np.zeros_like(base))
    assembly.to_start(pb)
    np.testing.assert_allclose(assembly.current.get_element_local_frame(), frame)
    np.testing.assert_allclose(assembly.current.global_vector, force, atol=1e-10)
    np.testing.assert_allclose(
        assembly.current.global_matrix.toarray(), matrix, atol=1e-9
    )


def test_mitc_shell_large_deflection_equilibrium():
    space = fd.ModelingSpace("3D")
    mesh = fd.mesh.rectangle_mesh(3, 2, 0, 2, 0, 1, "quad4", ndim=3)
    law = fd.constitutivelaw.ShellHomogeneous(
        fd.constitutivelaw.ElasticIsotrop(1200, 0.25), 0.15
    )
    wf = fd.weakform.PlateEquilibrium(
        law, nlgeom=True, consistent_tangent=True, space=space
    )
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.set_solver("direct_scipy")
    pb.bc.add("Dirichlet", mesh.node_sets["left"], ["Disp", "Rot"], 0)
    pb.bc.add("Dirichlet", mesh.node_sets["right"], "DispZ", 0.3)
    pb.set_nr_criterion("Displacement", tol=1e-8, max_subiter=20)
    pb.nlsolve(dt=0.5, update_dt=False, print_info=0)
    np.testing.assert_allclose(pb.get_disp()[2, mesh.node_sets["right"]], 0.3)
    np.testing.assert_allclose(
        assembly.current.global_vector[pb._dof_free], 0, atol=1e-6
    )
