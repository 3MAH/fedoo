"""Check the optional shell tangent against the evaluated force residual."""

from copy import deepcopy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import fedoo as fd
from fedoo.lib_elements.element_list import get_element


@pytest.mark.parametrize("damping", [False, True])
@pytest.mark.parametrize("element", ["ptri3mitc", "pquad4mitc"])
def test_automatic_tangent_switch_matches_always_consistent(damping, element):
    """Switching at a deformed iterate preserves the residual and full Jacobian."""
    _, expected, _ = _shell(element, consistent=True, damping=damping)
    wf, assembly, pb = _shell(element, consistent=None, damping=damping)
    residual = assembly.current.get_global_vector().copy()
    assert not wf._use_consistent_tangent(assembly)
    assert pb._set_automatic_consistent_tangent(True) == 1
    assembly.current.assemble_global_mat("all")
    assert wf._use_consistent_tangent(assembly)
    np.testing.assert_allclose(
        assembly.current.get_global_vector(), residual, atol=1e-10
    )
    np.testing.assert_allclose(
        assembly.current.get_global_matrix().toarray(),
        expected.current.get_global_matrix().toarray(),
        rtol=1e-10,
        atol=1e-10,
    )
    _assert_force_derivative(assembly, pb)


@pytest.mark.parametrize("adaptive", [False, True])
def test_automatic_tangent_stagnation_and_attempt_reset(monkeypatch, adaptive):
    wf, assembly, pb = _shell("ptri3mitc", consistent=None, damping=True)
    # Exercise AssemblySum as well as the shell+damping WeakFormSum.
    previous_dU = pb._dU.copy()
    pb = fd.problem.NonLinear(fd.Assembly.sum(assembly))
    pb._dU = previous_dU
    pb.nr_parameters["adaptive_stiffness"] = adaptive
    errors = iter([1.0] * 8 + ([0.0] if adaptive else []))
    matrices = []
    monkeypatch.setattr(pb, "elastic_prediction", lambda: None)
    monkeypatch.setattr(pb, "_update_d", lambda: None)
    monkeypatch.setattr(pb, "compute_nr_error", lambda: next(errors))
    monkeypatch.setattr(pb, "_xbc_is_applied", lambda: True)
    monkeypatch.setattr(pb, "solve_nr_increment", lambda: None)
    monkeypatch.setattr(
        pb, "_elastic_reference_matrix", lambda **kwargs: assembly.current.global_matrix
    )

    def update(compute, **kwargs):
        if compute == "matrix":
            matrices.append(wf._use_consistent_tangent(assembly))

    monkeypatch.setattr(pb, "update", update)
    converged, subiter, _ = pb.solve_time_increment(max_subiter=20)
    assert bool(converged) == adaptive
    # Three nonproductive checks, then four fresh checks before failure or
    # adaptive rollback (which subtracts two from its iteration counter).
    assert subiter == (6 if adaptive else 8)
    assert matrices == [False, False, False, True, True, True, True]
    assert wf._use_consistent_tangent(assembly)
    errors = iter([0.0])
    converged, _, _ = pb.solve_time_increment(max_subiter=20)
    assert converged
    assert not wf._use_consistent_tangent(assembly)


@pytest.mark.parametrize(
    "trace,switch_iteration,converged",
    [
        # User's oscillating residual: temporary decreases do not hide stagnation.
        ([0.09261, 0.02166, 0.00904, 0.01427, 0.00939, 0.03107, 0.0], 6, True),
        # Four oscillating errors after switching exhaust the fresh grace period.
        (
            [
                0.09261,
                0.02166,
                0.00904,
                0.01427,
                0.00939,
                0.03107,
                0.05,
                0.04,
                0.06,
                0.05,
            ],
            6,
            False,
        ),
        # Sub-1% individual decreases accumulate into significant progress.
        ([0.995**i for i in range(10)] + [0.0], None, True),
        # Earliest activation and a fresh reference at the larger switching error.
        ([1.0, 2.0, 1.5, 2.0, 1.8, 1.6, 1.4, 0.0], 4, True),
    ],
)
def test_automatic_tangent_progress_reference(
    monkeypatch, trace, switch_iteration, converged
):
    wf, assembly, pb = _shell("ptri3mitc", consistent=None)
    errors = iter(trace)
    matrices = []
    monkeypatch.setattr(pb, "elastic_prediction", lambda: None)
    monkeypatch.setattr(pb, "_update_d", lambda: None)
    monkeypatch.setattr(pb, "compute_nr_error", lambda: next(errors))
    monkeypatch.setattr(pb, "_xbc_is_applied", lambda: True)
    monkeypatch.setattr(pb, "solve_nr_increment", lambda: None)

    def update(compute, **kwargs):
        if compute == "matrix":
            matrices.append(wf._use_consistent_tangent(assembly))

    monkeypatch.setattr(pb, "update", update)
    result, subiter, _ = pb.solve_time_increment(max_subiter=30)
    assert bool(result) == converged
    assert subiter == len(trace)
    assert matrices == [
        switch_iteration is not None and i >= switch_iteration
        for i in range(1, len(trace))
    ]


@pytest.mark.parametrize("consistent", [False, None])
@pytest.mark.parametrize("limit", [2, 4, 8])
def test_configurable_early_divergence(monkeypatch, consistent, limit):
    wf, assembly, pb = _shell("ptri3mitc", consistent=consistent)
    pb.set_nr_criterion("Force", early_divergence_niter=limit)
    monkeypatch.setattr(pb, "elastic_prediction", lambda: None)
    monkeypatch.setattr(pb, "_update_d", lambda: None)
    monkeypatch.setattr(pb, "compute_nr_error", lambda: 1.0)
    monkeypatch.setattr(pb, "update", lambda **kwargs: None)
    monkeypatch.setattr(pb, "solve_nr_increment", lambda: None)
    converged, subiter, _ = pb.solve_time_increment(max_subiter=30)
    assert not converged
    # Automatic mode still switches at iteration four, even for limit < 3.
    assert subiter == (4 if consistent is None else 1) + limit
    assert bool(wf._use_consistent_tangent(assembly)) == (consistent is None)


@pytest.mark.parametrize("limit", [0, -1, 2.5, True, None])
def test_invalid_early_divergence_limit(limit):
    _, _, pb = _shell("ptri3mitc", consistent=False)
    with pytest.raises(ValueError, match="positive integer"):
        pb.set_nr_criterion("Force", early_divergence_niter=limit)


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


@pytest.mark.parametrize("consistent", [False, True])
@pytest.mark.parametrize("alpha", [0.0, 0.25, 1.0])
def test_line_search_scales_shell_trial_rotations(consistent, alpha):
    from fedoo.problem.line_search import _evaluate_residual_norm

    _, assembly, pb = _shell("ptri3mitc", consistent=consistent)
    pb.init_bc_start_value()
    pb.apply_boundary_conditions(0.0, 0.0)
    pb._update_d()
    base_residual = pb._get_free_dof_residual().copy()
    base_du = pb._dU.copy()
    direction = np.random.default_rng(17).normal(size=len(base_du)) * 0.01
    pb.set_X(direction)
    _, trial = _evaluate_residual_norm(pb, direction, alpha)
    np.testing.assert_array_equal(pb.get_X(), direction)
    np.testing.assert_array_equal(pb._dU, base_du)
    if alpha == 0:
        np.testing.assert_allclose(trial, base_residual, rtol=1e-10, atol=1e-10)

    # Independent evaluation of the requested trial, including its rotations.
    _, expected_assembly, expected = _shell("ptri3mitc", consistent=consistent)
    expected.init_bc_start_value()
    expected.apply_boundary_conditions(0.0, 0.0)
    expected._dU = base_du + alpha * direction
    expected.set_X(alpha * direction)
    expected._line_search_update = True
    expected_assembly.update(expected, compute="vector")
    expected._update_d()
    np.testing.assert_allclose(
        trial, expected._get_free_dof_residual(), rtol=1e-10, atol=1e-10
    )


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


@pytest.mark.parametrize("drilling", [True, False])
@pytest.mark.parametrize("rotvec", [[0, 0, 0.3], [0.2, -0.1, 0.25]])
def test_rigid_rotation_has_zero_internal_force(drilling, rotvec):
    space = fd.ModelingSpace("3D")
    nodes = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    mesh = fd.Mesh(nodes, np.array([[0, 1, 2, 3]]), "quad4")
    material = fd.constitutivelaw.ElasticIsotrop(1200, 0.25)
    law = fd.constitutivelaw.ShellHomogeneous(material, 0.15)
    wf = fd.weakform.PlateEquilibrium(
        law, nlgeom=True, true_drilling_rotation=drilling, space=space
    )
    assembly = fd.Assembly.create(wf, mesh, elm_type="pquad4mitc")
    pb = fd.problem.NonLinear(assembly)
    assembly.initialize(pb)
    pb._dU = np.zeros(space.nvar * mesh.n_nodes)
    disp = space.get_rank_vector("Disp")
    rot = space.get_rank_vector("Rot")
    steps = 3
    for step in range(1, steps + 1):
        # Fixed rotation axis: nodal rotation vectors are additive.
        increment = step / steps * np.asarray(rotvec, dtype=float)
        rotation = Rotation.from_rotvec(increment).as_matrix()
        total = np.zeros((space.nvar, mesh.n_nodes))
        total[disp] = (nodes @ rotation.T - nodes).T
        total[rot] = increment[:, None]
        delta = total.ravel() - pb._dU
        pb._dU = total.ravel()
        pb.set_X(delta)
        assembly.update(pb)
        np.testing.assert_allclose(assembly.current.global_vector, 0, atol=1e-10)


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


def test_consistent_tangent_accepts_time_integrator_and_rejects_split_forms():
    # A time integrator derives a subclass whose mixin wraps get_weak_equation:
    # the plate equation is unchanged and the consistent tangent stays valid.
    space = fd.ModelingSpace("3D")
    mesh = fd.mesh.rectangle_mesh(3, 3, 0, 1, 0, 1, "quad4", ndim=3)
    material = fd.constitutivelaw.ElasticIsotrop(1200, 0.25)
    material.set_density(1e-3)
    law = fd.constitutivelaw.ShellHomogeneous(material, 0.05)
    wf = fd.weakform.PlateEquilibrium(
        law, nlgeom="UL", consistent_tangent=True, space=space
    )
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly, nlgeom="UL")
    pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0), ["Disp", "Rot"], 0)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 1), "DispZ", 0.05)
    pb.nlsolve(dt=0.5, tmax=1, update_dt=False, print_info=0)
    assert np.abs(pb.get_disp()).max() == pytest.approx(0.05, rel=0.2)

    assert wf._plate_weak_equation_owner() is fd.weakform.PlateEquilibriumFI
    split = fd.weakform.PlateShearEquilibrium.__new__(fd.weakform.PlateShearEquilibrium)
    assert split._plate_weak_equation_owner() is fd.weakform.PlateShearEquilibrium


@pytest.mark.parametrize("drilling", [False, True])
def test_drilling_penalty_is_fixed_and_independent_of_tangent_option(drilling):
    forces = []
    for consistent in (False, True):
        law = fd.constitutivelaw.ShellHomogeneousNonLinear(
            fd.constitutivelaw.ElasticIsotrop(1200, 0.25),
            0.15,
            n_thickness_points=2,
        )
        wf, assembly, pb = _shell(drilling=drilling, consistent=consistent, section=law)
        current = assembly.current
        initial_scale = wf._drill_reference(current).copy()
        force = current.global_vector.copy()
        assert np.linalg.norm(current.sv["_DrillConstraint"]) > 0
        # Change the section tangent without changing stress or kinematics:
        # the stabilization force must retain its initial scale.
        current.sv["_ShellStiffnessMatrix"] = (
            np.asarray(current.sv["_ShellStiffnessMatrix"]) * 0.1
        )
        current.assemble_global_mat("vector")
        np.testing.assert_allclose(current.global_vector, force, atol=1e-10)
        np.testing.assert_array_equal(wf._drill_reference(current), initial_scale)
        forces.append(force)
    np.testing.assert_allclose(forces[0], forces[1], atol=1e-10)


def test_consistent_tangent_with_law_defined_at_initialization():
    # A Simcoon law only provides its elastic matrix once initialized, which
    # the assembly does after the weak form: the drilling scale is deferred.
    pytest.importorskip("simcoon")
    material = fd.constitutivelaw.Simcoon("EPICP", [1200, 0.25, 0, 1e6, 100, 0.3])
    reference = fd.constitutivelaw.ShellHomogeneousNonLinear(
        fd.constitutivelaw.ElasticIsotrop(1200, 0.25), 0.15, n_thickness_points=2
    )
    law = fd.constitutivelaw.ShellHomogeneousNonLinear(
        material, 0.15, n_thickness_points=2
    )
    wf, assembly, pb = _shell(section=law)
    _assert_force_derivative(assembly, pb)
    wf_ref, assembly_ref, _ = _shell(section=reference)
    np.testing.assert_allclose(
        wf._drill_reference(assembly), wf_ref._drill_reference(assembly_ref)
    )
    # The scale is fixed: a rollback before any accepted increment keeps it.
    scale = wf._drill_reference(assembly).copy()
    assembly.to_start(pb)
    np.testing.assert_allclose(wf._drill_reference(assembly), scale)
