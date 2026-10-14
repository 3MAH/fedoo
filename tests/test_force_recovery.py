"""Forces must belong to the evaluated state, not a Newton/history vector."""

from types import SimpleNamespace

import numpy as np
import pytest

import fedoo as fd
from fedoo.core.problem import Problem


def _model(damping=False):
    space = fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=3, ny=2, elm_type="quad4")
    law = fd.constitutivelaw.ElasticIsotrop(100.0, 0.0)
    law.set_density(2.0)
    wf = fd.weakform.StressEquilibrium(law, space=space)
    if damping:
        wf.set_damping(alpha=0.2, beta=0.01)
    return mesh, fd.Assembly.create(wf, mesh)


def _supports(pb, mesh, mpc=False):
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    right = mesh.find_nodes("X", mesh.bounding_box.xmax)
    pb.bc.add("Dirichlet", left, "Disp", 0.0)
    if mpc:
        pb.bc.add(fd.MPC([[right[1]], [right[0]]], ["DispX", "DispX"], [1, -1]))
        right = right[:1]
    pb.bc.add("Dirichlet", right, "DispX", 0.01)


@pytest.mark.parametrize("mpc", [False, True])
def test_nonlinear_forces_match_static_linear_and_survive_rollback(mpc):
    mesh, assembly = _model()
    reference = fd.problem.Linear(assembly)
    _supports(reference, mesh, mpc)
    reference.solve()
    expected_raw = reference.get_ext_forces(include_mpc=False).copy()
    expected_mapped = reference.get_ext_forces().copy()

    mesh, assembly = _model()
    pb = fd.problem.NonLinear(assembly)
    _supports(pb, mesh, mpc)
    pb.set_nr_criterion("Force", tol=1e-10)
    callback_forces = []
    pb.nlsolve(
        dt=0.5,
        tmax=1.0,
        print_info=0,
        callback=lambda p: callback_forces.append(p.get_ext_forces().copy()),
    )
    np.testing.assert_allclose(
        pb.get_ext_forces(include_mpc=False), expected_raw, atol=1e-10
    )
    np.testing.assert_allclose(pb.get_ext_forces(), expected_mapped, atol=1e-10)
    np.testing.assert_allclose(callback_forces[-1], expected_mapped, atol=1e-10)

    # A remaining correction must never pollute evaluated nonlinear forces.
    pb.set_X(np.ones(pb.n_dof))
    np.testing.assert_allclose(pb.get_ext_forces(), expected_mapped, atol=1e-10)
    pb._dU = np.zeros(pb.n_dof)
    pb._dU[: mesh.n_nodes] = 0.02 * mesh.nodes[:, 0]
    pb.update(compute="vector")
    assert not np.allclose(pb.get_ext_forces(include_mpc=False), expected_raw)
    pb.to_start()
    np.testing.assert_allclose(
        pb.get_ext_forces(include_mpc=False), expected_raw, atol=1e-10
    )


def test_force_reference_updates_and_zero_reference_checks_residual():
    pb = SimpleNamespace(
        nr_parameters={"criterion": "Force", "norm_type": 2, "err0": None},
        _dof_free=np.array([0]),
        _err0=None,
        get_ext_forces=lambda **kw: np.array([0.0]),
        _get_free_dof_residual=lambda: np.array([1.0]),
    )
    assert fd.problem.NonLinear.compute_nr_error(pb) == 1.0
    assert pb._err0 is None  # fallback must not become the force reference
    pb.get_ext_forces = lambda **kw: np.array([0.5])
    assert fd.problem.NonLinear.compute_nr_error(pb) == 2.0
    pb.get_ext_forces = lambda **kw: np.array([20.0])
    assert fd.problem.NonLinear.compute_nr_error(pb) == 0.05
    pb.get_ext_forces = lambda **kw: np.array([10.0])
    assert fd.problem.NonLinear.compute_nr_error(pb) == 0.1
    pb.nr_parameters["err0"] = 5.0
    pb._err0 = 5.0  # explicitly supplied reference remains fixed
    pb.get_ext_forces = lambda **kw: np.array([100.0])
    pb._get_free_dof_residual = lambda: np.array([1.0])
    assert fd.problem.NonLinear.compute_nr_error(pb) == 0.2
    pb.nr_parameters["err0"] = None
    pb._err0 = None
    pb.get_ext_forces = lambda **kw: np.array([0.0])
    pb._get_free_dof_residual = lambda: np.array([0.0])
    assert fd.problem.NonLinear.compute_nr_error(pb) == 0.0


@pytest.mark.parametrize(
    "integrator",
    [fd.time.Newmark(), fd.time.GeneralizedAlpha(alpha_m=0.1, alpha_f=0.2)],
)
def test_implicit_linear_completed_forces_include_inertia_and_damping(integrator):
    mesh, assembly = _model(damping=True)
    pb = fd.problem.Linear(assembly, time_step=0.1, integrator=integrator)
    _supports(pb, mesh)
    for factor in (0.5, 1.0):
        old_u = pb.get_X().copy() if not np.isscalar(pb.get_X()) else np.zeros(pb.n_dof)
        old_v = pb.get_velocity().copy()
        old_a = pb.get_acceleration().copy()
        pb.solve_time_increment(t_fact=factor)
        u = (1 - integrator.alpha_f) * pb.get_X() + integrator.alpha_f * old_u
        v = (1 - integrator.alpha_f) * pb.get_velocity() + integrator.alpha_f * old_v
        a = (
            1 - integrator.alpha_m
        ) * pb.get_acceleration() + integrator.alpha_m * old_a
        expected = (
            pb._dynamic_stiffness @ u
            + pb._dynamic_damping @ v
            + pb._dynamic_mass @ a
            - pb._dynamic_constant_force
        )
        np.testing.assert_allclose(pb.get_ext_forces(), expected, atol=1e-10)
        np.testing.assert_allclose(
            pb.get_ext_forces()[pb._dof_free], pb.get_B()[pb._dof_free], atol=1e-10
        )


@pytest.mark.parametrize("mass_lumping", [True, False])
@pytest.mark.parametrize(
    "integrator", [fd.time.CentralDifference(), fd.time.ExplicitGeneralizedAlpha()]
)
def test_explicit_solved_forces_remain_available_after_update(mass_lumping, integrator):
    mesh, assembly = _model(damping=True)
    pb = fd.problem.ExplicitDynamic(
        assembly, 0.001, integrator=integrator, mass_lumping=mass_lumping
    )
    _supports(pb, mesh)
    pb.apply_boundary_conditions(t_fact=0.5)
    pb.prepare_time_increment()
    predicted_end_velocity = pb._state.velocity + pb.time_step * pb._state.acceleration
    pb.solve()
    if mass_lumping:
        expected = pb.get_A() * pb.get_X() - pb.get_D()
    else:
        expected = pb.get_A() @ pb.get_X() - pb.get_D()
    np.testing.assert_allclose(pb.get_ext_forces(), expected, atol=1e-9)
    pb.update()
    np.testing.assert_allclose(
        pb.get_ext_forces()[pb._dof_free], pb.get_B()[pb._dof_free], atol=1e-9
    )
    np.testing.assert_allclose(pb.get_ext_forces(), expected, atol=1e-9)
    if integrator.needs_end_step_acceleration:
        # Prescribed acceleration must also enter the consistent-mass free
        # equations at the end-step closure, independently of force extraction.
        end_balance = (
            pb._mass_action(pb._state.acceleration)
            - pb._linear_internal_force(pb._state.displacement)
            + pb._damping_force(predicted_end_velocity)
        )
        np.testing.assert_allclose(
            end_balance[pb._dof_free], pb.get_B()[pb._dof_free], atol=1e-9
        )
    assert not hasattr(pb, "_dynamic_force")
    assert not hasattr(pb, "_force_start")
    pb.set_start()
    committed = pb.get_X().copy()
    np.testing.assert_allclose(pb.get_ext_forces(), expected, atol=1e-9)
    pb.solve_time_increment(t_fact=1.0)
    pb.to_start()
    # Rollback restores kinematics, not a previous effective-system balance.
    np.testing.assert_allclose(pb.get_X(), committed, atol=1e-9)


def test_base_linear_diagonal_force_vector():
    mesh, _ = _model()
    pb = Problem(A=np.arange(1, 1 + 2 * mesh.n_nodes, dtype=float), mesh=mesh)
    pb.set_X(np.ones(pb.n_dof))
    np.testing.assert_array_equal(pb.get_ext_forces(), pb.get_A())


def test_nonlinear_implicit_forces_match_linear_newmark():
    mesh, assembly = _model(damping=True)
    reference = fd.problem.Linear(assembly, time_step=0.1, integrator=fd.time.Newmark())
    _supports(reference, mesh)
    expected = []
    for factor in (0.5, 1.0):
        reference.solve_time_increment(t_fact=factor)
        expected.append(reference.get_ext_forces().copy())

    mesh, assembly = _model(damping=True)
    pb = fd.problem.NonLinear(assembly)
    pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())
    _supports(pb, mesh)
    pb.set_nr_criterion("Force", tol=1e-10)
    actual = []
    pb.nlsolve(
        dt=0.1,
        tmax=0.2,
        update_dt=False,
        print_info=0,
        callback=lambda p: actual.append(p.get_ext_forces().copy()),
    )
    np.testing.assert_allclose(actual, expected, atol=1e-9)
