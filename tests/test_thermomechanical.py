"""Coupled block derivatives, energy balance, history and solver regressions."""

import numpy as np
import pytest

import fedoo as fd
from fedoo.core.assembly import Assembly


def _native_elastic(kind, thermal=None):
    if kind == "isotropic":
        return fd.constitutivelaw.ElasticIsotrop(100, 0.2, thermal=thermal)
    if kind == "orthotropic":
        return fd.constitutivelaw.ElasticOrthotropic(
            150, 100, 90, 30, 35, 40, 0.2, 0.15, 0.25, thermal=thermal
        )
    if kind == "anisotropic":
        rng = np.random.default_rng(27)
        A = rng.normal(size=(6, 6))
        H = 20 * (A @ A.T + np.eye(6))
        return fd.constitutivelaw.ElasticAnisotropic(H, thermal=thermal)
    return fd.constitutivelaw.CompositeUD(
        Vf=0.4, E_f=200, E_m=50, angle=25, thermal=thermal
    )


def _model(material=None, heat_first=False, dynamic=False):
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=3, ny=2, elm_type="quad4")
    if material is None:
        material = fd.constitutivelaw.ElasticIsotrop(
            100,
            0.2,
            thermal=fd.constitutivelaw.ThermalProperties(
                1, 2, 1, alpha=0.001, reference_temperature=300
            ),
        )
    mechanical = fd.weakform.StressEquilibrium(material)
    thermal = fd.weakform.HeatEquation(material)
    wf = thermal + mechanical if heat_first else mechanical + thermal
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.set_solver("direct_scipy")
    pb.set_time_integrator(fd.time.FIRST_ORDER, fd.time.BackwardEuler())
    if dynamic:
        pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())
    pb.set_dof("Temp", 300.0)
    return pb, assembly, material


@pytest.mark.parametrize("heat_first", [False, True])
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "kind", ["isotropic", "orthotropic", "anisotropic", "composite"]
)
def test_all_coupled_blocks_match_residual_derivatives(heat_first, dynamic, kind):
    material = _native_elastic(
        kind,
        fd.constitutivelaw.ThermalProperties(
            1,
            2,
            1,
            alpha=[0.001, 0.002, 0.003, 0.0003, 0, 0],
            reference_temperature=300,
        ),
    )
    pb, assembly, _ = _model(material, heat_first=heat_first, dynamic=dynamic)
    assert isinstance(assembly, Assembly)  # one material history, not an AssemblySum
    pb.initialize()
    pb.dtime = 0.1
    pb.set_start()
    rng = np.random.default_rng(12)
    trial = rng.normal(size=pb.n_dof) * 0.001
    pb._dU = trial.copy()
    pb.update()
    tangent = assembly.get_global_matrix().copy()
    assert not assembly.assume_sym
    rank = assembly.space.variable_rank("Temp")
    thermal = np.arange(rank * pb.mesh.n_nodes, (rank + 1) * pb.mesh.n_nodes)
    mechanical = np.setdiff1d(np.arange(pb.n_dof), thermal)
    assert np.linalg.norm(tangent[mechanical][:, thermal].toarray()) > 0
    assert np.linalg.norm(tangent[thermal][:, mechanical].toarray()) > 0
    for indices in (mechanical, thermal):
        direction = np.zeros(pb.n_dof)
        direction[indices] = rng.normal(size=len(indices))
        h = 1e-5
        pb._dU = trial + h * direction
        pb.update(compute="vector")
        plus = -pb.get_D().copy()
        pb._dU = trial - h * direction
        pb.update(compute="vector")
        minus = -pb.get_D().copy()
        np.testing.assert_allclose(
            (plus - minus) / (2 * h), tangent @ direction, rtol=1e-6, atol=1e-6
        )


def _field_criterion(pb):
    pb.set_nr_criterion(
        "Fields",
        fields={
            "Disp": {"atol": 1e-9, "rtol": 1e-9},
            "Temp": {"atol": 1e-9, "rtol": 1e-9},
        },
    )


def test_adiabatic_thermoelastic_heating_and_rollback():
    pb, assembly, material = _model()
    # Uniform plane-strain extension with adiabatic thermal boundaries.
    nodes = np.arange(pb.mesh.n_nodes)
    pb.bc.add("Dirichlet", nodes, "DispX", 0.01 * pb.mesh.nodes[:, 0])
    pb.bc.add("Dirichlet", nodes, "DispY", 0.0)
    _field_criterion(pb)
    pb.nlsolve(dt=0.05, tmax=0.1, update_dt=False, print_info=0)
    C = material.get_tangent_matrix(assembly)
    beta = C @ np.array([material.alpha] * 3 + [0.0] * 3)
    expected = 300 - 300 * beta[0] * 0.01 / (material.density * material.specific_heat)
    np.testing.assert_allclose(pb.get_temp(), expected, atol=1e-10)
    saved = pb.get_temp().copy()
    pb._dU = np.ones(pb.n_dof) * 0.1
    pb.update()
    pb.dtime = 0.025  # retry with a new time step
    pb.to_start()
    np.testing.assert_array_equal(pb.get_temp(), saved)
    np.testing.assert_allclose(assembly.sv["HeatSource"], 0, atol=1e-12)
    assert max(pb.nr_errors.values()) < 1


@pytest.mark.parametrize("simcoon", [False, True])
def test_uniform_heat_input_does_not_double_count_capacity(simcoon):
    capacity = 2.0
    if simcoon:
        material = fd.constitutivelaw.SimcoonThermoMechanical(
            "ELISO",
            [1, capacity, 100, 0.2, 0],
            1,
            reference_temperature=300,
        )
    else:
        material = fd.constitutivelaw.ElasticIsotrop(
            100,
            0.2,
            thermal=fd.constitutivelaw.ThermalProperties(
                1, capacity, 1, alpha=0, reference_temperature=300
            ),
        )
    pb, assembly, _ = _model(material)
    pb.bc.add("Dirichlet", np.arange(pb.mesh.n_nodes), "Disp", 0)
    # Unit square, constant integrated heat input: Q*dt = C*delta_T.
    # Small delta_T also tests Simcoon's continuum thermal tangent accurately.
    pb.bc.add(
        "Neumann",
        np.arange(pb.mesh.n_nodes),
        "Temp",
        0.001 / pb.mesh.n_nodes,
        time_func=lambda t: 1,
    )
    _field_criterion(pb)
    pb.nlsolve(dt=0.1, tmax=0.1, update_dt=False, print_info=0)
    # Nodal equal loading is not uniform on this mesh. Compare integrated energy.
    temperature_gp = assembly.convert_data(pb.get_temp(), "Node", "GaussPoint")
    mean_temperature = float(np.mean(temperature_gp))  # equal-size elements/weights
    np.testing.assert_allclose(
        mean_temperature, 300 + 0.001 * 0.1 / capacity, atol=1e-9
    )


def test_prescribed_temperature_mechanics_and_separate_thermal_problem():
    material = fd.constitutivelaw.ElasticIsotrop(
        100,
        0.2,
        thermal=fd.constitutivelaw.ThermalProperties(
            1, 2, 1, alpha=0.001, reference_temperature=300
        ),
    )
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=3, ny=2, elm_type="quad4")
    assembly = fd.Assembly.create(fd.weakform.StressEquilibrium(material), mesh)
    assembly.sv["Temp"] = 310.0  # externally supplied thermal history
    pb = fd.problem.NonLinear(assembly)
    pb.set_solver("direct_scipy")
    pb.bc.add("Dirichlet", np.arange(mesh.n_nodes), "Disp", 0)
    pb.nlsolve(dt=0.1, tmax=0.1, print_info=0)
    beta = material.get_tangent_matrix(assembly) @ np.array([0.001] * 3 + [0] * 3)
    np.testing.assert_allclose(
        assembly.sv["Stress"].asarray(),
        -10 * beta[:, None] * np.ones((1, assembly.n_gauss_points)),
    )

    fd.ModelingSpace("2Dplane")
    thermal = fd.Assembly.create(fd.weakform.HeatEquation(material), mesh)
    heat = fd.problem.NonLinear(thermal)
    heat.set_solver("direct_scipy")
    heat.set_dof("Temp", 300)
    heat.set_time_integrator(fd.time.FIRST_ORDER, fd.time.BackwardEuler())
    heat.bc.add("Dirichlet", np.arange(mesh.n_nodes), "Temp", 310, start_value=300)
    heat.nlsolve(dt=0.1, tmax=0.1, print_info=0)
    np.testing.assert_allclose(heat.get_temp(), 310)


def test_incompatible_quadrature_is_rejected():
    pb, assembly, material = _model()
    mechanical = fd.weakform.StressEquilibrium(material)
    thermal = fd.weakform.HeatEquation(material)
    thermal.assembly_options["n_elm_gp"] = 1
    with pytest.raises(ValueError, match="matching quadrature"):
        fd.Assembly.create(mechanical + thermal, pb.mesh)


def test_field_criterion_does_not_hide_a_thermal_residual_and_handles_mpc():
    pb, _, _ = _model()
    pb.bc.add(fd.MPC([[1], [0]], ["Temp", "Temp"], [1, -1]))
    pb.apply_boundary_conditions()
    pb.set_D(np.zeros(pb.n_dof))
    _field_criterion(pb)
    assert pb.compute_nr_error() == 0
    residual = np.zeros(pb.n_dof)
    pb._set_vect_component(residual, "Temp", 1e-6)
    pb.set_D(residual)
    assert pb.compute_nr_error() > 1
    pb.set_nr_criterion("Fields", fields={"Disp": {"atol": 1e-9}})
    with pytest.raises(ValueError, match="cover every free DOF"):
        pb.compute_nr_error()


def test_shared_material_is_evaluated_once_per_trial(monkeypatch):
    pb, _, material = _model()
    pb.initialize()
    count = []
    original = material.update

    def evaluate(*args, **kwargs):
        count.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(material, "update", evaluate)
    pb.dtime = 0.1
    pb.update()
    assert len(count) == 1


@pytest.mark.parametrize(
    "kernel, n_statev, extra", [("EPICP", 8, []), ("EPKCP", 14, [10])]
)
def test_simcoon_plastic_trials_do_not_modify_committed_history(
    kernel, n_statev, extra
):
    material = fd.constitutivelaw.SimcoonThermoMechanical(
        kernel,
        [1, 10, 1000, 0.3, 0.001, 1, 10, 1] + extra,
        1,
        n_statev=n_statev,
        reference_temperature=300,
    )
    pb, assembly, _ = _model(material)
    pb.initialize()
    pb.dtime = 0.1
    pb.set_start()
    committed = assembly.sv_start["Statev"].copy()
    pb._dU = np.zeros(pb.n_dof)
    pb._dU[: pb.mesh.n_nodes] = 0.02 * pb.mesh.nodes[:, 0]
    pb.update()
    plastic = assembly.sv["Statev"].copy()
    assert np.max(plastic[1]) > 0
    np.testing.assert_array_equal(assembly.sv_start["Statev"], committed)
    pb.update()
    np.testing.assert_allclose(assembly.sv["Statev"], plastic)
    pb._dU *= 0.01
    pb.update()
    np.testing.assert_allclose(assembly.sv["Statev"][1], 0, atol=1e-12)
    pb.dtime = 0.05
    pb.to_start()
    np.testing.assert_array_equal(assembly.sv["Statev"], committed)


def test_simcoon_temperature_initialization_at_time_zero_is_preserved():
    material = fd.constitutivelaw.SimcoonThermoMechanical(
        "ELISO",
        [1, 2, 100, 0.2, 0.001],
        1,
        reference_temperature=300,
    )
    pb, assembly, _ = _model(material)
    nodes = np.arange(pb.mesh.n_nodes)
    pb.bc.add("Dirichlet", nodes, "Disp", 0)
    pb.bc.add("Dirichlet", nodes, "Temp", 305, start_value=300)
    pb.nlsolve(dt=0.1, tmax=0.1, update_dt=False, print_info=0)
    np.testing.assert_allclose(assembly.sv["Statev"][0], 300)
    np.testing.assert_allclose(assembly.sv["Stress"].asarray()[:3], -5 / 6, atol=1e-12)


def test_weak_coupling_matches_monolithic_without_thermal_feedback():
    material = fd.constitutivelaw.ElasticIsotrop(
        100,
        0.2,
        thermal=fd.constitutivelaw.ThermalProperties(
            1, 2, 1, alpha=0.001, reference_temperature=300, coupling=False
        ),
    )
    mono, _, _ = _model(material)
    mesh = mono.mesh
    left, right = mesh.find_nodes("X", 0), mesh.find_nodes("X", 1)
    mono.bc.add("Dirichlet", left, "Disp", 0)
    for nodes, value in ((left, 300), (right, 310)):
        mono.bc.add("Dirichlet", nodes, "Temp", value, start_value=300)
    _field_criterion(mono)
    mono.nlsolve(dt=0.05, tmax=0.1, update_dt=False, print_info=0)

    fd.ModelingSpace("2Dplane")
    heat_assembly = fd.Assembly.create(fd.weakform.HeatEquation(material), mesh)
    heat = fd.problem.NonLinear(heat_assembly)
    heat.set_solver("direct_scipy")
    heat.set_dof("Temp", 300)
    heat.set_time_integrator(fd.time.FIRST_ORDER, fd.time.BackwardEuler())
    for nodes, value in ((left, 300), (right, 310)):
        heat.bc.add("Dirichlet", nodes, "Temp", value, start_value=300)
    heat.set_nr_criterion("Fields", fields={"Temp": {"atol": 1e-9}})
    heat.nlsolve(dt=0.05, tmax=0.1, update_dt=False, print_info=0)
    np.testing.assert_allclose(mono.get_temp(), heat.get_temp(), atol=1e-10)

    fd.ModelingSpace("2Dplane")
    mechanical_assembly = fd.Assembly.create(
        fd.weakform.StressEquilibrium(material), mesh
    )
    mechanical_assembly.sv["Temp"] = heat_assembly.convert_data(
        heat.get_temp(), "Node", "GaussPoint"
    )
    mechanical = fd.problem.NonLinear(mechanical_assembly)
    mechanical.set_solver("direct_scipy")
    mechanical.bc.add("Dirichlet", left, "Disp", 0)
    mechanical.set_nr_criterion("Fields", fields={"Disp": {"atol": 1e-9}})
    mechanical.nlsolve(dt=0.1, tmax=0.1, update_dt=False, print_info=0)
    np.testing.assert_allclose(mono.get_disp(), mechanical.get_disp(), atol=1e-10)


def test_guarded_modes_and_invalid_field_tolerances():
    pb, _, material = _model()
    for fields in ({"Temp": {"atol": -1}}, {"Temp": {"rtol": None}}, {"Temp": {}}):
        with pytest.raises(ValueError):
            pb.set_nr_criterion("Fields", fields=fields)
    pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.GeneralizedAlpha(alpha_f=0.1))
    with pytest.raises(NotImplementedError, match="endpoint Newmark"):
        pb.initialize()
    with pytest.raises(ValueError, match="same material instance"):
        other = fd.constitutivelaw.ThermalProperties(1, 2, 1)
        fd.Assembly.create(
            fd.weakform.StressEquilibrium(material) + fd.weakform.HeatEquation(other),
            pb.mesh,
        )


def test_fields_with_blocked_temperature_work_with_infinity_norm():
    pb, _, _ = _model()
    pb.bc.add("Dirichlet", np.arange(pb.mesh.n_nodes), "Temp", 300)
    pb.apply_boundary_conditions()
    pb.set_D(np.zeros(pb.n_dof))
    _field_criterion(pb)
    pb.nr_parameters["norm_type"] = np.inf
    assert pb.compute_nr_error() == 0


@pytest.mark.parametrize("simcoon", [False, True])
def test_three_dimensional_stress_free_thermal_expansion(simcoon):
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, elm_type="hex8")
    if simcoon:
        material = fd.constitutivelaw.SimcoonThermoMechanical(
            "ELISO", [1, 2, 100, 0.2, 0.001], 1
        )
    else:
        material = fd.constitutivelaw.ElasticIsotrop(
            100,
            0.2,
            thermal=fd.constitutivelaw.ThermalProperties(
                1, 2, 1, alpha=0.001, reference_temperature=300
            ),
        )
    assembly = fd.Assembly.create(
        fd.weakform.StressEquilibrium(material) + fd.weakform.HeatEquation(material),
        mesh,
    )
    pb = fd.problem.NonLinear(assembly)
    pb.set_solver("direct_scipy")
    pb.set_dof("Temp", 300)
    pb.set_time_integrator(fd.time.FIRST_ORDER, fd.time.BackwardEuler())
    _field_criterion(pb)
    for axis in "XYZ":
        pb.bc.add("Dirichlet", mesh.find_nodes(axis, 0), "Disp" + axis, 0)
    pb.bc.add("Dirichlet", np.arange(mesh.n_nodes), "Temp", 310, start_value=300)
    pb.nlsolve(dt=0.1, tmax=0.1, print_info=0)
    np.testing.assert_allclose(pb.get_disp(), 0.01 * mesh.nodes.T, atol=1e-10)
    np.testing.assert_allclose(assembly.sv["Stress"].asarray(), 0, atol=1e-10)


def test_fields_reject_mpc_that_mixes_thermal_and_mechanical_units():
    pb, _, _ = _model()
    pb.bc.add(fd.MPC([[1], [0]], ["Temp", "DispX"], [1, -1]))
    pb.apply_boundary_conditions()
    pb.set_D(np.zeros(pb.n_dof))
    _field_criterion(pb)
    with pytest.raises(ValueError, match="mixing different fields"):
        pb.compute_nr_error()


def test_volumetric_heat_input_with_zero_external_force():
    material = fd.constitutivelaw.ElasticIsotrop(
        100,
        0.2,
        thermal=fd.constitutivelaw.ThermalProperties(
            1, 2, 1, alpha=0.001, reference_temperature=300, heat_source=2
        ),
    )
    pb, _, _ = _model(material)
    pb.bc.add("Dirichlet", np.arange(pb.mesh.n_nodes), "Disp", 0)
    _field_criterion(pb)
    pb.nlsolve(dt=0.1, tmax=0.1, update_dt=False, print_info=0)
    np.testing.assert_allclose(pb.get_temp(), 300.1, atol=1e-10)


@pytest.mark.parametrize(
    "kind", ["isotropic", "orthotropic", "anisotropic", "composite"]
)
@pytest.mark.parametrize("attachment", ["constructor", "method", "keywords"])
def test_native_material_free_thermal_expansion(kind, attachment):
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2, elm_type="hex8")
    properties = dict(
        thermal_conductivity=1,
        specific_heat=2,
        density=1,
        alpha=[0.001, 0.002, 0.003],
        reference_temperature=300,
    )
    thermal = fd.constitutivelaw.ThermalProperties(**properties)
    if attachment == "constructor":
        material = _native_elastic(kind, thermal)
    elif attachment == "method":
        material = _native_elastic(kind).set_thermal_properties(thermal)
    else:
        material = _native_elastic(kind).set_thermal_properties(**properties)
    assembly = fd.Assembly.create(
        fd.weakform.StressEquilibrium(material) + fd.weakform.HeatEquation(material),
        mesh,
    )
    pb = fd.problem.NonLinear(assembly)
    pb.set_solver("direct_scipy")
    pb.set_dof("Temp", 300)
    pb.set_time_integrator(fd.time.FIRST_ORDER, fd.time.BackwardEuler())
    _field_criterion(pb)
    for axis in "XYZ":
        pb.bc.add("Dirichlet", mesh.find_nodes(axis, 0), "Disp" + axis, 0)
    pb.bc.add("Dirichlet", np.arange(mesh.n_nodes), "Temp", 310, start_value=300)
    pb.nlsolve(dt=0.1, tmax=0.1, print_info=0)
    expected = np.asarray(properties["alpha"])[:, None] * 10 * mesh.nodes.T
    np.testing.assert_allclose(pb.get_disp(), expected, atol=1e-10)
    np.testing.assert_allclose(assembly.sv["Stress"].asarray(), 0, atol=1e-10)
    np.testing.assert_allclose(
        material.get_stress_from_strain(assembly, assembly.sv["Strain"]).asarray(),
        assembly.sv["Stress"].asarray(),
        atol=1e-10,
    )


def test_thermal_elasticity_preserves_spatially_varying_moduli():
    E = np.linspace(100, 150, 6)  # six nodes of the plane-strain test mesh
    material = fd.constitutivelaw.ElasticIsotrop(
        E,
        0.2,
        thermal=fd.constitutivelaw.ThermalProperties(
            1, 2, 1, alpha=0.001, reference_temperature=300
        ),
    )
    pb, assembly, _ = _model(material)
    nodes = np.arange(pb.mesh.n_nodes)
    pb.bc.add("Dirichlet", nodes, "Disp", 0)
    pb.bc.add("Dirichlet", nodes, "Temp", 310, start_value=300)
    pb.nlsolve(dt=0.1, tmax=0.1, print_info=0)
    C_unit = fd.constitutivelaw.ElasticIsotrop(1, 0.2).get_elastic_matrix()
    beta_unit = C_unit @ np.array([0.001] * 3 + [0] * 3)
    E_gp = assembly.convert_data(E, "Node", "GaussPoint")
    np.testing.assert_allclose(
        assembly.sv["Stress"].asarray(),
        -10 * beta_unit[:, None] * E_gp,
        atol=1e-10,
    )
