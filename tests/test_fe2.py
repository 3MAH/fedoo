"""Light FE2 test: macro problem driven by micro cells (no full nonlinear solve).

Formerly ``examples/FE2/test_fe2.py``.
"""

import numpy as np
import pytest

import fedoo as fd
from fedoo.util.voigt_tensors import StrainTensorList


@pytest.fixture(autouse=True)
def _clear_assembly_cache():
    fd.Assembly.delete_memory()
    yield
    fd.Assembly.delete_memory()


def test_fe2_uses_problem_level_mean_strain_dofs():
    fd.ModelingSpace("3D")

    micro_mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2)
    micro_material = fd.constitutivelaw.ElasticIsotrop(200_000.0, 0.3)
    micro_weakform = fd.weakform.StressEquilibrium(micro_material)
    micro_assembly = fd.Assembly.create(micro_weakform, micro_mesh, n_elm_gp=1)

    fe2 = fd.constitutivelaw.FE2(micro_assembly)
    macro_mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2)
    macro_weakform = fd.weakform.StressEquilibrium(fe2)
    macro_assembly = fd.Assembly.create(macro_weakform, macro_mesh, n_elm_gp=1)
    problem = fd.problem.NonLinear(macro_assembly)

    problem.initialize()

    assert "_StrainNodes" not in micro_mesh.node_sets
    assert "MeanStrain" in fe2.list_problem[0].global_dof._vector
    expected = micro_material.get_elastic_matrix()
    np.testing.assert_allclose(
        macro_assembly.sv["TangentMatrix"][:, :, 0],
        expected,
        rtol=1e-10,
        atol=1e-8,
    )

    imposed_strain = np.array([1e-4, 0, 0, 0, 0, 0])
    macro_assembly.sv["Strain"] = StrainTensorList(imposed_strain[:, None])
    problem.dtime = 1.0
    fe2._update_pb(0, macro_assembly, problem)

    np.testing.assert_allclose(
        macro_assembly.sv["Stress"].asarray()[:, 0],
        expected @ imposed_strain,
        rtol=1e-8,
        atol=1e-8,
    )


def _micro_assembly(material):
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2)
    weakform = fd.weakform.StressEquilibrium(material, nlgeom=False)
    return fd.Assembly.create(weakform, mesh)


def _solve_bar(law, history):
    """Solve a clamped block under a transverse displacement history."""
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2)
    weakform = fd.weakform.StressEquilibrium(law, nlgeom=False)
    assembly = fd.Assembly.create(weakform, mesh)
    problem = fd.problem.NonLinear(assembly)
    problem.set_nr_criterion("Displacement", max_subiter=50)
    problem.bc.add("Dirichlet", mesh.find_nodes("X", 0), "Disp", 0)
    tip = mesh.find_nodes("X", 1)
    previous, condition = 0.0, None
    for target in history:
        if condition is not None:
            problem.bc.remove(condition)
        condition = problem.bc.add(
            "Dirichlet", tip, "DispY", target, start_value=previous
        )
        problem.nlsolve(dt=1.0, tmax=1, update_dt=False, tol_nr=1e-7, print_info=0)
        previous = target
    return (
        np.array(problem.get_disp()),
        assembly.sv["Stress"].asarray().copy(),
        np.sum(problem.get_ext_forces("DispY")[tip]),
    )


def test_fe2_with_homogeneous_cell_matches_single_scale():
    """A cell made of the macroscopic solid gives the single-scale solution."""
    history = [0.02, -0.005]
    fd.ModelingSpace("3D")
    material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
    disp_ref, stress_ref, force_ref = _solve_bar(material, history)
    fd.Assembly.delete_memory()
    fd.ModelingSpace("3D")
    fe2 = fd.constitutivelaw.FE2(
        _micro_assembly(fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)),
        micro_nr_options={"tol": 1e-8},
    )
    disp, stress, force = _solve_bar(fe2, history)

    np.testing.assert_allclose(disp, disp_ref, atol=1e-9 * abs(disp_ref).max())
    np.testing.assert_allclose(stress, stress_ref, atol=1e-8 * abs(stress_ref).max())
    assert force == pytest.approx(force_ref, rel=1e-8)


def test_fe2_elastoplastic_cell_tangent_and_unloading():
    """One yielded cell: consistent tangent, then elastic unloading."""
    pytest.importorskip("simcoon")
    fd.ModelingSpace("3D")
    props = np.array([1e5, 0.3, 1e-5, 300.0, 1000.0, 1.0])  # yield stress 300
    material = fd.constitutivelaw.Simcoon("EPICP", props)
    material.tangent_mode = 2  # algorithmic tangent
    fe2 = fd.constitutivelaw.FE2(
        _micro_assembly(material), micro_nr_options={"tol": 1e-10}
    )
    macro_mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2)
    macro_weakform = fd.weakform.StressEquilibrium(fe2)
    macro_assembly = fd.Assembly.create(macro_weakform, macro_mesh, n_elm_gp=1)
    problem = fd.problem.NonLinear(macro_assembly)
    problem.initialize()
    problem.set_start()
    problem.dtime = 1.0
    elastic = macro_assembly.sv["ElasticMatrix"][:, :, 0].copy()

    def response(strain):
        macro_assembly.sv["Strain"] = StrainTensorList(np.array(strain)[:, None])
        fe2.update(macro_assembly, problem)
        return (
            macro_assembly.sv["Stress"].asarray()[:, 0].copy(),
            macro_assembly.sv["TangentMatrix"][:, :, 0].copy(),
        )

    strain = np.array([6e-3, -1e-3, 0.0, 2e-3, 0.0, 1e-3])
    stress, tangent = response(strain)
    # the cell has yielded and the tangent is not the elastic one
    assert abs(stress - elastic @ strain).max() > 0.1 * abs(stress).max()
    assert abs(tangent - elastic).max() > 0.1 * abs(elastic).max()
    # the tangent is the derivative of the macroscopic stress
    h = 1e-7
    for j in range(6):
        step = np.zeros(6)
        step[j] = h
        column = (response(strain + step)[0] - response(strain - step)[0]) / (2 * h)
        np.testing.assert_allclose(
            tangent[:, j], column, atol=1e-5 * abs(elastic).max()
        )

    # commit the yielded state, then unload: the increment is elastic
    response(strain)
    problem.set_start()
    unloading = -1e-4 * strain
    stress_unloaded = response(strain + unloading)[0]
    np.testing.assert_allclose(
        stress_unloaded - stress, elastic @ unloading, atol=1e-6 * abs(stress).max()
    )


def test_fe2_update_is_repeatable_and_keeps_the_start_state():
    fd.ModelingSpace("3D")
    micro_assembly = _micro_assembly(fd.constitutivelaw.ElasticIsotrop(1e5, 0.3))
    fe2 = fd.constitutivelaw.FE2(micro_assembly)
    macro_mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2)
    macro_weakform = fd.weakform.StressEquilibrium(fe2)
    macro_assembly = fd.Assembly.create(macro_weakform, macro_mesh, n_elm_gp=1)
    problem = fd.problem.NonLinear(macro_assembly)
    problem.initialize()
    problem.set_start()
    problem.dtime = 1.0
    start_stress = macro_assembly.sv_start["Stress"].asarray().copy()
    start_tangent = macro_assembly.sv_start["TangentMatrix"].copy()

    strain = np.array([1e-3, 0, 0, 0, 2e-4, 0])
    macro_assembly.sv["Strain"] = StrainTensorList(strain[:, None])
    stresses = []
    for _ in range(3):  # as successive Newton-Raphson iterations
        fe2.update(macro_assembly, problem)
        stresses.append(macro_assembly.sv["Stress"].asarray()[:, 0].copy())

    expected = fe2.list_assembly[0].weakform.constitutivelaw.get_elastic_matrix()
    np.testing.assert_allclose(stresses[0], expected @ strain, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(stresses[1], stresses[0], rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(stresses[2], stresses[0], rtol=1e-12, atol=1e-10)
    # the state at the start of the increment is not modified by the updates
    assert np.array_equal(macro_assembly.sv_start["Stress"].asarray(), start_stress)
    assert np.array_equal(macro_assembly.sv_start["TangentMatrix"], start_tangent)
    # one strain condition per cell, whatever the number of updates
    strain_conditions = [
        bc
        for bc in fe2.list_problem[0].bc.list_all()
        if getattr(bc, "variable_name", "").startswith("E")
    ]
    assert len(strain_conditions) == 6


def test_fe2_advances_a_viscoelastic_cell_during_a_strain_hold():
    pytest.importorskip("simcoon")
    fd.ModelingSpace("3D")
    props = np.array([3000.0, 0.35, 0.0, 1500.0, 0.35, 3000.0, 1200.0])
    material = fd.constitutivelaw.Simcoon("ZENER", props)
    fe2 = fd.constitutivelaw.FE2(_micro_assembly(material))
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2)
    assembly = fd.Assembly.create(fd.weakform.StressEquilibrium(fe2), mesh, n_elm_gp=1)
    problem = fd.problem.NonLinear(assembly)
    problem.initialize()
    problem.set_start()
    problem.dtime = 1.0
    assembly.sv["Strain"] = StrainTensorList(np.array([[0.001, 0, 0, 0, 0, 0]]).T)
    fe2.update(assembly, problem)
    problem.time = 1.0
    problem.set_start()
    initial_stress = assembly.sv["Stress"].asarray()[0, 0]

    fe2.update(assembly, problem)

    assert 0 < assembly.sv["Stress"].asarray()[0, 0] < initial_stress
    assert fe2.list_problem[0].time == 2.0


def test_fe2_micro_failure_cuts_the_macro_step_and_saves_only_accepted_times(
    monkeypatch,
):
    fd.ModelingSpace("3D")
    material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
    fe2 = fd.constitutivelaw.FE2(
        _micro_assembly(material),
        micro_nr_options={"criterion": "Displacement", "tol": 1e-8, "max_subiter": 9},
    )
    mesh = fd.mesh.box_mesh(nx=2, ny=2, nz=2)
    assembly = fd.Assembly.create(fd.weakform.StressEquilibrium(fe2), mesh, n_elm_gp=1)
    problem = fd.problem.NonLinear(assembly)
    problem.initialize()
    problem.bc.add("Dirichlet", mesh.find_nodes("X", 0), "Disp", 0)
    problem.bc.add("Dirichlet", mesh.find_nodes("X", 1), "DispX", 0.001)
    micro = fe2.list_problem[0]
    assert micro.nr_parameters["criterion"] == "Displacement"
    solve = micro.solve_time_increment
    attempts = []
    saved_times = []
    restore = problem.to_start

    def solve_trial():
        assert micro.nr_parameters["max_subiter"] == 9
        assert micro.nr_parameters["tol"] == 1e-8
        span = micro.tmax - micro.t0
        attempts.append((span, micro.dtime))
        if span > 0.25:
            # Accept one provisional quarter step, then exhaust the retries.
            if micro.dtime != 0.25 or micro.time != micro.t0:
                return False, 1, 1.0
        elif micro.dtime > 0.0625:
            return False, 1, 1.0
        return solve()

    def check_rollback():
        restore()
        np.testing.assert_array_equal(micro.get_disp(), 0)
        np.testing.assert_array_equal(problem.get_disp(), 0)

    monkeypatch.setattr(micro, "solve_time_increment", solve_trial)
    monkeypatch.setattr(
        micro, "save_results", lambda *_: saved_times.append(micro.time)
    )
    monkeypatch.setattr(problem, "to_start", check_rollback)
    problem.nlsolve(dt=1.0, tmax=1.0, dt_increase_niter=0, print_info=0)

    assert (1.0, 1.0) in attempts
    assert (1.0, 0.25) in attempts
    assert (0.25, 0.0625) in attempts
    assert saved_times == [0.25, 0.5, 0.75, 1.0]
