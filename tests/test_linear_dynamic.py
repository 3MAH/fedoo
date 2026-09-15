import numpy as np
import pytest

import fedoo as fd


def _dynamic_model():
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=3, ny=2, x_min=0.0, x_max=2.0, elm_type="quad4")
    material = fd.constitutivelaw.ElasticIsotrop(100.0, 0.3)
    material.set_density(2.0)
    weakform = fd.weakform.StressEquilibrium(material)
    stiffness = fd.Assembly.create(weakform, mesh)
    return mesh, stiffness


def test_linear_without_integrator_keeps_static_path():
    mesh, stiffness = _dynamic_model()
    problem = fd.problem.Linear(stiffness)
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    right = mesh.find_nodes("X", mesh.bounding_box.xmax)
    problem.bc.add("Dirichlet", left, ["DispX", "DispY"], 0.0)
    problem.bc.add("Dirichlet", right, "DispX", 0.1)
    problem.solve()

    assert problem.time_integrator is None
    assert np.max(problem.get_disp("DispX")) == 0.1


def test_repeated_static_solve_replaces_the_previous_solution():
    mesh, stiffness = _dynamic_model()
    problem = fd.problem.Linear(stiffness)
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    right = mesh.find_nodes("X", mesh.bounding_box.xmax)
    problem.bc.add("Dirichlet", left, ["DispX", "DispY"], 0.0)
    problem.bc.add("Neumann", right, "DispX", 1.0)

    problem.solve()
    first_solution = problem.get_X().copy()
    problem.solve()

    np.testing.assert_allclose(problem.get_X(), first_solution)


def test_static_linear_registered_output_is_automatic_and_can_be_cleared(tmp_path):
    mesh, stiffness = _dynamic_model()
    problem = fd.problem.Linear(stiffness)
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    right = mesh.find_nodes("X", mesh.bounding_box.xmax)
    problem.bc.add("Dirichlet", left, ["DispX", "DispY"], 0.0)
    problem.bc.add("Neumann", right, "DispX", 1.0)

    filename = tmp_path / "linear_results.fdh5"
    results = problem.add_output(filename, ["Disp", "Stress"])
    saved = []
    problem.save_results = lambda iteration=None: saved.append(iteration)
    problem.solve()

    assert saved == [0]

    assert problem.clear_outputs() is problem
    problem.solve()
    assert saved == [0]

    continued_results = problem.add_output(filename, ["Disp"])
    problem.solve()
    assert continued_results is results
    assert saved == [0, 1]


def test_fdh5_write_modes_across_problems(tmp_path):
    filename = tmp_path / "continued_results.fdh5"
    _, stiffness = _dynamic_model()
    first = fd.problem.Linear(stiffness)
    first_results = first.add_output(
        filename,
        ["Disp"],
        write_mode="overwrite",
    )
    first.save_results()
    first.save_results()
    assert first_results.n_iter == 2

    continued = fd.problem.Linear(stiffness)
    continued_results = continued.add_output(
        filename,
        ["Disp"],
        write_mode="append",
    )
    assert continued_results.n_iter == 2
    continued.save_results()
    assert continued_results.n_iter == 3

    loaded = fd.read_data(filename)
    assert loaded.n_iter == 3

    refusing = fd.problem.Linear(stiffness)
    with pytest.raises(FileExistsError, match="already exists"):
        refusing.add_output(filename, ["Disp"], write_mode="error")

    replacing = fd.problem.Linear(stiffness)
    replacing.add_output(filename, ["Disp"], write_mode="overwrite")
    replacing.save_results()

    from fedoo.util.fdh5 import FDH5Reader

    assert FDH5Reader(filename).list_iterations() == [0]


def test_fdz_write_modes_without_continuation(tmp_path):
    filename = tmp_path / "results.fdz"
    _, stiffness = _dynamic_model()

    first = fd.problem.Linear(stiffness)
    first_results = first.add_output(filename, ["Disp"])
    first.save_results(0)
    first.save_results(1)
    assert first_results.n_iter == 2
    assert fd.read_data(filename).n_iter == 2

    continued = fd.problem.Linear(stiffness)
    with pytest.raises(ValueError, match="only.*supported for 'fdh5'"):
        continued.add_output(filename, ["Disp"], write_mode="append")

    refusing = fd.problem.Linear(stiffness)
    with pytest.raises(ValueError, match="only.*supported for 'fdh5'"):
        refusing.add_output(filename, ["Disp"], write_mode="error")

    replacing = fd.problem.Linear(stiffness)
    replacing.add_output(filename, ["Disp"], write_mode="overwrite")
    replacing.save_results()

    assert fd.read_data(filename).n_iter == 1


def test_linear_newmark_regression():
    mesh, stiffness = _dynamic_model()
    dt = 0.01
    initial = np.zeros(stiffness.space.nvar * mesh.n_nodes)
    initial[mesh.n_nodes :] = 0.01 * mesh.nodes[:, 0]
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)

    current = fd.problem.LinearNewmark(
        stiffness,
        time_step=dt,
        integrator=fd.time.Newmark(beta=0.5, gamma=0.5),
    )
    current.set_initial_displacement("all", initial)
    current.set_initial_velocity("all", 0.0)
    current.set_initial_acceleration("all", 0.0)
    current.bc.add("Dirichlet", left, ["DispX", "DispY"], 0.0)
    current.initialize()

    for _ in range(4):
        current.apply_boundary_conditions()
        current.solve()
        current.update()

    expected_displacement = np.array(
        [
            0.0,
            0.00089796163343715,
            0.00062719081987743,
            0.0,
            -0.00089796163343715,
            -0.00062719081987744,
            0.0,
            0.01009419414724506,
            0.01959847348281653,
            0.0,
            0.01009419414724508,
            0.01959847348281667,
        ]
    )
    expected_velocity = np.array(
        [
            0.0,
            0.04679766109243953,
            0.03338869541156735,
            0.0,
            -0.04679766109243937,
            -0.0333886954115676,
            0.0,
            0.00476651432328706,
            -0.02098287202429786,
            0.0,
            0.00476651432328844,
            -0.02098287202429334,
        ]
    )
    expected_acceleration = np.array(
        [
            0.0,
            1.110621676576292,
            0.8959881602381707,
            0.0,
            -1.1106216765762822,
            -0.8959881602381826,
            0.0,
            0.09139970800809555,
            -0.5060340972111443,
            0.0,
            0.09139970800802616,
            -0.5060340972112137,
        ]
    )
    np.testing.assert_allclose(current.get_X(), expected_displacement, rtol=1e-11)
    np.testing.assert_allclose(current.get_velocity(), expected_velocity, rtol=1e-11)
    np.testing.assert_allclose(
        current.get_acceleration(), expected_acceleration, rtol=1e-11
    )


def test_linear_generalized_alpha_history():
    mesh, stiffness = _dynamic_model()
    problem = fd.problem.Linear(
        stiffness,
        time_step=0.01,
        integrator=fd.time.GeneralizedAlpha(alpha_m=-0.1, alpha_f=0.0),
    )
    initial = np.zeros(stiffness.space.nvar * mesh.n_nodes)
    initial[mesh.n_nodes :] = 0.01 * mesh.nodes[:, 0]
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    problem.set_initial_displacement("all", initial)
    problem.bc.add("Dirichlet", left, ["DispX", "DispY"], 0.0)

    returned = problem.solve_history(tmax=0.03)

    assert returned is problem
    assert np.isclose(problem.time, 0.03)
    assert np.all(np.isfinite(problem.get_X()))
    assert np.all(np.isfinite(problem.get_velocity()))


def test_linear_history_automatically_saves_registered_outputs(tmp_path):
    mesh, stiffness = _dynamic_model()
    problem = fd.problem.Linear(
        stiffness,
        time_step=0.01,
        integrator=fd.time.Newmark(),
    )
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    problem.bc.add("Dirichlet", left, ["DispX", "DispY"], 0.0)
    problem.add_output(tmp_path / "linear_history.fdh5", ["Disp"])
    saved = []
    problem.save_results = lambda iteration=None: saved.append(iteration)

    problem.solve_history(tmax=0.01)

    assert saved == [0]
