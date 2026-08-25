import numpy as np

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
