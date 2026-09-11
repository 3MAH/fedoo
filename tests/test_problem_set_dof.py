import numpy as np

import fedoo as fd


def _assembly():
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="quad4")
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3)
    weakform = fd.weakform.StressEquilibrium(material)
    return fd.Assembly.create(weakform, mesh)


def test_set_dof_initializes_and_sets_scalar_and_vector_fields():
    problem = fd.problem.Linear(_assembly())
    n_nodes = problem.mesh.n_nodes

    problem.set_dof("DispX", np.arange(n_nodes))
    np.testing.assert_allclose(problem.get_dof_solution("DispX"), np.arange(n_nodes))

    displacement = np.arange(2 * n_nodes).reshape(2, n_nodes)
    assert problem.set_dof("Disp", displacement) is problem
    np.testing.assert_allclose(problem.get_dof_solution("Disp"), displacement)


def test_nonlinear_set_dof_defines_the_preinitialization_state():
    problem = fd.problem.NonLinear(_assembly())
    initial = np.linspace(0.0, 0.01, problem.mesh.n_nodes)

    problem.set_dof("DispX", initial)
    np.testing.assert_allclose(problem.get_dof_solution("DispX"), initial)

    problem.initialize()
    with np.testing.assert_raises_regex(RuntimeError, "before problem initialization"):
        problem.set_dof("DispX", 0.0)


def test_set_dof_reports_an_incompatible_field_shape():
    problem = fd.problem.Linear(_assembly())

    with np.testing.assert_raises_regex(ValueError, "cannot assign values"):
        problem.set_dof("Disp", np.zeros(3))
