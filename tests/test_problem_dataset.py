"""Live Problem results, detached exports, BC management and time labels."""

import numpy as np
import pytest
from scipy import sparse
import fedoo as fd


def make_problem():
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(3, 2, elm_type="quad4")
    assembly = fd.Assembly.create(
        fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3)),
        mesh,
    )
    problem = fd.problem.Linear(assembly)
    problem.set_dof("DispX", np.arange(mesh.n_nodes) * 0.01)
    return problem


def test_live_problem_and_detached_single_frame_export(tmp_path):
    problem = make_problem()
    data = fd.ProblemDataSet(problem, fields=["Disp"])
    assert isinstance(data, fd.DataSet)
    assert data.mesh is problem.mesh
    assert not hasattr(data, "list_data")
    assert not hasattr(data, "loaded_iter")
    snapshot = data.deepcopy()
    assert type(snapshot) is fd.DataSet
    problem.set_dof("DispX", 2.0)
    problem.time = 1.25
    np.testing.assert_allclose(data.get_data("Disp", "X"), 2.0)
    assert data.get_data("Time") == 1.25
    np.testing.assert_allclose(
        snapshot["Disp", "X"], np.arange(problem.mesh.n_nodes) * 0.01
    )
    filename = str(tmp_path / "problem.fdh5")
    data.save(filename)
    saved = fd.read_data(filename)
    assert type(saved) is fd.DataSet
    np.testing.assert_allclose(saved["Disp", "X"], 2.0)
    assert saved.scalar_data["Time"] == 1.25
    with pytest.raises((NameError, ValueError)):
        data.get_data("NotAField")
    np.testing.assert_allclose(data["Disp", "X"], 2.0)


def test_unassembled_problem_and_scalar_expressions():
    problem = make_problem()
    data = fd.ProblemDataSet(problem)
    np.testing.assert_allclose(data["DispX"], np.arange(problem.mesh.n_nodes) * 0.01)
    data.add_scalar_expression("Total", "sum(Disp_X)")
    problem.set_dof("DispX", 0.5)
    assert data.get_data("Total") == pytest.approx(problem.mesh.n_nodes * 0.5)


def test_problem_identity_survives_reloaded_base_class(monkeypatch):
    import fedoo.core.problem_dataset as module
    from types import SimpleNamespace

    problem = make_problem()
    replacement = type("ProblemBase", (), {"__module__": "fedoo.core.base"})
    monkeypatch.setattr(module, "ProblemBase", replacement)
    assert not isinstance(problem, replacement)
    data = module.ProblemDataSet(problem)
    np.testing.assert_allclose(
        data.get_data("DispX"), problem.get_dof_solution("DispX")
    )
    imitation = SimpleNamespace(mesh=problem.mesh, get_results=problem.get_results)
    assert not module._is_problem(imitation)
    with pytest.raises(TypeError, match="requires a Problem"):
        module.ProblemDataSet(imitation)


def test_solved_problem_stress_snapshot(tmp_path):
    problem = make_problem()
    problem.set_solver("direct_scipy")
    problem.bc.add("Dirichlet", "left", "Disp", 0)
    problem.bc.add("Dirichlet", "right", "DispX", 0.01)
    problem.solve()
    data = fd.ProblemDataSet(problem)
    expected = problem.get_results(["Stress", "Disp"])
    np.testing.assert_allclose(
        data.get_data("Stress", "XX"), expected.get_data("Stress", "XX")
    )
    filename = str(tmp_path / "solved.fdh5")
    data.save(filename)
    np.testing.assert_allclose(
        fd.read_data(filename).get_data("Stress", "XX"),
        expected.get_data("Stress", "XX"),
    )


def test_raw_matrix_problem_and_global_dofs(tmp_path):
    problem = make_problem()
    raw = fd.Problem(sparse.eye(problem.n_dof), mesh=problem.mesh, space=problem.space)
    raw.add_global_dof("Driver", raw.mesh.n_nodes)
    raw.set_A(sparse.eye(raw.n_dof))
    raw.set_dof("DispX", 1.0)
    raw.set_dof("Driver", np.arange(raw.mesh.n_nodes))
    data = fd.ProblemDataSet(raw)
    np.testing.assert_allclose(data["DispX"], 1.0)
    np.testing.assert_array_equal(data["Driver"], np.arange(raw.mesh.n_nodes))
    assert "Driver" in data.scalar_data
    assert "Fext(Driver)" in data.scalar_data
    filename = str(tmp_path / "raw.fdh5")
    data.save(filename, compressed=False)
    np.testing.assert_array_equal(
        fd.read_data(filename)["Driver"], np.arange(raw.mesh.n_nodes)
    )


def test_problem_assembly_sum_preserves_complete_mesh_and_sets(tmp_path):
    problem = make_problem()
    first = problem.assembly
    second_mesh = fd.Mesh(
        first.mesh.nodes, first.mesh.elements[:1], "quad4", name="extra"
    )
    second = fd.Assembly.create(first.weakform, second_mesh)
    combined = fd.problem.Linear(first + second)
    data = fd.ProblemDataSet(combined, fields=["Disp"])
    assert isinstance(data.mesh, fd.MultiMesh)
    assert data.mesh.n_elements == first.mesh.n_elements + 1
    data.mesh.node_sets["custom"] = np.array([0, 1])
    data.mesh.submeshes[1].element_sets["extra_only"] = np.array([0])
    data.sync_problem_sets()
    for _ in range(2):
        data.refresh()
        np.testing.assert_array_equal(
            data.mesh.submeshes[1].element_sets["extra_only"], [0]
        )
    np.testing.assert_array_equal(combined.mesh.node_sets["custom"], [0, 1])
    filename = str(tmp_path / "sum.fdh5")
    data.save(filename)
    saved = fd.read_data(filename)
    assert saved.mesh.n_elements == data.mesh.n_elements
    np.testing.assert_array_equal(
        saved.mesh.submeshes[1].element_sets["extra_only"], [0]
    )
