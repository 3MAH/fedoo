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


@pytest.fixture
def problem_window(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    problem = make_problem()
    window = viewer.MainWindow(problem)
    monkeypatch.setattr(viewer.QtWidgets.QMessageBox, "information", lambda *a: None)
    monkeypatch.setattr(
        viewer.QtWidgets.QMessageBox, "warning", lambda *a: pytest.fail(str(a[-1]))
    )
    try:
        yield viewer, problem, window
    finally:
        window.close()
        app.processEvents()


def test_viewer_problem_reload_time_toggle_and_save(problem_window, tmp_path):
    viewer, problem, window = problem_window
    assert isinstance(window.data, fd.ProblemDataSet)
    assert window.boundary_conditions_action.isEnabled()
    assert window.iter_slider.maximum() == window.iter_slider.minimum()
    assert "_fedoo_time" not in window.plotter.actors
    window.open_plot_dialog()
    assert not window._plot_dialog.show_time_cb.isChecked()
    window._plot_dialog.show_time_cb.setChecked(False)
    window._apply_plot_options()
    assert "_fedoo_time" not in window.plotter.actors
    problem.set_dof("DispX", 0.2)
    problem.time = 2.5
    window.reload_results()
    window._plot_dialog.show_time_cb.setChecked(True)
    window._apply_plot_options()
    text = window.plotter.actors["_fedoo_time"].GetText(3)
    assert "2.5" in text
    filename = str(tmp_path / "viewer_problem.fdh5")
    assert window._save_data_to_fdh5(filename)
    saved = fd.read_data(filename)
    np.testing.assert_allclose(saved["DispX"], 0.2)
    assert saved.scalar_data["Time"] == 2.5
    window.add_dataset_dock(saved, "Saved snapshot")
    assert not window.boundary_conditions_action.isEnabled()


def test_viewer_wraps_problem_from_before_base_reload(problem_window, monkeypatch):
    viewer, problem, window = problem_window
    import fedoo.core.problem_dataset as module

    replacement = type("ProblemBase", (), {"__module__": "fedoo.core.base"})
    monkeypatch.setattr(viewer, "ProblemBase", replacement)
    monkeypatch.setattr(module, "ProblemBase", replacement)
    second = viewer.MainWindow(problem)
    try:
        assert isinstance(second.data, fd.ProblemDataSet)
        assert second.data.problem is problem
        assert second.boundary_conditions_action.isEnabled()
        dock = window.add_dataset_dock(problem, "Existing notebook problem")
        assert isinstance(dock.data, fd.ProblemDataSet)
        assert "Disp" in dock.data.field_names()
    finally:
        second.close()


def test_modal_problem_current_mode_in_viewer(problem_window):
    viewer, _, window = problem_window
    fd.ModelingSpace("3D")
    mesh = fd.mesh.rectangle_mesh(5, 5, elm_type="quad4", ndim=3)
    material = fd.constitutivelaw.ElasticIsotrop(210e9, 0.3)
    material.set_density(7800.0)
    shell = fd.constitutivelaw.ShellHomogeneous(material, 0.01)
    assembly = fd.Assembly.create(fd.weakform.PlateEquilibrium(shell), mesh)
    modal = fd.problem.Modal(assembly)
    boundary = np.unique(
        np.concatenate(
            [mesh.node_sets[name] for name in ("left", "right", "bottom", "top")]
        )
    )
    modal.bc.add("Dirichlet", boundary, ["Disp", "Rot"], 0.0)
    modal.solve(n_modes=2, sigma=0.0)
    dock = window.add_dataset_dock(modal, "Current mode")
    assert isinstance(dock.data, fd.ProblemDataSet)
    np.testing.assert_allclose(
        dock.data.get_data("Disp"), modal.get_dof_solution("Disp")
    )
    modal.set_mode(1)
    window.reload_results()
    np.testing.assert_allclose(
        dock.data.get_data("Disp"), modal.get_dof_solution("Disp")
    )
    assert not hasattr(dock.data, "list_data")


def test_boundary_conditions_add_edit_highlight_and_remove(problem_window):
    viewer, problem, window = problem_window
    window.open_boundary_conditions_dialog()
    dialog = window.boundary_conditions_dialog
    dialog.name_edit.setText("fixed")
    dialog.variable_combo.setCurrentText("DispX")
    index = next(
        i
        for i in range(dialog.target_combo.count())
        if dialog.target_combo.itemData(i) == ("Node", "left")
    )
    dialog.target_combo.setCurrentIndex(index)
    dialog.value_edit.setText("0.1 + 0.2")
    dialog.apply_condition()
    assert len(problem.bc) == 1
    boundary = list(problem.bc)[0]
    assert boundary.name == "fixed"
    assert boundary.node_set_name == "left"
    assert boundary.value == pytest.approx(0.3)
    dialog.table.selectRow(0)
    dialog.highlight_selected()
    assert window.node_sets_dialog.current_selection == set(
        problem.mesh.node_sets["left"]
    )
    time_func = boundary.time_func
    dialog.value_edit.setText("0.5")
    dialog.apply_condition()
    assert len(problem.bc) == 1
    assert list(problem.bc)[0].value == 0.5
    assert list(problem.bc)[0].time_func is time_func
    dialog.table.selectRow(0)
    dialog.remove_selected()
    assert len(problem.bc) == 0


def test_boundary_condition_from_element_set_is_nodal(problem_window):
    viewer, problem, window = problem_window
    window._ensure_dock_element_sets(window.active_dock)
    window.active_dock.element_sets["first"] = [0]
    window.open_boundary_conditions_dialog()
    dialog = window.boundary_conditions_dialog
    dialog.type_combo.setCurrentText("Neumann")
    dialog.variable_combo.setCurrentText("DispY")
    index = next(
        i
        for i in range(dialog.target_combo.count())
        if dialog.target_combo.itemData(i) == ("Element", "first")
    )
    dialog.target_combo.setCurrentIndex(index)
    dialog.value_edit.setText("2")
    dialog.apply_condition()
    boundary = list(problem.bc)[0]
    np.testing.assert_array_equal(
        np.sort(boundary.node_set), np.unique(problem.mesh.elements[0])
    )
    assert boundary.bc_type == "Neumann"


def test_boundary_condition_global_ids_have_no_physical_nodes(problem_window):
    viewer, problem, window = problem_window
    problem.add_global_dof("Driver", 3)
    window.open_boundary_conditions_dialog()
    dialog = window.boundary_conditions_dialog
    dialog.variable_combo.setCurrentText("Driver")
    dialog.target_combo.setCurrentText("Global DOF ID 2")
    dialog.value_edit.setText("1.5")
    dialog.apply_condition()
    boundary = list(problem.bc)[0]
    np.testing.assert_array_equal(boundary.node_set_name, [2])
    dialog.table.selectRow(0)
    dialog.highlight_selected()
    assert "no mesh-node" in dialog.status.text()
    dialog.value_edit.setText("2")
    dialog.apply_condition()
    assert len(problem.bc) == 1
    np.testing.assert_array_equal(list(problem.bc)[0].node_set_name, [2])


def test_time_overlay_not_added_when_time_absent(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    data = fd.DataSet(fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3))
    data.node_data["Value"] = np.zeros(data.mesh.n_nodes)
    window = viewer.MainWindow(data)
    try:
        assert "_fedoo_time" not in window.plotter.actors
    finally:
        window.close()
        app.processEvents()
