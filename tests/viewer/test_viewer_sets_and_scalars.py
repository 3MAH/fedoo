"""Node/element set tools, conversions and scalar expressions through Save As."""

import numpy as np
import pytest
import fedoo as fd


@pytest.fixture(params=[False, True], ids=["mesh", "multimesh"])
def viewer_window(request, monkeypatch, tmp_path):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    mesh = fd.mesh.rectangle_mesh(3, 2, elm_type="quad4", ndim=3)
    if request.param:
        mesh = fd.MultiMesh(
            mesh.nodes, {"quad4": mesh.elements[:1], "tri3": mesh.elements[1:, :3]}
        )
    data = fd.DataSet(mesh)
    data.node_data["Disp"] = np.vstack(
        (np.arange(mesh.n_nodes), np.zeros((2, mesh.n_nodes)))
    )
    stress = np.zeros((6, 2))
    stress[0] = [10, 20]
    data.element_data["Stress"] = (
        {0: stress[:, :1], 1: stress[:, 1:]} if request.param else stress
    )
    source = str(tmp_path / "source.fdh5")
    data.to_fdh5(source, iteration=0, overwrite=True)
    data.node_data["Disp"][0] *= 2
    data.to_fdh5(source, iteration=1)
    window = viewer.MainWindow(fd.read_data(source))
    window.iter_slider.setValue(0)
    monkeypatch.setattr(viewer.QtWidgets.QMessageBox, "information", lambda *args: None)
    monkeypatch.setattr(
        viewer.QtWidgets.QMessageBox,
        "warning",
        lambda *args: pytest.fail(str(args[-1])),
    )
    try:
        yield viewer, window
    finally:
        window.close()
        app.processEvents()


def choose_set(viewer, dialog, name):
    for row in range(dialog.sets_list.count()):
        if dialog.sets_list.item(row).data(viewer.Qt.UserRole) == name:
            dialog.sets_list.setCurrentRow(row)
            return
    pytest.fail(f"Missing set {name}")


def test_node_manager_create_rename_remove_and_conversion(viewer_window, monkeypatch):
    viewer, window = viewer_window
    window.open_node_sets_dialog()
    nodes = window.node_sets_dialog
    nodes.use_def_mesh.setChecked(False)
    nodes.expr_edit.setText("ID <= 2")
    nodes._apply_expression_to_selection()
    assert nodes.current_selection == {0, 1, 2}
    monkeypatch.setattr(
        viewer.QtWidgets.QInputDialog, "getText", lambda *a, **k: ("edge", True)
    )
    nodes._save_selection_as_new_set()
    np.testing.assert_array_equal(window.data.mesh.node_sets["edge"], [0, 1, 2])
    assert not any(
        box.title() == "Show/Hide elements" and box.isVisible()
        for box in nodes.findChildren(viewer.QtWidgets.QGroupBox)
    )
    choose_set(viewer, nodes, "edge")
    monkeypatch.setattr(
        viewer.QtWidgets.QInputDialog, "getText", lambda *a, **k: ("BottomCustom", True)
    )
    nodes._rename_selected_set()
    assert "BottomCustom" in window.data.mesh.node_sets
    assert "edge" not in window.data.mesh.node_sets
    choose_set(viewer, nodes, "BottomCustom")
    window.open_element_sets_dialog()
    elements = window.element_sets_dialog
    elements.conversion_all_nodes.setChecked(False)
    elements.conversion_sources.setCurrentText("BottomCustom")
    monkeypatch.setattr(
        viewer.QtWidgets.QInputDialog, "getText", lambda *a, **k: ("touching", True)
    )
    elements._convert_selected_set()
    assert set(window.data.mesh.element_sets["touching"]) == {0, 1}
    nodes.conversion_sources.setCurrentText("touching")
    monkeypatch.setattr(
        viewer.QtWidgets.QInputDialog, "getText", lambda *a, **k: ("connected", True)
    )
    nodes._convert_selected_set()
    blocks = getattr(window.data.mesh, "submeshes", (window.data.mesh,))
    expected = np.unique(np.concatenate([block.elements.ravel() for block in blocks]))
    np.testing.assert_array_equal(window.data.mesh.node_sets["connected"], expected)
    choose_set(viewer, nodes, "BottomCustom")
    nodes._remove_selected_set()
    assert "BottomCustom" not in window.data.mesh.node_sets


def test_field_selection_and_scalar_history_saved_from_viewer(
    viewer_window, monkeypatch, tmp_path
):
    viewer, window = viewer_window
    window.open_element_sets_dialog()
    elements = window.element_sets_dialog
    elements.expr_edit.setText("Stress_XX >= 15")
    elements._apply_expression_to_selection()
    assert elements.current_selection == {1}
    window.open_node_sets_dialog()
    nodes = window.node_sets_dialog
    nodes.use_def_mesh.setChecked(False)
    nodes.expr_edit.setText("Y == 1")
    nodes._apply_expression_to_selection()
    monkeypatch.setattr(
        viewer.QtWidgets.QInputDialog, "getText", lambda *a, **k: ("Top", True)
    )
    nodes._save_selection_as_new_set()
    window.open_scalar_expression_dialog()
    scalar = window.scalar_expression_dialog
    scalar.name_edit.setText("TopDisplacement")
    scalar.expression_edit.setText('sum(Disp_X[nodes("Top")])')
    scalar.compute()
    assert window.data.scalar_data["TopDisplacement"] == 12.0
    window.open_history_plot_dialog()
    assert window._history_plot_dialog.field_combo.findText("TopDisplacement") >= 0
    np.testing.assert_allclose(window.data.get_history("TopDisplacement"), [12.0, 24.0])
    window.iter_slider.setValue(1)
    assert window.data.scalar_data["TopDisplacement"] == 24.0
    destination = str(tmp_path / "saved.fdh5")
    assert window._save_data_to_fdh5(destination, compressed=False)
    reopened = fd.read_data(destination)
    assert "Top" in reopened.mesh.node_sets
    np.testing.assert_allclose(reopened.get_history("TopDisplacement"), [12.0, 24.0])
    assert (
        reopened.scalar_expressions["TopDisplacement"]["expression"]
        == 'sum(Disp_X[nodes("Top")])'
    )


def test_sets_are_isolated_when_switching_datasets(viewer_window):
    viewer, window = viewer_window
    first = window.active_dock
    window.open_node_sets_dialog()
    window.node_sets_dialog.element_sets["FirstOnly"] = [0]
    window.node_sets_dialog._refresh_sets_list()
    other = fd.DataSet(fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3))
    other.node_data["Value"] = np.ones(other.mesh.n_nodes)
    window.add_dataset_dock(other, "Other")
    assert "FirstOnly" not in window.node_sets_dialog.element_sets
    window._set_active(first)
    assert window.node_sets_dialog.element_sets["FirstOnly"] == [0]


def test_invalid_selection_preserves_current_selection(viewer_window, monkeypatch):
    viewer, window = viewer_window
    window.open_node_sets_dialog()
    dialog = window.node_sets_dialog
    dialog.current_selection = {0}
    errors = []
    monkeypatch.setattr(
        viewer.QtWidgets.QMessageBox, "warning", lambda *a: errors.append(a[-1])
    )
    dialog.expr_edit.setText("unknown_field > 0")
    dialog._apply_expression_to_selection()
    assert dialog.current_selection == {0}
    assert errors
