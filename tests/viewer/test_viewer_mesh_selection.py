"""Connected exterior surface/edge selection and viewer node-set workflows."""

import numpy as np
import pytest
import fedoo as fd
from fedoo.lib_elements.element_list import get_node_elm_coordinates


def node_at(mesh, point):
    return int(np.flatnonzero(np.all(np.isclose(mesh.nodes, point), axis=1))[0])


@pytest.fixture
def surface_window(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    mesh = fd.mesh.box_mesh(3, 3, 3)
    data = fd.DataSet(mesh)
    data.node_data["Value"] = np.arange(mesh.n_nodes, dtype=float)
    window = viewer.MainWindow(data)
    monkeypatch.setattr(viewer.QtWidgets.QMessageBox, "information", lambda *a: None)
    monkeypatch.setattr(
        viewer.QtWidgets.QMessageBox, "warning", lambda *a: pytest.fail(str(a[-1]))
    )
    try:
        yield viewer, window
    finally:
        window.close()
        app.processEvents()


def test_viewer_node_seed_and_save_set(surface_window, monkeypatch, tmp_path):
    viewer, window = surface_window
    window.open_node_sets_dialog()
    dialog = window.node_sets_dialog
    seed = node_at(window.data.mesh, [0.5, 0.5, 1])
    monkeypatch.setattr(window, "_pick_node", lambda on_pick: on_pick(seed))
    dialog._surface_seed_from_selection()
    assert dialog.surface_seed_id.value() == seed
    dialog._select_surface_nodes()
    assert dialog.current_selection == set(window.data.mesh.find_nodes("Z", 1))
    monkeypatch.setattr(
        viewer.QtWidgets.QInputDialog, "getText", lambda *a, **k: ("TopSurface", True)
    )
    dialog._save_selection_as_new_set()
    filename = str(tmp_path / "surface.fdh5")
    assert window._save_data_to_fdh5(filename)
    saved = fd.read_data(filename)
    np.testing.assert_array_equal(
        saved.mesh.node_sets["TopSurface"], sorted(dialog.current_selection)
    )


def test_viewer_node_seed_chooses_face_and_returns_nodes(surface_window, monkeypatch):
    viewer, window = surface_window
    window.open_element_sets_dialog()
    assert not hasattr(window.element_sets_dialog, "surface_seed_id")
    window.open_node_sets_dialog()
    dialog = window.node_sets_dialog
    assert not hasattr(dialog, "surface_seed_type")
    monkeypatch.setattr(
        viewer.QtWidgets.QInputDialog, "getItem", lambda *a, **k: (a[3][0], True)
    )
    dialog._select_surface_nodes()
    nodes = window.node_sets_dialog.current_selection
    assert len(nodes) == 9
    assert np.any(np.ptp(window.data.mesh.nodes[sorted(nodes)], axis=0) == 0)


def test_viewer_cancel_ambiguous_seed_keeps_selection(surface_window, monkeypatch):
    viewer, window = surface_window
    window.open_node_sets_dialog()
    dialog = window.node_sets_dialog
    dialog.update_selection({0})
    monkeypatch.setattr(
        viewer.QtWidgets.QInputDialog, "getItem", lambda *a, **k: ("", False)
    )
    dialog._select_surface_nodes()
    assert dialog.current_selection == {0}


def test_viewer_uses_deformed_geometry_without_mutating_source(
    surface_window, monkeypatch
):
    viewer, window = surface_window
    # Start with a flat shell and fold its second cell through the Disp field.
    nodes = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0.0]]
    )
    data = fd.DataSet(
        fd.Mesh(nodes.copy(), np.array([[0, 1, 2, 3], [1, 4, 5, 2]]), "quad4")
    )
    data.node_data["Disp"] = np.zeros((3, 6))
    data.node_data["Disp"][2, 4:] = 1
    window.add_dataset_dock(data, "Folded shell")
    window.open_node_sets_dialog()
    dialog = window.node_sets_dialog
    dialog.surface_mode.setCurrentIndex(dialog.surface_mode.findData("faces"))
    dialog.surface_seed_id.setValue(0)
    dialog.surface_angle.setValue(30)
    dialog.use_def_mesh.setChecked(True)
    dialog._select_surface_nodes()
    assert dialog.current_selection == {0, 1, 2, 3}
    dialog.use_def_mesh.setChecked(False)
    dialog._select_surface_nodes()
    assert dialog.current_selection == set(range(6))
    np.testing.assert_array_equal(data.mesh.nodes, nodes)


def test_edge_viewer_mode_selects_nodes_and_cancel_preserves_selection(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    mesh = fd.mesh.box_mesh(3, 3, 3, elm_type="hex20")
    data = fd.DataSet(mesh)
    data.node_data["Value"] = np.arange(mesh.n_nodes, dtype=float)
    window = viewer.MainWindow(data)
    monkeypatch.setattr(viewer.QtWidgets.QMessageBox, "information", lambda *a: None)
    try:
        window.open_node_sets_dialog()
        dialog = window.node_sets_dialog
        assert not hasattr(dialog, "surface_seed_type")
        dialog.surface_mode.setCurrentIndex(dialog.surface_mode.findData("edges"))
        assert not dialog.surface_feature_angle.isHidden()
        assert dialog.surface_angle_label.text() == "Turn limit:"
        assert "edge" in dialog.surface_group.title()
        seed = node_at(mesh, [0.5, 0, 0])
        monkeypatch.setattr(window, "_pick_node", lambda on_pick: on_pick(seed))
        dialog.surface_select_button.click()
        app.processEvents()
        dialog._select_surface_nodes()
        expected = set(mesh.find_nodes("Y == 0 and Z == 0"))
        assert dialog.current_selection == expected
        monkeypatch.setattr(
            viewer.QtWidgets.QInputDialog, "getText", lambda *a, **k: ("Edge", True)
        )
        dialog._save_selection_as_new_set()
        saved = str(tmp_path / "edge.fdh5")
        assert window._save_data_to_fdh5(saved)
        np.testing.assert_array_equal(
            fd.read_data(saved).mesh.node_sets["Edge"], sorted(expected)
        )
        dialog.surface_seed_id.setValue(0)
        monkeypatch.setattr(
            viewer.QtWidgets.QInputDialog, "getItem", lambda *a, **k: ("", False)
        )
        dialog._select_surface_nodes()
        assert dialog.current_selection == expected
        monkeypatch.setattr(
            viewer.QtWidgets.QInputDialog, "getItem", lambda *a, **k: (a[3][0], True)
        )
        dialog._select_surface_nodes()
        assert len(dialog.current_selection) == 5
        assert 0 in dialog.current_selection
        dialog.surface_mode.setCurrentIndex(dialog.surface_mode.findData("solid"))
        assert dialog.surface_feature_angle.isHidden()
        assert dialog.surface_angle_label.text() == "Angle limit:"
    finally:
        window.close()
        app.processEvents()
