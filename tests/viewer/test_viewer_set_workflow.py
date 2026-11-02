"""Incoming set creation, plot time controls and single-pick surface seeds."""

from types import SimpleNamespace
import numpy as np
import pytest
import fedoo as fd


@pytest.fixture(params=[2, 3], ids=["xy", "xyz"])
def planar_window(request, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", False)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    mesh = fd.mesh.rectangle_mesh(5, 3, elm_type="quad4", ndim=request.param)
    data = fd.DataSet(mesh)
    data.node_data["Value"] = np.arange(mesh.n_nodes, dtype=float)
    data.scalar_data["Time"] = 1.5
    window = viewer.MainWindow(data)
    monkeypatch.setattr(viewer.QtWidgets.QMessageBox, "information", lambda *a: None)
    try:
        yield viewer, app, window
    finally:
        window.close()
        app.processEvents()


def test_planar_viewer_grows_boundary_line_and_rejects_interior(
    planar_window, monkeypatch
):
    viewer, app, window = planar_window
    window.open_node_sets_dialog()
    dialog = window.node_sets_dialog
    assert dialog.surface_mode.currentData() == "boundary"
    assert not hasattr(dialog, "surface_seed_type")
    mesh = window.data.mesh
    top = mesh.find_nodes("Y", 1)
    dialog.surface_seed_id.setValue(int(top[2]))
    dialog.surface_angle.setValue(0)
    dialog._select_surface_nodes()
    assert dialog.current_selection == set(top)
    surface = fd.mesh.extract_surface(mesh, node_set=sorted(dialog.current_selection))
    assert surface.elm_type == "lin2"
    assert surface.n_elements == 4
    interior = int(mesh.find_nodes("X == 0.5 and Y == 0.5")[0])
    dialog.surface_seed_id.setValue(interior)
    warnings = []
    monkeypatch.setattr(
        viewer.QtWidgets.QMessageBox, "warning", lambda *a: warnings.append(a[-1])
    )
    dialog._select_surface_nodes()
    assert "inside" in warnings[0]
    assert dialog.current_selection == set(top)


def test_surface_select_reuses_node_picker_and_populates_clicked_id(
    planar_window, monkeypatch
):
    viewer, app, window = planar_window
    window.open_node_sets_dialog()
    dialog = window.node_sets_dialog
    callbacks = {}
    monkeypatch.setattr(
        window.plotter,
        "enable_point_picking",
        lambda **kwargs: callbacks.update(kwargs),
    )
    dialog.surface_select_button.click()
    assert callbacks["picker"] == "point"
    assert callbacks["use_picker"]
    assert dialog.surface_select_button.text() == "Cancel"
    window.eventFilter(window.plotter.interactor, viewer.QEvent(viewer.QEvent.FocusIn))
    window.eventFilter(
        window.plotter.interactor, viewer.QEvent(viewer.QEvent.MouseButtonPress)
    )
    assert dialog._surface_pick_dock is window.active_dock
    node = int(window.data.mesh.find_nodes("X == 0.5 and Y == 1")[0])
    displayed = window.active_dock.pv_mesh
    point = window.data.mesh.as_3d().nodes[node]
    display_id = displayed.find_closest_point(point)
    picker = SimpleNamespace(
        GetDataSet=lambda: displayed, GetPointId=lambda: display_id
    )
    callbacks["callback"](point, picker)
    assert dialog.surface_seed_id.value() == node
    app.processEvents()
    assert dialog.surface_select_button.text() == "Select"
    assert dialog._surface_pick_dock is None


def test_rubber_band_overlay_tracks_drag_and_is_removed_on_stop(planar_window):
    viewer, app, window = planar_window
    from qtpy.QtCore import QPoint, QEvent, Qt

    window.open_element_sets_dialog()
    dialog = window.element_sets_dialog
    assert not hasattr(dialog, "surface_seed_id")
    dialog._start_picking()
    overlay = dialog._selection_band
    assert overlay is not None
    window.eventFilter(overlay.widget, QEvent(QEvent.FocusIn))
    window.eventFilter(overlay.widget, QEvent(QEvent.MouseButtonPress))
    assert dialog._selection_band is overlay
    assert dialog.stop_pick_btn.isEnabled()
    event = lambda kind, point: SimpleNamespace(
        type=lambda: kind, button=lambda: Qt.LeftButton, pos=lambda: point
    )
    overlay.eventFilter(overlay.widget, event(QEvent.MouseButtonPress, QPoint(20, 30)))
    overlay.eventFilter(overlay.widget, event(QEvent.MouseMove, QPoint(90, 110)))
    assert not overlay.band.isHidden()
    assert overlay.band.geometry().width() == 71
    assert overlay.band.geometry().height() == 81
    overlay.eventFilter(
        overlay.widget, event(QEvent.MouseButtonRelease, QPoint(90, 110))
    )
    assert overlay.band.isHidden()
    dialog._stop_picking()
    assert dialog._selection_band is None


def test_time_is_plot_option_and_respects_apply_target(planar_window):
    viewer, app, window = planar_window
    view_menu = next(
        action.menu()
        for action in window.menuBar().actions()
        if action.text() == "View"
    )
    assert "Show time" not in [action.text() for action in view_menu.actions()]
    first = window.active_dock
    assert "_fedoo_time" not in first.plotter.actors
    first.opts["show_time"] = True
    first.update_plot()
    window.add_dataset_dock(window.data.deepcopy(), "Second view")
    second = window.active_dock
    window.apply_options_to_all = False
    window.open_plot_dialog()
    assert not window._plot_dialog.show_time_cb.isChecked()
    window._plot_dialog.show_time_cb.setChecked(False)
    window._apply_plot_options()
    assert "_fedoo_time" not in second.plotter.actors
    assert "_fedoo_time" in first.plotter.actors
    window._set_active(first)
    assert window._plot_dialog.show_time_cb.isChecked()
    window.apply_options_to_all = True
    window._plot_dialog.show_time_cb.setChecked(False)
    window._apply_plot_options()
    assert all("_fedoo_time" not in dock.plotter.actors for dock in window.all_docks)


def test_create_from_source_dropdown_and_all_any_checkbox(planar_window, monkeypatch):
    viewer, app, window = planar_window
    window.open_element_sets_dialog()
    window.open_node_sets_dialog()
    nodes, elements = window.node_sets_dialog, window.element_sets_dialog
    window.active_dock.element_sets["One"] = [0]
    window.active_dock.element_sets["Two"] = [1]
    nodes._refresh_conversion_sources()
    assert isinstance(nodes.conversion_sources, viewer.QtWidgets.QComboBox)
    assert not hasattr(nodes, "conversion_all_nodes")
    assert nodes.conversion_sources.findText("Two") >= 0
    nodes.conversion_sources.setCurrentText("One")
    nodes._refresh_conversion_sources()
    assert nodes.conversion_sources.currentText() == "One"
    monkeypatch.setattr(
        viewer.QtWidgets.QInputDialog, "getText", lambda *a, **k: ("Combined", True)
    )
    nodes._convert_selected_set()
    expected = np.unique(window.data.mesh.elements[:1])
    np.testing.assert_array_equal(window.data.mesh.node_sets["Combined"], expected)
    elements.conversion_sources.setCurrentText("Combined")
    assert elements.conversion_all_nodes.isChecked()
    assert "All nodes" in elements.conversion_all_nodes.text()
    monkeypatch.setattr(
        viewer.QtWidgets.QInputDialog,
        "getText",
        lambda *a, **k: ("BackToElements", True),
    )
    elements._convert_selected_set()
    np.testing.assert_array_equal(window.data.mesh.element_sets["BackToElements"], [0])
    elements.conversion_all_nodes.click()
    assert "At least one" in elements.conversion_all_nodes.text()
    monkeypatch.setattr(
        viewer.QtWidgets.QInputDialog, "getText", lambda *a, **k: ("AnyNode", True)
    )
    elements._convert_selected_set()
    expected_any = np.flatnonzero(
        np.isin(window.data.mesh.elements, expected).any(axis=1)
    )
    np.testing.assert_array_equal(
        window.data.mesh.element_sets["AnyNode"], expected_any
    )
    elements.conversion_all_nodes.click()
    assert "All nodes" in elements.conversion_all_nodes.text()


def test_shell_seed_select_always_uses_node_picker(planar_window, monkeypatch):
    viewer, app, window = planar_window
    mesh = fd.mesh.rectangle_mesh(3, 2, elm_type="quad4", ndim=3)
    data = fd.DataSet(mesh)
    data.node_data["Value"] = np.ones(mesh.n_nodes)
    data.field_metadata["_ShellSection"] = {"family": "shell", "thickness": 0.1}
    window.add_dataset_dock(data, "Shell")
    window.open_node_sets_dialog()
    dialog = window.node_sets_dialog
    assert dialog.surface_mode.currentData() == "faces"
    assert not hasattr(dialog, "surface_seed_type")
    picked = []

    def pick_node(on_pick):
        picked.append(True)
        on_pick(1)

    monkeypatch.setattr(window, "_pick_node", pick_node)
    monkeypatch.setattr(
        window,
        "_pick_element",
        lambda *a, **k: pytest.fail("Surface seeds must use nodes"),
    )
    dialog.surface_select_button.click()
    assert picked
    assert dialog.surface_seed_id.value() == 1
    app.processEvents()
    dialog._select_surface_nodes()
    assert dialog.current_selection == set(range(mesh.n_nodes))
