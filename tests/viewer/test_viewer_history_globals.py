"""Global histories use value indices, independent of mesh associations."""

from types import SimpleNamespace
import numpy as np
import pytest
import fedoo as fd


@pytest.fixture
def global_results(tmp_path):
    mesh = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    mesh = fd.MultiMesh(
        mesh.nodes, {"quad4": mesh.elements, "tri3": mesh.elements[:, :3]}
    )
    data = fd.DataSet(mesh)
    data.node_data["Value"] = np.arange(mesh.n_nodes, dtype=float)
    filename = str(tmp_path / "global_history.fdh5")
    for frame in range(2):
        data.scalar_data.update(
            Time=float(frame),
            driver_dof=np.arange(7, dtype=float) + frame,
            **{"Fext(driver_dof)": np.arange(21).reshape(3, 7) + frame},
        )
        data.to_fdh5(filename, iteration=frame, overwrite=frame == 0)
    results = fd.read_data(filename)
    results.load(0)
    return results


def test_history_search_and_global_ids(global_results, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    parent = viewer.QtWidgets.QWidget()
    parent.data = global_results
    parent.active_dock = SimpleNamespace(data=global_results, beam_csys="local")
    parent.get_components = lambda field: viewer.get_field_components(
        field,
        global_results[field].shape[0] if np.ndim(global_results[field]) > 1 else 1,
    )
    dialog = viewer.HistoryPlotDialog(parent)
    try:
        dialog.data_type_combo.setCurrentText("GaussPoint")
        for text, field, component in (
            ("driver_dof", "driver_dof", "All components"),
            ("Fext(driver_dof)", "Fext(driver_dof)", "All components"),
            ("Fext(driver_dof)_2", "Fext(driver_dof)", "2"),
        ):
            dialog.filter_edit.setText(text)
            assert dialog.field_combo.currentText() == field
            assert dialog.comp_combo.currentText() == component
            assert dialog.id_label.text() == "ID:"
            assert dialog.id_spin.maximum() == 6
            assert dialog.id_spin.isEnabled()
            assert not dialog.pick_mesh_btn.isEnabled()
            dialog.id_spin.setValue(6)
            dialog.add_y_data()
            assert len(dialog.y_data_list) == 1
            assert dialog.y_data_list[0]["id"] == 6
            assert dialog.check_valide(field, "0", "GaussPoint", 6)
            assert not dialog.check_valide(field, "0", "GaussPoint", 7)
            dialog.y_data_list.clear()
            dialog.y_table.setRowCount(0)
        dialog.field_combo.setCurrentText("Time")
        assert dialog.id_spin.value() == 0
        assert not dialog.id_spin.isEnabled()
        dialog.field_combo.setCurrentText("Value")
        assert dialog.id_label.text() == "Cell/Node ID:"
        assert dialog.pick_mesh_btn.isEnabled()
        assert dialog.data_type_combo.isEnabled()
    finally:
        dialog.close()
        parent.close()
        app.processEvents()


@pytest.mark.parametrize("parallel_projection", [False, True])
def test_clipped_animation_preserves_zoom_and_plane_count(
    tmp_path, monkeypatch, parallel_projection
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    # Plane widgets require a VTK interactor, absent in OFF_SCREEN plotters.
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", False)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    mesh = fd.mesh.box_mesh(2, 2, 2, elm_type="hex8")
    data = fd.DataSet(mesh)
    filename = str(tmp_path / "clipped.fdh5")
    for frame in range(2):
        data.node_data["Value"] = np.arange(mesh.n_nodes, dtype=float) + frame
        data.node_data["Disp"] = mesh.nodes.T * frame * 0.1
        data.to_fdh5(filename, iteration=frame, overwrite=frame == 0)
    window = viewer.MainWindow(fd.read_data(filename))
    try:
        window.open_clip_dialog()
        window._on_dialog_clip_changed(True, (0.5, 0.5, 0.5), (1, 0, 0), False)
        camera = window.plotter.camera
        camera.parallel_projection = parallel_projection
        camera.zoom(1.7)
        position = np.array(list(window.plotter.camera_position))
        scale = camera.parallel_scale
        window.anim_wrap_check.setChecked(True)
        window.start_animation()
        for _ in range(4):
            window._on_anim_tick()
            np.testing.assert_allclose(list(window.plotter.camera_position), position)
            assert camera.parallel_scale == pytest.approx(scale)
            assert camera.parallel_projection == parallel_projection
            widgets = getattr(window.plotter, "widgets", window.plotter)
            assert len(widgets.plane_widgets) == 1
    finally:
        window.stop_animation()
        window.close()
        app.processEvents()


@pytest.mark.parametrize("association", ["Node", "Element"])
def test_history_picking_maps_display_ids_and_stops_after_pick(
    global_results, monkeypatch, association
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    window = viewer.MainWindow(global_results)
    window.open_history_plot_dialog()
    dialog = window._history_plot_dialog
    dialog.field_combo.setCurrentText("Value")
    dialog.data_type_combo.setCurrentText(association)
    callbacks = {}
    stopped = []
    monkeypatch.setattr(window.plotter, "disable_picking", lambda: stopped.append(True))
    try:
        if association == "Node":
            displayed = viewer.pv.PolyData(np.array([[0.0, 0.0, 0.0]]))
            displayed.point_data["_fedoo_node_ids"] = np.array([3])
            picker = SimpleNamespace(GetDataSet=lambda: displayed, GetPointId=lambda: 0)
            monkeypatch.setattr(
                window.plotter,
                "enable_point_picking",
                lambda **kwargs: callbacks.update(kwargs),
            )
            dialog.on_pick_mesh()
            callbacks["callback"](displayed.points[0], picker)
            assert dialog.id_spin.value() == 3
            assert callbacks["use_picker"]
            assert len(stopped) == 2
        else:
            displayed = viewer.pv.UnstructuredGrid(
                np.array([3, 0, 1, 2]),
                np.array([5]),
                np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            )
            displayed.cell_data["_fedoo_global_cell_ids"] = np.array([1])
            picker = SimpleNamespace(
                SetTolerance=lambda value: None,
                Pick=lambda *args: None,
                GetCellId=lambda: 0,
                GetActor=lambda: None,
            )
            monkeypatch.setattr(viewer.vtk, "vtkCellPicker", lambda: picker)
            monkeypatch.setattr(
                window.active_dock,
                "picked_actor_mesh",
                lambda actor: ("data", displayed),
            )
            interactor = window.plotter.interactor
            removed = []
            monkeypatch.setattr(interactor, "GetEventPosition", lambda: (0, 0))

            def observe(event, callback):
                callbacks[event] = callback
                return 42

            monkeypatch.setattr(interactor, "AddObserver", observe)
            monkeypatch.setattr(interactor, "RemoveObserver", removed.append)
            dialog.on_pick_mesh()
            dialog.on_pick_mesh()
            assert removed == [42]
            callbacks["LeftButtonPressEvent"](None, None)
            assert dialog.id_spin.value() == 1
            assert removed == [42, 42]
            assert window._lbp_tag is None
    finally:
        window.close()
        app.processEvents()
