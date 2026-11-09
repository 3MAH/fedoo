"""Local frames display as basis glyphs and follow geometry/visibility."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import fedoo as fd


def dataset():
    mesh = fd.mesh.rectangle_mesh(3, 2, elm_type="quad4", ndim=3)
    result = fd.DataSet(mesh)
    frames = np.eye(3).reshape(9, 1)
    result.gausspoint_data["ShellLocalFrame"] = np.tile(
        frames, (1, mesh.n_elements * 4)
    )
    result.node_data["Disp"] = np.tile([[1.0], [0.0], [0.0]], (1, mesh.n_nodes))
    return result


def test_frame_samples_placement_masks_and_deformation():
    pytest.importorskip("pyvistaqt")
    from fedoo.util.frame_plot import frame_samples

    result = dataset()
    positions, frames = frame_samples(result, {"scale": 0})
    np.testing.assert_allclose(positions, result.mesh.element_centers)
    np.testing.assert_allclose(frames, np.broadcast_to(np.eye(3), (2, 3, 3)))
    moved, _ = frame_samples(result, {"scale": 2})
    np.testing.assert_allclose(moved, positions + [2, 0, 0])
    positions, _ = frame_samples(result, {"scale": 0, "frame_gauss_points": True})
    np.testing.assert_allclose(positions, result.mesh.gausspoint_coordinates(4))
    positions, _ = frame_samples(
        result, {"scale": 0, "element_set": [1], "frame_gauss_points": True}
    )
    assert len(positions) == 4
    positions, _ = frame_samples(result, {"frame_max": 1})
    assert len(positions) == 1


def test_selected_local_frame_renders_triads_and_overlay(monkeypatch):
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    window = viewer.MainWindow(dataset())
    try:
        dock = window.active_dock
        dock.current_field = "ShellLocalFrame"
        dock.current_comp = None
        dock.update_plot()
        assert all(f"_local_frame_{axis}" in dock.plotter.actors for axis in range(3))
        assert "_local_frame_legend" not in dock.plotter.actors
        window.open_local_frame_dialog()
        dialog = window._local_frame_dialog
        assert not dialog.frame_legend_cb.isChecked()
        dialog.frame_legend_cb.setChecked(True)
        window.apply_local_frame_options_to_dock(dock)
        dock.update_plot()
        assert "_local_frame_legend" in dock.plotter.actors
        dialog.update_values()
        assert dialog.frame_legend_cb.isChecked()
        dialog.frame_legend_cb.setChecked(False)
        window.apply_local_frame_options_to_dock(dock)
        dock.update_plot()
        assert "_local_frame_legend" not in dock.plotter.actors
        dock.opts["frame_axes"] = (False, False, True)
        dock.update_plot()
        assert "_local_frame_0" not in dock.plotter.actors
        assert "_local_frame_2" in dock.plotter.actors
        dock.current_field = "Disp"
        dock.current_comp = "X"
        dock.opts["show_local_frames"] = True
        dock.update_plot()
        assert "_local_frame_2" in dock.plotter.actors
        dock.opts["show_local_frames"] = False
        dock.update_plot()
        assert "_local_frame_2" not in dock.plotter.actors
    finally:
        window.close()
        app.processEvents()


def test_save_precision_controls(tmp_path, monkeypatch):
    import h5py

    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    window = viewer.MainWindow(dataset())
    path = str(tmp_path / "packed.fdh5")
    monkeypatch.setattr(viewer.QtWidgets.QMessageBox, "information", lambda *a: None)

    def accept(dialog):
        for key, value in (
            ("float_dtype", None),
            ("coordinate_dtype", "float32"),
            ("integer_min_bits", 16),
            ("local_frame", "matrix"),
        ):
            combo = dialog.findChild(viewer.QtWidgets.QComboBox, "fdh5_" + key)
            assert combo is not None
            combo.setCurrentIndex(combo.findData(value))
        dialog.selectFile(path)
        return viewer.QtWidgets.QDialog.Accepted

    monkeypatch.setattr(viewer.QtWidgets.QFileDialog, "exec_", accept)
    try:
        window.save_current_data_as()
        with h5py.File(path) as f:
            assert f["mesh/nodes"].dtype == np.float32
            assert f["mesh/submesh_0/elements"].dtype == np.int16
            assert f["results/iter_0/node_data/Disp"].dtype == np.float64
        assert window.active_dock.opts["fdh5_options"]["integer_min_bits"] == 16
    finally:
        window.close()
        app.processEvents()


def test_two_component_displacements_on_embedded_mesh():
    pytest.importorskip("pyvistaqt")
    from fedoo.util.frame_plot import frame_samples

    result = dataset()
    result.node_data["Disp"] = result.node_data["Disp"][:2]
    origins, _ = frame_samples(result, {"scale": 1})
    np.testing.assert_allclose(origins, result.mesh.element_centers + [1, 0, 0])


@pytest.mark.parametrize("family", ["Shell", "Beam"])
def test_private_frames_hidden_but_overlay_available(tmp_path, monkeypatch, family):
    import h5py

    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    data = dataset()
    private_name = "_" + family + "LocalFrame"
    data.gausspoint_data[private_name] = data.gausspoint_data.pop("ShellLocalFrame")
    path = str(tmp_path / "private_frame.fdh5")
    data.save(path)
    with h5py.File(path) as f:
        ds = f["results/iter_0/gausspoint_data/submesh_0/" + private_name]
        assert ds.attrs["frame_encoding"] == "quaternion-xyzw-rows"
    loaded = fd.read_data(path)
    window = viewer.MainWindow(loaded)
    try:
        assert private_name not in viewer._viewer_field_names(loaded)
        assert window.field_combo.findText(private_name) == -1
        dock = window.active_dock
        dock.opts["show_local_frames"] = True
        dock.update_plot()
        assert all(f"_local_frame_{axis}" in dock.plotter.actors for axis in range(3))
        loaded.show_internal_fields = True
        assert private_name in viewer._viewer_field_names(loaded)
        dock.current_field = private_name
        dock.update_plot()
        assert "_local_frame_2" in dock.plotter.actors
    finally:
        window.close()
        app.processEvents()


@pytest.mark.parametrize("representation", ["matrix", "quaternion", "rotvec"])
def test_tagged_fields_in_viewer_field_list_and_components(monkeypatch, representation):
    from scipy.spatial.transform import Rotation

    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    data = dataset()
    data.gausspoint_data.pop("ShellLocalFrame")
    rotation = Rotation.from_rotvec(np.tile([0, 0, 0.3], (8, 1)))
    data.gausspoint_data["Fibres"] = {
        "matrix": rotation.as_matrix().reshape(-1, 9).T,
        "quaternion": rotation.as_quat().T,
        "rotvec": rotation.as_rotvec().T,
    }[representation]
    data.field_metadata["Fibres"] = {"type": "local_frame"}
    window = viewer.MainWindow(data)
    try:
        assert window.field_combo.findText("Fibres") >= 0
        window.field_combo.setCurrentText("Fibres")
        assert [
            window.comp_combo.itemText(i) for i in range(window.comp_combo.count())
        ] == [
            "Whole frame",
            "Local 1",
            "Local 2",
            "Local 3",
            "XX",
            "XY",
            "XZ",
            "YX",
            "YY",
            "YZ",
            "ZX",
            "ZY",
            "ZZ",
        ]
        for axis in range(3):
            window.comp_combo.setCurrentText(f"Local {axis + 1}")
            assert f"_local_frame_{axis}" in window.active_dock.plotter.actors
            assert all(
                f"_local_frame_{other}" not in window.active_dock.plotter.actors
                for other in range(3)
                if other != axis
            )
        window.comp_combo.setCurrentText("Whole frame")
        assert all(
            f"_local_frame_{axis}" in window.active_dock.plotter.actors
            for axis in range(3)
        )
        inspector = window.object_inspector
        assert inspector._component_labels("GaussPoint", "Fibres", data["Fibres"]) == [
            "Whole frame",
            "Local 1",
            "Local 2",
            "Local 3",
            "XX",
            "XY",
            "XZ",
            "YX",
            "YY",
            "YZ",
            "ZX",
            "ZY",
            "ZZ",
        ]
        window.comp_combo.setCurrentText("XX")
        assert "_local_frame_0" not in window.active_dock.plotter.actors
        assert window.active_dock.current_comp == "XX"
        np.testing.assert_allclose(
            data.get_data("Fibres", "XX"), np.cos(0.3), atol=1e-14
        )
        window.open_local_frame_dialog()
        dialog = window._local_frame_dialog
        assert dialog.frame_field.findData("Fibres") >= 0
        dialog.frame_field.setCurrentIndex(dialog.frame_field.findData("Fibres"))
        dialog.local_frames_cb.setChecked(True)
        window.apply_local_frame_options_to_dock(window.active_dock)
        assert window.active_dock.opts["frame_field"] == "Fibres"
    finally:
        window.close()
        app.processEvents()


def test_private_custom_frame_available_in_overlay_selector(monkeypatch):
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    data = dataset()
    data.element_data["_Fibres"] = np.zeros((3, data.mesh.n_elements))
    data.field_metadata["_Fibres"] = {"type": "local_frame"}
    window = viewer.MainWindow(data)
    try:
        assert window.field_combo.findText("_Fibres") == -1
        window.open_local_frame_dialog()
        dialog = window._local_frame_dialog
        assert dialog.frame_field.findData("_Fibres") >= 0
        dialog.frame_field.setCurrentIndex(dialog.frame_field.findData("_Fibres"))
        dialog.local_frames_cb.setChecked(True)
        window.apply_local_frame_options_to_dock(window.active_dock)
        window.active_dock.update_plot()
        assert "_local_frame_0" in window.active_dock.plotter.actors
    finally:
        window.close()
        app.processEvents()


def test_custom_frames_only_on_one_submesh():
    pytest.importorskip("pyvistaqt")
    from fedoo.util.frame_plot import frame_samples

    data = dataset()
    base = data.mesh
    mesh = fd.MultiMesh(
        base.nodes, {"quad4": base.elements, "tri3": base.elements[:, :3]}
    )
    mixed = fd.DataSet(mesh)
    mixed.element_data["Fibres"] = {1: np.tile([[0], [0], [0]], (1, 2))}
    mixed.field_metadata["Fibres"] = {"submeshes": {"1": {"type": "local_frame"}}}
    positions, matrices = frame_samples(mixed, {"frame_field": "Fibres", "scale": 0})
    assert len(positions) == 2
    np.testing.assert_allclose(matrices, np.broadcast_to(np.eye(3), (2, 3, 3)))


def test_separate_frame_options_apply_to_chosen_windows(monkeypatch):
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    window = viewer.MainWindow(dataset())
    try:
        first = window.active_dock
        window.add_dataset_dock(dataset(), "Second")
        second = window.active_dock
        first.opts["frame_size"] = 3
        second.opts["frame_size"] = 7
        window.apply_options_to_all = False
        window.open_plot_dialog()
        assert not hasattr(window._plot_dialog, "frame_size")
        window.open_local_frame_dialog()
        dialog = window._local_frame_dialog
        assert dialog.frame_size.value() == 7
        dialog.frame_size.setValue(9)
        before = second.opts["show_edges"]
        dialog.button_box.button(viewer.QtWidgets.QDialogButtonBox.Apply).click()
        assert second.opts["frame_size"] == 9
        assert first.opts["frame_size"] == 3
        assert second.opts["show_edges"] == before
        window.apply_options_to_all = True
        dialog.frame_size.setValue(11)
        dialog.button_box.button(viewer.QtWidgets.QDialogButtonBox.Apply).click()
        assert first.opts["frame_size"] == second.opts["frame_size"] == 11
        window._set_active(first)
        assert dialog.frame_size.value() == 11
        window._plot_dialog.edges_cb.setChecked(not before)
        window._apply_plot_options()
        assert first.opts["frame_size"] == second.opts["frame_size"] == 11
        dialog.frame_size.setValue(15)
        dialog.button_box.button(viewer.QtWidgets.QDialogButtonBox.Cancel).click()
        assert first.opts["frame_size"] == 11
    finally:
        window.close()
        app.processEvents()


def test_viewer_delegates_overlay_to_dataset_plot(monkeypatch):
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    data = dataset()
    window = viewer.MainWindow(data)
    try:
        calls = []
        original = data.plot

        def record(*args, **kwargs):
            calls.append(kwargs)
            return original(*args, **kwargs)

        monkeypatch.setattr(data, "plot", record)
        dock = window.active_dock
        dock.current_field = "Disp"
        dock.current_comp = "X"
        dock.opts.update(
            show_local_frames=True,
            frame_field="ShellLocalFrame",
            frame_axes=(False, False, True),
        )
        dock.update_plot()
        assert calls[-1]["field"] == "Disp"
        assert calls[-1]["component"] == "X"
        assert calls[-1]["show_local_frame"] == 2.0
        assert calls[-1]["local_frame_axis"] == [3]
        assert "_local_frame_2" in dock.plotter.actors
    finally:
        window.close()
        app.processEvents()
