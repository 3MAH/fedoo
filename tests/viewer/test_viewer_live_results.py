"""Live frame discovery and display-only surface thickness regressions."""

import numpy as np
import pytest
import fedoo as fd


def test_viewer_refreshes_controls_without_recomputing_all_frame_limits(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    pv = pytest.importorskip("pyvista")
    monkeypatch.setattr(pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    mesh = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    saved = fd.DataSet(mesh)
    saved.node_data["Value"] = np.ones(mesh.n_nodes)
    filename = str(tmp_path / "controls.fdh5")
    saved.to_fdh5(filename, iteration=0, overwrite=True)
    saved.to_fdh5(filename, iteration=1)
    window = viewer.MainWindow(fd.read_data(filename))
    try:
        dock = window.active_dock
        dock.opts.update(clim_mode="all", clim=[0, 5])

        def unexpected_scan(*args, **kwargs):
            pytest.fail("Frame navigation must not scan all frame fields")

        monkeypatch.setattr(dock.data, "get_all_frame_lim", unexpected_scan)
        saved.node_data["Value"] *= 3
        saved.to_fdh5(filename, iteration=2)
        window.iter_slider.setValue(0)
        assert window.iter_slider.maximum() == 2
        assert window.iter_spin.maximum() == 2
        assert dock.opts["clim"] == [0, 5]
        saved.to_fdh5(filename, iteration=0)
        window.reload_results()
        np.testing.assert_allclose(dock.data["Value"], 3)
        assert dock.opts["clim"] == [0, 5]
        window.open_plot_dialog()
        assert not hasattr(window._plot_dialog, "thickness_spin")
    finally:
        if window._plot_dialog:
            window._plot_dialog.close()
        window.close()
        app.processEvents()


def test_live_file_grows_without_reloading_fields_or_resetting_options(tmp_path):
    viewer = pytest.importorskip("fedoo.util.viewer")
    mesh = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    saved = fd.DataSet(mesh)
    saved.node_data["Value"] = np.arange(mesh.n_nodes, dtype=float)
    filename = str(tmp_path / "growing.fdh5")
    saved.to_fdh5(filename, iteration=4, overwrite=True)
    data = fd.read_data(filename)
    data.beam_options.update(reduction="signed_abs_max")
    loaded = data.node_data["Value"]

    saved.node_data["Value"] += 10
    saved.to_fdh5(filename, iteration=9)
    data = viewer._refresh_result_frames(data)
    assert data.n_iter == 2
    assert data.loaded_iter == 0
    assert data.node_data["Value"] is loaded
    assert data.beam_options["reduction"] == "signed_abs_max"
    data.load(1)
    np.testing.assert_allclose(data["Value"], np.arange(mesh.n_nodes) + 10)

    saved.node_data["Value"] += 10
    saved.to_fdh5(filename, iteration=9)
    viewer._refresh_result_frames(data)
    np.testing.assert_allclose(data["Value"], np.arange(mesh.n_nodes) + 10)
    viewer._refresh_result_frames(data, reload_current=True)
    np.testing.assert_allclose(data["Value"], np.arange(mesh.n_nodes) + 20)


def test_shell_menu_uses_saved_thickness_after_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    pv = pytest.importorskip("pyvista")
    monkeypatch.setattr(pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    fd.ModelingSpace("3D")
    surface = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    mesh = fd.Mesh(
        np.vstack((surface.nodes, [[10, 10, 10]])), surface.elements, "quad4"
    )
    section = fd.constitutivelaw.ShellHomogeneous(
        fd.constitutivelaw.ElasticIsotrop(1000, 0.25), 0.4
    )
    assembly = fd.Assembly.create(fd.weakform.PlateEquilibrium(section), mesh)
    pb = fd.problem.Linear(assembly)
    result = pb.get_results(assembly, ["Stress"])
    filename = str(tmp_path / "shell.fdh5")
    result.save(filename)
    window = viewer.MainWindow(fd.read_data(filename))
    try:

        def accept_shell_options(dialog):
            checkbox = dialog.findChild(viewer.QCheckBox, "shell_display_thickness")
            assert checkbox is not None and checkbox.isEnabled()
            checkbox.setChecked(True)
            return viewer.QtWidgets.QDialog.Accepted

        monkeypatch.setattr(viewer.QtWidgets.QDialog, "exec", accept_shell_options)
        window.open_shell_section_dialog()
        assert window.active_dock.opts["shell_thickness"] is True
        displayed = window.active_dock.pv_mesh
        assert displayed.n_cells == 1
        np.testing.assert_allclose(displayed.bounds[4:], [-0.2, 0.2])
        callbacks = []

        def capture_picker(**kwargs):
            assert kwargs["use_picker"] is True
            callbacks.append(kwargs["callback"])

        monkeypatch.setattr(window.plotter, "enable_point_picking", capture_picker)
        window._pick_node()
        for index in (0, 1):
            from types import SimpleNamespace

            picker = SimpleNamespace(
                GetDataSet=lambda: displayed, GetPointId=lambda: index
            )
            callbacks[0](displayed.points[index], picker)
            assert "Node id=0" in window.statusBar().currentMessage()
            assert "Stress" in window.statusBar().currentMessage()
    finally:
        window.close()
        app.processEvents()
