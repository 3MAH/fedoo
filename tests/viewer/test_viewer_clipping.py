"""Plane interaction must not destroy a VTK widget inside its own callback."""

import numpy as np
import pytest
import fedoo as fd


@pytest.fixture
def clip_window(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", False)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    mesh = fd.mesh.box_mesh(3, 3, 3)
    data = fd.DataSet(mesh)
    data.node_data["Value"] = np.arange(mesh.n_nodes, dtype=float)
    window = viewer.MainWindow(data)
    window.open_clip_dialog()
    window._clip_dialog.enableClipChk.setChecked(True)
    try:
        yield viewer, app, window
    finally:
        window.close()
        app.processEvents()


def test_interaction_defers_and_coalesces_redraw_without_replacing_widget(
    clip_window, monkeypatch
):
    viewer, app, window = clip_window
    widget = window._plane_widget
    original_plot = window.active_dock.update_plot
    calls = []

    def counted_plot(*args, **kwargs):
        calls.append(1)
        return original_plot(*args, **kwargs)

    monkeypatch.setattr(window.active_dock, "update_plot", counted_plot)
    camera = np.array(list(window.plotter.camera_position))
    for x in np.linspace(0.1, 0.9, 10):
        widget.SetOrigin(x, 0.5, 0.5)
        widget.InvokeEvent("EndInteractionEvent")
        assert window._plane_widget is widget
        assert widget.GetEnabled()
        assert not calls  # do not rebuild the scene on VTK's callback stack
    app.processEvents()
    assert len(calls) == 1
    assert window._plane_widget is widget
    assert len(getattr(window.plotter, "widgets", window.plotter).plane_widgets) == 1
    np.testing.assert_allclose(window.opts["clip_origin"], [0.9, 0.5, 0.5])
    np.testing.assert_allclose(list(window.plotter.camera_position), camera)
    assert window.opts["clip_args"]["cell_ids"]


def test_dialog_changes_keep_widget_and_disable_cancels_pending_redraw(
    clip_window, monkeypatch
):
    viewer, app, window = clip_window
    widget = window._plane_widget
    window._clip_dialog.originX.setValue(0.3)
    assert window._plane_widget is widget
    np.testing.assert_allclose(widget.GetOrigin(), [0.3, 0.5, 0.5])
    widget.InvokeEvent("EndInteractionEvent")
    assert window._clip_redraw_timer.isActive()
    window.disable_plane_widget()
    assert not window._clip_redraw_timer.isActive()
    app.processEvents()
    assert not widget.GetEnabled()


def test_switch_dock_discards_pending_interaction(clip_window):
    viewer, app, window = clip_window
    old_widget = window._plane_widget
    old_widget.InvokeEvent("EndInteractionEvent")
    assert window._clip_redraw_timer.isActive()
    window.add_dataset_dock(window.data.deepcopy(), "Other dataset")
    assert not window._clip_redraw_timer.isActive()
    assert window._plane_widget is not old_widget
    app.processEvents()
    assert not old_widget.GetEnabled()


def test_close_cancels_pending_clip_redraw(clip_window):
    viewer, app, window = clip_window
    window._plane_widget.InvokeEvent("EndInteractionEvent")
    assert window._clip_redraw_timer.isActive()
    window.close()
    assert not window._clip_redraw_timer.isActive()
    app.processEvents()
