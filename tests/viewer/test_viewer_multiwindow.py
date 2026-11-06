"""Multiple result views keep layout, active selection and frame independence."""

import numpy as np
import pytest
import fedoo as fd


@pytest.fixture
def multiwindow(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    mesh = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    results = []
    for count in (3, 2, 1):
        path = str(tmp_path / f"frames_{count}.fdh5")
        data = fd.DataSet(mesh)
        for index in range(count):
            data.node_data["Value"] = np.full(mesh.n_nodes, index, dtype=float)
            data.to_fdh5(path, iteration=index, overwrite=index == 0)
        results.append(fd.read_data(path))
    window = viewer.MainWindow(results[0])
    window.resize(1400, 900)
    window.show()
    first = window.active_dock
    second = window.add_dataset_dock(results[1], "Two frames")
    third = window.add_dataset_dock(results[2], "Static")
    app.processEvents()
    try:
        yield app, window, (first, second, third)
    finally:
        window.close()
        app.processEvents()


@pytest.mark.parametrize("layout", ["horizontal", "vertical", "tabified"])
def test_multiple_views_layout_switch_close_and_duplicate(multiwindow, layout):
    app, window, docks = multiwindow
    first, second, third = docks
    window._set_active(first)
    getattr(window, f"_distribute_{layout}")()
    for _ in range(3):
        app.processEvents()
    assert set(window.all_docks) == set(docks)
    assert window.active_dock in docks
    if layout == "tabified":
        assert set(window._view_area.tabifiedDockWidgets(first)) == {second, third}
    else:
        assert all(dock.isVisible() for dock in docks)
        for index, dock in enumerate(docks):
            for other in docks[index + 1 :]:
                assert not dock.geometry().intersects(other.geometry())
        centers = [dock.geometry().center() for dock in docks]
        if layout == "horizontal":
            assert len({point.x() for point in centers}) == 3
            assert len({point.y() for point in centers}) == 1
        else:
            assert len({point.y() for point in centers}) == 3
            assert len({point.x() for point in centers}) == 1
    for dock in docks:
        window._set_active(dock)
        assert window.data is dock.data
        assert window.dock_selector_combo.currentIndex() == window.all_docks.index(dock)
    second.close()
    app.processEvents()
    assert set(window.all_docks) == {first, third}
    window._set_active(first)
    window.copy_active_dock()
    app.processEvents()
    copied = window.active_dock
    assert copied not in docks
    assert len(window.all_docks) == 3
    assert copied.data is not first.data
    assert window.window_layout == getattr(window, f"_distribute_{layout}")


@pytest.mark.parametrize("control", ["iter_slider", "iter_spin"])
def test_multiple_views_independent_and_linked_iterations(multiwindow, control):
    app, window, (first, second, third) = multiwindow
    for dock in (first, second, third):
        window._set_active(dock)
        window.iter_slider.setValue(0)
    window._set_active(first)
    getattr(window, control).setValue(1)
    assert window.iter_spin.value() == window.iter_slider.value() == 1
    assert [dock.current_iter for dock in (first, second, third)] == [1, 0, 0]
    np.testing.assert_allclose(first.data["Value"], 1)
    np.testing.assert_allclose(second.data["Value"], 0)
    window._set_active(second)
    assert window.iter_spin.value() == window.iter_slider.value() == 0
    window._set_active(first)
    assert window.iter_spin.value() == window.iter_slider.value() == 1
    window._toggle_sync_iter(True)
    getattr(window, control).setValue(0)
    getattr(window, control).setValue(1)
    assert [dock.current_iter for dock in (first, second, third)] == [1, 1, 0]
    np.testing.assert_allclose(second.data["Value"], 1)
    getattr(window, control).setValue(2)
    # Shorter results retain their last valid frame.
    assert [dock.current_iter for dock in (first, second, third)] == [2, 1, 0]
    np.testing.assert_allclose(first.data["Value"], 2)
    window._toggle_sync_iter(False)
    getattr(window, control).setValue(0)
    assert [dock.current_iter for dock in (first, second, third)] == [0, 1, 0]
