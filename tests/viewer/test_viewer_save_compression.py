"""Save As compression applies to every FDH5 frame and mesh array."""

import h5py
import numpy as np
import pytest
import fedoo as fd
from fedoo.util.fdh5 import CompressionConfig, ZSTD_FILTER_ID, zstd_available


def _saved_data(tmp_path, multi_frame):
    mesh = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    data = fd.DataSet(mesh)
    data.node_data["Value"] = np.arange(mesh.n_nodes, dtype=float)
    if not multi_frame:
        return data
    source = str(tmp_path / "source.fdh5")
    data.to_fdh5(source, iteration=0, overwrite=True)
    data.node_data["Value"] += 10
    data.to_fdh5(source, iteration=1)
    return fd.read_data(source)


def _check_compression(filename, compressed):
    arrays = []
    config = CompressionConfig.from_compressed(
        True if compressed is None else compressed
    )
    with h5py.File(filename) as handle:

        def check(name, value):
            if (
                isinstance(value, h5py.Dataset)
                and value.shape
                and value.dtype.kind != "O"
            ):
                arrays.append(name)
                if config.compression == "zstd":
                    filters = value.id.get_create_plist()
                    assert filters.get_filter(0)[0] == ZSTD_FILTER_ID
                    assert filters.get_filter(0)[2] == (config.compression_opts,)
                else:
                    assert value.compression == config.compression
                    if config.compression == "gzip":
                        assert value.compression_opts == config.compression_opts

        handle.visititems(check)
    assert any(name.startswith("mesh/") for name in arrays)
    assert any(name.startswith("results/") for name in arrays)


@pytest.mark.parametrize("multi_frame", [False, True])
@pytest.mark.parametrize(
    "compressed",
    [False, 1, 9, CompressionConfig("zstd", 3), CompressionConfig("zstd", 9)],
)
def test_save_as_compression_controls(tmp_path, monkeypatch, multi_frame, compressed):
    config = CompressionConfig.from_compressed(compressed)
    if config.compression == "zstd" and not zstd_available():
        pytest.skip("Optional hdf5plugin is not installed")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    data = _saved_data(tmp_path, multi_frame)
    # Recompress the source file itself to check that later frames survive.
    filename = str(tmp_path / ("source.fdh5" if multi_frame else "saved.fdh5"))
    window = viewer.MainWindow(data)
    monkeypatch.setattr(viewer.QtWidgets.QMessageBox, "information", lambda *a: None)

    def accept(dialog):
        combo = dialog.findChild(viewer.QtWidgets.QComboBox, "fdh5_compression")
        level = dialog.findChild(viewer.QtWidgets.QSpinBox, "fdh5_compression_level")
        assert combo.currentData() == "gzip"
        assert level.value() == 4
        config = CompressionConfig.from_compressed(compressed)
        combo.setCurrentIndex(combo.findData(config.compression))
        assert level.isEnabled() == (compressed is not False)
        if compressed is not False:
            assert level.maximum() == (22 if config.compression == "zstd" else 9)
            level.setValue(config.compression_opts)
        dialog.selectFile(filename)
        return viewer.QtWidgets.QDialog.Accepted

    monkeypatch.setattr(viewer.QtWidgets.QFileDialog, "exec_", accept)
    try:
        window.save_current_data_as()
        _check_compression(filename, compressed)
        saved_options = window.active_dock.opts["fdh5_options"]
        assert saved_options["compression"] == config.compression
        if config.compression is not None:
            assert saved_options["compression_level"] == config.compression_opts

        def check_remembered(dialog):
            combo = dialog.findChild(viewer.QtWidgets.QComboBox, "fdh5_compression")
            level = dialog.findChild(
                viewer.QtWidgets.QSpinBox, "fdh5_compression_level"
            )
            assert combo.currentData() == config.compression
            if config.compression is not None:
                assert level.value() == config.compression_opts
            return viewer.QtWidgets.QDialog.Rejected

        monkeypatch.setattr(viewer.QtWidgets.QFileDialog, "exec_", check_remembered)
        window.save_current_data_as()
        saved = fd.read_data(filename)
        if multi_frame:
            assert saved.n_iter == 2
            saved.load(1)
            np.testing.assert_allclose(saved["Value"], np.arange(4) + 10)
            saved.load(0)
        np.testing.assert_allclose(saved["Value"], np.arange(4))
    finally:
        window.close()
        app.processEvents()


def test_save_as_disables_unavailable_zstd(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    monkeypatch.setattr(viewer, "zstd_available", lambda: False)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    window = viewer.MainWindow(_saved_data(tmp_path, False))

    def cancel(dialog):
        combo = dialog.findChild(viewer.QtWidgets.QComboBox, "fdh5_compression")
        assert combo.currentData() == "gzip"
        file_menu = next(
            action.menu()
            for action in window.menuBar().actions()
            if action.text() == "File"
        )
        actions = file_menu.actions()
        assert not any(action.text() == "Save" for action in actions)
        save_as = next(action for action in actions if action.text() == "Save as...")
        assert "Ctrl+S" in [shortcut.toString() for shortcut in save_as.shortcuts()]
        assert not dialog.testOption(viewer.QtWidgets.QFileDialog.DontConfirmOverwrite)
        assert not combo.model().item(combo.findData("zstd")).isEnabled()
        return viewer.QtWidgets.QDialog.Rejected

    monkeypatch.setattr(viewer.QtWidgets.QFileDialog, "exec_", cancel)
    try:
        window.save_current_data_as()
    finally:
        window.close()
        app.processEvents()


def test_cancel_save_as_does_not_write(tmp_path, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    window = viewer.MainWindow(_saved_data(tmp_path, False))
    monkeypatch.setattr(
        viewer.QtWidgets.QFileDialog,
        "exec_",
        lambda dialog: viewer.QtWidgets.QDialog.Rejected,
    )
    monkeypatch.setattr(
        window, "_save_data_to_fdh5", lambda *a, **k: pytest.fail("Save cancelled")
    )
    try:
        window.save_current_data_as()
        assert "fdh5_compression" not in window.active_dock.opts
    finally:
        window.close()
        app.processEvents()
