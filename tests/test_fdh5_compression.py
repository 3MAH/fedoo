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
    "compressed", [None, False, True, 1, 9, "zstd", CompressionConfig("zstd", 9)]
)
def test_save_propagates_fdh5_compression(tmp_path, multi_frame, compressed):
    config = CompressionConfig.from_compressed(
        True if compressed is None else compressed
    )
    if config.compression == "zstd" and not zstd_available():
        pytest.skip("Optional hdf5plugin is not installed")
    data = _saved_data(tmp_path, multi_frame)
    filename = str(tmp_path / "saved.fdh5")
    if multi_frame:
        data.save_all(filename, **_save_options(compressed))
    else:
        data.save(filename, **_save_options(compressed))
    _check_compression(filename, compressed)


def test_zstd_missing_filter_preserves_existing_file(tmp_path, monkeypatch):
    from fedoo.util import fdh5

    data = _saved_data(tmp_path, False)
    filename = tmp_path / "existing.fdh5"
    data.to_fdh5(str(filename), overwrite=True)
    original = filename.read_bytes()
    monkeypatch.setattr(fdh5, "zstd_available", lambda: False)
    with pytest.raises(ImportError, match=r"pip install fedoo\[zstd\]"):
        data.to_fdh5(
            str(filename), overwrite=True, fdh5_options={"compression": "zstd"}
        )
    assert filename.read_bytes() == original


def test_zstd_reader_auto_registers_filter_and_explains_missing_dependency(
    tmp_path, monkeypatch
):
    from fedoo.util import fdh5

    if not zstd_available():
        pytest.skip("Optional hdf5plugin is not installed")
    data = _saved_data(tmp_path, False)
    filename = str(tmp_path / "zstd.fdh5")
    data.to_fdh5(filename, overwrite=True, fdh5_options={"compression": "zstd"})
    imported = []
    import_module = fdh5.importlib.import_module

    def register(name):
        imported.append(name)
        return import_module(name)

    monkeypatch.setattr(fdh5.h5py.h5z, "filter_avail", lambda filter_id: bool(imported))
    monkeypatch.setattr(fdh5.importlib, "import_module", register)
    np.testing.assert_array_equal(fd.read_data(filename)["Value"], np.arange(4))
    assert imported == ["hdf5plugin"]
    monkeypatch.setattr(fdh5, "zstd_available", lambda: False)
    with pytest.raises(ImportError, match=r"pip install fedoo\[zstd\]"):
        fd.read_data(filename)
    # A failed registration must release the reader's HDF5 handle.
    with h5py.File(filename, "r+"):
        pass
    gzip_file = str(tmp_path / "gzip.fdh5")
    data.to_fdh5(gzip_file)
    np.testing.assert_array_equal(fd.read_data(gzip_file)["Value"], np.arange(4))


@pytest.mark.parametrize(
    "codec,level", [("invalid", 3), ("gzip", 10), ("zstd", 0), ("zstd", 23)]
)
def test_invalid_compression_settings(codec, level):
    with pytest.raises(ValueError):
        CompressionConfig(codec, level)


def _save_options(compressed):
    if compressed is None or isinstance(compressed, bool):
        return {"compressed": compressed}
    config = CompressionConfig.from_compressed(compressed)
    return {
        "fdh5_options": {
            "compression": config.compression,
            "compression_level": config.compression_opts,
        }
    }
