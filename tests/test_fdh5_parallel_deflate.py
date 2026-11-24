"""Parallel deflate/inflate of FDH5 chunks: same file, same arrays."""

import filecmp
import zlib
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import fedoo as fd
from fedoo.util import fdh5
from fedoo.util.fdh5 import CompressionConfig, FDH5Reader, FDH5Writer


ARRAYS = {
    # smooth float fields, an edge chunk on the last axis, several ranks
    "vector": np.cumsum(np.linspace(-1, 1, 3 * 5001)).reshape(3, 5001),
    "tensor": np.sin(np.arange(6 * 4 * 3001) * 1e-3).reshape(6, 4, 3001),
    "scalar_field": np.cos(np.arange(20011) * 1e-2),
    "transposed": np.arange(4 * 9001, dtype=float).reshape(9001, 4).T,
    "int64": np.arange(3 * 7001, dtype=np.int64).reshape(3, 7001),
    "int32": np.arange(20001, dtype=np.int32),
    # integers and nearly incompressible floats
    "random_bytes": np.random.default_rng(0).integers(
        0, 2**63, size=(2, 9001), dtype=np.int64
    ),
    "random_float": np.random.default_rng(1).random((2, 20001)),
    # below the threshold: left to HDF5
    "small": np.arange(10.0),
}


@pytest.fixture
def n_threads():
    config = fd.get_config()
    previous = config["N_THREADS"]
    yield lambda n: config.__setitem__("N_THREADS", n)
    config["N_THREADS"] = previous


def _same_zlib(tmp_path):
    """Whether Python's zlib deflates like HDF5's filter (same library/level).

    Only then are the parallel chunks byte-identical to HDF5's; otherwise
    (h5py wheels with their own zlib, libdeflate...) they differ but remain
    valid deflate streams.
    """
    value = np.cos(np.arange(4096) * 1e-2)
    with h5py.File(tmp_path / "zlib_probe.h5", "w") as f:
        ds = f.create_dataset("x", data=value, chunks=(4096,), compression="gzip")
        _, stored = ds.id.read_direct_chunk((0,))
    return stored == zlib.compress(value.tobytes(), 4)


def _bare_writer(compression):
    writer = FDH5Writer.__new__(FDH5Writer)
    writer.compression = compression
    # dtypes kept: the arrays are compared with what was given
    writer.options = dict(fdh5.get_fdh5_options(), integer_min_bits=None)
    writer._compression_kwargs = compression.dataset_kwargs()
    return writer


def _write(path, compression):
    # No FDH5 header: its creation time would differ between two files.
    writer = _bare_writer(compression)
    with h5py.File(path, "w") as f:
        group = f.create_group("x")
        for name, value in ARRAYS.items():
            writer._create_dataset(group, name, value)


@pytest.mark.parametrize("level", [True, 1, 9])
def test_parallel_deflate_writes_the_same_file(tmp_path, n_threads, level, monkeypatch):
    compression = CompressionConfig.from_compressed(level)
    serial, parallel = tmp_path / "serial.h5", tmp_path / "parallel.h5"
    calls = []
    deflated = fdh5._create_deflated_dataset
    monkeypatch.setattr(
        fdh5,
        "_create_deflated_dataset",
        lambda *args: calls.append(args[1]) or deflated(*args),
    )
    n_threads(1)
    _write(serial, compression)
    assert not calls
    n_threads(4)
    _write(parallel, compression)
    assert calls
    for path in (serial, parallel):
        with h5py.File(path) as f:
            for name, value in ARRAYS.items():
                np.testing.assert_array_equal(f["x"][name][...], value)
                assert f["x"][name].dtype == value.dtype
                assert f["x"][name].compression_opts == compression.compression_opts
    if _same_zlib(tmp_path):  # else the bytes may differ
        assert filecmp.cmp(serial, parallel, shallow=False)


@pytest.mark.parametrize("chunks", [None, True, False, (7, 3, 1000)])
def test_parallel_deflate_chunk_options(tmp_path, n_threads, chunks):
    compression = CompressionConfig("gzip", 4, chunks=chunks)
    data = np.random.default_rng(0).random((20, 6, 2500))
    writer = _bare_writer(compression)
    n_threads(4)
    with h5py.File(tmp_path / "chunks.h5", "w") as f:
        try:
            ds = writer._create_dataset(f, "a", data)
        except ValueError:  # h5py refuses gzip without chunks
            assert chunks is False
            return
        np.testing.assert_array_equal(ds[...], data)
        if isinstance(chunks, tuple):
            assert ds.chunks == chunks


@pytest.mark.parametrize("compressed", [True, False])
def test_parallel_inflate_reads_the_same_arrays(tmp_path, n_threads, compressed):
    path = tmp_path / "data.h5"
    n_threads(1)
    _write(path, CompressionConfig.from_compressed(compressed))
    n_threads(4)
    with FDH5Reader(path).open() as f:
        tree = {name: f["x"][name] for name in ARRAYS}
        for name, value in ARRAYS.items():
            assert fdh5._inflate_ok(tree[name], 4) == (
                compressed
                and value.dtype.kind == "f"
                and value.nbytes >= fdh5._PARALLEL_INFLATE_MIN_BYTES
            )
        read = fdh5._materialize_hdf5(tree)
    for name, value in ARRAYS.items():
        np.testing.assert_array_equal(read[name], value)
        assert read[name].dtype == value.dtype
        assert read[name].shape == value.shape


def test_parallel_inflate_falls_back_on_other_filters(tmp_path, n_threads):
    path = tmp_path / "filters.h5"
    value = ARRAYS["tensor"].reshape(24, -1)
    with h5py.File(path, "w") as f:
        f.create_dataset("shuffle", data=value, compression="gzip", shuffle=True)
        f.create_dataset("lzf", data=value, compression="lzf")
        partial = f.create_dataset(
            "partial",
            shape=value.shape,
            dtype=float,
            chunks=(1, 3001),
            compression="gzip",
        )
        partial[0] = value[0]  # unallocated chunks keep the fill value
    n_threads(4)
    with h5py.File(path) as f:
        assert not fdh5._inflate_ok(f["shuffle"], 4)
        assert not fdh5._inflate_ok(f["lzf"], 4)
        read = fdh5._materialize_hdf5({k: f[k] for k in f})
        np.testing.assert_array_equal(read["partial"], f["partial"][...])
    np.testing.assert_array_equal(read["shuffle"], value)
    np.testing.assert_array_equal(read["lzf"], value)
    np.testing.assert_array_equal(read["partial"][0], value[0])
    np.testing.assert_array_equal(read["partial"][1:], 0)


def test_parallel_inflate_skipped_filter_chunk(tmp_path, n_threads):
    """A chunk whose deflate filter was skipped (mask bit 0) is read raw."""
    path = tmp_path / "mask.h5"
    value = np.arange(2 * 20000, dtype=float).reshape(2, 20000)
    with h5py.File(path, "w") as f:
        ds = f.create_dataset(
            "x", shape=value.shape, dtype=float, chunks=(1, 20000), compression="gzip"
        )
        ds.id.write_direct_chunk((0, 0), fdh5.zlib.compress(value[0].tobytes()))
        ds.id.write_direct_chunk((1, 0), value[1].tobytes(), filter_mask=1)
    n_threads(4)
    with h5py.File(path) as f:
        assert fdh5._inflate_ok(f["x"], 4)
        np.testing.assert_array_equal(fdh5._materialize_hdf5(f["x"]), value)


def test_dataset_roundtrip_threads(tmp_path, n_threads):
    """A DataSet written and read with several threads matches the serial file."""
    mesh = fd.mesh.box_mesh(12, 12, 12)
    data = fd.DataSet(mesh)
    n_gp = mesh.n_elements * 8
    data.node_data["Disp"] = np.sin(mesh.nodes.T * 3.0)
    data.gausspoint_data["Stress"] = np.cos(np.arange(6 * n_gp) * 1e-2).reshape(6, -1)
    data.gausspoint_data["Statev"] = np.arange(12 * n_gp, dtype=float).reshape(12, -1)
    data.scalar_data["Time"] = 0.5
    n_threads(1)
    data.to_fdh5(tmp_path / "serial.fdh5", overwrite=True)
    n_threads(4)
    data.to_fdh5(tmp_path / "parallel.fdh5", overwrite=True)
    for name in ("serial", "parallel"):
        read = fd.read_data(str(tmp_path / f"{name}.fdh5"))
        for field in ("Stress", "Statev"):
            stored = read.gausspoint_data[field]
            np.testing.assert_array_equal(
                stored, data.gausspoint_data[field].astype(stored.dtype)
            )
        stored = read.node_data["Disp"]
        np.testing.assert_array_equal(
            stored, data.node_data["Disp"].astype(stored.dtype)
        )
    if not _same_zlib(tmp_path):
        return
    with (
        FDH5Reader(tmp_path / "serial.fdh5").open() as a,
        FDH5Reader(tmp_path / "parallel.fdh5").open() as b,
    ):
        for field in ("Stress", "Statev"):
            path = f"results/iter_0/gausspoint_data/submesh_0/{field}"
            for i in range(a[path].id.get_num_chunks()):
                offset = a[path].id.get_chunk_info(i).chunk_offset
                assert a[path].id.read_direct_chunk(offset) == b[
                    path
                ].id.read_direct_chunk(offset)


def test_output_writer_reuse(tmp_path):
    """Frames written through add_output reuse one FDH5Writer per file; a file
    removed between two frames is written again with its header and mesh."""
    fd.ModelingSpace("2Dstress")
    mesh = fd.mesh.rectangle_mesh(5, 5)
    material = fd.constitutivelaw.ElasticIsotrop(200e3, 0.3)
    wf = fd.weakform.StressEquilibrium(material)
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.Linear(assembly)
    pb.bc.add("Dirichlet", "left", "Disp", 0)
    pb.bc.add("Dirichlet", "right", "DispX", 0.1)
    pb.solve()
    filename = tmp_path / "frames"
    pb.add_output(filename, assembly, ["Disp", "Stress"])
    pb.save_results(0)
    writer = pb._problem_output._multiframe_files[str(filename) + ".fdh5"]["writer"]
    pb.save_results(1)
    state = pb._problem_output._multiframe_files[str(filename) + ".fdh5"]
    assert state["writer"] is writer
    (tmp_path / "frames.fdh5").unlink()
    pb.save_results(2)
    assert state["writer"] is not writer
    with h5py.File(tmp_path / "frames.fdh5") as f:
        assert f.attrs["format"] == "FDH5"
        assert "mesh" in f
    data = fd.read_data(str(tmp_path / "frames.fdh5"))
    stored = data.node_data["Disp"]
    np.testing.assert_array_equal(stored, pb.get_disp().astype(stored.dtype))


def test_parallel_deflate_failure_links_no_dataset(tmp_path, n_threads, monkeypatch):
    """A compression error raises, as h5py does, and leaves no dataset."""
    path = tmp_path / "failure.h5"
    calls = []

    def failing_compress(buffer, level):
        calls.append(level)
        if len(calls) == 3:
            raise zlib.error("simulated failure")
        return zlib.compress(buffer, level)

    writer = _bare_writer(CompressionConfig())
    n_threads(4)
    with h5py.File(path, "w") as f:
        writer._create_dataset(f, "kept", ARRAYS["tensor"])
        monkeypatch.setattr(
            fdh5, "zlib", SimpleNamespace(compress=failing_compress, error=zlib.error)
        )
        with pytest.raises(zlib.error, match="simulated failure"):
            writer._create_dataset(f, "failed", ARRAYS["tensor"])
        monkeypatch.undo()
        assert "failed" not in f
        assert len(calls) >= 3
        # the group stays usable: the same name can be written again
        writer._create_dataset(f, "failed", ARRAYS["tensor"])
    with h5py.File(path) as f:
        assert set(f) == {"kept", "failed"}
        for name in f:
            np.testing.assert_array_equal(f[name][...], ARRAYS["tensor"])


def test_thread_pools_are_kept_per_size():
    """Changing N_THREADS never shuts down a pool another thread may use."""
    pool2 = fdh5._executor(2)
    assert fdh5._executor(2) is pool2
    pool3 = fdh5._executor(3)
    assert pool3 is not pool2
    assert pool2.submit(lambda: 1).result() == 1
    assert pool3.submit(lambda: 2).result() == 2
