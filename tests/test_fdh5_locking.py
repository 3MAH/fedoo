"""FDH5 lock contention must not abort ordinary live-viewer output."""

import errno
import multiprocessing
import os
import threading

import h5py
import numpy as np
import pytest

from fedoo.util import fdh5


@pytest.mark.parametrize(
    "error",
    [
        OSError(
            0,
            "Unable to synchronously open file (unable to lock file, Win32 GetLastError() = 33)",
        ),
        BlockingIOError(errno.EAGAIN, "Resource temporarily unavailable"),
    ],
)
def test_retry_only_temporary_lock_conflicts(monkeypatch, error):
    attempts = []
    handle = object()

    def open_file(*args):
        attempts.append(args)
        if len(attempts) < 3:
            raise error
        return handle

    monkeypatch.setattr(fdh5.h5py, "File", open_file)
    assert fdh5._open_hdf5("result.fdh5", "a", 1) is handle
    assert len(attempts) == 3


def test_permanent_io_errors_are_not_retried(monkeypatch):
    error = OSError("file signature not found")

    def fail(*args):
        raise error

    monkeypatch.setattr(fdh5.h5py, "File", fail)
    monkeypatch.setattr(fdh5._time, "sleep", lambda _: pytest.fail("Unexpected retry"))
    with pytest.raises(OSError) as caught:
        fdh5._open_hdf5("result.fdh5", "a", 1)
    assert caught.value is error


def test_persistent_lock_has_bounded_timeout(monkeypatch):
    error = OSError(0, "unable to lock file")
    attempts = []

    def fail(*args):
        attempts.append(args)
        raise error

    monkeypatch.setattr(fdh5.h5py, "File", fail)
    with pytest.raises(OSError) as caught:
        fdh5._open_hdf5("result.fdh5", "a", 0.02)
    assert caught.value is error
    assert len(attempts) >= 2


def test_frame_read_is_one_snapshot_and_releases_its_lock(tmp_path, monkeypatch):
    filename = tmp_path / "snapshot.fdh5"
    writer = fdh5.FDH5Writer(filename)
    writer.write_mesh(np.zeros((2, 3)))
    writer.write_iteration(0, node_data={"Value": np.array([1.0, 2.0])})
    reader = fdh5.FDH5Reader(filename)
    opens = []
    original = h5py.File

    def counted(path, mode):
        opens.append(mode)
        return original(path, mode)

    monkeypatch.setattr(fdh5.h5py, "File", counted)
    frame = reader.read_iteration(0)
    assert opens == ["r"]
    np.testing.assert_array_equal(frame["node_data"]["Value"], [1, 2])
    assert isinstance(frame["node_data"]["Value"], np.ndarray)
    writer.write_iteration(1)
    assert reader.list_iterations() == [0, 1]


def _hold_reader(filename, ready, release):
    with h5py.File(filename, "r"):
        ready.set()
        release.wait(10)


@pytest.mark.skipif(
    os.environ.get("HDF5_USE_FILE_LOCKING", "").upper() in ("FALSE", "0"),
    reason="HDF5 locks explicitly disabled",
)
def test_writer_waits_for_viewer_in_another_process(tmp_path):
    filename = str(tmp_path / "live.fdh5")
    writer = fdh5.FDH5Writer(filename, lock_timeout=5)
    writer.write_mesh(np.zeros((2, 3)))
    writer.write_iteration(0)
    context = multiprocessing.get_context("spawn")
    ready, release = context.Event(), context.Event()
    process = context.Process(target=_hold_reader, args=(filename, ready, release))
    process.start()
    timer = None
    try:
        assert ready.wait(15), "Reader process did not acquire its lock"
        # Prove the reader holds a conflicting lock before exercising retry.
        with pytest.raises(OSError):
            fdh5._open_hdf5(filename, "a", 0)
        timer = threading.Timer(0.2, release.set)
        timer.start()
        writer.write_iteration(1, scalars={"Time": 1.0})
        assert fdh5.FDH5Reader(filename).list_iterations() == [0, 1]
    finally:
        release.set()
        if timer is not None:
            timer.cancel()
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join()
    assert process.exitcode == 0
