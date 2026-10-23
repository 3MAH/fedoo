"""Optional HDF5 Zstandard preserves multimesh fields and mixed-codec histories."""

import numpy as np
import pytest

import fedoo as fd
from fedoo.util.fdh5 import CompressionConfig, FDH5Reader, zstd_available


@pytest.mark.skipif(not zstd_available(), reason="Optional hdf5plugin is not installed")
def test_zstd_multimesh_lazy_reads_and_mixed_codec_append(tmp_path):
    surface = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    mesh = fd.MultiMesh(
        surface.nodes,
        {"quad4": surface.elements, "tri3": surface.elements[:, :3]},
    )
    data = fd.DataSet(mesh)
    data.node_data["Disp"] = np.arange(12, dtype=float).reshape(3, 4)
    data.element_data["ElementValue"] = {0: np.array([1.0]), 1: np.array([2.0])}
    data.gausspoint_data["GPValue"] = {0: np.arange(4.0), 1: np.arange(3.0)}
    data.scalar_data.update(Time=0.0, driver_dof=np.array([1.0, 2.0, 3.0]))
    data.field_metadata["GPValue"] = {"description": "Zstandard round trip"}
    filename = str(tmp_path / "multi.fdh5")
    data.to_fdh5(filename, compressed=CompressionConfig("zstd", 5), overwrite=True)
    data.scalar_data["Time"] = 1.0
    data.to_fdh5(filename, iteration=1, compressed=True)
    reader = FDH5Reader(filename)
    with reader.open() as handle:
        assert handle.attrs["fedoo_requires_zstd"]
        frame = reader.read_iteration(0, lazy=True, file=handle)
        np.testing.assert_array_equal(frame["node_data"]["Disp"][...], data["Disp"])
        for sid, expected in (
            ("submesh_0", np.arange(4.0)),
            ("submesh_1", np.arange(3.0)),
        ):
            np.testing.assert_array_equal(
                frame["gausspoint_data"][sid]["GPValue"][...], expected
            )
        np.testing.assert_array_equal(
            frame["scalars"]["driver_dof"][...], [1.0, 2.0, 3.0]
        )
        assert (
            frame["field_metadata"]["GPValue"]["description"] == "Zstandard round trip"
        )
        assert handle["results/iter_1/node_data/Disp"].compression == "gzip"
    loaded = fd.read_data(filename)
    np.testing.assert_array_equal(loaded.get_history("Time"), [0.0, 1.0])


def test_missing_zstd_rejected_when_registering_simulation_output(
    tmp_path, monkeypatch
):
    from fedoo.core.output import _ProblemOutput
    from fedoo.util import fdh5

    monkeypatch.setattr(fdh5, "zstd_available", lambda: False)
    output = _ProblemOutput()
    with pytest.raises(ImportError, match="hdf5plugin"):
        output.add_output(
            str(tmp_path / "missing.fdh5"), None, ["Disp"], compressed="zstd"
        )
    assert not (tmp_path / "missing.fdh5").exists()
