"""FDH5 storage policies change disk arrays, never computation arrays."""

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import fedoo as fd
from fedoo.util.fdh5 import FDH5Reader, resolve_fdh5_options


@pytest.fixture(autouse=True)
def defaults():
    previous = fd.get_fdh5_options()
    fd.reset_fdh5_options()
    yield
    fd.set_fdh5_options(**previous)


def data():
    mesh = fd.mesh.rectangle_mesh(3, 2, elm_type="quad4", ndim=3)
    mesh.node_sets["left"] = np.array([0, 1], dtype=np.int64)
    mesh.element_sets["first"] = np.array([0], dtype=np.int64)
    result = fd.DataSet(mesh)
    result.node_data["Disp"] = np.linspace(0, 1, 18).reshape(3, 6)
    result.element_data["IDs"] = np.array([-1, 40000], dtype=np.int64)
    result.scalar_data["Time"] = 0.123456789012345
    result.scalar_data["driver"] = np.array([0.123456789012345])
    return result


def test_storage_defaults_and_overrides(tmp_path):
    result = data()
    original = result["Disp"].copy()
    path = tmp_path / "compact.fdh5"
    result.save(path)
    with h5py.File(path) as f:
        assert f["mesh/nodes"].dtype == np.float64
        assert f["mesh/submesh_0/elements"].dtype == np.int32
        assert f["mesh/node_sets/left"].dtype == np.int32
        assert f["results/iter_0/node_data/Disp"].dtype == np.float32
        assert f["results/iter_0/scalars/Time"].dtype == np.float64
        assert f["results/iter_0/scalars/driver"].dtype == np.float64
    assert result["Disp"].dtype == np.float64
    np.testing.assert_array_equal(result["Disp"], original)
    loaded = fd.read_data(path)
    assert loaded["Disp"].dtype == np.float32
    result.save(
        path,
        fdh5_options={
            "float_dtype": None,
            "coordinate_dtype": "float32",
            "integer_min_bits": 16,
        },
    )
    with h5py.File(path) as f:
        assert f["mesh/nodes"].dtype == np.float32
        assert f["mesh/submesh_0/elements"].dtype == np.int16
        assert f["results/iter_0/node_data/Disp"].dtype == np.float64
        assert f["results/iter_0/element_data/submesh_0/IDs"].dtype == np.int32


@pytest.mark.parametrize(
    "dtype,values,expected",
    [
        ("int64", [-32768, 32767], "int16"),
        ("int64", [-32769, 32768], "int32"),
        ("int64", [-1, 2**40], "int64"),
        ("uint64", [0, 65535], "uint16"),
        ("uint64", [0, 65536], "uint32"),
        ("uint64", [0, 2**63], "uint64"),
        ("int8", [-1, 1], "int8"),
    ],
)
def test_lossless_integer_ranges(dtype, values, expected):
    from fedoo.util.fdh5 import _storage_array

    array = np.array(values, dtype=dtype)
    packed = _storage_array(array, {"integer_min_bits": 16})
    assert packed.dtype == np.dtype(expected)
    np.testing.assert_array_equal(array, packed)
    assert _storage_array(array, {"integer_min_bits": None}) is array


def test_global_defaults_copy_reset_and_boolean_shorthand(tmp_path):
    fd.set_fdh5_options(compression=None, float_dtype=None)
    assert resolve_fdh5_options()["compression"] is None
    assert resolve_fdh5_options(compressed=True)["compression"] == "gzip"
    assert resolve_fdh5_options(compressed=True)["compression_level"] == 4
    assert resolve_fdh5_options(compressed=False)["compression"] is None
    copy = fd.get_fdh5_options()
    copy["float_dtype"] = "float32"
    assert fd.get_fdh5_options()["float_dtype"] is None
    result = data()
    result.save(tmp_path / "none.fdh5")
    with h5py.File(tmp_path / "none.fdh5") as f:
        assert f["mesh/nodes"].compression is None
    result.save(tmp_path / "gzip.fdh5", compressed=True)
    with h5py.File(tmp_path / "gzip.fdh5") as f:
        assert f["mesh/nodes"].compression == "gzip"
    fd.reset_fdh5_options()
    assert fd.get_fdh5_options()["float_dtype"] == "float32"


@pytest.mark.parametrize(
    "options",
    [
        {"typo": True},
        {"float_dtype": "float16"},
        {"integer_min_bits": True},
        {"integer_min_bits": 12},
        {"integer_min_bits": 32.0},
        {"compression": "bad"},
        {"compression": "zstd", "compression_level": 0},
        {"local_frame": "bad"},
    ],
)
def test_invalid_policy_does_not_replace_file(tmp_path, options):
    path = tmp_path / "existing.fdh5"
    path.write_bytes(b"preserve me")
    with pytest.raises(ValueError):
        data().save(path, fdh5_options=options)
    assert path.read_bytes() == b"preserve me"


def test_conflicting_compression_and_unsupported_shortcuts():
    with pytest.raises(ValueError, match="either"):
        resolve_fdh5_options({"compression": "gzip"}, True)
    with pytest.raises(TypeError, match="compressed"):
        resolve_fdh5_options(compressed=4)
    assert resolve_fdh5_options({"float_dtype": None}, True)["float_dtype"] is None
    fd.set_fdh5_options(compression="zstd")
    assert fd.get_fdh5_options()["compression_level"] == 3


@pytest.mark.parametrize(
    "association", ["node_data", "element_data", "gausspoint_data"]
)
@pytest.mark.parametrize("encoding", ["matrix", "quaternion"])
def test_frame_encoding_roundtrip(tmp_path, association, encoding):
    result = data()
    count = 6 if association == "node_data" else 2
    frames = (
        Rotation.from_rotvec([[0.1, 0.2, 0.3]] * count).as_matrix().reshape(-1, 9).T
    )
    if association == "gausspoint_data":
        frames = np.tile(frames, (1, 4))
    getattr(result, association)["ShellLocalFrame"] = frames
    path = tmp_path / "frames.fdh5"
    result.save(path, fdh5_options={"local_frame": encoding, "float_dtype": None})
    subgroup = "" if association == "node_data" else "submesh_0/"
    with h5py.File(path) as f:
        stored = f[f"results/iter_0/{association}/{subgroup}ShellLocalFrame"]
        assert stored.shape[0] == (4 if encoding == "quaternion" else 9)
        if encoding == "quaternion" and association == "gausspoint_data":
            assert stored.shape == (4, 2)
            assert stored.attrs["frame_gp_repeats"] == 4
    loaded = fd.read_data(path)
    np.testing.assert_allclose(
        getattr(loaded, association)["ShellLocalFrame"], frames, atol=1e-14
    )
    reader = FDH5Reader(path)
    with reader.open() as handle:
        lazy = reader.read_iteration(0, lazy=True, file=handle)[association]
        values = (
            lazy["ShellLocalFrame"]
            if association == "node_data"
            else lazy["submesh_0"]["ShellLocalFrame"]
        )
        np.testing.assert_allclose(np.asarray(values), frames, atol=1e-14)


def test_invalid_frames_retained_without_projection(tmp_path):
    result = data()
    result.element_data["ShellLocalFrame"] = np.ones((9, 2))
    path = tmp_path / "invalid.fdh5"
    result.save(path)
    with h5py.File(path) as f:
        ds = f["results/iter_0/element_data/submesh_0/ShellLocalFrame"]
        assert ds.shape == (9, 2)
        assert "frame_encoding" not in ds.attrs


def test_registered_output_captures_defaults(tmp_path):
    mesh = data().mesh
    space = fd.ModelingSpace("3D")
    law = fd.constitutivelaw.ElasticIsotrop(1000, 0.3)
    assembly = fd.Assembly.create(fd.weakform.StressEquilibrium(law, space=space), mesh)
    pb = fd.problem.Linear(assembly)
    fd.set_fdh5_options(compression=None, float_dtype=None)
    path = tmp_path / "registered.fdh5"
    pb.add_output(path, ["Disp"])
    fd.set_fdh5_options(compression="gzip", float_dtype="float32")
    pb.set_X(np.arange(pb.n_dof, dtype=float))
    pb.save_results()
    with h5py.File(path) as f:
        ds = f["results/iter_0/node_data/Disp"]
        assert ds.dtype == np.float64
        assert ds.compression is None


@pytest.mark.parametrize("dtype", [int, float])
def test_integer_orientation_and_varying_gauss_frames(tmp_path, dtype):
    result = data()
    rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=dtype)
    frames = np.tile(rotation.reshape(9, 1), (1, 8))
    frames[:, 2:4] = np.eye(3).reshape(9, 1)
    result.gausspoint_data["BeamLocalFrame"] = frames
    path = tmp_path / "varying.fdh5"
    result.save(path)
    loaded = fd.read_data(path)
    np.testing.assert_allclose(loaded["BeamLocalFrame"], frames, atol=1e-7)
    with h5py.File(path) as f:
        assert f.attrs["version"] == "1.1"
        ds = f["results/iter_0/gausspoint_data/submesh_0/BeamLocalFrame"]
        assert ds.shape == (4, 8)
        assert ds.dtype == np.float32
        assert ds.attrs["frame_gp_repeats"] == 1


def test_multiframe_policy_and_shared_file_validation(tmp_path):
    result = data()
    source = tmp_path / "source.fdh5"
    result.to_fdh5(
        source, iteration=0, overwrite=True, fdh5_options={"float_dtype": None}
    )
    result.node_data["Disp"] *= 2
    result.to_fdh5(source, iteration=1, fdh5_options={"float_dtype": None})
    frames = fd.read_data(source)
    target = tmp_path / "target.fdh5"
    frames.save_all(
        str(target),
        fdh5_options={"compression": None, "float_dtype": None, "integer_min_bits": 16},
    )
    with h5py.File(target) as f:
        assert f["mesh/submesh_0/elements"].dtype == np.int16
        for index in (0, 1):
            assert f[f"results/iter_{index}/node_data/Disp"].dtype == np.float64
            assert f[f"results/iter_{index}/node_data/Disp"].compression is None
