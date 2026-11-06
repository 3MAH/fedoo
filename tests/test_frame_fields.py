"""Metadata-driven orientation fields and material output."""

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import fedoo as fd
from fedoo.util.frame_fields import frame_matrices, is_frame_field


def _data():
    mesh = fd.mesh.rectangle_mesh(3, 2, elm_type="quad4", ndim=3)
    return fd.DataSet(mesh)


@pytest.mark.parametrize("input_type", ["matrix", "quaternion", "rotvec"])
@pytest.mark.parametrize("storage", ["matrix", "quaternion", "rotvec"])
@pytest.mark.parametrize(
    "association", ["node_data", "element_data", "gausspoint_data"]
)
def test_named_frame_input_and_storage_roundtrip(
    tmp_path, input_type, storage, association
):
    data = _data()
    count = {"node_data": 6, "element_data": 2, "gausspoint_data": 8}[association]
    rotation = Rotation.from_rotvec(
        np.column_stack(
            [np.linspace(0, 0.3, count), np.zeros(count), np.linspace(0, 3.5, count)]
        )
    )
    values = {
        "matrix": rotation.as_matrix().reshape(-1, 9).T,
        "quaternion": rotation.as_quat().T,
        "rotvec": rotation.as_rotvec().T,
    }[input_type]
    getattr(data, association)["FiberOrientation"] = values
    data.field_metadata["FiberOrientation"] = {
        "type": "local_frame",
        "description": "fibre basis",
    }
    path = str(tmp_path / "frame.fdh5")
    data.save(path, fdh5_options={"local_frame": storage, "float_dtype": None})
    prefix = "" if association == "node_data" else "submesh_0/"
    with h5py.File(path) as f:
        ds = f[f"results/iter_0/{association}/{prefix}FiberOrientation"]
        assert ds.shape == ({"matrix": 9, "quaternion": 4, "rotvec": 3}[storage], count)
        assert (
            ds.attrs["frame_encoding"]
            == {
                "matrix": "matrix-rows",
                "quaternion": "quaternion-xyzw-rows",
                "rotvec": "rotvec-radians-rows",
            }[storage]
        )
    loaded = fd.read_data(path)
    assert loaded.field_metadata["FiberOrientation"]["type"] == "local_frame"
    np.testing.assert_allclose(
        getattr(loaded, association)["FiberOrientation"],
        rotation.as_matrix().reshape(-1, 9).T,
        atol=1e-14,
    )
    np.testing.assert_array_equal(
        getattr(data, association)["FiberOrientation"], values
    )


def test_shapes_alone_are_not_local_frame_tags(tmp_path):
    data = _data()
    for components in (3, 4, 9):
        data.node_data[f"Ordinary{components}"] = (
            np.arange(components * 6).reshape(components, 6).astype(float)
        )
    path = str(tmp_path / "ordinary.fdh5")
    data.save(path, fdh5_options={"local_frame": "rotvec"})
    with h5py.File(path) as f:
        for name in data.node_data:
            assert "frame_encoding" not in f["results/iter_0/node_data/" + name].attrs
    assert not is_frame_field("Ordinary3", data.field_metadata)


def _material_problem(material, space=None):
    space = space or fd.ModelingSpace("3D")
    mesh = fd.Mesh(
        np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]]),
        np.array([[0, 1, 2, 3]]),
        "tet4",
    )
    assembly = fd.Assembly.create(
        fd.weakform.StressEquilibrium(material, space=space), mesh
    )
    return fd.problem.Linear(assembly), assembly


def test_material_frame_requested_only_and_row_convention(tmp_path):
    material = fd.constitutivelaw.ElasticAnisotropic(np.eye(6))
    rotation = Rotation.from_rotvec([0, 0, np.pi / 3])
    material.set_local_frame(rotation)
    problem, assembly = _material_problem(material)
    plain = problem.get_results(assembly, [])
    assert "MaterialLocalFrame" not in plain.field_names()
    data = problem.get_results(assembly, ["MaterialLocalFrame"])
    expected = np.broadcast_to(rotation.as_matrix().T, (assembly.n_gauss_points, 3, 3))
    np.testing.assert_allclose(data["MaterialLocalFrame"].T.reshape(-1, 3, 3), expected)
    assert data.field_metadata["MaterialLocalFrame"] == {"type": "local_frame"}
    assert "_BeamSection" not in data.field_metadata
    path = str(tmp_path / "material.fdh5")
    data.save(path, fdh5_options={"local_frame": "rotvec"})
    np.testing.assert_allclose(
        fd.read_data(path)["MaterialLocalFrame"], expected.reshape(-1, 9).T, atol=1e-7
    )


def test_material_frame_uses_current_orientation_and_identity(monkeypatch):
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    problem, assembly = _material_problem(material)
    data = problem.get_results(assembly, ["MaterialLocalFrame"])
    np.testing.assert_allclose(
        frame_matrices(data["MaterialLocalFrame"]),
        np.broadcast_to(np.eye(3), (assembly.n_gauss_points, 3, 3)),
    )
    current = Rotation.from_rotvec([0.2, 0.4, 0.6]).as_matrix()[None]
    monkeypatch.setattr(material, "get_current_local_frame", lambda a: current)
    data = problem.get_results(assembly, ["MaterialLocalFrame"])
    np.testing.assert_allclose(
        frame_matrices(data["MaterialLocalFrame"]),
        np.broadcast_to(current.transpose(0, 2, 1), (assembly.n_gauss_points, 3, 3)),
    )


def test_multimesh_frame_metadata(tmp_path):
    base = _data()
    mesh = fd.MultiMesh(
        base.mesh.nodes,
        {"quad4": base.mesh.elements, "tri3": base.mesh.elements[:, :3]},
    )
    data = fd.DataSet(mesh)
    values = Rotation.from_rotvec([[0, 0, 0.3], [0, 0, 0.7]]).as_quat().T
    data.element_data["Fibre"] = {1: values}
    data.field_metadata["Fibre"] = {"submeshes": {"1": {"type": "local_frame"}}}
    path = str(tmp_path / "mixed.fdh5")
    data.save(path, fdh5_options={"local_frame": "rotvec"})
    loaded = fd.read_data(path)
    np.testing.assert_allclose(
        loaded.element_data["Fibre"][1],
        frame_matrices(values).reshape(-1, 9).T,
        atol=1e-7,
    )
    assert is_frame_field("Fibre", loaded.field_metadata)


@pytest.mark.parametrize("encoding", ["quaternion", "rotvec"])
def test_repeated_rotations_and_encoding_attributes(tmp_path, encoding):
    data = _data()
    matrices = Rotation.from_rotvec([[0, 0, 0.4], [0.1, 0.2, 0.3]]).as_matrix()
    data.gausspoint_data["Fibres"] = np.tile(matrices.reshape(-1, 9).T, (1, 4))
    data.field_metadata["Fibres"] = {"type": "local_frame"}
    path = str(tmp_path / "repeat.fdh5")
    data.save(path, fdh5_options={"local_frame": encoding, "float_dtype": None})
    with h5py.File(path) as f:
        field = f["results/iter_0/gausspoint_data/submesh_0/Fibres"]
        assert field.shape == (4 if encoding == "quaternion" else 3, 2)
        assert field.attrs["frame_gp_repeats"] == 4
    np.testing.assert_allclose(fd.read_data(path)["Fibres"], data["Fibres"], atol=1e-14)


def test_material_frame_metadata_on_assemblysum():
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    _, first = _material_problem(material)
    _, second = _material_problem(material, first.space)
    assembly = fd.Assembly.sum(first, second)
    problem = fd.problem.Linear(assembly)
    result = problem.get_results(assembly, ["MaterialLocalFrame"])
    assert is_frame_field("MaterialLocalFrame", result.field_metadata)
    assert "_BeamSection" not in result.field_metadata
    assert "_ShellSection" not in result.field_metadata
    assert set(result.gausspoint_data["MaterialLocalFrame"]) == {0, 1}


def test_encoded_frames_validate_declared_encoding(tmp_path):
    from fedoo.util.fdh5 import FDH5Reader

    data = _data()
    data.element_data["Fibres"] = np.zeros((3, 2))
    data.field_metadata["Fibres"] = {"type": "local_frame"}
    path = str(tmp_path / "encoding.fdh5")
    data.save(path, fdh5_options={"local_frame": "rotvec"})
    with h5py.File(path, "r+") as f:
        ds = f["results/iter_0/element_data/submesh_0/Fibres"]
        ds.attrs["frame_encoding"] = "quaternion-xyzw-rows"
    with pytest.raises(ValueError, match="Invalid shape"):
        FDH5Reader(path).read_iteration(0)
