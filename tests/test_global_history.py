"""Global histories use value indices, independent of mesh associations."""

from types import SimpleNamespace
import numpy as np
import pytest
import fedoo as fd


@pytest.fixture
def global_results(tmp_path):
    mesh = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    mesh = fd.MultiMesh(
        mesh.nodes, {"quad4": mesh.elements, "tri3": mesh.elements[:, :3]}
    )
    data = fd.DataSet(mesh)
    data.node_data["Value"] = np.arange(mesh.n_nodes, dtype=float)
    filename = str(tmp_path / "global_history.fdh5")
    for frame in range(2):
        data.scalar_data.update(
            Time=float(frame),
            driver_dof=np.arange(7, dtype=float) + frame,
            **{"Fext(driver_dof)": np.arange(21).reshape(3, 7) + frame},
        )
        data.to_fdh5(filename, iteration=frame, overwrite=frame == 0)
    results = fd.read_data(filename)
    results.load(0)
    return results


@pytest.mark.parametrize("dtype", ["Node", "Element", "GaussPoint"])
def test_global_history_ignores_mesh_conversion(global_results, dtype):
    data = global_results
    for field, component, expected in (
        ("Time", None, [0, 1]),
        ("driver_dof", 0, [6, 7]),
        ("Fext(driver_dof)", 2, [20, 21]),
    ):
        np.testing.assert_allclose(
            data.get_history(field, 6, component, dtype), expected
        )
        _, association = data.get_data(field, component, dtype, return_data_type=True)
        assert association == "Scalar"
