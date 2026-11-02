"""Displayed shell points must retain physical node ids and displayed values."""

from types import SimpleNamespace
import numpy as np
import pytest
import fedoo as fd


@pytest.mark.parametrize("data_type", ["Node", "Element", "GaussPoint"])
def test_top_bottom_picks_use_displayed_values(data_type):
    viewer = pytest.importorskip("fedoo.util.viewer")
    pv = pytest.importorskip("pyvista")
    surface = fd.mesh.rectangle_mesh(3, 2, elm_type="quad4", ndim=3)
    # Start with an unused node so compact display indices differ from node ids.
    mesh = fd.Mesh(
        np.vstack(([[10, 10, 10]], surface.nodes)), surface.elements + 1, "quad4"
    )
    data = fd.DataSet(mesh)
    data.field_metadata["_ShellSection"] = {"family": "shell", "thickness": 0.2}
    values = {
        "Node": 10.0 * np.arange(mesh.n_nodes),
        "Element": np.array([11.0, 22.0]),
        "GaussPoint": np.tile([11.0, 22.0], 4),
    }[data_type]
    data.dict_data[data_type]["Value"] = values
    dock = SimpleNamespace(
        data=data,
        opts={"clip_args": None, "scale": 1},
        current_field="Value",
        current_data_type=data_type,
    )
    plotter = pv.Plotter(off_screen=True)
    try:
        data.plot(
            "Value",
            data_type=data_type,
            shell_thickness=True,
            show=False,
            plotter=plotter,
        )
        displayed = pv.wrap(plotter.actors["data1"].GetMapper().GetInput())
        picked = []
        for bottom in range(0, displayed.n_points, 2):
            pair = []
            for index in (bottom, bottom + 1):
                picker = SimpleNamespace(
                    GetDataSet=lambda: displayed, GetPointId=lambda: index
                )
                node_id, value, association = viewer._picked_node_value(
                    dock, displayed.points[index], picker
                )
                assert node_id > 0
                if data_type == "Node":
                    np.testing.assert_allclose(value, values[node_id])
                elif data_type == "GaussPoint":
                    assert np.isclose(value, 11) or np.isclose(value, 22)
                else:
                    assert association == "adjacent elements"
                    assert set(value).issubset({11, 22})
                pair.append((node_id, value))
                picked.append((node_id, value))
            assert pair[0][0] == pair[1][0]
            np.testing.assert_allclose(pair[0][1], pair[1][1])
        if data_type == "GaussPoint":
            shared = np.intersect1d(mesh.elements[0], mesh.elements[1])[0]
            # Read the clicked side of a discontinuous field, rather than its
            # averaged nodal value of 16.5.
            np.testing.assert_allclose(
                sorted(set(value for node, value in picked if node == shared)), [11, 22]
            )
    finally:
        plotter.close()
