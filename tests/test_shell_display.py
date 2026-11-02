"""Live frame discovery and display-only surface thickness regressions."""

import numpy as np
import pytest
import fedoo as fd


@pytest.mark.parametrize("elm_type", ["tri3", "quad4"])
def test_thickened_display_preserves_fields_and_cell_selection(elm_type):
    pytest.importorskip("pyvista")
    from fedoo.core.dataset import _thicken_plot_mesh

    mesh = fd.mesh.rectangle_mesh(3, 2, elm_type=elm_type, ndim=3)
    meshplot = mesh.to_pyvista()
    # Tilt the deformed midsurface: offsets must follow its current normal.
    meshplot.points[:, 2] = meshplot.points[:, 0]
    original = meshplot.points.copy()
    meshplot.point_data["Value"] = np.arange(mesh.n_nodes)
    meshplot.cell_data["Value"] = np.arange(mesh.n_elements)
    meshplot.cell_data["_fedoo_global_cell_ids"] = np.arange(mesh.n_elements) + 7
    volume = _thicken_plot_mesh(meshplot, elm_type, 0.2)
    np.testing.assert_allclose(
        (volume.points[0::2] + volume.points[1::2]) / 2, original
    )
    np.testing.assert_allclose(
        np.linalg.norm(volume.points[1::2] - volume.points[0::2], axis=1), 0.2
    )
    np.testing.assert_array_equal(
        volume.point_data["Value"], np.repeat(np.arange(mesh.n_nodes), 2)
    )
    np.testing.assert_array_equal(volume.cell_data["Value"], np.arange(mesh.n_elements))
    np.testing.assert_array_equal(
        volume.cell_data["_fedoo_global_cell_ids"], np.arange(mesh.n_elements) + 7
    )
    np.testing.assert_array_equal(meshplot.points, original)


@pytest.mark.parametrize("data_type", ["Node", "Element", "GaussPoint"])
def test_thickened_plot_with_selected_cells(data_type):
    pv = pytest.importorskip("pyvista")
    mesh = fd.mesh.rectangle_mesh(3, 2, elm_type="quad4", ndim=3)
    data = fd.DataSet(mesh)
    data.field_metadata["_ShellSection"] = {"family": "shell", "thickness": 0.2}
    size = {
        "Node": mesh.n_nodes,
        "Element": mesh.n_elements,
        "GaussPoint": 4 * mesh.n_elements,
    }[data_type]
    data.dict_data[data_type]["Value"] = np.arange(size, dtype=float)
    original = mesh.nodes.copy()
    plotter = pv.Plotter(off_screen=True)
    try:
        data.plot(
            "Value",
            data_type=data_type,
            shell_thickness=True,
            element_set=[0],
            plotter=plotter,
            show=False,
        )
        displayed = pv.wrap(plotter.actors["data1"].GetMapper().GetInput())
        assert displayed.n_cells == 1
        np.testing.assert_allclose(displayed.bounds[4:], [-0.1, 0.1])
    finally:
        plotter.close()
    np.testing.assert_array_equal(mesh.nodes, original)


def test_thickness_in_mixed_mesh_keeps_beams_and_element_ids():
    pv = pytest.importorskip("pyvista")
    surface = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    beam = fd.Mesh(surface.nodes, np.array([[0, 1]]), "lin2")
    data = fd.DataSet(fd.MultiMesh.from_mesh_list([surface, beam]))
    data.field_metadata["_ShellSection"] = {
        "submeshes": {"0": {"family": "shell", "thickness": 0.2}}
    }
    plotter = pv.Plotter(off_screen=True)
    try:
        data.plot(shell_thickness=True, plotter=plotter, show=False, name="data")
        displayed = [
            pv.wrap(plotter.actors[f"data_{i}"].GetMapper().GetInput())
            for i in range(2)
        ]
        assert displayed[0].n_points == 2 * surface.n_nodes
        np.testing.assert_allclose(displayed[0].bounds[4:], [-0.1, 0.1])
        np.testing.assert_allclose(displayed[1].bounds[4:], [0, 0])
        np.testing.assert_array_equal(
            displayed[0].cell_data["_fedoo_global_cell_ids"], [0]
        )
        np.testing.assert_array_equal(
            displayed[1].cell_data["_fedoo_global_cell_ids"], [1]
        )
    finally:
        plotter.close()


def test_shell_display_ignores_unused_shared_nodes():
    pytest.importorskip("pyvista")
    from fedoo.core.dataset import _thicken_plot_mesh

    surface = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    mesh = fd.Mesh(
        np.vstack((surface.nodes, [[10, 10, 10]])), surface.elements, "quad4"
    )
    displayed = _thicken_plot_mesh(mesh.to_pyvista(), "quad4", 0.2)
    assert displayed.n_cells == 1
    assert displayed.n_points == 8


@pytest.mark.parametrize("data_type", ["Node", "GaussPoint"])
def test_display_uses_varying_saved_shell_thickness(data_type):
    pv = pytest.importorskip("pyvista")
    nodes = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [3, 0, 0],
            [4, 0, 0],
            [4, 1, 0],
            [3, 1, 0],
        ],
        dtype=float,
    )
    mesh = fd.Mesh(nodes, np.arange(8).reshape(2, 4), "quad4")
    data = fd.DataSet(mesh)
    data.dict_data[data_type]["Value"] = np.arange(8, dtype=float)
    data.field_metadata["_ShellSection"] = {
        "family": "shell",
        "thickness": np.tile([0.2, 0.6], 4),
        "n_elm_gp": 4,
    }
    plotter = pv.Plotter(off_screen=True)
    try:
        data.plot(
            "Value",
            data_type=data_type,
            shell_thickness=True,
            plotter=plotter,
            show=False,
        )
        displayed = pv.wrap(plotter.actors["data1"].GetMapper().GetInput())
        widths = np.linalg.norm(displayed.points[1::2] - displayed.points[0::2], axis=1)
        np.testing.assert_allclose(widths[displayed.points[0::2, 0] < 2], 0.2)
        np.testing.assert_allclose(widths[displayed.points[0::2, 0] > 2], 0.6)
    finally:
        plotter.close()
