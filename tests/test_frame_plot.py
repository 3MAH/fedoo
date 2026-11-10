"""Local-frame overlays share the regular DataSet plot and need no Qt."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import fedoo as fd
from fedoo.util.frame_fields import frame_axis_mask

pv = pytest.importorskip("pyvista")


@pytest.fixture
def plotter(monkeypatch):
    monkeypatch.setattr(pv, "OFF_SCREEN", True)
    pl = pv.Plotter(off_screen=True)
    yield pl
    pl.close()


def data():
    mesh = fd.mesh.rectangle_mesh(3, 2, elm_type="quad4", ndim=3)
    result = fd.DataSet(mesh)
    result.node_data["Stress"] = np.tile(
        [[10.0], [0.0], [0.0], [0.0], [0.0], [0.0]], (1, mesh.n_nodes)
    )
    result.gausspoint_data["_ShellLocalFrame"] = np.tile(
        np.eye(3).reshape(9, 1), (1, mesh.n_elements * 4)
    )
    return result


def test_scalar_field_and_private_frame_overlay(plotter):
    result = data()
    result.set_current_local_frame("ShellLocalFrame")
    result.plot(
        field="Stress",
        component="vm",
        show_local_frame=True,
        local_frame_axis=3,
        plotter=plotter,
        show=False,
        scale=0,
    )
    assert "_local_frame_2" in plotter.actors
    assert "_local_frame_0" not in plotter.actors
    assert "_local_frame_legend" not in plotter.actors
    displayed = plotter.actors["data1"].mapper.dataset
    np.testing.assert_allclose(displayed.point_data["Data"], 10)


def test_selected_axes_and_legend(plotter):
    result = data()
    result.plot(
        show_local_frame=True,
        local_frame_axis=[1, 3],
        local_frame_options={"legend": True},
        plotter=plotter,
        show=False,
    )
    assert "_local_frame_0" in plotter.actors
    assert "_local_frame_1" not in plotter.actors
    assert "_local_frame_2" in plotter.actors
    assert "_local_frame_legend" in plotter.actors


@pytest.mark.parametrize("representation", ["matrix", "quaternion", "rotvec"])
def test_raw_matrix_component_access_and_plot(plotter, representation):
    result = data()
    rotation = Rotation.from_rotvec(np.tile([0, 0, 0.3], (8, 1)))
    result.gausspoint_data["Fibres"] = {
        "matrix": rotation.as_matrix().reshape(-1, 9).T,
        "quaternion": rotation.as_quat().T,
        "rotvec": rotation.as_rotvec().T,
    }[representation]
    result.field_metadata["Fibres"] = {"type": "local_frame"}
    result.set_current_local_frame("Fibres")
    np.testing.assert_allclose(result.get_data("Fibres", "XX"), np.cos(0.3))
    np.testing.assert_allclose(result.get_data("Fibres", "XY", "Node"), -np.sin(0.3))
    result.plot(
        field="Fibres",
        component="XX",
        data_type="Node",
        show_local_frame=True,
        local_frame_axis=1,
        plotter=plotter,
        show=False,
    )
    assert "_local_frame_0" in plotter.actors
    np.testing.assert_allclose(
        plotter.actors["data1"].mapper.dataset.point_data["Data"], np.cos(0.3)
    )


def test_multimesh_overlay_matches_visible_cells(plotter, monkeypatch):
    from fedoo.util import frame_plot

    base = data()
    mesh = fd.MultiMesh(
        base.mesh.nodes,
        {"quad4": base.mesh.elements, "tri3": base.mesh.elements[:, :3]},
    )
    result = fd.DataSet(mesh)
    result.node_data = base.node_data
    result.element_data["Fibres"] = {0: np.zeros((3, 2)), 1: np.zeros((3, 2))}
    result.field_metadata["Fibres"] = {"type": "local_frame"}
    result.set_current_local_frame("Fibres")
    samples = []
    draw = frame_plot.plot_local_frames

    def record(data, pl, options, field=None, component=None):
        samples.append(frame_plot.frame_samples(data, options, field)[0])
        return draw(data, pl, options, field, component)

    monkeypatch.setattr(frame_plot, "plot_local_frames", record)
    result.plot(
        field=None,
        show_local_frame=True,
        selected_submeshes=[1],
        element_set=[1],
        global_element_set=True,
        plotter=plotter,
        show=False,
        scale=0,
    )
    # Global element 1 belongs to submesh 0 and is hidden by selected_submeshes.
    assert len(samples) == 1 and len(samples[0]) == 0
    result.plot(
        field="Stress",
        component="vm",
        show_local_frame=True,
        selected_submeshes=[1],
        element_set=[3],
        global_element_set=True,
        plotter=plotter,
        show=False,
        scale=0,
    )
    assert len(samples) == 2 and len(samples[1]) == 1
    np.testing.assert_allclose(samples[1][0], mesh[1].element_centers[1])


def test_overlay_exists_before_screenshot(plotter, monkeypatch):
    def capture(*args, **kwargs):
        assert "_local_frame_2" in plotter.actors

    monkeypatch.setattr(plotter, "screenshot", capture)
    data().plot(
        show_local_frame=True,
        local_frame_axis=3,
        plotter=plotter,
        show=False,
        screenshot="unused.png",
    )


@pytest.mark.parametrize("axis", [0, 4, True, [1, 4], 1.5])
def test_invalid_axis_rejected(axis):
    with pytest.raises(ValueError, match="local_frame_axis"):
        frame_axis_mask(axis)


@pytest.mark.parametrize("ndim", [2, 3])
def test_geometric_line_fallback(plotter, ndim):
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])[:, :ndim]
    result = fd.DataSet(fd.Mesh(points, np.array([[0, 1]]), "lin2"))
    result.plot(show_local_frame=True, plotter=plotter, show=False)
    assert "_local_frame_0" in plotter.actors
    assert "_local_frame_1" in plotter.actors
    assert ("_local_frame_2" in plotter.actors) == (ndim == 3)
    if ndim == 2:
        with pytest.raises(ValueError, match="Axis 3"):
            result.plot(
                show_local_frame=True, local_frame_axis=3, plotter=plotter, show=False
            )


def test_geometric_shell_and_normal_fallback(plotter):
    result = data()
    result.gausspoint_data.clear()
    result.plot(show_local_frame=True, plotter=plotter, show=False)
    assert all(f"_local_frame_{axis}" in plotter.actors for axis in range(3))
    plotter.clear()
    result.element_data["MaterialLocalFrame"] = np.tile(
        Rotation.from_rotvec([0, 1.0, 0]).as_matrix().reshape(9, 1),
        (1, result.mesh.n_elements),
    )
    result.set_current_local_frame("MaterialLocalFrame")
    result.plot(show_normals=2.0, plotter=plotter, show=False)
    assert "_local_frame_0" not in plotter.actors
    assert "_local_frame_2" in plotter.actors
    from fedoo.util.frame_plot import frame_samples

    _, frames = frame_samples(result, {"geometric_normals": True})
    np.testing.assert_allclose(frames[:, 2], np.tile([0, 0, 1], (2, 1)))


@pytest.mark.parametrize("ndim", [2, 3])
def test_full_dimension_geometry_has_no_fallback(plotter, ndim):
    if ndim == 2:
        mesh = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=2)
    else:
        mesh = fd.Mesh(
            np.array([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]]),
            np.array([[0, 1, 2, 3]]),
            "tet4",
        )
    result = fd.DataSet(mesh)
    result.plot(show_local_frame=True, plotter=plotter, show=False)
    assert not any(name.startswith("_local_frame_") for name in plotter.actors)
    result.element_data["MaterialLocalFrame"] = np.eye(3).reshape(9, 1)
    result.plot(show_local_frame=True, plotter=plotter, show=False)
    assert "_local_frame_0" in plotter.actors
    assert ("_local_frame_2" in plotter.actors) == (ndim == 3)


def test_selection_copies_and_percentage_size(plotter, monkeypatch):
    result = data().set_current_local_frame("ShellLocalFrame")
    assert result.copy().current_local_frame == "ShellLocalFrame"
    assert result.deepcopy().current_local_frame == "ShellLocalFrame"
    result.set_current_local_frame(None)
    assert result.current_local_frame is None
    from fedoo.util import frame_plot

    original = frame_plot.plot_local_frames
    sizes = []

    def record(data, pl, options, *args):
        sizes.append(options["frame_size"])
        return original(data, pl, options, *args)

    monkeypatch.setattr(frame_plot, "plot_local_frames", record)
    result.plot(show_local_frame=True, plotter=plotter, show=False)
    result.plot(show_local_frame=1.0, plotter=plotter, show=False)
    assert sizes == [2.0, 1.0]


def test_frame_selection_survives_iteration_changes(tmp_path):
    result = data()
    path = str(tmp_path / "selection.fdh5")
    result.to_fdh5(path, iteration=0, overwrite=True)
    result.to_fdh5(path, iteration=1)
    history = fd.read_data(path)
    history.set_current_local_frame("ShellLocalFrame")
    history.load(1)
    assert history.current_local_frame == "ShellLocalFrame"
    assert history.copy().current_local_frame == "ShellLocalFrame"
    assert history.deepcopy().current_local_frame == "ShellLocalFrame"


def test_geometric_gauss_points_and_private_normal_selection():
    from fedoo.util.frame_plot import frame_samples

    result = data()
    result.gausspoint_data.clear()
    origins, _ = frame_samples(result, {"frame_gauss_points": True})
    assert len(origins) == result.mesh.n_elements * 4
    result.gausspoint_data["_ShellLocalFrame"] = np.tile(
        Rotation.from_rotvec([0, 1, 0]).as_matrix().reshape(9, 1), (1, 8)
    )
    result.set_current_local_frame("ShellLocalFrame")
    _, frames = frame_samples(result, {"geometric_normals": True})
    np.testing.assert_allclose(frames[:, 2], np.tile([0, 0, 1], (2, 1)))


def test_mixed_mesh_auto_geometry_and_material():
    from fedoo.util.frame_plot import frame_samples

    base = data().mesh
    mesh = fd.MultiMesh(
        base.nodes,
        {
            "quad4": base.elements,
            "lin2": base.elements[:, :2],
            "tet4": base.elements[:1],
        },
    )
    result = fd.DataSet(mesh)
    origins, _ = frame_samples(result, {})
    assert len(origins) == 4  # two shells, two lines; no geometric volume frame
    result.element_data["MaterialLocalFrame"] = {2: np.eye(3).reshape(9, 1)}
    origins, _ = frame_samples(result, {})
    assert len(origins) == 5
