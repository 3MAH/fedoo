"""Connected exterior surface/edge selection and viewer node-set workflows."""

import numpy as np
import pytest

import fedoo as fd
from fedoo.lib_elements.element_list import get_node_elm_coordinates


def node_at(mesh, point):
    return int(np.flatnonzero(np.all(np.isclose(mesh.nodes, point), axis=1))[0])


def pick_normal(error, direction):
    direction = np.asarray(direction)
    return int(error.face_ids[np.argmax(error.normals @ direction)])


@pytest.mark.parametrize("kind", ["hex8", "hex20"])
def test_box_patch_includes_midside_nodes_and_reconstructs(kind):
    mesh = fd.mesh.box_mesh(3, 3, 3, elm_type=kind)
    seed = node_at(mesh, [0.5, 0.5, 1])
    selected = mesh.find_surface_nodes(seed, angle=0)
    np.testing.assert_array_equal(selected, mesh.find_nodes("Z", 1))
    surface = fd.mesh.extract_surface(mesh, node_set=selected, quad2tri=False)
    assert surface.n_elements == 4
    assert surface.elm_type == ("quad4" if kind == "hex8" else "quad8")
    assert len(selected) == (9 if kind == "hex8" else 21)
    all_surface_nodes = mesh.find_surface_nodes(seed, angle=90)
    assert len(all_surface_nodes) > len(selected)
    assert node_at(mesh, [0.5, 0.5, 0.5]) not in all_surface_nodes


@pytest.mark.parametrize("seed_type", ["Node", "Element"])
def test_corner_ambiguity_is_explicit(seed_type):
    mesh = fd.mesh.box_mesh(3, 3, 3)
    with pytest.raises(fd.mesh.AmbiguousSurfaceSeed) as error:
        mesh.find_surface_nodes(0, seed_type, 30)
    face = pick_normal(error.value, [0, 0, -1])
    selected = mesh.find_surface_nodes(0, seed_type, 30, seed_face=face)
    np.testing.assert_array_equal(selected, mesh.find_nodes("Z", 0))


def test_multimesh_interfaces_removed_and_global_element_seed():
    mesh = fd.mesh.box_mesh(4, 4, 4)
    # Split the original grid across two submeshes without renumbering nodes.
    blocks = [
        fd.Mesh(mesh.nodes, mesh.elements[::2], "hex8"),
        fd.Mesh(mesh.nodes, mesh.elements[1::2], "hex8"),
    ]
    multi = fd.MultiMesh.from_mesh_list(blocks, register_name=False)
    seed = node_at(mesh, [1 / 3, 1 / 3, 1])
    np.testing.assert_array_equal(
        multi.find_surface_nodes(seed, angle=0), mesh.find_nodes("Z", 1)
    )
    # The center element has no exterior face, even at a submesh interface.
    local = int(
        np.flatnonzero(np.all(np.isclose(blocks[1].element_centers, 0.5), axis=1))[0]
    )
    with pytest.raises(ValueError, match="exterior"):
        multi.find_surface_nodes(blocks[0].n_elements + local, "Element")
    global_seed = blocks[0].n_elements
    with pytest.raises(fd.mesh.AmbiguousSurfaceSeed) as error:
        multi.find_surface_nodes(global_seed, "Element", 0)
    face = int(error.value.face_ids[0])
    nodes = multi.find_surface_nodes(global_seed, "Element", 0, seed_face=face)
    assert set(blocks[1].elements[0]) & set(nodes)


@pytest.mark.parametrize("reverse", [False, True])
def test_shell_fold_threshold_and_reversed_winding(reverse):
    points = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 1], [2, 1, 1]]
    mesh = fd.Mesh(
        np.array(points, dtype=float), np.array([[0, 1, 2, 3], [1, 4, 5, 2]]), "quad4"
    )
    if reverse:
        mesh.elements[1] = mesh.elements[1, ::-1]
    np.testing.assert_array_equal(
        mesh.find_surface_nodes(0, "Element", 44), [0, 1, 2, 3]
    )
    np.testing.assert_array_equal(
        mesh.find_surface_nodes(0, "Element", 45), np.arange(6)
    )


def test_shell_vertex_contact_does_not_connect_faces():
    mesh = fd.Mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 2, 0.0]]),
        np.array([[0, 1, 2], [2, 3, 4]]),
        "tri3",
    )
    np.testing.assert_array_equal(mesh.find_surface_nodes(0, "Element", 180), [0, 1, 2])
    with pytest.raises(fd.mesh.AmbiguousSurfaceSeed):
        mesh.find_surface_nodes(2, angle=180)


def test_local_angle_grows_around_curved_surface():
    theta = np.arange(8) * np.pi / 4
    ring = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(8)))
    nodes = np.vstack((ring, ring + [0, 0, 1]))
    cells = np.array([[i, (i + 1) % 8, (i + 1) % 8 + 8, i + 8] for i in range(8)])
    mesh = fd.Mesh(nodes, cells, "quad4")
    np.testing.assert_array_equal(
        mesh.find_surface_nodes(0, "Element", 44), [0, 1, 8, 9]
    )
    np.testing.assert_array_equal(
        mesh.find_surface_nodes(0, "Element", 45), np.arange(16)
    )


def test_nonmanifold_edges_stop_traversal():
    mesh = fd.Mesh(
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0.5, 0.5, 1]]),
        np.array([[0, 1, 2], [1, 0, 3], [0, 1, 4]]),
        "tri3",
    )
    np.testing.assert_array_equal(mesh.find_surface_nodes(0, "Element", 180), [0, 1, 2])


@pytest.mark.parametrize("kind", ["quad4", "quad8", "quad9", "tri3", "tri6"])
def test_two_dimensional_boundary_edges(kind):
    mesh = fd.mesh.rectangle_mesh(5, 3, elm_type=kind)
    selected = mesh.find_surface_nodes(node_at(mesh, [0.5, 1]), angle=0)
    np.testing.assert_array_equal(selected, mesh.find_nodes("Y", 1))
    reconstructed = fd.mesh.extract_surface(mesh, node_set=selected)
    assert reconstructed.n_elements > 0
    assert np.allclose(mesh.nodes[np.unique(reconstructed.elements), 1], 1)


@pytest.mark.parametrize("kind", ["tet4", "tet10", "wed6", "wed15", "wed18"])
def test_solid_families_keep_face_nodes(kind):
    points = get_node_elm_coordinates(kind)
    mesh = fd.Mesh(points, np.arange(len(points))[None, :], kind)
    with pytest.raises(fd.mesh.AmbiguousSurfaceSeed) as error:
        mesh.find_surface_nodes(0, "Element", 0)
    face = int(error.value.face_ids[0])
    nodes = mesh.find_surface_nodes(0, "Element", 0, seed_face=face)
    assert len(nodes) == (3 if kind in ("tet4", "wed6") else 6)


@pytest.mark.parametrize(
    "options",
    [
        dict(seed=-1),
        dict(seed=100000),
        dict(seed=0.5),
        dict(seed=0, seed_type="Face"),
        dict(seed=0, angle=-1),
        dict(seed=0, angle=181),
        dict(seed=0, angle=np.nan),
        dict(seed=0, seed_face=100000),
    ],
)
def test_invalid_arguments(options):
    mesh = fd.mesh.box_mesh(3, 3, 3)
    with pytest.raises(ValueError):
        fd.mesh.find_surface_nodes(mesh, **options)


def test_interior_node_cannot_seed_surface():
    mesh = fd.mesh.box_mesh(3, 3, 3)
    with pytest.raises(ValueError, match="exterior"):
        mesh.find_surface_nodes(node_at(mesh, [0.5, 0.5, 0.5]))


def test_degenerate_face_fails_clearly():
    mesh = fd.Mesh(np.zeros((3, 3)), np.array([[0, 1, 2]]), "tri3")
    with pytest.raises(ValueError, match="Degenerate"):
        mesh.find_surface_nodes(0)


@pytest.fixture
def surface_window(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    mesh = fd.mesh.box_mesh(3, 3, 3)
    data = fd.DataSet(mesh)
    data.node_data["Value"] = np.arange(mesh.n_nodes, dtype=float)
    window = viewer.MainWindow(data)
    monkeypatch.setattr(viewer.QtWidgets.QMessageBox, "information", lambda *a: None)
    monkeypatch.setattr(
        viewer.QtWidgets.QMessageBox, "warning", lambda *a: pytest.fail(str(a[-1]))
    )
    try:
        yield viewer, window
    finally:
        window.close()
        app.processEvents()


def test_viewer_node_seed_and_save_set(surface_window, monkeypatch, tmp_path):
    viewer, window = surface_window
    window.open_node_sets_dialog()
    dialog = window.node_sets_dialog
    seed = node_at(window.data.mesh, [0.5, 0.5, 1])
    monkeypatch.setattr(window, "_pick_node", lambda on_pick: on_pick(seed))
    dialog._surface_seed_from_selection()
    assert dialog.surface_seed_id.value() == seed
    dialog._select_surface_nodes()
    assert dialog.current_selection == set(window.data.mesh.find_nodes("Z", 1))
    monkeypatch.setattr(
        viewer.QtWidgets.QInputDialog, "getText", lambda *a, **k: ("TopSurface", True)
    )
    dialog._save_selection_as_new_set()
    filename = str(tmp_path / "surface.fdh5")
    assert window._save_data_to_fdh5(filename)
    saved = fd.read_data(filename)
    np.testing.assert_array_equal(
        saved.mesh.node_sets["TopSurface"], sorted(dialog.current_selection)
    )


def test_viewer_node_seed_chooses_face_and_returns_nodes(surface_window, monkeypatch):
    viewer, window = surface_window
    window.open_element_sets_dialog()
    assert not hasattr(window.element_sets_dialog, "surface_seed_id")
    window.open_node_sets_dialog()
    dialog = window.node_sets_dialog
    assert not hasattr(dialog, "surface_seed_type")
    monkeypatch.setattr(
        viewer.QtWidgets.QInputDialog, "getItem", lambda *a, **k: (a[3][0], True)
    )
    dialog._select_surface_nodes()
    nodes = window.node_sets_dialog.current_selection
    assert len(nodes) == 9
    assert np.any(np.ptp(window.data.mesh.nodes[sorted(nodes)], axis=0) == 0)


def test_viewer_cancel_ambiguous_seed_keeps_selection(surface_window, monkeypatch):
    viewer, window = surface_window
    window.open_node_sets_dialog()
    dialog = window.node_sets_dialog
    dialog.update_selection({0})
    monkeypatch.setattr(
        viewer.QtWidgets.QInputDialog, "getItem", lambda *a, **k: ("", False)
    )
    dialog._select_surface_nodes()
    assert dialog.current_selection == {0}


def test_viewer_uses_deformed_geometry_without_mutating_source(
    surface_window, monkeypatch
):
    viewer, window = surface_window
    # Start with a flat shell and fold its second cell through the Disp field.
    nodes = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [2, 0, 0], [2, 1, 0.0]]
    )
    data = fd.DataSet(
        fd.Mesh(nodes.copy(), np.array([[0, 1, 2, 3], [1, 4, 5, 2]]), "quad4")
    )
    data.node_data["Disp"] = np.zeros((3, 6))
    data.node_data["Disp"][2, 4:] = 1
    window.add_dataset_dock(data, "Folded shell")
    window.open_node_sets_dialog()
    dialog = window.node_sets_dialog
    dialog.surface_mode.setCurrentIndex(dialog.surface_mode.findData("faces"))
    dialog.surface_seed_id.setValue(0)
    dialog.surface_angle.setValue(30)
    dialog.use_def_mesh.setChecked(True)
    dialog._select_surface_nodes()
    assert dialog.current_selection == {0, 1, 2, 3}
    dialog.use_def_mesh.setChecked(False)
    dialog._select_surface_nodes()
    assert dialog.current_selection == set(range(6))
    np.testing.assert_array_equal(data.mesh.nodes, nodes)


@pytest.mark.parametrize("kind", ["hex8", "hex20"])
def test_box_edge_excludes_surface_tessellation_and_keeps_midside_nodes(kind):
    mesh = fd.mesh.box_mesh(3, 3, 3, elm_type=kind)
    seed = node_at(mesh, [0.5, 0, 0])
    selected = mesh.find_edge_nodes(seed, angle=0)
    np.testing.assert_array_equal(selected, mesh.find_nodes("Y == 0 and Z == 0"))
    assert len(selected) == (3 if kind == "hex8" else 5)
    if kind == "hex20":
        midside = node_at(mesh, [0.25, 0, 0])
        np.testing.assert_array_equal(mesh.find_edge_nodes(midside), selected)
    for point in ([0.5, 0.5, 1], [0.5, 0.5, 0.5]):
        with pytest.raises(ValueError, match="not on an exterior feature edge"):
            mesh.find_edge_nodes(node_at(mesh, point))


def test_corner_seed_requires_edge_choice_and_stops_at_branch():
    mesh = fd.mesh.box_mesh(3, 3, 3)
    with pytest.raises(fd.mesh.AmbiguousEdgeSeed) as error:
        mesh.find_edge_nodes(0)
    assert len(error.value.edge_ids) == 3
    along_x = int(np.argmax(np.abs(error.value.directions[:, 0])))
    edge = int(error.value.edge_ids[along_x])
    selected = mesh.find_edge_nodes(0, angle=180, seed_edge=edge)
    np.testing.assert_array_equal(selected, mesh.find_nodes("Y == 0 and Z == 0"))


@pytest.mark.parametrize("kind", ["tet4", "tet10", "wed6", "wed15", "wed18"])
def test_other_solid_families_keep_high_order_edge_nodes(kind):
    points = get_node_elm_coordinates(kind)
    mesh = fd.Mesh(points, np.arange(len(points))[None, :], kind)
    with pytest.raises(fd.mesh.AmbiguousEdgeSeed) as error:
        mesh.find_edge_nodes(0)
    edge = int(error.value.edge_ids[0])
    nodes = mesh.find_edge_nodes(0, seed_edge=edge)
    assert len(nodes) == (2 if kind in ("tet4", "wed6") else 3)
    assert 0 in nodes


def test_multimesh_shared_solid_faces_do_not_become_feature_edges():
    mesh = fd.mesh.box_mesh(3, 3, 3)
    blocks = [
        fd.Mesh(mesh.nodes, mesh.elements[::2], "hex8"),
        fd.Mesh(mesh.nodes, mesh.elements[1::2], "hex8"),
    ]
    multi = fd.MultiMesh.from_mesh_list(blocks, register_name=False)
    seed = node_at(mesh, [0.5, 0, 0])
    np.testing.assert_array_equal(
        multi.find_edge_nodes(seed), mesh.find_edge_nodes(seed)
    )
    with pytest.raises(ValueError, match="not on an exterior feature edge"):
        multi.find_edge_nodes(node_at(mesh, [0.5, 0.5, 1]))


@pytest.mark.parametrize("kind", ["quad4", "quad8", "quad9"])
def test_shell_open_boundary_includes_midside_nodes(kind):
    mesh = fd.mesh.rectangle_mesh(5, 3, elm_type=kind, ndim=3)
    selected = mesh.find_edge_nodes(node_at(mesh, [0.5, 1, 0]), angle=0)
    np.testing.assert_array_equal(selected, mesh.find_nodes("Y", 1))


@pytest.mark.parametrize("reverse", [False, True])
def test_shell_crease_sharpness_threshold_and_winding(reverse):
    mesh = fd.mesh.rectangle_mesh(5, 3, ndim=3, elm_type="quad4")
    mesh.nodes[:, 2] = np.maximum(mesh.nodes[:, 0] - 0.5, 0) * 2
    if reverse:
        right = mesh.element_centers[:, 0] > 0.5
        mesh.elements[right] = mesh.elements[right, ::-1]
    seed = node_at(mesh, [0.5, 0.5, 0])
    np.testing.assert_array_equal(
        mesh.find_edge_nodes(seed, feature_angle=30), mesh.find_nodes("X", 0.5)
    )
    with pytest.raises(ValueError, match="sharpness threshold"):
        mesh.find_edge_nodes(seed, feature_angle=80)


def cylinder_shell():
    theta = np.arange(36) * 2 * np.pi / 36
    ring = np.column_stack((np.cos(theta), np.sin(theta), np.zeros(36)))
    nodes = np.vstack([ring + [0, 0, z] for z in (0.0, 0.5, 1.0)])
    cells = [
        [
            level * 36 + i,
            level * 36 + (i + 1) % 36,
            (level + 1) * 36 + (i + 1) % 36,
            (level + 1) * 36 + i,
        ]
        for level in range(2)
        for i in range(36)
    ]
    return fd.Mesh(nodes, np.asarray(cells), "quad4")


def test_curved_shell_rim_turn_limit_and_smooth_wall():
    mesh = cylinder_shell()
    top = np.arange(72, 108)
    np.testing.assert_array_equal(
        mesh.find_edge_nodes(72, angle=11, feature_angle=30), top
    )
    with pytest.raises(ValueError, match="not on an exterior feature edge"):
        mesh.find_edge_nodes(36, feature_angle=30)
    with pytest.raises(fd.mesh.AmbiguousEdgeSeed) as error:
        mesh.find_edge_nodes(72, angle=5)
    nodes = mesh.find_edge_nodes(72, angle=5, seed_edge=int(error.value.edge_ids[0]))
    assert len(nodes) == 2
    assert 72 in nodes


@pytest.mark.parametrize(
    "options",
    [
        dict(seed=-1),
        dict(seed=0.5),
        dict(seed=True),
        dict(seed=100000),
        dict(seed=0, angle=-1),
        dict(seed=0, angle=181),
        dict(seed=0, feature_angle=np.nan),
        dict(seed=0, feature_angle=181),
        dict(seed=0, seed_edge=100000),
    ],
)
def test_invalid_edge_arguments(options):
    mesh = fd.mesh.box_mesh(3, 3, 3)
    with pytest.raises(ValueError):
        fd.mesh.find_edge_nodes(mesh, **options)


def test_two_dimensional_mesh_uses_existing_boundary_selector():
    mesh = fd.mesh.rectangle_mesh(3, 3)
    with pytest.raises(ValueError, match="2D"):
        mesh.find_edge_nodes(0)


def test_edge_viewer_mode_selects_nodes_and_cancel_preserves_selection(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    mesh = fd.mesh.box_mesh(3, 3, 3, elm_type="hex20")
    data = fd.DataSet(mesh)
    data.node_data["Value"] = np.arange(mesh.n_nodes, dtype=float)
    window = viewer.MainWindow(data)
    monkeypatch.setattr(viewer.QtWidgets.QMessageBox, "information", lambda *a: None)
    try:
        window.open_node_sets_dialog()
        dialog = window.node_sets_dialog
        assert not hasattr(dialog, "surface_seed_type")
        dialog.surface_mode.setCurrentIndex(dialog.surface_mode.findData("edges"))
        assert not dialog.surface_feature_angle.isHidden()
        assert dialog.surface_angle_label.text() == "Turn limit:"
        assert "edge" in dialog.surface_group.title()
        seed = node_at(mesh, [0.5, 0, 0])
        monkeypatch.setattr(window, "_pick_node", lambda on_pick: on_pick(seed))
        dialog.surface_select_button.click()
        app.processEvents()
        dialog._select_surface_nodes()
        expected = set(mesh.find_nodes("Y == 0 and Z == 0"))
        assert dialog.current_selection == expected
        monkeypatch.setattr(
            viewer.QtWidgets.QInputDialog, "getText", lambda *a, **k: ("Edge", True)
        )
        dialog._save_selection_as_new_set()
        saved = str(tmp_path / "edge.fdh5")
        assert window._save_data_to_fdh5(saved)
        np.testing.assert_array_equal(
            fd.read_data(saved).mesh.node_sets["Edge"], sorted(expected)
        )
        dialog.surface_seed_id.setValue(0)
        monkeypatch.setattr(
            viewer.QtWidgets.QInputDialog, "getItem", lambda *a, **k: ("", False)
        )
        dialog._select_surface_nodes()
        assert dialog.current_selection == expected
        monkeypatch.setattr(
            viewer.QtWidgets.QInputDialog, "getItem", lambda *a, **k: (a[3][0], True)
        )
        dialog._select_surface_nodes()
        assert len(dialog.current_selection) == 5
        assert 0 in dialog.current_selection
        dialog.surface_mode.setCurrentIndex(dialog.surface_mode.findData("solid"))
        assert dialog.surface_feature_angle.isHidden()
        assert dialog.surface_angle_label.text() == "Angle limit:"
    finally:
        window.close()
        app.processEvents()
