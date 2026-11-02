"""Optional inspector follows active results and disappears in fullscreen."""

import numpy as np
import pytest
import fedoo as fd


@pytest.fixture
def inspected_window(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    mesh = fd.mesh.rectangle_mesh(3, 2, elm_type="quad4", ndim=3)
    mesh.node_sets["Left"] = mesh.find_nodes("X", 0)
    mesh.element_sets["First"] = [0]
    data = fd.DataSet(mesh)
    data.node_data["Value"] = np.arange(mesh.n_nodes, dtype=float)
    data.node_data["Disp"] = np.zeros((3, mesh.n_nodes))
    data.element_data["Pressure"] = np.arange(mesh.n_elements, dtype=float)
    data.scalar_data["Time"] = 1.5
    data.add_scalar_expression("Total", "sum(Value)")
    window = viewer.MainWindow(data)
    window.resize(1100, 700)
    window.show()
    app.processEvents()
    try:
        yield viewer, app, window
    finally:
        window.close()
        app.processEvents()


def select(inspector, kind, key=""):
    from qtpy.QtWidgets import QTreeWidgetItemIterator
    from qtpy.QtCore import Qt

    iterator = QTreeWidgetItemIterator(inspector.tree)
    while iterator.value():
        item = iterator.value()
        if item.data(0, Qt.UserRole) == (kind, key):
            inspector.tree.setCurrentItem(item)
            return item
        iterator += 1
    pytest.fail(f"Inspector item missing: {kind}, {key}")


def properties(inspector):
    return {
        inspector.properties.item(row, 0).text(): inspector.properties.item(
            row, 1
        ).text()
        for row in range(inspector.properties.rowCount())
    }


def show_inspector(app, window):
    window.object_inspector_action.trigger()
    app.processEvents()
    assert window.object_inspector.isVisible()
    return window.object_inspector


def test_optional_inspector_mesh_properties_and_set_highlighting(inspected_window):
    viewer, app, window = inspected_window
    assert window.object_inspector.isHidden()
    assert not window.object_inspector_action.isChecked()
    inspector = show_inspector(app, window)
    assert inspector.geometry().right() < window.active_dock.geometry().left()
    assert inspector.width() < window.width() / 2
    assert inspector.width() == max(
        inspector.minimumWidth(), inspector.minimumSizeHint().width()
    )
    select(inspector, "mesh")
    assert properties(inspector)["Nodes"] == str(window.data.mesh.n_nodes)
    select(inspector, "nodes", "Left")
    assert properties(inspector)["Count"] == "2"
    assert "_inspector_highlight" in window.plotter.actors
    camera = list(window.plotter.camera_position)
    select(inspector, "elements", "First")
    assert properties(inspector)["Count"] == "1"
    assert list(window.plotter.camera_position) == camera
    inspector.open_manager()
    manager = window.element_sets_dialog
    assert manager._selected_set_name() == "First"
    inspector.hide()
    assert "_inspector_highlight" not in window.plotter.actors


def test_inspector_menu_and_minimum_width_after_window_resize(inspected_window):
    viewer, app, window = inspected_window
    menus = {action.text(): action.menu() for action in window.menuBar().actions()}
    assert window.object_inspector_action in menus["Windows"].actions()
    assert window.object_inspector_action not in menus["View"].actions()
    inspector = show_inspector(app, window)
    for width in (1500, 900, 1250):
        window.resize(width, 700)
        app.processEvents()
        app.processEvents()
        assert inspector.width() == max(
            inspector.minimumWidth(), inspector.minimumSizeHint().width()
        )
    inspector.setFloating(True)
    inspector.resize(450, 500)
    app.processEvents()
    window.resize(1400, 700)
    app.processEvents()
    app.processEvents()
    assert inspector.width() == 450


def test_field_properties_plot_and_expression_editor(inspected_window):
    viewer, app, window = inspected_window
    inspector = show_inspector(app, window)
    select(inspector, "field", ("Node", "Value"))
    assert properties(inspector)["Association"] == "Node"
    assert properties(inspector)["Stored range"] == "0 … 5"
    assert inspector.plot_button.isEnabled()
    inspector.plot_field()
    app.processEvents()
    assert window.active_dock.current_field == "Value"
    select(inspector, "field", ("Scalar", "Total"))
    assert properties(inspector)["Expression"] == "sum(Value)"
    assert not inspector.plot_button.isEnabled()
    inspector.open_manager()
    assert window.scalar_expression_dialog.name_edit.text() == "Total"


@pytest.mark.parametrize("floating", [False, True])
def test_fullscreen_hides_and_restores_visible_inspector(inspected_window, floating):
    viewer, app, window = inspected_window
    inspector = show_inspector(app, window)
    inspector.setFloating(floating)
    app.processEvents()
    window.toggle_fullscreen()
    app.processEvents()
    assert window.isFullScreen()
    assert inspector.isHidden()
    window.toggle_fullscreen()
    app.processEvents()
    assert inspector.isVisible()
    assert inspector.isFloating() == floating
    inspector.hide()
    window.toggle_fullscreen()
    window.toggle_fullscreen()
    app.processEvents()
    assert inspector.isHidden()


def test_inspector_updates_active_pane_sets_and_current_values(inspected_window):
    viewer, app, window = inspected_window
    inspector = show_inspector(app, window)
    select(inspector, "nodes", "Left")
    old_dock = window.active_dock
    second = old_dock.data.deepcopy()
    second.node_data["Value"] += 10
    second.mesh.node_sets.pop("Left")
    second.mesh.node_sets["Other"] = [1, 2, 3]
    new_dock = window.add_dataset_dock(second, "Other dataset")
    app.processEvents()
    window._set_active(new_dock)
    app.processEvents()
    app.processEvents()
    assert "Other dataset" in inspector.context_label.text()
    assert "_inspector_highlight" not in old_dock.plotter.actors
    select(inspector, "nodes", "Other")
    select(inspector, "field", ("Node", "Value"))
    assert properties(inspector)["Stored range"] == "10 … 15"
    window.data.node_data["Value"] += 1
    window.active_dock.update_plot()
    app.processEvents()
    assert properties(inspector)["Stored range"] == "11 … 16"
    window._ensure_dock_node_sets(window.active_dock)
    window.active_dock.node_sets["New"] = [0]
    window._sync_viewer_sets(window.active_dock)
    app.processEvents()
    select(inspector, "nodes", "New")


def test_inspector_live_boundary_conditions_and_global_dofs(inspected_window):
    viewer, app, window = inspected_window
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(3, 2, elm_type="quad4")
    assembly = fd.Assembly.create(
        fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3)),
        mesh,
    )
    problem = fd.problem.Linear(assembly)
    boundary = problem.bc.add("Dirichlet", "left", "DispX", 0, name="Fixed")
    problem.add_global_dof("Driver", 2)
    global_bc = problem.bc.add("Dirichlet", [0], "Driver", 0, name="Global")
    live_dock = window.add_dataset_dock(problem, "Live problem")
    app.processEvents()
    window._set_active(live_dock)
    app.processEvents()
    inspector = show_inspector(app, window)
    select(inspector, "bc", str(id(boundary)))
    assert properties(inspector)["Target"] == "left"
    assert "_inspector_highlight" in window.plotter.actors
    inspector.open_manager()
    assert window.boundary_conditions_dialog.editing[2] is boundary
    select(inspector, "bc", str(id(global_bc)))
    assert "Global DOFs" in properties(inspector)["Association"]
    assert "_inspector_highlight" not in window.plotter.actors


def test_empty_viewer_and_closed_last_pane(inspected_window):
    viewer, app, window = inspected_window
    inspector = show_inspector(app, window)
    window.active_dock.close()
    app.processEvents()
    assert window.active_dock is None
    assert inspector.context_label.text() == "No active dataset"
    assert inspector.tree.topLevelItemCount() == 0
    assert not inspector.edit_button.isEnabled()


def test_multimesh_field_range_includes_every_block(inspected_window):
    viewer, app, window = inspected_window
    source = window.data.mesh
    mesh = fd.MultiMesh.from_mesh_list(
        [
            fd.Mesh(source.nodes, source.elements[:1], "quad4", name="A"),
            fd.Mesh(source.nodes, source.elements[1:], "quad4", name="B"),
        ],
        register_name=False,
    )
    data = fd.DataSet(mesh)
    data.node_data["Value"] = np.arange(mesh.n_nodes, dtype=float)
    data.element_data["Pressure"] = fd.MultiMeshData(
        mesh, {0: np.array([-3.0]), 1: np.array([8.0])}
    )
    new_dock = window.add_dataset_dock(data, "MultiMesh")
    app.processEvents()
    window._set_active(new_dock)
    app.processEvents()
    inspector = show_inspector(app, window)
    select(inspector, "field", ("Element", "Pressure"))
    assert properties(inspector)["Stored range"] == "-3 … 8"
    window.active_dock.element_sets["Second"] = [1]
    window._sync_viewer_sets(window.active_dock)
    app.processEvents()
    select(inspector, "elements", "Second")
    actor = window.plotter.actors["_inspector_highlight"]
    assert actor.GetMapper().GetInput().GetNumberOfCells() == 1


def test_element_context_menu_show_only_mask_and_unmask(inspected_window):
    viewer, app, window = inspected_window
    inspector = show_inspector(app, window)
    select(inspector, "elements", "First")
    menu = inspector._visibility_menu()
    assert [action.text() for action in menu.actions()] == ["Show only", "Mask"]
    assert menu.actions()[1].isCheckable()
    assert not menu.actions()[1].isChecked()
    camera = list(window.plotter.camera_position)
    menu.actions()[0].trigger()
    app.processEvents()
    assert window.opts["element_set"] == [0]
    assert properties(inspector)["Visibility"] == "Visible"
    inspector._visibility_menu().actions()[1].trigger()
    app.processEvents()
    assert window.opts["element_set"] == []
    assert "_inspector_highlight" not in window.plotter.actors
    assert properties(inspector)["Visibility"] == "Masked"
    menu = inspector._visibility_menu()
    assert menu.actions()[1].isChecked()
    menu.actions()[1].trigger()
    app.processEvents()
    assert window.opts["element_set"] == [0]
    assert list(window.plotter.camera_position) == camera
    select(inspector, "mesh")
    inspector._visibility_menu().actions()[0].trigger()
    app.processEvents()
    assert window.opts["element_set"] is None


def test_mesh_context_menu_masks_and_restores_all_elements(inspected_window):
    viewer, app, window = inspected_window
    inspector = show_inspector(app, window)
    select(inspector, "mesh")
    inspector._visibility_menu().actions()[1].trigger()
    app.processEvents()
    assert window.opts["element_set"] == []
    assert properties(inspector)["Visibility"] == "Masked"
    menu = inspector._visibility_menu()
    assert menu.actions()[1].isChecked()
    menu.actions()[1].trigger()
    app.processEvents()
    assert window.opts["element_set"] is None
    assert properties(inspector)["Visibility"] == "Visible"


def test_mask_check_reflects_partial_and_empty_sets(inspected_window):
    viewer, app, window = inspected_window
    inspector = show_inspector(app, window)
    select(inspector, "elements", "First")
    inspector._visibility_menu().actions()[0].trigger()
    app.processEvents()
    select(inspector, "elements", "All")
    assert properties(inspector)["Visibility"].startswith("Partially masked")
    assert not inspector._visibility_menu().actions()[1].isChecked()
    window.active_dock.element_sets["Empty"] = []
    window._sync_viewer_sets(window.active_dock)
    app.processEvents()
    select(inspector, "elements", "Empty")
    assert all(
        not action.isEnabled() for action in inspector._visibility_menu().actions()
    )
    select(inspector, "nodes", "Left")
    assert inspector._visibility_menu() is None


def test_inspector_visibility_stays_in_sync_with_element_manager(inspected_window):
    viewer, app, window = inspected_window
    inspector = show_inspector(app, window)
    window.open_element_sets_dialog()
    manager = window.element_sets_dialog
    manager.update_selection({0})
    manager._apply_visibility("hide")
    app.processEvents()
    select(inspector, "elements", "First")
    menu = inspector._visibility_menu()
    assert menu.actions()[1].isChecked()
    menu.actions()[1].trigger()
    app.processEvents()
    assert window.opts["element_set"] is None
    assert manager.visibility_mode is None
    select(inspector, "elements", "First")
    inspector._visibility_menu().actions()[0].trigger()
    app.processEvents()
    assert manager.visibility_mode == "show"
    assert manager.current_selection == {0}


def test_multimesh_submesh_visibility_uses_global_ids(inspected_window):
    viewer, app, window = inspected_window
    original = window.data.mesh
    mesh = fd.MultiMesh.from_mesh_list(
        [
            fd.Mesh(original.nodes, original.elements[:1], "quad4", name="A"),
            fd.Mesh(original.nodes, original.elements[1:], "quad4", name="B"),
        ],
        register_name=False,
    )
    data = fd.DataSet(mesh)
    data.node_data["Value"] = np.arange(mesh.n_nodes, dtype=float)
    new_dock = window.add_dataset_dock(data, "Meshes")
    app.processEvents()
    window._set_active(new_dock)
    app.processEvents()
    inspector = show_inspector(app, window)
    select(inspector, "submesh", "1")
    inspector._visibility_menu().actions()[0].trigger()
    app.processEvents()
    assert window.opts["element_set"] == [1]
    select(inspector, "submesh", "0")
    assert inspector._visibility_menu().actions()[1].isChecked()
    assert "_inspector_highlight" not in window.plotter.actors
    inspector._visibility_menu().actions()[1].trigger()
    app.processEvents()
    assert window.opts["element_set"] is None


def test_old_context_menu_does_not_modify_new_active_view(inspected_window):
    viewer, app, window = inspected_window
    inspector = show_inspector(app, window)
    select(inspector, "elements", "First")
    menu = inspector._visibility_menu()
    original = window.active_dock
    second = window.add_dataset_dock(window.data.deepcopy(), "Other")
    app.processEvents()
    window._set_active(second)
    app.processEvents()
    menu.actions()[0].trigger()
    assert original.opts["element_set"] is None
    assert second.opts["element_set"] is None


def test_right_click_targets_item_under_mouse(inspected_window, monkeypatch):
    viewer, app, window = inspected_window
    inspector = show_inspector(app, window)
    target = select(inspector, "elements", "First")
    inspector.tree.scrollToItem(target)
    app.processEvents()
    position = inspector.tree.visualItemRect(target).center()
    select(inspector, "mesh")
    menus = []
    monkeypatch.setattr(
        viewer.QtWidgets.QMenu,
        "exec_",
        lambda menu, point: menus.append([action.text() for action in menu.actions()]),
    )
    inspector._show_context_menu(position)
    assert inspector._key() == ("elements", "First")
    assert menus == [["Show only", "Mask"]]


@pytest.mark.parametrize("floating", [False, True])
def test_inspector_width_after_closing_and_reopening_last_view(
    inspected_window, floating
):
    viewer, app, window = inspected_window
    inspector = show_inspector(app, window)
    inspector.setFloating(floating)
    if floating:
        inspector.resize(450, 500)
    data = window.data
    for _ in range(2):
        window.active_dock.close()
        app.processEvents()
        app.processEvents()
        assert window.active_dock is None
        window.add_dataset_dock(data, "Reopened")
        for _ in range(3):
            app.processEvents()
        expected = (
            450
            if floating
            else max(inspector.minimumWidth(), inspector.minimumSizeHint().width())
        )
        assert inspector.width() == expected
        if not floating:
            assert inspector.geometry().right() < window.active_dock.geometry().left()


def test_inspector_omits_only_generated_submesh_element_sets(inspected_window):
    viewer, app, window = inspected_window
    source = window.data.mesh
    mesh = fd.MultiMesh.from_mesh_list(
        [
            fd.Mesh(source.nodes, source.elements[:1], "quad4", name="A"),
            fd.Mesh(source.nodes, source.elements[1:], "quad4", name="B"),
        ],
        register_name=False,
    )
    mesh.submeshes[0].element_sets["submesh_custom"] = [0]
    data = fd.DataSet(mesh)
    data.node_data["Value"] = np.arange(mesh.n_nodes, dtype=float)
    dock = window.add_dataset_dock(data, "Blocks")
    window._set_active(dock)
    app.processEvents()
    inspector = show_inspector(app, window)
    window._set_active(dock)
    inspector.refresh()
    generated = {
        name for name in dock.unsaved_element_set_names if name.startswith("submesh_")
    }
    assert len(generated) == 2
    assert not any(("elements", name) in inspector._objects for name in generated)
    assert ("elements", "submesh_custom") in inspector._objects
    assert ("elements", "All") in inspector._objects
    for index in range(2):
        select(inspector, "submesh", str(index))
        assert properties(inspector)["Elements"] == "1"
    # The set manager still exposes block sets for selection and conversion.
    window.open_element_sets_dialog()
    assert generated.issubset(window.element_sets_dialog.element_sets)
    # A generated set deliberately saved by the user becomes a visible set.
    saved_name = sorted(generated)[0]
    dock.unsaved_element_set_names.remove(saved_name)
    inspector.refresh()
    select(inspector, "elements", saved_name)
