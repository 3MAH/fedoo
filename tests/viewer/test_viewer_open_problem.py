"""Discover live workspace Problems and open them through the Problem menu."""

import sys
from types import SimpleNamespace
import numpy as np
import pytest
import fedoo as fd
from fedoo.core.base import ProblemBase


def make_problem(name=""):
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(3, 2, elm_type="quad4")
    assembly = fd.Assembly.create(
        fd.weakform.StressEquilibrium(fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3)),
        mesh,
    )
    problem = fd.problem.Linear(assembly, name=name)
    problem.set_dof("DispX", 0.1)
    return problem


def test_discovery_named_namespace_aliases_and_unnamed_active(monkeypatch):
    viewer = pytest.importorskip("fedoo.util.viewer")
    named, workspace, active = (
        make_problem("registered"),
        make_problem(),
        make_problem(),
    )
    monkeypatch.setattr(
        ProblemBase,
        "get_all",
        staticmethod(lambda: {"registered": named, "duplicate": named}),
    )
    monkeypatch.setattr(ProblemBase, "get_active", staticmethod(lambda: active))
    monkeypatch.setattr(
        sys.modules["__main__"], "workspace_pb", workspace, raising=False
    )
    monkeypatch.setitem(
        sys.modules,
        "IPython",
        SimpleNamespace(
            get_ipython=lambda: SimpleNamespace(
                user_ns={"pb": workspace, "alias": workspace, "number": 42}
            )
        ),
    )
    found = viewer._workspace_problems()
    assert len(found) == 3
    labels = {id(problem): label for label, problem in found}
    assert "registered" in labels[id(named)]
    assert "workspace_pb" in labels[id(workspace)]
    assert "alias" in labels[id(workspace)]
    assert labels[id(active)] == "Active problem"


@pytest.fixture
def empty_window(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    window = viewer.MainWindow()
    try:
        yield viewer, app, window
    finally:
        window.close()
        app.processEvents()


def test_problem_menu_opens_live_problem_from_empty_viewer(empty_window, monkeypatch):
    viewer, app, window = empty_window
    problem = make_problem()
    monkeypatch.setattr(viewer, "_workspace_problems", lambda: [("pb", problem)])
    menus = {action.text(): action.menu() for action in window.menuBar().actions()}
    assert window.open_problem_action in menus["Problem"].actions()
    assert window.boundary_conditions_action in menus["Problem"].actions()
    assert window.boundary_conditions_action not in menus["Tools"].actions()
    assert window.open_problem_action.isEnabled()
    assert not window.boundary_conditions_action.isEnabled()
    window.open_problem_action.trigger()
    dialog = window._problem_dialog
    assert dialog.table.rowCount() == 1
    assert dialog.open_button.isEnabled()
    dialog.open_button.click()
    assert len(window.all_docks) == 1
    assert window.data.problem is problem
    assert window.active_dock.title == "pb"
    assert window.boundary_conditions_action.isEnabled()
    np.testing.assert_allclose(window.data["DispX"], 0.1)
    # Opening changes the viewer selection, not the solver's active problem.
    other = make_problem()
    problem.set_dof("DispX", 0.2)
    window.open_existing_problem_dialog()
    dialog.open_button.click()
    assert len(window.all_docks) == 1
    assert ProblemBase.get_active() is other
    np.testing.assert_allclose(window.data["DispX"], 0.2)


def test_empty_list_refresh_and_cancel(empty_window, monkeypatch):
    viewer, app, window = empty_window
    available = []
    monkeypatch.setattr(viewer, "_workspace_problems", lambda: list(available))
    window.open_existing_problem_dialog()
    dialog = window._problem_dialog
    assert not dialog.open_button.isEnabled()
    assert "No Problem" in dialog.status.text()
    problem = make_problem()
    available.append(("new_pb", problem))
    dialog.refresh()
    assert dialog.selected_problem() is problem
    assert dialog.open_button.isEnabled()
    dialog.reject()
    assert not window.all_docks


def test_extraction_failure_keeps_dialog_open_without_partial_pane(
    empty_window, monkeypatch
):
    viewer, app, window = empty_window
    problem = make_problem()
    monkeypatch.setattr(viewer, "_workspace_problems", lambda: [("pb", problem)])

    def fail(*args, **kwargs):
        raise RuntimeError("Results unavailable")

    monkeypatch.setattr(problem, "get_results", fail)
    messages = []
    monkeypatch.setattr(
        viewer.QtWidgets.QMessageBox, "warning", lambda *args: messages.append(args[-1])
    )
    window.open_existing_problem_dialog()
    window._problem_dialog.open_button.click()
    assert messages == ["Results unavailable"]
    assert not window.all_docks
    assert window._problem_dialog.isVisible()
