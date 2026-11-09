"""Stored eigenmodes are navigable in a live Problem inspector."""

import numpy as np
import pytest
import fedoo as fd


def make_modal(n_modes=3):
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(3, 2, x_max=3.0, elm_type="quad4")
    material = fd.constitutivelaw.ElasticIsotrop(210e9, 0.3)
    material.set_density(7800.0)
    assembly = fd.Assembly.create(fd.weakform.StressEquilibrium(material), mesh)
    problem = fd.problem.Modal(assembly)
    problem.bc.add("Dirichlet", mesh.find_nodes("X", 0), ["DispX", "DispY"], 0.0)
    if n_modes:
        problem.solve(n_modes=n_modes)
    return problem


def make_buckling(n_modes=3):
    fd.ModelingSpace("2D")
    x = np.linspace(0.0, 1.0, 7)
    mesh = fd.Mesh(
        np.column_stack((x, np.zeros_like(x))),
        np.column_stack((np.arange(6), np.arange(1, 7))),
        "lin2",
    )
    section = fd.constitutivelaw.BeamCircular(
        fd.constitutivelaw.ElasticIsotrop(210e9, 0.3), 0.02, k=0.5
    )
    assembly = fd.Assembly.create(
        fd.weakform.BeamEquilibrium(section, nlgeom=False), mesh
    )
    preload = fd.problem.NonLinear(assembly, nlgeom=False)
    preload.set_solver("direct_scipy")
    preload.bc.add("Dirichlet", [0], ["DispX", "DispY"], 0.0)
    preload.bc.add("Dirichlet", [6], "DispY", 0.0)
    preload.bc.add("Neumann", [6], "DispX", -1.0)
    preload.nlsolve(dt=1.0, tmax=1.0, update_dt=False, print_info=0)
    problem = fd.problem.LinearBuckling(preload)
    problem.bc.add("Dirichlet", [0], ["DispX", "DispY"], 0.0)
    problem.bc.add("Dirichlet", [6], "DispY", 0.0)
    problem.solve(n_modes=n_modes)
    return problem


@pytest.fixture(params=[make_modal, make_buckling], ids=["modal", "buckling"])
def eigenproblem(request):
    fd.Assembly.delete_memory()
    problem = request.param()
    yield problem
    fd.Assembly.delete_memory()


def test_viewer_slider_spin_animation_and_external_reload(eigenproblem, monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    problem = eigenproblem
    problem.set_mode(2, scale=1.7)
    selected = problem.get_X().copy()
    monkeypatch.setattr(
        problem, "solve", lambda *a, **k: pytest.fail("Navigation must not solve")
    )
    window = viewer.MainWindow(problem)
    try:
        assert window.iter_slider.maximum() == 2
        assert window.iter_slider.value() == window.iter_spin.value() == 2
        np.testing.assert_allclose(problem.get_X(), selected)
        window.iter_slider.setValue(0)
        assert problem.current_mode == 0
        window.iter_spin.setValue(1)
        assert problem.current_mode == 1
        assert window.data.scalar_data["Mode"] == 2
        window.anim_wrap_check.setChecked(True)
        window.start_animation()
        window._on_anim_tick()
        assert problem.current_mode == 2
        window._on_anim_tick()
        assert problem.current_mode == 0
        window.stop_animation()
        problem.set_mode(2, scale=2.0)
        selected = problem.get_X().copy()
        window.reload_results()
        assert window.iter_slider.value() == window.iter_spin.value() == 2
        np.testing.assert_allclose(problem.get_X(), selected)
    finally:
        window.close()
        app.processEvents()


def test_modes_computed_after_open_are_discovered_on_reload(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    viewer = pytest.importorskip("fedoo.util.viewer")
    monkeypatch.setattr(viewer.pv, "OFF_SCREEN", True)
    app = viewer.QtWidgets.QApplication.instance() or viewer.QtWidgets.QApplication([])
    problem = make_modal(0)
    window = viewer.MainWindow(problem)
    try:
        assert window.iter_slider.maximum() == 0
        problem.solve(n_modes=3)
        window.reload_results()
        assert window.iter_slider.maximum() == 2
        window.iter_slider.setValue(2)
        assert problem.current_mode == 2
        problem.solve(n_modes=1)
        window.reload_results()
        assert window.iter_slider.maximum() == 0
        assert window.iter_spin.value() == problem.current_mode == 0
    finally:
        window.close()
        app.processEvents()
