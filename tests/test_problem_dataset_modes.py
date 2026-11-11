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


def test_live_mode_selection_metadata_scalars_and_snapshot(eigenproblem, tmp_path):
    problem = eigenproblem
    data = fd.ProblemDataSet(problem, fields=["Disp"])
    assert data.n_iter == data.n_modes == 3
    assert not hasattr(data, "list_data")
    data.add_scalar_expression("NormX", "sum(Disp_X**2)")
    for mode in (2, 0, 1):
        data.set_mode(mode)
        assert data.current_mode == problem.current_mode == mode
        np.testing.assert_allclose(problem.get_X(), problem.modes[mode])
        assert data.get_data("Mode") == mode + 1
        metadata = (
            "Frequency" if isinstance(problem, fd.problem.Modal) else "LoadFactor"
        )
        expected = (
            problem.frequencies[mode]
            if metadata == "Frequency"
            else problem.load_factors[mode]
        )
        assert data.get_data(metadata) == pytest.approx(expected)
        assert data.get_data("NormX") == pytest.approx(
            np.sum(problem.get_disp("DispX") ** 2)
        )
    path = str(tmp_path / "selected_mode.fdh5")
    data.save(path)
    saved = fd.read_data(path)
    assert type(saved) is fd.DataSet
    assert saved.scalar_data["Mode"] == 2
    np.testing.assert_allclose(saved.get_data("Disp"), data.get_data("Disp"))
    with pytest.raises(IndexError):
        data.set_mode(3)
    with pytest.raises(IndexError):
        data.set_mode(-1)
    assert problem.current_mode == 1
