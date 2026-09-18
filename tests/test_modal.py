import numpy as np
import pytest
from scipy import linalg

import fedoo as fd
import fedoo.problem.modal as modal_module


def _solid_model(with_density=True):
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(
        nx=3,
        ny=2,
        x_min=0.0,
        x_max=3.0,
        y_min=0.0,
        y_max=1.0,
        elm_type="quad4",
    )
    material = fd.constitutivelaw.ElasticIsotrop(210e9, 0.3)
    if with_density:
        material.set_density(7800.0)
    assembly = fd.Assembly.create(fd.weakform.StressEquilibrium(material), mesh)
    return mesh, assembly


def test_modal_matches_dense_solution_and_mass_normalizes_modes():
    mesh, assembly = _solid_model()
    problem = fd.problem.Modal(assembly)
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    problem.bc.add("Dirichlet", left, ["DispX", "DispY"], 0.0)

    returned = problem.solve(n_modes=3, sigma=0.0)

    transform = problem._MatCB
    reduced_stiffness = transform.T @ problem.stiffness_matrix @ transform
    reduced_mass = transform.T @ problem.mass_matrix @ transform
    expected = linalg.eigh(
        reduced_stiffness.toarray(),
        reduced_mass.toarray(),
        subset_by_index=(0, 2),
        eigvals_only=True,
    )

    assert returned is problem
    np.testing.assert_allclose(problem.eigenvalues, expected, rtol=1e-9)
    np.testing.assert_allclose(
        problem.frequencies,
        np.sqrt(expected) / (2.0 * np.pi),
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        np.diag(problem.modes @ (problem.mass_matrix @ problem.modes.T)),
        np.ones(3),
        rtol=1e-11,
        atol=1e-11,
    )
    np.testing.assert_allclose(problem.modes[:, problem._dof_slave], 0.0)

    for eigenvalue, mode in zip(problem.eigenvalues, problem.modes):
        residual = transform.T @ (
            problem.stiffness_matrix @ mode - eigenvalue * (problem.mass_matrix @ mode)
        )
        np.testing.assert_allclose(residual, 0.0, atol=1e-3)


def test_modal_set_solver_configures_sparse_eigensolver(monkeypatch):
    mesh, assembly = _solid_model()
    problem = fd.problem.Modal(assembly)
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    problem.bc.add("Dirichlet", left, "Disp", 0.0)
    captured = {}
    scipy_eigsh = modal_module.eigsh

    def recording_eigsh(*args, **kwargs):
        captured.update(kwargs)
        return scipy_eigsh(*args, **kwargs)

    monkeypatch.setattr(modal_module, "eigsh", recording_eigsh)
    returned = problem.set_solver("eigsh", which="lm", tol=1.0e-9, maxiter=100, ncv=6)
    problem.solve(n_modes=2, sigma=0.0)

    assert returned is problem
    assert problem._solver_type == "eigsh"
    assert captured["which"] == "LM"
    assert captured["tol"] == 1.0e-9
    assert captured["maxiter"] == 100
    assert captured["ncv"] == 6


def test_modal_set_solver_can_force_dense_eigensolver(monkeypatch):
    mesh, assembly = _solid_model()
    problem = fd.problem.Modal(assembly)
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    problem.bc.add("Dirichlet", left, "Disp", 0.0)

    def unexpected_eigsh(*args, **kwargs):
        raise AssertionError("eigsh should not be called")

    monkeypatch.setattr(modal_module, "eigsh", unexpected_eigsh)
    problem.set_solver("eigh")
    problem.solve(n_modes=2)

    assert len(problem.eigenvalues) == 2


def test_modal_set_solver_rejects_linear_solver_names():
    _, assembly = _solid_model()
    problem = fd.problem.Modal(assembly)

    with pytest.raises(ValueError, match="auto.*eigsh.*eigh"):
        problem.set_solver("direct")


def test_modal_set_mode_controls_current_solution_without_rescaling_storage():
    mesh, assembly = _solid_model()
    problem = fd.problem.Modal(assembly)
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    problem.bc.add("Dirichlet", left, "Disp", 0.0)
    problem.solve(n_modes=2, sigma=0.0)

    stored_mode = problem.modes[1].copy()
    returned = problem.set_mode(1, scale=2.5, update_weakform=False)

    assert problem.current_mode == 1
    np.testing.assert_allclose(returned, 2.5 * stored_mode)
    np.testing.assert_allclose(problem.get_X(), 2.5 * stored_mode)
    np.testing.assert_allclose(
        problem.get_disp(),
        problem.get_dof_solution("Disp"),
    )
    np.testing.assert_allclose(problem.modes[1], stored_mode)


def test_modal_rejects_nonhomogeneous_constraints():
    mesh, assembly = _solid_model()
    problem = fd.problem.Modal(assembly)
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    problem.bc.add("Dirichlet", left, "Disp", 0.1)

    with pytest.raises(ValueError, match="homogeneous"):
        problem.solve(n_modes=1)


def test_modal_requires_second_order_storage():
    mesh, assembly = _solid_model(with_density=False)
    problem = fd.problem.Modal(assembly)
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    problem.bc.add("Dirichlet", left, "Disp", 0.0)

    with pytest.raises(ValueError, match="density"):
        problem.solve(n_modes=1)


def test_modal_free_free_keeps_rigid_body_modes():
    _, assembly = _solid_model()
    problem = fd.problem.Modal(assembly)

    problem.solve(n_modes=4)

    np.testing.assert_allclose(problem.eigenvalues[:3], 0.0, atol=1e-6)
    np.testing.assert_allclose(problem.frequencies[:3], 0.0, atol=1e-10)
    assert problem.eigenvalues[3] > 0.0


def test_modal_supports_beam_translational_and_rotary_inertia():
    fd.ModelingSpace("3D")
    material = fd.constitutivelaw.ElasticIsotrop(210e9, 0.3, name="ModalBeamMaterial")
    material.set_density(7800.0)
    properties = fd.constitutivelaw.BeamCircular(material, 0.02, k=0.0)
    coordinates = np.column_stack((np.linspace(0.0, 1.0, 4), np.zeros(4), np.zeros(4)))
    elements = np.column_stack((np.arange(3), np.arange(1, 4)))
    fd.Mesh(coordinates, elements, "lin2", name="modal_beam_mesh")
    fd.weakform.BeamEquilibrium(properties, name="ModalBeamWeakForm")
    assembly = fd.Assembly.create(
        "ModalBeamWeakForm",
        "modal_beam_mesh",
        "bernoulliBeam",
        name="modal_beam_assembly",
        mesh_change=True,
    )
    problem = fd.problem.Modal(assembly)
    problem.bc.add("Dirichlet", [0], ["Disp", "Rot"], 0.0)

    problem.solve(n_modes=1, sigma=0.0)

    assert np.all(np.isfinite(problem.frequencies))
    assert np.all(problem.frequencies > 0.0)


def test_modal_saves_one_fdh5_frame_per_mode_with_frequency_metadata(tmp_path):
    mesh, assembly = _solid_model()
    problem = fd.problem.Modal(assembly)
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    problem.bc.add("Dirichlet", left, "Disp", 0.0)
    results = problem.add_output(tmp_path / "modes.fdh5", assembly, ["Disp"])

    returned = problem.solve(n_modes=2, sigma=0.0)

    assert returned is problem
    assert (tmp_path / "modes.fdh5").is_file()
    assert results.n_iter == 2
    for index in range(2):
        results.load(index)
        assert results.scalar_data["Mode"] == index + 1
        np.testing.assert_allclose(
            results.scalar_data["Eigenvalue"], problem.eigenvalues[index]
        )
        np.testing.assert_allclose(
            results.scalar_data["AngularFrequency"],
            problem.angular_frequencies[index],
        )
        np.testing.assert_allclose(
            results.scalar_data["Frequency"], problem.frequencies[index]
        )
        np.testing.assert_allclose(
            results.node_data["Disp"],
            problem.modes[index].reshape(problem.space.nvar, -1),
        )
