import numpy as np
import pytest

import fedoo as fd
from fedoo.util.voigt_tensors import StressTensorList


@pytest.fixture(autouse=True)
def _clear_assembly_cache():
    fd.Assembly.delete_memory()
    yield
    fd.Assembly.delete_memory()


def _prestressed_beam(geometric_stiffness=False):
    fd.ModelingSpace("2D")
    material = fd.constitutivelaw.ElasticIsotrop(210e9, 0.3)
    properties = fd.constitutivelaw.BeamCircular(material, 0.02, k=0.5)
    x = np.linspace(0.0, 1.0, 7)
    mesh = fd.Mesh(
        np.column_stack((x, np.zeros_like(x))),
        np.column_stack((np.arange(6), np.arange(1, 7))),
        "lin2",
    )
    weakform = fd.weakform.BeamEquilibrium(properties, nlgeom=False)
    weakform.geometric_stiffness = geometric_stiffness
    assembly = fd.Assembly.create(weakform, mesh)
    preload = fd.problem.NonLinear(assembly, nlgeom=False)
    _constrain_pinned_beam(preload, mesh)
    preload.bc.add("Neumann", [mesh.n_nodes - 1], "DispX", -1.0)
    preload.nlsolve(
        dt=1.0,
        tmax=1.0,
        update_dt=False,
        print_info=0,
    )
    return mesh, weakform, assembly, preload


def _constrain_pinned_beam(problem, mesh):
    problem.bc.add("Dirichlet", [0], ["DispX", "DispY"], 0.0)
    problem.bc.add("Dirichlet", [mesh.n_nodes - 1], "DispY", 0.0)


@pytest.mark.parametrize("geometric_stiffness", [False, True])
def test_linear_buckling_reuses_unambiguous_preload_matrix(geometric_stiffness):
    mesh, weakform, assembly, preload = _prestressed_beam(geometric_stiffness)
    problem = fd.problem.LinearBuckling(preload)
    _constrain_pinned_beam(problem, mesh)

    problem.solve(n_modes=2)

    expected_first_factor = np.pi**2 * 210e9 * np.pi * 0.02**4 / 4
    np.testing.assert_allclose(
        problem.load_factors[0], expected_first_factor, rtol=0.03
    )
    assert np.all(problem.load_factors > 0.0)
    for load_factor, mode in zip(problem.load_factors, problem.modes):
        residual = problem._MatCB.T @ (
            problem.material_stiffness_matrix @ mode
            + load_factor * problem.geometric_stiffness_matrix @ mode
        )
        reference = problem._MatCB.T @ (problem.material_stiffness_matrix @ mode)
        assert np.linalg.norm(residual) <= 1e-8 * np.linalg.norm(reference)
    np.testing.assert_allclose(
        problem.total_stiffness_matrix.toarray(),
        (
            problem.material_stiffness_matrix + problem.geometric_stiffness_matrix
        ).toarray(),
    )
    if geometric_stiffness:
        assert problem.total_assembly is assembly
    else:
        assert problem.material_assembly is assembly
    assert weakform.geometric_stiffness is geometric_stiffness


def test_linear_buckling_can_disable_matrix_reuse():
    mesh, _, assembly, _ = _prestressed_beam(False)
    problem = fd.problem.LinearBuckling(
        assembly,
        reuse_assembled_matrix=False,
    )
    _constrain_pinned_beam(problem, mesh)

    problem.solve(n_modes=1)

    assert problem.material_assembly is not assembly
    assert problem.total_assembly is not assembly


def test_linear_buckling_supports_solid_initial_stress_stiffness():
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(
        nx=5,
        ny=2,
        x_min=0.0,
        x_max=4.0,
        y_min=0.0,
        y_max=1.0,
        elm_type="quad4",
    )
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3)
    weakform = fd.weakform.StressEquilibrium(material, nlgeom="TL")
    assembly = fd.Assembly.create(weakform, mesh)
    preload = fd.problem.NonLinear(assembly, nlgeom="TL")
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    right = mesh.find_nodes("X", mesh.bounding_box.xmax)
    preload.bc.add("Dirichlet", left, ["DispX", "DispY"], 0.0)
    preload.bc.add("Neumann", right, "DispX", -0.1)
    preload.nlsolve(dt=1.0, tmax=1.0, update_dt=False, print_info=0)

    problem = fd.problem.LinearBuckling(preload)
    problem.bc.add("Dirichlet", left, ["DispX", "DispY"], 0.0)
    problem.solve(n_modes=2)

    assert np.all(np.isfinite(problem.load_factors))
    assert np.all(problem.load_factors > 0.0)
    assert problem.geometric_stiffness_matrix.nnz > 0


def test_mixed_weakform_sum_rebuilds_both_matrices():
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="quad4")
    material = fd.constitutivelaw.ElasticIsotrop(1000.0, 0.3)
    first = fd.weakform.StressEquilibrium(material, nlgeom=False)
    second = fd.weakform.StressEquilibrium(material, nlgeom=False)
    first.geometric_stiffness = False
    second.geometric_stiffness = True
    assembly = fd.Assembly.create(fd.WeakFormSum([first, second]), mesh)
    preload = fd.problem.NonLinear(assembly, nlgeom=False)
    preload.initialize()
    leaf_assembly = next(assembly.iter_leaf())
    stress = np.zeros((6, leaf_assembly.n_gauss_points))
    stress[0] = -1.0
    leaf_assembly.sv["Stress"] = StressTensorList(stress)
    assembly.assemble_global_mat("matrix")

    problem = fd.problem.LinearBuckling(assembly)
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    problem.bc.add("Dirichlet", left, ["DispX", "DispY"], 0.0)
    problem.solve(n_modes=1)

    assert problem.material_assembly is not assembly
    assert problem.total_assembly is not assembly


def test_weakform_sum_copy_recursively_copies_its_leaves():
    fd.ModelingSpace("2D")
    material = fd.constitutivelaw.ElasticIsotrop(1.0, 0.3)
    properties = fd.constitutivelaw.BeamCircular(material, 0.1, k=0.5)
    first = fd.weakform.BeamEquilibrium(properties)
    second = fd.weakform.BeamEquilibrium(properties)
    source = fd.WeakFormSum([first, second])

    copied = source.copy()
    copied.list_weakform[0].geometric_stiffness = True

    assert copied is not source
    assert copied.list_weakform[0] is not first
    assert copied.list_weakform[1] is not second
    assert first.geometric_stiffness is None
    assert copied.list_weakform[0].properties is first.properties


def test_linear_buckling_rejects_unsupported_weakform():
    fd.ModelingSpace("2D")
    mesh = fd.Mesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        np.array([[0, 1, 2]]),
        "tri3",
    )
    assembly = fd.Assembly(fd.weakform.Inertia(1.0), mesh)
    problem = fd.problem.LinearBuckling(assembly)

    with pytest.raises(NotImplementedError, match="Inertia"):
        problem.solve(n_modes=1)


def test_linear_buckling_saves_one_frame_per_mode(tmp_path):
    mesh, _, assembly, _ = _prestressed_beam(False)
    problem = fd.problem.LinearBuckling(assembly)
    _constrain_pinned_beam(problem, mesh)
    results = problem.add_output(tmp_path / "buckling.fdh5", assembly, ["Disp"])

    problem.solve(n_modes=2)

    assert results.n_iter == 2
    for index in range(2):
        results.load(index)
        assert results.scalar_data["Mode"] == index + 1
        np.testing.assert_allclose(
            results.scalar_data["LoadFactor"], problem.load_factors[index]
        )
