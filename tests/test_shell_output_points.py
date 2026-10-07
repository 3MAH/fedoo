"""Shell thickness extraction, saved interpolation and mixed-element recovery."""

import numpy as np
import pytest
import fedoo as fd
from fedoo.util.voigt_tensors import StressTensorList
from fedoo.weakform.plate import _ShellComponentList


def shell_problem(section=None, nx=2):
    fd.ModelingSpace("3D")
    if section is None:
        section = fd.constitutivelaw.ShellHomogeneous(
            fd.constitutivelaw.ElasticIsotrop(1000, 0.25), 2.0, k=5 / 6
        )
    mesh = fd.mesh.rectangle_mesh(nx=nx, ny=2, elm_type="quad4", ndim=3)
    assembly = fd.Assembly.create(fd.weakform.PlateEquilibrium(section), mesh)
    problem = fd.problem.Linear(assembly)
    n = assembly.n_gauss_points
    assembly.sv["ShellStrain"] = _ShellComponentList(
        [
            np.full(n, v)
            for v in (0.01, -0.003, 0.004, 0.02, 0.005, -0.002, 0.008, -0.006)
        ]
    )
    if section._recovery_model.endswith("nonlinear"):
        section.initialize(assembly, problem)
    section.update(assembly, problem)
    return problem, assembly, section


def test_linear_output_positions_and_resultant_recovery(tmp_path):
    pb, assembly, law = shell_problem()
    result = pb.get_results(
        assembly, ["Stress", "Strain", "ShellStress", "ShellStrain"]
    )
    assert result.gausspoint_data["Stress"].shape == (6, 3, assembly.n_gauss_points)
    assert "Stress" not in result.beam_derived_fields()
    assert "ShellLocalFrame" in result.gausspoint_data
    for z in (-1, 0, 0.23, 1):
        expected = law.get_stress(assembly, position=z).asarray()
        result.shell_options.update(reduction=None, position=z)
        np.testing.assert_allclose(result["Stress"], expected)
        result.shell_options.update(source="recomputed")
        np.testing.assert_allclose(result["Stress"], expected)
        np.testing.assert_allclose(
            result["Strain"], law.get_strain(assembly, position=z).asarray()
        )
        result.shell_options.update(source="auto")
    result.save(str(tmp_path / "linear.fdh5"))
    loaded = fd.read_data(str(tmp_path / "linear.fdh5"))
    loaded.shell_options.update(source="recomputed", reduction=None, position=0.23)
    np.testing.assert_allclose(
        loaded["Stress"], law.get_stress(assembly, position=0.23).asarray()
    )
    law.output_points = [-0.7, 0.4]
    selected = pb.get_results(assembly, ["Stress"])
    np.testing.assert_array_equal(
        selected.field_metadata["Stress"]["stored_points"], [-0.7, 0.4]
    )
    explicit = pb.get_results(assembly, ["Stress"], position=[-0.2, 0, 0.8, 1])
    assert explicit.gausspoint_data["Stress"].shape[1] == 4
    with pytest.raises(ValueError, match="Shell position"):
        pb.get_results(assembly, ["Stress"], position=2)


def test_signed_extrema_and_atomic_selection():
    pb, assembly, _ = shell_problem()
    result = pb.get_results(assembly, ["Stress"])
    tensor = result.gausspoint_data["Stress"]
    tensor[0] = np.array([-12.0, 4.0, 12.0])[:, None]
    np.testing.assert_allclose(result["Stress", "XX"], -12)
    result.shell_options.update(reduction="max", point_index=999)
    np.testing.assert_allclose(result["Stress", "XX"], 12)
    result.shell_options.update(reduction=None, point_index=1)
    np.testing.assert_allclose(result["Stress", "XX"], 4)
    result.shell_options.update(position=0.5)
    assert result.shell_options["point_index"] is None
    np.testing.assert_allclose(result["Stress", "XX"], 8)
    old = dict(result.shell_options)
    with pytest.raises(ValueError, match="not both"):
        result.shell_options.update(position=0.4, point_index=1)
    assert result.shell_options == old
    with pytest.raises(ValueError, match="Shell position"):
        result.shell_options.update(position=2)


@pytest.mark.parametrize("association", ["GaussPoint", "Element", "Node"])
def test_stored_associations_and_rotation(association):
    pb, assembly, _ = shell_problem()
    result = pb.get_results(
        assembly, ["Stress", "ShellStress"], output_type=association
    )
    result.shell_options.update(reduction=None, point_index=0)
    assert result.get_data("Stress", "XX", return_data_type=True)[1] == association
    rotation = np.array([[0.0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    result.gausspoint_data["ShellLocalFrame"] = np.tile(
        rotation.reshape(9, 1), (1, assembly.n_gauss_points)
    )
    if association == "Node":
        with pytest.raises(ValueError, match="nodal"):
            result["Stress_global", "YY"]
    else:
        np.testing.assert_allclose(
            result["Stress_global", "YY"], result["Stress_local", "XX"]
        )


def test_nonlinear_saves_native_points_and_interpolates(tmp_path):
    law = fd.constitutivelaw.ShellHomogeneousNonLinear(
        fd.constitutivelaw.ElasticIsotrop(1000, 0.25), 2, n_thickness_points=5
    )
    pb, assembly, law = shell_problem(law)
    points = law._z.copy()
    values = np.zeros((6, len(points), assembly.n_gauss_points))
    values[0] = np.array([8.0, -30.0, 2.0, 9.0, 1.0])[:, None]
    law._material_assembly.sv["Stress"] = StressTensorList(values.reshape(6, -1))
    result = pb.get_results(assembly, ["ShellStress"])
    assert "Stress" in result.shell_derived_fields()
    np.testing.assert_array_equal(result.gausspoint_data["_ShellStress"][0], values[0])
    result.save(str(tmp_path / "nonlinear.fdh5"))
    loaded = fd.read_data(str(tmp_path / "nonlinear.fdh5"))
    loaded.shell_options.update(reduction=None, position=0.15)
    expected = np.interp(0.15, points, values[0, :, 0])
    np.testing.assert_allclose(loaded["Stress", "XX"], expected)
    loaded.shell_options.update(position=1)
    np.testing.assert_allclose(loaded["Stress", "XX"], 1)
    loaded.shell_options.update(reduction="abs_max", samples=2)
    np.testing.assert_allclose(loaded["Stress", "XX"], -30)
    saved = pb.get_results(assembly, ["Stress"])
    assert saved.gausspoint_data["Stress"].shape[1] == 5
    saved.shell_options.update(reduction=None, position=0.15)
    np.testing.assert_allclose(saved["Stress", "XX"], expected)
    del loaded.gausspoint_data["_ShellStress"]
    # A nonlinear resultant does not invent an elastic stress distribution.
    assert "Stress" not in loaded.shell_derived_fields()


@pytest.mark.parametrize("nonlinear", [False, True])
def test_laminate_interfaces_remain_discontinuous(tmp_path, nonlinear):
    materials = [fd.constitutivelaw.ElasticIsotrop(e, 0.25) for e in (1000, 4000)]
    cls = (
        fd.constitutivelaw.ShellLaminateNonLinear
        if nonlinear
        else fd.constitutivelaw.ShellLaminate
    )
    law = cls(materials, [1.0, 1.0])
    pb, assembly, law = shell_problem(law)
    result = pb.get_results(assembly, ["Stress", "ShellStress", "ShellStrain"])
    assert "Stress" in result.shell_derived_fields()
    points = np.asarray(result.field_metadata["Stress"]["stored_points"])
    layers = np.asarray(result.field_metadata["Stress"]["point_layers"])
    result.shell_options.update(reduction=None, position=-1e-9)
    lower = result["Stress", "XX"]
    result.shell_options.update(position=1e-9)
    upper = result["Stress", "XX"]
    if not nonlinear:
        np.testing.assert_allclose(upper, 4 * lower, rtol=1e-7)
        indices = np.flatnonzero(points == 0)
        assert len(indices) == 2
        result.shell_options.update(point_index=int(indices[1]))
        np.testing.assert_allclose(result["Stress", "XX"], upper, rtol=1e-7)
    else:
        assert np.max(upper / lower) > 3
    result.save(str(tmp_path / "laminate.fdh5"))
    loaded = fd.read_data(str(tmp_path / "laminate.fdh5"))
    loaded.shell_options.update(source="recomputed", reduction=None, position=1e-9)
    np.testing.assert_allclose(loaded["Stress", "XX"], upper)
    assert set(layers) == {0, 1}


def test_zero_state_and_missing_source():
    pb, assembly, _ = shell_problem()
    assembly.sv["ShellStrain"] = assembly.sv["ShellStress"] = 0
    result = pb.get_results(assembly, ["ShellStress", "ShellStrain"])
    np.testing.assert_array_equal(result["Stress", "vm"], 0)
    np.testing.assert_array_equal(result["Strain", "XX"], 0)
    result.shell_options.update(source="stored")
    with pytest.raises(ValueError, match="No stored"):
        result["Stress", "XX"]


def test_strain_rotation_uses_engineering_shear():
    pb, assembly, law = shell_problem()
    assembly.sv["ShellStrain"] = _ShellComponentList(
        [np.zeros(assembly.n_gauss_points) for _ in range(8)]
    )
    assembly.sv["ShellStrain"][2][:] = 0.2
    result = pb.get_results(assembly, ["Strain", "ShellStrain"])
    frame = np.array([[1.0, 1, 0], [-1, 1, 0], [0, 0, np.sqrt(2)]]) / np.sqrt(2)
    result.gausspoint_data["ShellLocalFrame"] = np.tile(
        frame.reshape(9, 1), (1, assembly.n_gauss_points)
    )
    result.shell_options.update(reduction=None, position=0)
    for source in ("stored", "recomputed"):
        result.shell_options.update(source=source)
        np.testing.assert_allclose(result["Strain_global", "XX"], -0.1)
        np.testing.assert_allclose(result["Strain_global", "YY"], 0.1)
        np.testing.assert_allclose(result["Strain_global", "XY"], 0, atol=1e-14)
        np.testing.assert_allclose(result["Strain_global", "I"], 0.1)


def test_repeated_output_positions_and_options_iterable():
    pb, assembly, law = shell_problem()
    result = pb.get_results(assembly, ["Stress"], position=[-1, 0, 0, 1])
    result.shell_options.update(iter([("reduction", None), ("position", 0.2)]))
    np.testing.assert_allclose(
        result["Stress"], law.get_stress(assembly, position=0.2).asarray()
    )


def test_one_nonlinear_integration_point_is_constant():
    law = fd.constitutivelaw.ShellHomogeneousNonLinear(
        fd.constitutivelaw.ElasticIsotrop(1000, 0.25), 2, n_thickness_points=1
    )
    pb, assembly, law = shell_problem(law)
    result = pb.get_results(assembly, ["Stress", "ShellStress"])
    assert result.gausspoint_data["Stress"].shape[1] == 1
    result.shell_options.update(reduction=None, position=-1)
    bottom = result["Stress", "XX"]
    result.shell_options.update(position=1)
    np.testing.assert_allclose(result["Stress", "XX"], bottom)
    result.shell_options.update(source="recomputed")
    np.testing.assert_allclose(result["Stress", "XX"], bottom)


def test_varying_thickness_and_element_subset(tmp_path):
    pb, assembly, law = shell_problem(nx=3)
    law.thickness = np.array([1.0, 2.0])
    h = assembly.convert_data(law.thickness)
    middle = law.get_stress(assembly, position=0).asarray()
    top = law.get_stress(assembly, position=1).asarray()
    # Prescribe consistent resultants without involving the stiffness assembly.
    assembly.sv["ShellStress"] = _ShellComponentList(
        np.concatenate(
            (
                middle[[0, 1, 3]] * h,
                (top - middle)[[0, 1, 3]] * h**2 / 6,
                middle[4:6] * h * law.k,
            )
        )
    )
    result = pb.get_results(
        assembly, ["Stress", "ShellStress", "ShellStrain"], element_set=[1]
    )
    assert result.mesh.n_elements == 1
    result.save(str(tmp_path / "subset.fdh5"))
    loaded = fd.read_data(str(tmp_path / "subset.fdh5"))
    loaded.shell_options.update(source="recomputed", reduction=None, position=0.3)
    expected = (
        law.get_stress(assembly, position=0.3).asarray().reshape(6, -1, 2)[..., 1]
    )
    np.testing.assert_allclose(loaded["Stress"], expected)


@pytest.mark.parametrize("mixed", [False, True])
def test_assemblysum_and_multimesh_storage(tmp_path, mixed):
    fd.ModelingSpace("3D")
    nodes = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    laws = [fd.constitutivelaw.ShellHomogeneous(material, 2)]
    meshes = [fd.Mesh(nodes, np.array([[0, 1, 2, 3]]), "quad4")]
    if mixed:
        laws.append(
            fd.constitutivelaw.BeamCircular(material, 1.0, output_points=[[0, 0]])
        )
        meshes.append(fd.Mesh(nodes, np.array([[0, 1]]), "lin2"))
    else:
        laws.append(fd.constitutivelaw.ShellLaminate([material, material], [1.0, 1.0]))
        meshes.append(fd.Mesh(nodes, np.array([[0, 1, 2], [0, 2, 3]]), "tri3"))
    assemblies = [
        fd.Assembly.create(
            fd.weakform.PlateEquilibrium(law)
            if hasattr(law, "get_shell_stiffness_matrix")
            else fd.weakform.BeamEquilibrium(law),
            mesh,
        )
        for law, mesh in zip(laws, meshes)
    ]
    assembly = fd.Assembly.sum(*assemblies)
    pb = fd.problem.Linear(assembly)
    for part, law in zip(assemblies, laws):
        if hasattr(law, "get_shell_stiffness_matrix"):
            part.sv["ShellStrain"] = _ShellComponentList(
                [np.full(part.n_gauss_points, 0.01)] + [0] * 7
            )
            law.update(part, pb)
        else:
            part.sv["BeamStress"] = np.zeros((6, part.n_gauss_points))
            part.sv["BeamStress"][0] = 10 * law.A
    result = pb.get_results(assembly, ["Stress", "ShellStress"])
    result.save(str(tmp_path / "multi.fdh5"))
    loaded = fd.read_data(str(tmp_path / "multi.fdh5"))
    loaded.shell_options.update(source="recomputed", reduction=None, position=0.1)
    stress = loaded["Stress", "XX"]
    np.testing.assert_allclose(stress.submesh(0), 1000 * 0.01 / (1 - 0.25**2))
    np.testing.assert_allclose(
        stress.submesh(1), 10 if mixed else 1000 * 0.01 / (1 - 0.25**2)
    )
    if mixed:
        assert "_ShellSection" not in loaded._submesh_dataset(1).field_metadata
    else:
        assert "_ShellStress" in loaded._submesh_dataset(1).gausspoint_data


@pytest.mark.parametrize(
    "shell_kind", ["linear", "nonlinear", "laminate", "laminate_nonlinear"]
)
@pytest.mark.parametrize("association", ["GaussPoint", "Element"])
def test_stress_only_extrema_on_mixed_elements(tmp_path, shell_kind, association):
    fd.ModelingSpace("3D")
    nodes = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]])
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    beam = fd.constitutivelaw.BeamRectangular(material, 2.0, 4.0)
    shells = {
        "linear": lambda: fd.constitutivelaw.ShellHomogeneous(material, 2.0),
        "nonlinear": lambda: fd.constitutivelaw.ShellHomogeneousNonLinear(
            material, 2.0
        ),
        "laminate": lambda: fd.constitutivelaw.ShellLaminate(
            [material, material], [1.0, 1.0]
        ),
        "laminate_nonlinear": lambda: fd.constitutivelaw.ShellLaminateNonLinear(
            [material, material], [1.0, 1.0]
        ),
    }
    shell = shells[shell_kind]()
    meshes = [
        fd.Mesh(nodes, np.array([[0, 1]]), "lin2"),
        fd.Mesh(nodes, np.array([[0, 1, 2, 3]]), "quad4"),
        fd.Mesh(nodes, np.array([[0, 1, 3, 4]]), "tet4"),
    ]
    weakforms = [
        fd.weakform.BeamEquilibrium(beam),
        fd.weakform.PlateEquilibrium(shell),
        fd.weakform.StressEquilibrium(material),
    ]
    assemblies = [fd.Assembly.create(wf, mesh) for wf, mesh in zip(weakforms, meshes)]
    total = fd.Assembly.sum(*assemblies)
    problem = fd.problem.Linear(total)
    beam_assembly, shell_assembly, solid_assembly = assemblies
    beam_assembly.sv["BeamStress"] = np.zeros((6, beam_assembly.n_gauss_points))
    beam_assembly.sv["BeamStress"][0] = -5 * beam.A
    beam_assembly.sv["BeamStress"][1:3] = 2
    beam_assembly.sv["BeamStress"][5] = 3
    shell_assembly.sv["ShellStrain"] = _ShellComponentList(
        [
            np.full(shell_assembly.n_gauss_points, x)
            for x in (0.001, -0.003, 0.002, 0.01, -0.004, 0.006, 0.003, -0.005)
        ]
    )
    if shell._recovery_model.endswith("nonlinear"):
        shell.initialize(shell_assembly, problem)
    shell.update(shell_assembly, problem)
    solid = np.zeros((6, solid_assembly.n_gauss_points))
    solid[0], solid[1], solid[3] = -37.0, 13.0, 5.0
    solid_assembly.sv["Stress"] = StressTensorList(solid)
    result = problem.get_results(total, ["Stress"], output_type=association)
    assert "BeamStress" not in result.gausspoint_data
    assert "ShellStress" not in result.gausspoint_data
    assert "ShellStress" not in result.element_data
    result.save(str(tmp_path / "stress_only.fdh5"))
    loaded = fd.read_data(str(tmp_path / "stress_only.fdh5"))
    assert "BeamStress" not in loaded.gausspoint_data
    assert "ShellStress" not in loaded.gausspoint_data
    assert ("_ShellStress" in loaded.gausspoint_data) == (shell_kind != "linear")
    for reduction in ("min", "max", "abs_max"):
        loaded.beam_options.update(reduction=reduction)
        loaded.shell_options.update(reduction=reduction)
        for component in ("XX", "vm", "I"):
            values = loaded.get_data("Stress", component)
            for sid in range(3):
                raw = np.asarray(loaded.dict_data[association]["Stress"][sid])
                if raw.ndim == 2:
                    raw = raw[:, None, :]
                scalar = np.asarray(
                    StressTensorList(raw.reshape(6, -1))[component]
                ).reshape(raw.shape[1:])
                if reduction == "abs_max":
                    indices = np.argmax(np.abs(scalar), axis=0)
                    expected = np.take_along_axis(scalar, indices[None], axis=0)[0]
                else:
                    expected = getattr(np, reduction)(scalar, axis=0)
                np.testing.assert_allclose(values.submesh(sid), expected)
            assert loaded._submesh_has_field("Stress", 2)
            np.testing.assert_allclose(
                loaded.get_data("Stress_local", component).submesh(2), expected
            )
            np.testing.assert_allclose(
                loaded.get_data("Stress_global", component).submesh(2), expected
            )
            assert not loaded._submesh_has_field("Stress_global", 0)
