"""Section extension contracts, sampling validation and private metadata."""

import numpy as np
import pytest

import fedoo as fd


def test_custom_shell_without_recovery_keeps_generalized_outputs():
    class CustomShell(fd.constitutivelaw.ShellBase):
        def get_shell_stiffness_matrix(self):
            return np.eye(8)

    fd.ModelingSpace("3D")
    law = CustomShell(0.1)
    mesh = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    assembly = fd.Assembly.create(fd.weakform.PlateEquilibrium(law), mesh)
    pb = fd.problem.Linear(assembly)
    assembly.sv["ShellStress"] = np.ones((8, assembly.n_gauss_points))
    result = pb.get_results(assembly, ["ShellStress"])
    np.testing.assert_array_equal(result["ShellStress"], assembly.sv["ShellStress"])
    with pytest.raises(NotImplementedError, match="does not define"):
        law.section_description()
    with pytest.raises(NameError, match="not available"):
        pb.get_results(assembly, ["Stress"])
    result = pb.get_results(assembly, ["Stress", "ShellStress"], ignore_missing=True)
    assert "Stress" not in result.gausspoint_data
    assert "ShellStress" in result.gausspoint_data


def test_linear_laminate_recovery_uses_layer_material_contract():
    class CustomLaminate(fd.constitutivelaw.ShellBase):
        _recovery_model = "laminate_linear"

        def __init__(self, materials):
            self.list_thickness = [1.0, 1.0]
            self._materials = tuple(materials)
            super().__init__(2)

        @property
        def layer_materials(self):
            return self._materials

        def get_strain(self, assembly, **kwargs):
            from fedoo.util.voigt_tensors import StrainTensorList

            strain = np.zeros((6, assembly.n_gauss_points))
            strain[0] = 0.01
            return StrainTensorList(strain)

    from types import SimpleNamespace

    fd.ModelingSpace("3D")
    materials = [fd.constitutivelaw.ElasticIsotrop(e, 0) for e in (1000, 2000)]
    law = CustomLaminate(materials)
    stress = law.integration_stresses(
        SimpleNamespace(n_gauss_points=1, convert_data=np.asarray)
    )
    np.testing.assert_allclose(stress[0, :, 0], [10, 10, 20, 20])
    built_in = fd.constitutivelaw.ShellLaminate(materials, [1, 1])
    assert built_in.layer_materials == tuple(materials)


@pytest.mark.parametrize("value", [4, 8.5, "32"])
def test_invalid_beam_sampling_count_fails_at_construction(value):
    fd.ModelingSpace("3D")
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    with pytest.raises(ValueError, match="Beam n_points"):
        fd.constitutivelaw.BeamRectangular(material, 2, 1, n_points=value)
    section = fd.constitutivelaw.BeamCircular(material, 1, n_points=np.int64(8))
    with pytest.raises(ValueError, match="Beam n_points"):
        section.n_points = value
    assert section.n_points == 8


@pytest.mark.parametrize("value", [1, 2.5, "3"])
def test_invalid_shell_sampling_count_fails_at_construction(value):
    with pytest.raises(ValueError, match="Shell n_points"):
        fd.constitutivelaw.ShellBase(0.1, n_points=value)
    law = fd.constitutivelaw.ShellBase(0.1, n_points=np.int64(2))
    with pytest.raises(ValueError, match="Shell n_points"):
        law.n_points = value
    assert law.n_points == 2


@pytest.mark.parametrize("family", ["beam", "shell", "mixed"])
def test_private_section_output_renames_metadata(tmp_path, family):
    fd.ModelingSpace("3D")
    nodes = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    assemblies = []
    fields = ["Stress"]
    if family != "shell":
        law = fd.constitutivelaw.BeamRectangular(material, 2, 1)
        beam = fd.Assembly.create(
            fd.weakform.BeamEquilibrium(law),
            fd.Mesh(nodes, np.array([[0, 1]]), "lin2"),
        )
        beam.sv["BeamStress"] = np.zeros((6, beam.n_gauss_points))
        assemblies.append(beam)
        fields.append("BeamStress")
    if family != "beam":
        law = fd.constitutivelaw.ShellHomogeneous(material, 0.1)
        shell = fd.Assembly.create(
            fd.weakform.PlateEquilibrium(law),
            fd.Mesh(nodes, np.array([[0, 1, 2, 3]]), "quad4"),
        )
        shell.sv["ShellStress"] = np.zeros((8, shell.n_gauss_points))
        assemblies.append(shell)
        fields.append("ShellStress")
    assembly = assemblies[0] if len(assemblies) == 1 else fd.Assembly.sum(*assemblies)
    pb = fd.problem.Linear(assembly)
    history = pb.add_output(str(tmp_path / "private"), assembly, fields, private=True)
    pb.save_results()
    history.load(-1)
    for name in fields:
        assert name not in history.gausspoint_data
        assert name not in history.field_metadata
        assert "_" + name in history.gausspoint_data
        assert "_" + name in history.field_metadata
