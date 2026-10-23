"""Unavailable section recovery follows ordinary and mixed output semantics."""

import numpy as np
import pytest

import fedoo as fd
from fedoo.util.voigt_tensors import StressTensorList


def beam_output_problem(torsion=False, mixed=False):
    fd.ModelingSpace("3D")
    nodes = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    section = fd.constitutivelaw.BeamProperties(
        material,
        1,
        1,
        1,
        1,
        section_points=[[0, 0], [1, 1]] if torsion else None,
    )
    beam = fd.Assembly.create(
        fd.weakform.BeamEquilibrium(section),
        fd.Mesh(nodes, np.array([[0, 1]]), "lin2"),
    )
    assembly = beam
    solid = None
    if mixed:
        solid = fd.Assembly.create(
            fd.weakform.StressEquilibrium(material),
            fd.Mesh(nodes, np.array([[0, 1, 2, 3]]), "tet4"),
        )
        assembly = fd.Assembly.sum(beam, solid)
    problem = fd.problem.Linear(assembly)
    beam.sv["BeamStress"] = np.zeros((6, beam.n_gauss_points))
    if torsion:
        beam.sv["BeamStress"][3] = 1
    if solid is not None:
        stress = np.zeros((6, solid.n_gauss_points))
        stress[0] = 7
        solid.sv["Stress"] = StressTensorList(stress)
    return problem, assembly, beam, solid


@pytest.mark.parametrize("torsion", [False, True])
def test_unavailable_beam_tensor_obeys_ignore_missing(torsion):
    problem, assembly, beam, _ = beam_output_problem(torsion=torsion)
    with pytest.raises(NameError, match='Field "Stress" not available'):
        problem.get_results(assembly, ["Stress"])
    result = problem.get_results(
        assembly,
        ["Stress", "Stress_vm", "Stress_pc", "BeamStress"],
        ignore_missing=True,
    )
    assert set(result.gausspoint_data) == {"BeamStress", "BeamLocalFrame"}
    np.testing.assert_array_equal(result["BeamStress"], beam.sv["BeamStress"])


@pytest.mark.parametrize("torsion", [False, True])
@pytest.mark.parametrize("ignore_missing", [False, True])
def test_mixed_output_keeps_tensor_available_on_one_submesh(torsion, ignore_missing):
    problem, assembly, _, solid = beam_output_problem(torsion=torsion, mixed=True)
    result = problem.get_results(
        assembly, ["Stress", "Stress_vm"], ignore_missing=ignore_missing
    )
    assert set(result.gausspoint_data["Stress"]) == {1}
    np.testing.assert_array_equal(
        result.gausspoint_data["Stress"][1], solid.sv["Stress"].asarray()
    )
    np.testing.assert_allclose(result.gausspoint_data["Stress_vm"][1], 7)


def test_mixed_output_keeps_automatic_beam_frames(tmp_path):
    problem, assembly, beam, _ = beam_output_problem(mixed=True)
    result = problem.get_results(
        assembly, ["BeamStress", "Stress"], ignore_missing=True
    )
    expected = np.tile(
        beam.current.get_element_local_frame().reshape(-1, 9).T,
        (1, beam.n_elm_gp),
    )
    np.testing.assert_allclose(result.gausspoint_data["BeamLocalFrame"][0], expected)
    assert set(result.gausspoint_data["BeamLocalFrame"]) == {0}
    path = str(tmp_path / "mixed.fdh5")
    result.save(path)
    loaded = fd.read_data(path)
    np.testing.assert_allclose(loaded.gausspoint_data["BeamLocalFrame"][0], expected)


def test_field_unavailable_on_every_submesh_obeys_ignore_missing():
    problem, assembly, _, _ = beam_output_problem(mixed=True)
    with pytest.raises(NameError, match="not available on any assembly"):
        problem.get_results(assembly, ["Unavailable"])
    result = problem.get_results(
        assembly, ["Unavailable", "Stress"], ignore_missing=True
    )
    assert "Unavailable" not in result.field_names()
    assert "Stress" in result.field_names()


@pytest.mark.parametrize("mixed", [False, True])
def test_invalid_beam_position_is_not_treated_as_missing(mixed):
    problem, assembly, _, _ = beam_output_problem(torsion=True, mixed=mixed)
    with pytest.raises(ValueError, match="scalar"):
        problem.get_results(assembly, ["Stress"], position=1, ignore_missing=True)


@pytest.mark.parametrize("position", [1, (1, 1), np.array([[0, 0], [1, 1]])])
def test_mixed_beam_shell_position_has_explicit_error(position):
    problem, _, beam, _ = beam_output_problem(torsion=True)
    beam.sv["BeamStress"][:] = 0
    nodes = np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    shell = fd.Assembly.create(
        fd.weakform.PlateEquilibrium(
            fd.constitutivelaw.ShellHomogeneous(material, 0.1)
        ),
        fd.Mesh(nodes, np.array([[0, 1, 2, 3]]), "quad4"),
    )
    shell.sv["Stress"] = StressTensorList(np.zeros((6, shell.n_gauss_points)))
    shell.sv["ShellStress"] = np.zeros((8, shell.n_gauss_points))
    assembly = fd.Assembly.sum(beam, shell)
    with pytest.raises(
        ValueError, match="Mixed beam/shell output requires position=None"
    ):
        problem.get_results(
            assembly, ["Stress_vm"], position=position, ignore_missing=True
        )
    result = problem.get_results(assembly, ["Stress"], position=None)
    assert set(result.gausspoint_data["Stress"]) == {0, 1}
    # A position is irrelevant to resultants and valid for a single family.
    problem.get_results(assembly, ["BeamStress", "ShellStress"], position=position)
    problem.get_results(beam, ["Stress"], position=(1, 1))
    problem.get_results(shell, ["Stress"], position=1)
