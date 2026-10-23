"""Beam output carries the frames required for saved global tensor recovery."""

import numpy as np
import pytest
import fedoo as fd


@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("field", ["BeamStress", "BeamStrain", "Stress", "Strain"])
@pytest.mark.parametrize("subset", [False, True])
def test_automatic_beam_frames_roundtrip(tmp_path, dimension, field, subset):
    fd.ModelingSpace("2D" if dimension == 2 else "3D")
    nodes = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])[:, :dimension]
    mesh = fd.Mesh(nodes, np.array([[0, 1], [1, 2]]), "lin2")
    section = fd.constitutivelaw.BeamCircular(
        fd.constitutivelaw.ElasticIsotrop(1000, 0.25), 1
    )
    assembly = fd.Assembly.create(fd.weakform.BeamEquilibrium(section), mesh)
    problem = fd.problem.Linear(assembly)
    for name in ("BeamStress", "BeamStrain"):
        values = np.zeros((6, assembly.n_gauss_points))
        values[0] = 1
        assembly.sv[name] = values
    ids = [0] if subset else None
    result = problem.get_results(assembly, [field], element_set=ids)
    expected = section.get_local_frame(assembly).reshape(-1, 9).T
    if subset:
        expected = expected.reshape(9, -1, 2)[..., ids].reshape(9, -1)
    np.testing.assert_allclose(result.gausspoint_data["BeamLocalFrame"], expected)
    kind = "Strain" if "Strain" in field else "Stress"
    result.beam_options.update(position=(0, 0), reduction=None)
    global_values = result[kind + "_global"].copy()
    assert np.max(np.abs(global_values[1])) > 0  # axial beam along global Y
    path = str(tmp_path / "beam.fdh5")
    result.save(path)
    loaded = fd.read_data(path)
    loaded.beam_options.update(position=(0, 0), reduction=None)
    np.testing.assert_allclose(loaded[kind + "_global"], global_values)
