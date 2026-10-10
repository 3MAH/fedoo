"""Section-point storage, normalized coordinates and unified result selection."""

from types import SimpleNamespace
import numpy as np
import pytest
import fedoo as fd


def beam_data(stored=True, generalized=True):
    section = fd.constitutivelaw.BeamRectangular(
        fd.constitutivelaw.ElasticIsotrop(1000, 0.25),
        np.array([2.0, 4.0]),
        np.array([4.0, 2.0]),
        output_points=[[-1, 0], [0, 0], [1, 0]],
    )
    values = np.array(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [2.0, -3.0]]
    )
    assembly = SimpleNamespace(sv={"BeamStress": values}, n_gauss_points=2)
    data = fd.DataSet(
        fd.Mesh(np.array([[0.0, 0.0], [1.0, 0.0]]), np.array([[0, 1]]), "lin2")
    )
    description = section.section_description()
    data.field_metadata["_BeamSection"] = description
    if generalized:
        data.gausspoint_data["BeamStress"] = values
    if stored:
        data.gausspoint_data["Stress"] = section.get_stress(assembly).asarray()
        data.field_metadata["Stress"] = dict(
            description, stored_points=section.output_points.tolist()
        )
    return data, section, assembly


def test_normalized_multi_point_output_and_scalar_rejection():
    data, section, assembly = beam_data()
    assert data.gausspoint_data["Stress"].shape == (6, 3, 2)
    assert section.get_stress(assembly, [[0, 0], [1, 1]]).asarray().shape == (6, 2, 2)
    with pytest.raises(ValueError, match="scalar"):
        section.get_stress(assembly, 1)


@pytest.mark.parametrize("source", ["stored", "recomputed", "auto"])
def test_signed_reduction_and_unambiguous_updates(source):
    data, _, _ = beam_data()
    data.beam_options.update(source=source)
    raw = data.gausspoint_data["Stress"][0]
    expected = raw[0]  # first point wins equal-magnitude ties
    np.testing.assert_allclose(data["Stress", "XX"], expected)
    data.beam_options.update(reduction=None, point_index=2)
    np.testing.assert_allclose(data["Stress", "XX"], raw[2])
    data.beam_options.update(position=(0, 0))
    assert data.beam_options["point_index"] is None
    np.testing.assert_allclose(data["Stress", "XX"], 0)
    before = dict(data.beam_options)
    with pytest.raises(ValueError, match="not both"):
        data.beam_options.update(position=(1, 0), point_index=1)
    assert data.beam_options == before
    with pytest.raises(ValueError, match="scalar"):
        data.beam_options.update(position=1)
    data.beam_options.update(reduction="max", point_index=999)
    np.testing.assert_allclose(data["Stress", "XX"], np.max(raw, axis=0))


def test_stored_selection_matches_coordinates_and_ignores_samples(tmp_path):
    data, _, _ = beam_data()
    data.save(str(tmp_path / "points.fdh5"))
    restored = fd.read_data(str(tmp_path / "points.fdh5"))
    assert restored.gausspoint_data["Stress"].shape == (6, 3, 2)
    restored.beam_options.update(
        source="stored", samples=-123, reduction=None, position=(1, 0)
    )
    np.testing.assert_allclose(
        restored["Stress", "XX"], data.gausspoint_data["Stress"][0, 2]
    )
    restored.beam_options.update(position=(0.3, 0))
    np.testing.assert_allclose(restored["Stress", "XX"], 0)
    restored.beam_options.update(stored_position="exact")
    with pytest.raises(ValueError, match="does not match"):
        restored["Stress", "XX"]
    restored.beam_options.update(source="recomputed")
    np.testing.assert_allclose(
        restored["Stress", "XX"], 0.3 * data.gausspoint_data["Stress"][0, 2]
    )


def test_missing_sources_and_missing_point_selection():
    data, _, _ = beam_data(stored=True, generalized=False)
    assert "Stress" in data.field_names()
    assert np.all(np.isfinite(data["Stress", "vm"]))
    data.beam_options.update(source="recomputed")
    with pytest.raises(ValueError, match="BeamStress"):
        data["Stress", "XX"]
    data, _, _ = beam_data(stored=False)
    data.beam_options.update(source="stored")
    with pytest.raises(ValueError, match="unavailable"):
        data["Stress", "XX"]
    data.beam_options.update(source="recomputed", reduction=None)
    with pytest.raises(ValueError, match="provide position"):
        data["Stress", "XX"]


def test_custom_identity_bounds_and_explicit_bounds():
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    custom = fd.constitutivelaw.BeamProperties(material, 1, 1, 1, 1)
    np.testing.assert_allclose(custom.section_coordinates((3, -5)), [3, -5])
    data, _, _ = beam_data(stored=False)
    data.set_beam_section(custom)
    with pytest.raises(ValueError, match="geometry"):
        data["Stress", "XX"]
    data.beam_options.update(reduction=None, position=(3, -5))
    assert np.all(np.isfinite(data["Stress", "XX"]))
    custom = fd.constitutivelaw.BeamProperties(
        material, 1, 1, 1, 1, output_points=[[0, 0]], section_bounds=(-2, 6, -3, 1)
    )
    np.testing.assert_allclose(custom.section_coordinates((-0.5, 0.5)), [-1, 0.5])
    assert custom.output_points.shape == (1, 2)


@pytest.mark.parametrize("kind", ["Stress", "Strain"])
def test_nearest_stored_point_varies_at_each_location(kind):
    data, _, _ = beam_data(generalized=False)
    points = np.array(
        [[[-0.75, 0.8], [0, 0]], [[0.25, -0.4], [0, 0]], [[0.9, 0.2], [0, 0]]]
    )
    tensors = np.arange(36.0).reshape(6, 3, 2)
    data.gausspoint_data = {kind: tensors}
    data.field_metadata[kind] = dict(
        data.field_metadata["_BeamSection"], stored_points=points.tolist()
    )
    data.beam_options.update(reduction=None, position=(0.15, 0))
    expected = tensors[:, [1, 2], [0, 1]]
    np.testing.assert_array_equal(data[kind], expected)
    # Equal normalized distances select the first point deterministically.
    data.beam_options.update(position=(-0.25, 0))
    np.testing.assert_array_equal(data[kind][:, 0], tensors[:, 0, 0])
    data.beam_options.update(stored_position="exact", position=(0.25, 0))
    with pytest.raises(ValueError, match="every location"):
        data[kind]
    before = dict(data.beam_options)
    with pytest.raises(ValueError, match="stored_position"):
        data.beam_options.update(stored_position="interpolated")
    assert data.beam_options == before


def test_stored_only_capability_survives_fdh5(tmp_path):
    data, section, _ = beam_data()
    section.recovery = "stored_only"
    description = section.section_description()
    data.field_metadata["_BeamSection"] = description
    data.field_metadata["Stress"]["recovery"] = "stored_only"
    path = tmp_path / "stored_only.fdh5"
    data.save(str(path))
    restored = fd.read_data(str(path))
    assert restored.beam_section_description()["recovery"] == "stored_only"
    assert type(section).from_section_description(description).recovery == "stored_only"
    restored.beam_options.update(reduction=None, position=(0.7, 0))
    np.testing.assert_allclose(
        restored["Stress", "XX"], data.gausspoint_data["Stress"][0, 2]
    )
    restored.beam_options.update(source="recomputed")
    with pytest.raises(ValueError, match="stored_only"):
        restored["Stress", "XX"]
    del restored.gausspoint_data["Stress"]
    assert "Stress" not in restored.beam_derived_fields()
    assert "Stress_local" not in restored.beam_derived_fields()
    section.recovery = "invalid"
    with pytest.raises(ValueError, match="recovery"):
        section.section_description()


@pytest.mark.parametrize("preferred", ["stored", "recomputed"])
def test_source_preference_falls_back_only_outside_reference(preferred):
    reference, _, _ = beam_data()
    stored_only, _, _ = beam_data()
    recomputed_only, _, _ = beam_data(stored=False)
    stored_only.field_metadata["_BeamSection"]["recovery"] = "stored_only"
    for index, data in enumerate((reference, stored_only)):
        data.gausspoint_data["Stress"][:] = 100 + index
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    combined = fd.DataSet(
        fd.MultiMesh.from_mesh_list(
            [fd.Mesh(nodes, np.array([[i, i + 1]]), "lin2") for i in range(3)]
        )
    )
    sections = (reference, stored_only, recomputed_only)
    combined.gausspoint_data = {
        "BeamStress": {
            i: data.gausspoint_data["BeamStress"] for i, data in enumerate(sections)
        },
        "Stress": {
            i: data.gausspoint_data["Stress"] for i, data in enumerate(sections[:2])
        },
    }
    combined.field_metadata = {
        name: {
            "submeshes": {
                str(i): data.field_metadata[name]
                for i, data in enumerate(sections)
                if name in data.field_metadata
            }
        }
        for name in ("_BeamSection", "Stress")
    }
    combined.beam_options.update(
        source=preferred,
        source_fallback=True,
        reference_submesh=0,
        reduction=None,
        position=(0.7, 0),
    )
    result = combined.get_data("Stress", "XX")
    reference.beam_options.update(source=preferred, reduction=None, position=(0.7, 0))
    np.testing.assert_allclose(result.submesh(0), reference["Stress", "XX"])
    np.testing.assert_allclose(result.submesh(1), 101)
    recomputed_only.beam_options.update(
        source="recomputed", reduction=None, position=(0.7, 0)
    )
    np.testing.assert_allclose(result.submesh(2), recomputed_only["Stress", "XX"])
    # Choosing an incompatible reference must still reject the explicit source.
    combined.beam_options.update(
        reference_submesh=1 if preferred == "recomputed" else 2
    )
    with pytest.raises(ValueError, match="stored_only|unavailable"):
        combined.get_data("Stress", "XX")
    # API explicit sources remain strict unless fallback is enabled.
    combined.beam_options.update(source_fallback=False, reference_submesh=0)
    with pytest.raises(ValueError, match="stored_only|unavailable"):
        combined.get_data("Stress", "XX")


def test_solver_output_default_and_shell_default(tmp_path):
    fd.ModelingSpace("2D")
    section = fd.constitutivelaw.BeamCircular(
        fd.constitutivelaw.ElasticIsotrop(1000, 0.25), 2, output_points=[[0, 0], [1, 0]]
    )
    mesh = fd.Mesh(np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([[0, 1]]), "lin2")
    assembly = fd.Assembly.create(fd.weakform.BeamEquilibrium(section), mesh)
    pb = fd.problem.Linear(assembly)
    pb.bc.add("Dirichlet", [0], ["Disp", "Rot"], 0)
    pb.bc.add("Dirichlet", [1], "DispX", 0.1)
    pb.apply_boundary_conditions()
    pb.solve()
    result = pb.get_results(assembly, ["Stress", "BeamStress", "BeamLocalFrame"])
    assert result.gausspoint_data["Stress"].shape[:2] == (6, 2)
    np.testing.assert_allclose(result["Stress", "XX"], 10)
    result.beam_options.update(source="recomputed")
    np.testing.assert_allclose(result["Stress", "XX"], 10)
    with pytest.raises(ValueError, match="scalar"):
        pb.get_results(assembly, ["Stress"], position=1)
    node = pb.get_results(assembly, ["Stress"], output_type="Node")
    assert node.node_data["Stress"].shape == (6, 2, 2)
    np.testing.assert_allclose(node["Stress", "XX"], 10)


def test_pipe_sampling_follows_both_varying_radii(tmp_path):
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    section = fd.constitutivelaw.BeamPipe(
        material, np.array([0.5, 1.5]), np.array([2.0, 3.0])
    )
    points = section.output_points
    assert points.shape[1:] == (2, 2)
    physical = [section.section_coordinates(tuple(point)) for point in points]
    radius = np.hypot(np.array(physical)[:, 0], np.array(physical)[:, 1])
    np.testing.assert_allclose(np.min(radius, axis=0), section.r_int)
    np.testing.assert_allclose(np.max(radius, axis=0), section.r_ext)
    data, _, assembly = beam_data()
    assembly.sv["BeamStress"][:] = 0
    assembly.sv["BeamStress"][3] = [1, 2]
    data.gausspoint_data["Stress"] = section.get_stress(assembly).asarray()
    description = section.section_description()
    data.field_metadata = {
        "_BeamSection": description,
        "Stress": dict(description, stored_points=points.tolist()),
    }
    data.save(str(tmp_path / "pipe.fdh5"))
    restored = fd.read_data(str(tmp_path / "pipe.fdh5"))
    restored.beam_options.update(reduction=None, point_index=3)
    expected = data.gausspoint_data["Stress"][:, 3]
    np.testing.assert_allclose(restored["Stress"], expected)


def test_shell_none_still_selects_top_face():
    fd.ModelingSpace("3D")
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    section = fd.constitutivelaw.ShellHomogeneous(material, thickness=2)
    mesh = fd.mesh.rectangle_mesh(nx=2, ny=2, elm_type="quad4", ndim=3)
    assembly = fd.Assembly.create(fd.weakform.PlateEquilibrium(section), mesh)
    pb = fd.problem.Linear(assembly)
    assembly.sv["ShellStrain"] = [
        np.full(assembly.n_gauss_points, 0.01),
        0,
        0,
        0.002,
        0,
        0,
        0,
        0,
    ]
    top = pb.get_results(assembly, ["Stress"], position=1)
    default = pb.get_results(assembly, ["Stress"])
    bottom = pb.get_results(assembly, ["Stress"], position=-1)
    np.testing.assert_allclose(default["Stress", "XX"], top["Stress", "XX"])
    assert np.all(default["Stress", "XX"] > bottom["Stress", "XX"])


def test_stored_element_recovery_and_ambiguous_nodal_frames():
    data, _, _ = beam_data()
    rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    data.gausspoint_data["BeamLocalFrame"] = np.tile(rotation.reshape(9, 1), (1, 2))
    tensors = data.gausspoint_data.pop("Stress")
    averaged = np.mean(tensors, axis=-1, keepdims=True)
    data.element_data["Stress"] = averaged
    data.beam_options.update(source="stored", reduction=None, point_index=0)
    np.testing.assert_allclose(data["Stress", "XX"], averaged[0, 0])
    np.testing.assert_allclose(data["Stress_global", "YY"], averaged[0, 0])
    data.element_data.clear()
    data.node_data["Stress"] = np.repeat(averaged, 2, axis=-1)
    np.testing.assert_allclose(data["Stress", "XX"], np.repeat(averaged[0, 0], 2))
    with pytest.raises(ValueError, match="ambiguous"):
        data["Stress_global", "YY"]
    data.beam_options.update(source="recomputed", position=(0, 0))
    assert np.all(np.isfinite(data.get_data("Stress_global", "YY", "Node")))


@pytest.mark.parametrize("association", ["GaussPoint", "Element"])
def test_stored_points_on_multiple_submeshes(tmp_path, association):
    first, _, _ = beam_data(generalized=False)
    second, _, _ = beam_data(generalized=False)
    second.field_metadata["Stress"]["stored_points"] = [[-1, 0], [1, 0], [0, 0]]
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    meshes = [
        fd.Mesh(nodes, np.array([[0, 1]]), "lin2"),
        fd.Mesh(nodes, np.array([[2, 3]]), "lin2"),
    ]
    combined = fd.DataSet(fd.MultiMesh.from_mesh_list(meshes))
    raw = first.gausspoint_data["Stress"]
    if association == "Element":
        raw = np.mean(raw, axis=-1, keepdims=True)
    storage = (
        combined.gausspoint_data
        if association == "GaussPoint"
        else combined.element_data
    )
    storage["Stress"] = {0: raw, 1: 2 * raw}
    combined.field_metadata = {
        name: {
            "submeshes": {
                "0": first.field_metadata[name],
                "1": second.field_metadata[name],
            }
        }
        for name in ("_BeamSection", "Stress")
    }
    combined.save(str(tmp_path / "multi.fdh5"))
    restored = fd.read_data(str(tmp_path / "multi.fdh5"))
    restored.beam_options.update(source="stored", reduction=None, point_index=0)
    value, category = restored.get_data("Stress", "XX", return_data_type=True)
    assert category == association
    np.testing.assert_allclose(value.submesh(0), raw[0, 0])
    np.testing.assert_allclose(value.submesh(1), 2 * raw[0, 0])
    restored.beam_options.update(position=(0.8, 0))
    nearest = restored.get_data("Stress", "XX")
    np.testing.assert_allclose(nearest.submesh(0), raw[0, 2])
    np.testing.assert_allclose(nearest.submesh(1), 2 * raw[0, 1])
