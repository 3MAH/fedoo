"""Lazy saved-field recovery, section envelopes and backwards compatibility."""

import h5py
import json


from types import SimpleNamespace
import numpy as np
import pytest
import fedoo as fd
from fedoo.util.beam_recovery import SavedBeamSection, section_description
from fedoo.util import BeamStressList


def test_native_section_storage_is_shared_and_typed(tmp_path):
    data, section = dataset()
    path = tmp_path / "native.fdh5"
    data.save(str(path))
    with h5py.File(path) as file:
        iteration = file["results/iter_0"]
        assert "field_metadata" not in iteration
        metadata = iteration["section_metadata/submesh_0"]
        assert metadata["geometry"].asstr()[()] == "circle"
        assert metadata["properties/r"].shape == ()
        assert metadata["properties/r"][()] == section.r
        np.testing.assert_allclose(metadata["reference_bounds"], [-1, 1, -1, 1])


@pytest.mark.parametrize("association", ["Node", "Element", "GaussPoint"])
def test_varying_section_properties_keep_association(tmp_path, association):
    data, section = dataset()
    radius = {
        "Node": np.array([1.0, 2.0]),
        "Element": np.array([2.0]),
        "GaussPoint": np.arange(1.0, 5.0),
    }[association]
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    section = fd.constitutivelaw.BeamCircular(material, radius)
    section.section_property_associations = {
        name: association for name in ("r", "A", "Iyy", "Izz", "Jx")
    }
    data.set_beam_section(section)
    area = data.mesh.convert_data(
        section.A, convert_from=association, convert_to="GaussPoint", n_elm_gp=4
    )
    data.gausspoint_data["BeamStress"][:] = 0
    data.gausspoint_data["BeamStress"][0] = 7 * area
    data.beam_options["position"] = (0, 0)
    expected = data.get_data("Stress", "XX")
    path = tmp_path / "varying.fdh5"
    data.save(str(path))
    category = {
        "Node": "node_data",
        "Element": "element_data",
        "GaussPoint": "gausspoint_data",
    }[association]
    with h5py.File(path) as file:
        iteration = file["results/iter_0"]
        field_path = (
            f"{category}/_Section_r"
            if association == "Node"
            else f"{category}/submesh_0/_Section_r"
        )
        np.testing.assert_allclose(iteration[field_path], radius)
        assert (
            iteration["section_metadata/submesh_0/properties/r"].attrs["association"]
            == association
        )
    restored = fd.read_data(str(path))
    restored.beam_options.update(reduction=None, position=(1, 0))
    restored.beam_options["position"] = (0, 0)
    np.testing.assert_allclose(restored.get_data("Stress", "XX"), expected)
    np.testing.assert_allclose(expected, 7)


def test_custom_points_round_trip_and_centroid_scaling(tmp_path):
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    points = np.array([[-0.2, -0.5], [0.8, -0.5], [-0.2, 0.5]])
    section = fd.constitutivelaw.BeamProperties(
        material,
        4,
        2,
        3,
        5,
        k=0.9,
        section_points=points,
        section_scale=(np.array([1, 2, 3, 4]), 2),
    )
    data, _ = dataset()
    data.set_beam_section(section)
    data.gausspoint_data["BeamStress"][:] = 0
    data.gausspoint_data["BeamStress"][5] = -section.Izz
    data.beam_options.update(reduction="max", samples=32)
    np.testing.assert_allclose(data.get_data("Stress", "XX"), [0.8, 1.6, 2.4, 3.2])
    path = tmp_path / "custom.fdh5"
    data.save(str(path))
    restored = fd.read_data(str(path))
    restored.beam_options.update(reduction=None, position=(1, 0))
    restored.beam_options.update(reduction="max", samples=32)
    np.testing.assert_allclose(restored.get_data("Stress", "XX"), [0.8, 1.6, 2.4, 3.2])
    restored.beam_options.update(reduction=None, point_index=0)
    np.testing.assert_allclose(
        restored.get_data("Stress", "XX"), [-0.2, -0.4, -0.6, -0.8]
    )


@pytest.mark.parametrize("shape", ["circle", "rectangle", "custom"])
def test_normalized_yz_uses_directional_bounds_with_varying_dimensions(shape):
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    if shape == "circle":
        section = fd.constitutivelaw.BeamCircular(material, np.arange(1.0, 5.0))
        expected = 1.5 * np.arange(1.0, 5.0)
    elif shape == "rectangle":
        section = fd.constitutivelaw.BeamRectangular(material, 2, 6)
        expected = np.full(4, 3.5)
    else:
        section = fd.constitutivelaw.BeamProperties(
            material,
            4,
            2,
            3,
            5,
            section_points=[[-0.2, -0.5], [0.8, 0.5]],
            section_scale=(np.arange(1.0, 5.0), 2),
            section_bounds=(
                -0.2 * np.arange(1.0, 5.0),
                0.8 * np.arange(1.0, 5.0),
                -1,
                1,
            ),
        )
        expected = 0.1 * np.arange(1.0, 5.0) + 1
    data, _ = dataset()
    data.set_beam_section(section)
    data.gausspoint_data["BeamStress"][:] = 0
    data.gausspoint_data["BeamStress"][4] = section.Iyy
    data.gausspoint_data["BeamStress"][5] = section.Izz
    data.beam_options.update(position=(-0.5, 1), position_coordinates="normalized")
    np.testing.assert_allclose(data.get_data("Stress", "XX"), expected)
    data.beam_options["position"] = (0, 0)
    np.testing.assert_allclose(data.get_data("Stress", "XX"), 0)


def test_old_json_section_metadata_remains_readable(tmp_path):
    data, _ = dataset()
    path = tmp_path / "old_json.fdh5"
    data.save(str(path))
    with h5py.File(path, "a") as file:
        group = file["results/iter_0"]
        del group["section_metadata"]
        group.create_dataset(
            "field_metadata",
            data=json.dumps(data.field_metadata),
            dtype=h5py.string_dtype("utf-8"),
        )
    restored = fd.read_data(str(path))
    restored.beam_options.update(reduction=None, position=(1, 0))
    restored.beam_options["position"] = (0.5, 0.25)
    np.testing.assert_allclose(restored.get_data("Stress", "XX"), 3)


def test_section_mesh_interface_supports_mesher_overrides():
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    section = fd.constitutivelaw.BeamRectangular(material, 2, 4)
    mesh = section.section_mesh(n_points=32, nx=7, ny=5)
    assert len(mesh.nodes) == 35
    np.testing.assert_allclose(mesh.nodes.min(axis=0), [-0.5, -0.5])
    np.testing.assert_allclose(mesh.nodes.max(axis=0), [0.5, 0.5])


def test_registered_section_round_trip_requires_no_shape_specific_reader(tmp_path):
    class ExampleSection(fd.constitutivelaw.BeamProperties):
        section_type = "test_registered_section"
        section_dimensions = ("height",)

        def sample_points(self, n_points=32):
            return np.array([[-0.25, 0], [0.75, 0]])

        def scale_section_points(self, points):
            for y, z in points:
                yield y * self.height, z * self.height

    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    section = ExampleSection(material, 4, 2, 3, 5)
    section.height = 2
    data, _ = dataset()
    data.set_beam_section(section)
    data.gausspoint_data["BeamStress"][:] = 0
    data.gausspoint_data["BeamStress"][5] = -5
    path = tmp_path / "registered.fdh5"
    data.save(str(path))
    restored = fd.read_data(str(path))
    restored.beam_options.update(reduction=None, position=(1, 0))
    restored.beam_options.update(reduction="max")
    np.testing.assert_allclose(restored.get_data("Stress", "XX"), 1.5)


def dataset(section=None, frame=False):
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    section = section or fd.constitutivelaw.BeamCircular(material, 2)
    mesh = fd.Mesh(np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([[0, 1]]), "lin2")
    data = fd.DataSet(mesh)
    values = [5 * section.A, section.A, 0, 0, 0, 2 * section.Izz]
    data.gausspoint_data["BeamStress"] = np.tile(np.array(values)[:, None], (1, 4))
    data.gausspoint_data["BeamStrain"] = (
        data.gausspoint_data["BeamStress"]
        / np.array(section.get_beam_rigidity())[:, None]
    )
    description = section_description(section, SimpleNamespace())
    data.field_metadata = {"BeamStress": description, "BeamStrain": description}
    if frame:
        rotation = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
        data.gausspoint_data["BeamLocalFrame"] = np.tile(rotation.reshape(9, 1), (1, 4))
    data.beam_options.update(reduction=None, position=(1, 0))
    return data, section


def test_local_without_frames_and_legacy_file(tmp_path):
    data, sec = dataset()
    data.save(str(tmp_path / "new.fdh5"))
    restored = fd.read_data(str(tmp_path / "new.fdh5"))
    restored.beam_options.update(reduction=None, position=(1, 0))
    assert "Stress" in restored.field_names()
    assert "Stress_global" not in restored.field_names()
    restored.beam_options["position"] = (0.5, 0.25)
    np.testing.assert_allclose(restored["Stress", "XX"], 3)
    np.testing.assert_allclose(restored["Strain", "XX"], 0.003)
    np.testing.assert_allclose(restored["Strain", "YY"], 0)
    assert not {"E", "G", "nu"} & set(
        restored.field_metadata["BeamStrain"]["properties"]
    )
    data.field_metadata = {}
    data.save(str(tmp_path / "legacy.fdh5"))
    legacy = fd.read_data(str(tmp_path / "legacy.fdh5"))
    legacy.beam_options.update(reduction=None, position=(1, 0))
    assert set(legacy.field_names()) == {"BeamStress", "BeamStrain"}
    with pytest.raises(NameError):
        legacy.get_data("Stress", "vm")
    legacy.set_beam_section(sec)
    legacy.beam_options["position"] = (0.5, 0.25)
    np.testing.assert_allclose(legacy["Stress", "0"], 3)
    legacy.save(str(tmp_path / "upgraded.fdh5"))
    upgraded = fd.read_data(str(tmp_path / "upgraded.fdh5"))
    upgraded.beam_options.update(reduction=None, position=(1, 0))
    assert "Stress" in upgraded.field_names()


def test_saved_strain_recovers_with_geometry_only(tmp_path):
    data, _ = dataset(frame=True)
    data.gausspoint_data.pop("BeamStress")
    data.field_metadata = {
        "BeamStrain": {"version": 1, "geometry": "circle", "properties": {"r_ext": 2.0}}
    }
    data.gausspoint_data["BeamStrain"][:] = np.array(
        [0.01, 0.02, -0.03, 0.004, 0.002, 0.003]
    )[:, None]
    path = str(tmp_path / "kinematic.fdh5")
    data.save(path)
    restored = fd.read_data(path)
    restored.beam_options.update(reduction=None, position=(1, 0))
    restored.beam_options["position"] = (0.5, -0.25)
    np.testing.assert_allclose(
        restored["Strain_local"],
        np.tile(np.array([0.006, 0, 0, 0.022, -0.026, 0])[:, None], (1, 4)),
    )
    np.testing.assert_allclose(restored["Strain_global", "YY"], 0.006)
    np.testing.assert_allclose(restored["Strain_global", "XY"], -0.022)
    restored.beam_options["reduction"] = "max"
    np.testing.assert_allclose(restored["Strain_local", "XY"], 0.028)
    np.testing.assert_allclose(restored["Strain_local", "YY"], 0)


def test_circle_envelopes_evaluate_vm_before_reduction():
    data, sec = dataset(frame=True)
    data.gausspoint_data["BeamStress"][:] = np.array([0, sec.A, 0, sec.Jx, 0, 0])[
        :, None
    ]
    data.beam_options.update(reduction="max", samples=32)
    np.testing.assert_allclose(data["Stress", "vm"], 10 / 3 * np.sqrt(3))
    np.testing.assert_allclose(data["Stress", "I"], 10 / 3)
    np.testing.assert_allclose(data["Stress", "III"], -1 / 3, atol=1e-14)
    np.testing.assert_allclose(data["Stress_global", "vm"], data["Stress", "vm"])
    # Independent shear maxima combine stresses at different section points.
    assert np.all(data["Stress", "vm"] < np.sqrt(3 * ((10 / 3) ** 2 + 2**2)))
    data.beam_options["reduction"] = "min"
    np.testing.assert_allclose(data["Stress", "III"], -10 / 3)
    with pytest.raises(ValueError, match="scalar component"):
        data.get_data("Stress")


def test_compression_maximum_and_maximum_absolute_are_distinct():
    data, sec = dataset()
    data.gausspoint_data["BeamStress"][0] = -5 * sec.A
    data.beam_options["reduction"] = "max"
    np.testing.assert_allclose(data["Stress", "XX"], -1)
    data.beam_options["reduction"] = "min"
    np.testing.assert_allclose(data["Stress", "XX"], -9)
    data.beam_options["reduction"] = "abs_max"
    np.testing.assert_allclose(data["Stress", "XX"], -9)


def test_oblique_circle_maximum_converges_with_sampling():
    data, sec = dataset()
    angle = 0.37
    data.gausspoint_data["BeamStress"][4] = np.sin(angle) * sec.Iyy
    data.gausspoint_data["BeamStress"][5] = -np.cos(angle) * sec.Izz
    data.beam_options.update(reduction="max", samples=32)
    low = data["Stress", "XX"]
    data.beam_options["samples"] = 128
    high = data["Stress", "XX"]
    exact = 5 + sec.r
    assert np.all(high >= low)
    assert np.max(np.abs(high - exact)) < 0.001


@pytest.mark.parametrize("shape", ["rectangle", "pipe"])
def test_known_section_geometry_envelopes(shape):
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    sec = (
        fd.constitutivelaw.BeamRectangular(material, 2, 4)
        if shape == "rectangle"
        else fd.constitutivelaw.BeamPipe(material, 1, 2)
    )
    data, _ = dataset(sec)
    data.beam_options["reduction"] = "max"
    extent = 1 if shape == "rectangle" else 2
    np.testing.assert_allclose(data["Stress", "XX"], 5 + 2 * extent)
    if shape == "pipe":
        for y, z in SavedBeamSection(data.field_metadata["BeamStress"]).points(32):
            assert np.hypot(y, z) >= 1 - 1e-14


@pytest.mark.parametrize("shape", ["circle", "pipe", "rectangle"])
def test_sampling_budget_counts_points_and_refines_section(shape):
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    section = {
        "circle": fd.constitutivelaw.BeamCircular(material, 2),
        "pipe": fd.constitutivelaw.BeamPipe(material, 1, 2),
        "rectangle": fd.constitutivelaw.BeamRectangular(material, 2, 4),
    }[shape]
    saved = SavedBeamSection(section_description(section))
    previous_points = set()
    previous_radii = 0
    previous_angles = 0
    for budget in (32, 128, 512):
        points = np.asarray(list(saved.points(budget)), dtype=float)
        assert abs(len(points) - budget) <= 0.35 * budget
        assert len(points) == len({tuple(point) for point in np.round(points, 12)})
        if shape in ("circle", "pipe"):
            radii = np.hypot(points[:, 0], points[:, 1])
            assert radii.max() == pytest.approx(2)
            assert radii.min() == pytest.approx(1 if shape == "pipe" else 0)
            n_radii = len(np.unique(np.round(radii, 12)))
            n_angles = len(
                np.unique(
                    np.round(np.arctan2(points[radii > 0, 1], points[radii > 0, 0]), 12)
                )
            )
            assert n_radii > previous_radii
            assert n_angles > previous_angles
            current = {tuple(point) for point in np.round(points, 12)}
            assert previous_points <= current
            previous_points, previous_radii, previous_angles = (
                current,
                n_radii,
                n_angles,
            )
        else:
            assert np.all(np.abs(points[:, 0]) <= 1)
            assert np.all(np.abs(points[:, 1]) <= 2)
            for corner in ((-1, -2), (-1, 2), (1, -2), (1, 2), (0, 0)):
                assert np.any(np.all(points == corner, axis=1))


@pytest.mark.parametrize("budget", [7, 10.5])
def test_sampling_budget_requires_a_valid_integer(budget):
    data, _ = dataset()
    data.beam_options.update(reduction="max", samples=budget)
    with pytest.raises(ValueError, match="integer budget"):
        data.get_data("Stress", "vm")


def test_generic_geometry_fails_explicitly_and_rectangular_torsion_is_recovered(
    tmp_path,
):
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    sec = fd.constitutivelaw.BeamProperties(material, 4, 2, 3, 5, k=0.9)
    data, _ = dataset(sec)
    data.beam_options.update(position=(0, 0))
    np.testing.assert_allclose(data["Stress", "XX"], 5)
    data.beam_options["reduction"] = "max"
    with pytest.raises(ValueError, match="geometry"):
        data.get_data("Stress", "vm")
    rect, _ = dataset(fd.constitutivelaw.BeamRectangular(material, 2, 4))
    rect.gausspoint_data["BeamStress"][3] = 1
    assert np.all(np.isfinite(rect.get_data("Stress", "vm")))
    rect.beam_options.update(position=(0.25, 0.5), reduction=None)
    expected = rect.get_data("Stress", "vm")
    rect.save(str(tmp_path / "rectangle.fdh5"))
    restored = fd.read_data(str(tmp_path / "rectangle.fdh5"))
    restored.beam_options.update(reduction=None, position=(1, 0))
    restored.beam_options.update(position=(0.25, 0.5), reduction=None)
    np.testing.assert_allclose(restored.get_data("Stress", "vm"), expected)


def test_metadata_nonuniform_properties_and_element_subset():
    fd.ModelingSpace("2D")
    material = fd.constitutivelaw.ElasticIsotrop(1000, 0.25)
    sec = fd.constitutivelaw.BeamCircular(material, np.arange(1.0, 9.0))
    mesh = fd.Mesh(
        np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        np.array([[0, 1], [1, 2]]),
        "lin2",
    )
    assembly = fd.Assembly.create(fd.weakform.BeamEquilibrium(sec), mesh)
    description = section_description(sec, assembly, [1])
    np.testing.assert_allclose(description["properties"]["r"], [2, 4, 6, 8])
    saved = SavedBeamSection(description)
    force = BeamStressList([saved.A * 10, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(force.get_stress(saved, position=(0, 0))["XX"], 10)


def test_multiframe_metadata_and_selection_survive_reload(tmp_path):
    data, _ = dataset(frame=True)
    path = str(tmp_path / "history.fdh5")
    data.to_fdh5(path, iteration=0, overwrite=True)
    data.gausspoint_data["BeamStress"] *= 2
    data.to_fdh5(path, iteration=1)
    history = fd.read_data(path)
    history.beam_options.update(reduction=None, position=(1, 0))
    history.load(0)
    history.beam_options.update(position=(0.0, 0.0), reduction="abs_max")
    np.testing.assert_allclose(history["Stress_global", "YY"], 9)
    history.load(1)
    np.testing.assert_allclose(history["Stress_global", "YY"], 18)
    np.testing.assert_allclose(
        history.get_all_frame_lim("Stress_global", "YY")[2], [9, 18]
    )
    clone = history.copy()
    assert clone.beam_options == history.beam_options
    np.testing.assert_allclose(clone["Stress_global", "YY"], 18)
    stored = data.copy()
    stored.gausspoint_data["Stress"] = np.ones((6, 4)) * 123
    np.testing.assert_allclose(stored["Stress", "XX"], 123)
    np.testing.assert_allclose(stored["Stress_local", "XX"], 2)


def test_mixed_mesh_recovery_keeps_properties_and_frames_per_submesh(tmp_path):
    first, _ = dataset()
    second, _ = dataset(frame=True)
    nodes = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.0, 1.0]])
    meshes = [
        fd.Mesh(nodes, np.array([[0, 1]]), "lin2"),
        fd.Mesh(nodes, np.array([[2, 3]]), "lin2"),
    ]
    mesh = fd.MultiMesh.from_mesh_list(meshes)
    combined = fd.DataSet(mesh)
    combined.gausspoint_data = {
        "BeamStress": {0: first["BeamStress"], 1: second["BeamStress"] * 2},
        "BeamLocalFrame": {1: second["BeamLocalFrame"]},
    }
    combined.field_metadata = {
        "BeamStress": {
            "submeshes": {
                "0": first.field_metadata["BeamStress"],
                "1": second.field_metadata["BeamStress"],
            }
        }
    }
    combined.save(str(tmp_path / "mixed.fdh5"))
    restored = fd.read_data(str(tmp_path / "mixed.fdh5"))
    restored.beam_options.update(reduction=None, position=(1, 0))
    restored.beam_options["position"] = (0.0, 0.0)
    local = restored.get_data("Stress", "XX", "Element")
    np.testing.assert_allclose(local.submesh(0), 5)
    np.testing.assert_allclose(local.submesh(1), 10)
    global_stress = restored.get_data("Stress_global", "YY", "Element")
    assert global_stress.submesh(0) is None
    np.testing.assert_allclose(global_stress.submesh(1), 10)
    assert not restored._submesh_has_field("Stress_global", 0)
    assert restored._submesh_has_field("Stress_global", 1)
    restored.active_submesh = 1
    node = restored.get_data("Stress_global", "YY", "Node")
    np.testing.assert_allclose(node[[2, 3]], 10)


def test_legacy_section_attachment_survives_frame_changes_and_resave(tmp_path):
    data, sec = dataset()
    data.field_metadata = {}
    path = str(tmp_path / "old_history.fdh5")
    data.to_fdh5(path, iteration=0, overwrite=True)
    data.gausspoint_data["BeamStress"] *= 2
    data.to_fdh5(path, iteration=1)
    history = fd.read_data(path)
    history.beam_options.update(reduction=None, position=(1, 0))
    history.set_beam_section(sec)
    history.beam_options["position"] = (0, 0)
    np.testing.assert_allclose(history["Stress", "XX"], 10)
    history.load(0)
    np.testing.assert_allclose(history["Stress", "XX"], 5)
    history.save_all(str(tmp_path / "upgraded_history.fdh5"))
    upgraded = fd.read_data(str(tmp_path / "upgraded_history.fdh5"))
    upgraded.beam_options.update(reduction=None, position=(1, 0))
    upgraded.beam_options["position"] = (0, 0)
    np.testing.assert_allclose(upgraded["Stress", "XX"], 10)
    upgraded.load(0)
    np.testing.assert_allclose(upgraded["Stress", "XX"], 5)
