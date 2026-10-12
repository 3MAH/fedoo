"""Analytical section recovery, frames, serialized outputs and field labels."""

from types import SimpleNamespace


import numpy as np
import pytest
import fedoo as fd
from fedoo.util import BeamStressList, BeamStrainList
from fedoo.util.field_components import get_field_components


def section(radius=2):
    return fd.constitutivelaw.BeamCircular(
        fd.constitutivelaw.ElasticIsotrop(1000, 0.25), radius
    )


def test_axial_bending_kinematic_recovery():
    sec = section()
    # Local bending sign: eps_xx = eps_0 + z*kappa_y - y*kappa_z.
    e = BeamStrainList([0.01, 0, 0, 0, 0.002, 0.003])
    s = e.get_stress(sec, (0.5, -0.25))
    np.testing.assert_allclose(s.asarray(), [6, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(
        e.get_strain(sec, (0.5, -0.25)).asarray(), [0.006, 0, 0, 0, 0, 0]
    )
    assert s.von_mises() == pytest.approx(6)


def test_kinematic_strain_needs_no_material_or_shear_correction():
    # A material exposing no isotropic elastic constants is sufficient.
    sec = fd.constitutivelaw.BeamCircular(
        SimpleNamespace(name="geometry_only"), 2, k=0.3
    )
    values = BeamStrainList([0.01, 0.02, -0.03, 0.004, 0.002, 0.003])
    local = values.get_strain(sec, (0.5, -0.25))
    np.testing.assert_allclose(local.asarray(), [0.006, 0, 0, 0.022, -0.026, 0])
    frame = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    rotated = values.get_strain(sec, (0.5, -0.25), local_frame=frame)
    np.testing.assert_allclose(rotated.asarray(), [0, 0.006, 0, -0.022, 0, -0.026])
    sec.k = 0
    np.testing.assert_allclose(
        values.get_strain(sec, (0.5, -0.25)).asarray(), local.asarray()
    )


def test_non_circular_strain_requires_warping_only_for_nonzero_twist():
    sec = fd.constitutivelaw.BeamProperties(section().material, 8, 2, 3, 4)
    np.testing.assert_allclose(
        BeamStrainList([0.01, 0.02, 0.03, 0, 0, 0]).get_strain(sec).asarray(),
        [0.01, 0, 0, 0.02, 0.03, 0],
    )
    with pytest.raises(NotImplementedError, match="warping"):
        BeamStrainList([0, 0, 0, 0.001, 0, 0]).get_strain(sec)


@pytest.mark.parametrize("pipe", [False, True])
def test_circular_torsion_and_average_shear(pipe):
    sec = section()
    if pipe:
        sec = fd.constitutivelaw.BeamPipe(sec.material, 1, 2)
    q = BeamStressList([0, 3 * sec.A, -2 * sec.A, 4 * sec.Jx, 0, 0])
    expected_z = 2 if pipe else 1.5
    np.testing.assert_allclose(
        q.get_stress(sec, (0.5, 0.25)).asarray(), [0, 0, 0, 1, expected_z, 0]
    )
    np.testing.assert_allclose(
        q.get_strain(sec, (0.5, 0.25)).asarray(),
        [0, 0, 0, 1 / sec.material.G, expected_z / sec.material.G, 0],
    )


def test_rectangular_axes_and_unsupported_torsion():
    sec = fd.constitutivelaw.BeamRectangular(section().material, a=2, b=4)
    assert sec.Iyy == pytest.approx(2 * 4**3 / 12)
    assert sec.Izz == pytest.approx(4 * 2**3 / 12)
    # Quadrature of section stress recovers both bending moments.
    points, weights = np.polynomial.legendre.leggauss(3)
    y, z = np.meshgrid(points * sec.a / 2, points * sec.b / 2)
    w = np.outer(weights, weights) * sec.A / 4
    stress = BeamStressList([5, 0, 0, 0, 7, 11]).get_stress(
        sec, (y / (sec.a / 2), z / (sec.b / 2))
    )[0]
    assert np.sum(stress * w) == pytest.approx(5)
    assert np.sum(z * stress * w) == pytest.approx(7)
    assert np.sum(-y * stress * w) == pytest.approx(11)
    np.testing.assert_allclose(sec.torsion_stress(1, 0, 0), [0, 0])


def test_local_to_global_tensor_rotation():
    sec = section()
    # Local x is global y; local y is negative global x.
    frame = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    q = BeamStressList([10 * sec.A, 2 * sec.A, 0, 0, 0, 0])
    np.testing.assert_allclose(
        q.get_stress(sec, local_frame=frame).asarray(), [0, 10, 0, -8 / 3, 0, 0]
    )
    np.testing.assert_allclose(
        q.get_strain(sec, local_frame=frame).asarray(),
        [-0.0025, 0.01, -0.0025, -8 / (3 * sec.material.G), 0, 0],
    )
    frames = np.tile(frame.reshape(9, 1), (1, 2))
    np.testing.assert_allclose(
        q.get_stress(sec, local_frame=frames).asarray(),
        np.tile(np.array([0, 10, 0, -8 / 3, 0, 0])[:, None], (1, 2)),
    )


@pytest.mark.parametrize("a,b", [(2, 4), (4, 2)])
def test_rectangle_torsion_resultant_faces_and_elastic_consistency(a, b):
    sec = fd.constitutivelaw.BeamRectangular(section().material, a, b)
    # Both section-axis branches, using enough quadrature for the resultant
    # tolerance without repeating the development-time aspect-ratio sweep.
    points, weights = np.polynomial.legendre.leggauss(32)
    y, z = np.meshgrid(points * a / 2, points * b / 2)
    w = np.outer(weights, weights) * sec.A / 4
    ty, tz = sec.torsion_stress(7, y, z)
    assert np.sum((y * tz - z * ty) * w) == pytest.approx(7, rel=2e-5)
    assert np.sum(ty * w) == pytest.approx(0, abs=1e-10)
    assert np.sum(tz * w) == pytest.approx(0, abs=1e-10)
    np.testing.assert_allclose(
        sec.torsion_stress(7, a / 2, points * b / 2)[0], 0, atol=1e-12
    )
    np.testing.assert_allclose(
        sec.torsion_stress(7, points * a / 2, b / 2)[1], 0, atol=1e-12
    )
    # Elastic consistency needs representative points, not another full grid.
    sample_y = np.array([0, 0.1, -0.2]) * a
    sample_z = np.array([0, -0.2, 0.1]) * b
    twist = 7 / (sec.material.G * sec.Jx)
    np.testing.assert_allclose(
        np.asarray(sec.torsion_strain(twist, sample_y, sample_z)) * sec.material.G,
        sec.torsion_stress(7, sample_y, sample_z),
        atol=1e-12,
    )


def test_square_and_thin_rectangle_torsion_constants():
    material = section().material
    square = fd.constitutivelaw.BeamRectangular(material, 2, 2)
    assert square.Jx / 2**4 == pytest.approx(0.140577, rel=1e-5)
    # Retain the thin-section limit without integrating a large stress grid.
    thin = fd.constitutivelaw.BeamRectangular(material, 100, 0.1)
    assert thin.Jx == pytest.approx(100 * 0.1**3 / 3, rel=1e-3)


def test_section_shear_resultants_and_varying_geometry():
    material = section().material
    rect = fd.constitutivelaw.BeamRectangular(material, 2, 4)
    points, weights = np.polynomial.legendre.leggauss(8)
    y, z = np.meshgrid(points, 2 * points)
    w = np.outer(weights, weights) * rect.A / 4
    ty, tz = rect.shear_stress(3, -5, y, z)
    np.testing.assert_allclose([np.sum(ty * w), np.sum(tz * w)], [3, -5])
    np.testing.assert_allclose(rect.shear_stress(3, -5, 1, 2), [0, 0])
    disk = section()
    # Integrate chord-average stress using exact circular area moments.
    assert disk.shear_stress(3, 0, 0, 0)[0] == pytest.approx(4 / disk.A)
    assert 4 * 3 / (3 * disk.A) * (disk.A - disk.Izz / disk.r**2) == pytest.approx(3)
    np.testing.assert_allclose(disk.shear_stress(3, -5, disk.r, disk.r), [0, 0])
    varying = fd.constitutivelaw.BeamRectangular(
        material, np.array([2.0, 4.0]), np.array([4.0, 2.0])
    )
    for index in range(2):
        constant = fd.constitutivelaw.BeamRectangular(
            material, varying.a[index], varying.b[index]
        )
        np.testing.assert_allclose(
            np.asarray(varying.torsion_stress(3, 0.2, 0.3))[:, index],
            constant.torsion_stress(3, 0.2, 0.3),
        )


def test_component_registry_and_mixed_zero_components():
    q = BeamStressList([np.array([1, 2]), 0, 0, 0, 0, 0])
    assert q.asarray().shape == (6, 2)
    np.testing.assert_array_equal(q["N"], [1, 2])
    ds = fd.DataSet()
    ds.gausspoint_data["BeamStress"] = q.asarray()
    ds.gausspoint_data["BeamStrain"] = BeamStrainList([1, 2, 3, 4, 5, 6]).asarray()[
        :, None
    ]
    ds.gausspoint_data["BeamLocalFrame"] = np.arange(9)[:, None]
    assert ds["BeamStrain", "kappaY"][0] == 5
    assert ds["BeamLocalFrame", "XY"][0] == 1
    np.testing.assert_array_equal(ds["BeamStress", "N"], [1, 2])
    assert get_field_components("BeamStress", 6) == list(q.components)
    assert get_field_components("Custom", 3) == ["0", "1", "2"]
    assert get_field_components("Rot", 1) == ["Z", "norm"]


@pytest.mark.parametrize("dimension", ["2D", "3D"])
def test_beam_output_and_offline_frame_roundtrip(tmp_path, dimension):
    fd.ModelingSpace(dimension)
    sec = section()
    # Beam aligned with global Y, to distinguish local XX from global YY.
    mesh = fd.Mesh(np.array([[0, 0], [0, 10]], dtype=float), np.array([[0, 1]]), "lin2")
    if dimension == "3D":
        mesh = mesh.as_ndim(3)
    wf = fd.weakform.BeamEquilibrium(sec)
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.Linear(assembly)
    pb.bc.add("Dirichlet", [0], ["Disp", "Rot"], 0)
    pb.bc.add("Dirichlet", [1], "DispY", 0.1)
    pb.apply_boundary_conditions()
    pb.solve()
    result = pb.get_results(
        assembly,
        ["Stress", "Strain", "BeamStress", "BeamStrain", "BeamLocalFrame"],
        position=(0, 0),
    )
    np.testing.assert_allclose(result["Stress", "XX"], 10)
    np.testing.assert_allclose(result["Strain", "XX"], 0.01)
    vm = pb.get_results(assembly, ["Stress_vm", "Stress_pc"], position=(0, 0))
    np.testing.assert_allclose(vm["Stress_vm"], 10)
    nodal = pb.get_results(
        assembly, ["Stress", "Strain"], output_type="Node", position=(0, 0)
    )
    np.testing.assert_allclose(nodal["Stress", "XX"], 10)
    path = tmp_path / "beam.fdh5"
    result.save(str(path))
    restored = fd.read_data(str(path))
    global_stress = BeamStressList(restored["BeamStress"]).get_stress(
        sec, position=(0, 0), local_frame=restored["BeamLocalFrame"]
    )
    np.testing.assert_allclose(global_stress["YY"], 10)
    np.testing.assert_allclose(global_stress["XX"], 0, atol=1e-12)

    # Generalized fields alone now suffice after an FDH5 round trip.
    only = pb.get_results(assembly, ["BeamStress", "BeamStrain", "BeamLocalFrame"])
    only.save(str(tmp_path / "generalized.fdh5"))
    automatic = fd.read_data(str(tmp_path / "generalized.fdh5"))
    assert "Stress" not in automatic.gausspoint_data
    assert {"Stress", "Strain", "Stress_local", "Stress_global"} <= set(
        automatic.field_names()
    )
    np.testing.assert_allclose(automatic["Stress", "vm"], 10)
    np.testing.assert_allclose(automatic["Stress", "I"], 10)
    np.testing.assert_allclose(automatic["Stress_global", "YY"], 10)
    np.testing.assert_allclose(automatic["Strain_global", "YY"], 0.01)
    np.testing.assert_allclose(automatic["Stress_global", "YY", "Node"], 10)


def test_nonlinear_current_frame_and_add_output(tmp_path):
    fd.ModelingSpace("2D")
    sec = section()
    mesh = fd.Mesh(np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([[0, 1]]), "lin2")
    wf = fd.weakform.BeamEquilibrium(sec)
    wf += fd.weakform.ArtificialDamping(
        c_stab=1e-6, variables=["Rot"], energy_fraction=False
    )
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly, nlgeom=True)
    pb.bc.add("Dirichlet", [0], "Disp", 0)
    pb.bc.add("Dirichlet", [0, 1], "RotZ", np.pi / 2)
    pb.bc.add("Dirichlet", [1], "DispX", -10)
    pb.bc.add("Dirichlet", [1], "DispY", 11)
    res = pb.add_output(
        str(tmp_path / "rotated"),
        assembly,
        ["BeamStress", "BeamStrain", "BeamLocalFrame", "Stress", "Strain"],
        position=(0, 0),
    )
    private_res = pb.add_output(
        str(tmp_path / "private_displacement"), assembly, ["Disp"], private=True
    )
    pb.nlsolve(dt=0.25, update_dt=False, print_info=0)
    private_res.load(-1)
    assert "Disp" not in private_res.node_data
    np.testing.assert_allclose(
        private_res.node_data["_Disp"], pb.get_results(assembly, ["Disp"])["Disp"]
    )
    res.load(-1)
    frame = res["BeamLocalFrame"].T.reshape(-1, 3, 3)
    np.testing.assert_allclose(
        frame[:, 0, :], np.tile([0, 1, 0], (len(frame), 1)), atol=1e-12
    )
    recovered = BeamStressList(res["BeamStress"]).get_stress(
        sec, position=(0, 0), local_frame=res["BeamLocalFrame"]
    )
    # The existing updated-geometry beam strain uses the current length (11).
    np.testing.assert_allclose(recovered["YY"], sec.material.E / 11, rtol=1e-10)
    np.testing.assert_allclose(recovered["XX"], 0, atol=1e-10)


def test_shell_position_recovery_is_preserved():
    sec = fd.constitutivelaw.ShellHomogeneous(section().material, thickness=2)
    assembly = SimpleNamespace(
        sv={"ShellStrain": [0.01, 0, 0, 0.002, 0, 0, 0, 0]},
        n_gauss_points=1,
        n_elm_gp=1,
        convert_data=lambda value: value,
    )
    assert sec.get_strain(assembly, position=1)[0] == pytest.approx(0.008)
    assert sec.get_strain(assembly, position=-1)[0] == pytest.approx(0.012)
    assert (
        sec.get_stress(assembly, position=1)[0]
        < sec.get_stress(assembly, position=-1)[0]
    )
