from types import SimpleNamespace
import importlib

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyRotation
from simcoon import Rotation as SimcoonRotation

import fedoo as fd
from fedoo.util.voigt_tensors import StrainTensorList, StressTensorList


def _assembly(mesh, n_elm_gp):
    return SimpleNamespace(mesh=mesh, n_elm_gp=n_elm_gp)


def _z_frames(angles):
    angles = np.asarray(angles, dtype=float)
    rotvec = np.zeros((angles.size, 3))
    rotvec[:, 2] = angles.reshape(-1)
    return ScipyRotation.from_rotvec(rotvec).as_matrix().reshape(angles.shape + (3, 3))


@pytest.mark.parametrize("rotation_class", [ScipyRotation, SimcoonRotation])
def test_constitutive_law_accepts_scipy_and_simcoon_rotations(rotation_class):
    material = fd.constitutivelaw.ElasticAnisotropic(np.eye(6))
    rotation = rotation_class.from_rotvec([0.0, 0.0, 0.25])

    assert material.set_local_frame(rotation) is material
    np.testing.assert_allclose(
        material.get_local_frame(), rotation.as_matrix()[np.newaxis]
    )


def test_element_frames_are_expanded_in_fedoo_gauss_point_order():
    mesh = fd.mesh.rectangle_mesh(3, 2, elm_type="quad4", ndim=3)
    assembly = _assembly(mesh, n_elm_gp=4)
    material = fd.constitutivelaw.ElasticAnisotropic(np.eye(6))
    element_frames = _z_frames([0.1, 0.2])

    material.set_local_frame(element_frames)

    expected = np.tile(element_frames, (assembly.n_elm_gp, 1, 1))
    np.testing.assert_allclose(material.get_local_frame(assembly), expected)


def test_element_by_gauss_point_array_is_reordered_to_fedoo_storage():
    mesh = fd.mesh.rectangle_mesh(3, 2, elm_type="quad4", ndim=3)
    assembly = _assembly(mesh, n_elm_gp=2)
    material = fd.constitutivelaw.ElasticAnisotropic(np.eye(6))
    frames = _z_frames([[0.1, 0.2], [0.3, 0.4]])

    material.set_local_frame(frames, location="GaussPoint")

    expected = frames.transpose(1, 0, 2, 3).reshape(-1, 3, 3)
    np.testing.assert_allclose(material.get_local_frame(assembly), expected)


def test_nodal_frames_are_smoothly_interpolated_on_rotation_manifold():
    mesh = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    assembly = _assembly(mesh, n_elm_gp=4)
    material = fd.constitutivelaw.ElasticAnisotropic(np.eye(6))
    nodal_angles = 0.5 * mesh.nodes[:, 0] + 0.25 * mesh.nodes[:, 1]

    material.set_local_frame(_z_frames(nodal_angles), location="Node")
    gp_frames = np.asarray(material.get_local_frame(assembly))

    expected_angles = mesh._get_node2gausspoint_mat(4) @ nodal_angles
    actual_angles = ScipyRotation.from_matrix(gp_frames).as_rotvec()[:, 2]
    np.testing.assert_allclose(actual_angles, expected_angles)
    np.testing.assert_allclose(
        gp_frames @ gp_frames.transpose(0, 2, 1),
        np.tile(np.eye(3), (len(gp_frames), 1, 1)),
        atol=1e-12,
    )
    np.testing.assert_allclose(np.linalg.det(gp_frames), 1.0, atol=1e-12)


def test_mechanical_field_rotations_round_trip_at_gauss_points():
    mesh = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    assembly = _assembly(mesh, n_elm_gp=4)
    material = fd.constitutivelaw.ElasticAnisotropic(np.eye(6))
    material.set_local_frame(
        _z_frames(np.linspace(0.0, 0.6, mesh.n_nodes)), location="Node"
    )
    stress_global = np.arange(24.0).reshape(6, 4)
    strain_global = 0.01 * stress_global

    stress_local = material.global2local_stress(stress_global, assembly, current=False)
    strain_local = material.global2local_strain(strain_global, assembly, current=False)

    np.testing.assert_allclose(
        material.local2global_stress(stress_local, assembly, current=False),
        stress_global,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        material.local2global_strain(strain_local, assembly, current=False),
        strain_global,
        atol=1e-12,
    )


def test_rotation_increment_is_expressed_in_initial_material_frame():
    mesh = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    assembly = _assembly(mesh, n_elm_gp=1)
    material = fd.constitutivelaw.ElasticAnisotropic(np.eye(6))
    initial = ScipyRotation.from_rotvec([0.3, 0.2, -0.1]).as_matrix()
    increment = ScipyRotation.from_rotvec([-0.2, 0.4, 0.1]).as_matrix()
    material.set_local_frame(initial)

    actual = material.global2local_rotation_increment(increment[:, :, None], assembly)

    expected = initial.T @ increment @ initial
    np.testing.assert_allclose(actual[:, :, 0], expected, atol=1e-12)


def test_native_anisotropic_frame_follows_finite_rotation_without_initial_frame():
    mesh = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    increment = ScipyRotation.from_rotvec([0.0, 0.0, 0.35]).as_matrix()
    assembly = SimpleNamespace(
        mesh=mesh,
        n_elm_gp=1,
        n_gauss_points=1,
        _nlgeom="UL",
        sv={"DR": increment[:, :, None]},
        sv_start={},
    )
    H_local = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    material = fd.constitutivelaw.ElasticAnisotropic(H_local)

    H_global = material.local2global_H(H_local, assembly)

    expected = SimcoonRotation.from_matrix(increment).apply_stiffness(H_local)
    np.testing.assert_allclose(H_global, expected, atol=1e-12)
    np.testing.assert_allclose(assembly.sv["_MaterialFrame"][0], increment, atol=1e-12)


@pytest.mark.parametrize(
    ("umat_name", "expected"),
    [("ELISO", True), ("EPICP", True), ("ELORT", False), ("EPHIL", False)],
)
def test_simcoon_declares_known_material_symmetry(umat_name, expected):
    material = fd.constitutivelaw.Simcoon(umat_name, np.zeros(40))

    assert material.is_isotropic is expected


def test_simcoon_umat_boundary_uses_material_frame(monkeypatch):
    module = importlib.import_module("fedoo.constitutivelaw.simcoon_umat")
    material = fd.constitutivelaw.Simcoon("ELORT", np.zeros(12))
    frame = ScipyRotation.from_rotvec([0.0, 0.0, np.pi / 2]).as_matrix()
    material.set_local_frame(frame)
    strain_start = StrainTensorList(
        np.array([[0.01], [0.02], [0.03], [0.004], [0.005], [0.006]])
    )
    strain_increment = StrainTensorList(
        np.array([[0.001], [0.002], [0.003], [0.0004], [0.0005], [0.0006]])
    )
    stress_start = StressTensorList(
        np.array([[10.0], [20.0], [30.0], [4.0], [5.0], [6.0]])
    )
    local_stress = np.array([[11.0], [22.0], [33.0], [4.4], [5.5], [6.6]])
    local_tangent = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])[:, :, None]
    statev = np.zeros((material.n_statev, 1))
    wm = np.zeros((4, 1))
    captured = {}

    def fake_umat(
        umat_name,
        strain,
        dstrain,
        F0,
        F1,
        stress,
        DR,
        props,
        statev_start,
        time,
        dtime,
        wm_start,
        temp,
        **kwargs,
    ):
        captured.update(
            strain=strain,
            dstrain=dstrain,
            stress=stress,
            DR=DR,
            F0=F0,
            F1=F1,
        )
        return local_stress, statev_start.copy(), wm_start.copy(), local_tangent

    monkeypatch.setattr(module.sim, "umat", fake_umat)
    F0 = np.eye(3)[:, :, None]
    F1 = np.array([[1.1, 0.2, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 1.05]])[:, :, None]
    increment = ScipyRotation.from_rotvec([0.2, -0.1, 0.3]).as_matrix()
    DR = increment[:, :, None]
    assembly = SimpleNamespace(
        mesh=fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3),
        n_elm_gp=1,
        n_gauss_points=1,
        _nlgeom="UL",
        space=SimpleNamespace(get_dimension=lambda: "3D", list_variables=lambda: []),
        sv={
            "Strain": strain_start,
            "DStrain": strain_increment,
            "DR": DR,
            "F": F1,
            "Statev": statev.copy(),
            "Wm": wm.copy(),
        },
        sv_start={
            "Strain": strain_start,
            "Stress": stress_start,
            "Statev": statev.copy(),
            "Wm": wm.copy(),
            "F": F0,
        },
    )
    problem = SimpleNamespace(time=1.0, dtime=0.1)
    rotation = SimcoonRotation.from_matrix(frame)

    material.update(assembly, problem)

    np.testing.assert_allclose(
        captured["strain"],
        rotation.apply_strain(strain_start.asarray(), False).reshape(6, 1),
    )
    np.testing.assert_allclose(
        captured["dstrain"],
        rotation.apply_strain(strain_increment.asarray(), False).reshape(6, 1),
    )
    np.testing.assert_allclose(
        captured["stress"],
        rotation.apply_stress(stress_start.asarray(), False).reshape(6, 1),
    )
    np.testing.assert_allclose(captured["DR"][:, :, 0], frame.T @ increment @ frame)
    np.testing.assert_allclose(captured["F0"][:, :, 0], frame.T @ F0[:, :, 0] @ frame)
    np.testing.assert_allclose(captured["F1"][:, :, 0], frame.T @ F1[:, :, 0] @ frame)
    np.testing.assert_allclose(
        assembly.sv["Stress"].asarray(),
        rotation.apply_stress(local_stress).reshape(6, 1),
    )
    np.testing.assert_allclose(
        assembly.sv["TangentMatrix"][:, :, 0],
        rotation.apply_stiffness(local_tangent[:, :, 0].copy()),
    )


def test_real_simcoon_orthotropic_update_matches_explicit_basis_change():
    props = np.array([150e9, 12e9, 8e9, 0.25, 0.22, 0.30, 5e9, 4e9, 3e9, 0.0, 0.0, 0.0])
    problem = SimpleNamespace(time=0.0, dtime=1.0)

    def initialized_assembly(material):
        assembly = SimpleNamespace(
            mesh=fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3),
            n_elm_gp=1,
            n_gauss_points=1,
            _nlgeom=False,
            space=SimpleNamespace(
                get_dimension=lambda: "3D", list_variables=lambda: []
            ),
            sv={},
            sv_component={},
        )
        material.initialize(assembly, problem)
        assembly.sv["Strain"] = StrainTensorList(np.zeros((6, 1)))
        assembly.sv["Stress"] = StressTensorList(np.zeros((6, 1)))
        assembly.sv_start = dict(assembly.sv)
        return assembly

    frame = ScipyRotation.from_rotvec([0.2, -0.1, 0.4]).as_matrix()
    rotation = SimcoonRotation.from_matrix(frame)
    strain_global = np.array([[0.01], [-0.003], [0.002], [0.004], [0.001], [-0.002]])

    framed_material = fd.constitutivelaw.Simcoon("ELORT", props)
    framed_material.set_local_frame(frame)
    framed_assembly = initialized_assembly(framed_material)
    framed_assembly.sv["DStrain"] = StrainTensorList(strain_global)
    framed_material.update(framed_assembly, problem)

    local_material = fd.constitutivelaw.Simcoon("ELORT", props)
    local_assembly = initialized_assembly(local_material)
    strain_local = rotation.apply_strain(strain_global, active=False).reshape(6, 1)
    local_assembly.sv["DStrain"] = StrainTensorList(strain_local)
    local_material.update(local_assembly, problem)

    expected_stress = rotation.apply_stress(
        local_assembly.sv["Stress"].asarray()
    ).reshape(6, 1)
    expected_tangent = rotation.apply_stiffness(
        local_assembly.sv["TangentMatrix"][:, :, 0].copy()
    )
    np.testing.assert_allclose(
        framed_assembly.sv["Stress"].asarray(), expected_stress, rtol=1e-12
    )
    np.testing.assert_allclose(
        framed_assembly.sv["TangentMatrix"][:, :, 0],
        expected_tangent,
        rtol=1e-12,
    )


def test_invalid_material_frame_definitions_are_rejected():
    mesh = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3)
    assembly = _assembly(mesh, n_elm_gp=4)
    material = fd.constitutivelaw.ElasticAnisotropic(np.eye(6))
    material.set_local_frame(_z_frames(np.zeros(4)))

    with pytest.raises(ValueError, match="ambiguous"):
        material.get_local_frame(assembly)

    invalid = np.eye(3)
    invalid[0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        material.set_local_frame(invalid)

    with pytest.raises(ValueError, match="local-frame location"):
        material.set_local_frame(np.eye(3), location="node")


@pytest.mark.parametrize(
    "path", [fd.mesh.line_mesh_cylindric(3), fd.mesh.circle_mesh(r=10)]
)
def test_cylindrical_frame_generator_accepts_mesh(path):
    frames = fd.mesh.generate_cylindrical_local_frame(path)

    assert frames.shape == (path.n_nodes, 2, 2)
    np.testing.assert_allclose(
        frames @ frames.transpose(0, 2, 1),
        np.tile(np.eye(2), (path.n_nodes, 1, 1)),
        atol=1e-12,
    )


def test_extrude_accepts_rotation_object_without_storing_frames_on_mesh():
    profile = fd.mesh.line_mesh_1D(2, -1.0, 1.0)
    path = fd.Mesh(
        np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([[0, 1]]),
        "lin2",
    )
    rotation = ScipyRotation.identity()

    extruded = fd.mesh.extrude(profile, path, local_frame=rotation)

    np.testing.assert_allclose(
        extruded.nodes,
        [
            [0.0, 0.0, -1.0],
            [2.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
        ],
    )
    assert not hasattr(extruded, "local_frame")


def test_extrude_accepts_explicit_nodal_path_frames():
    profile = fd.mesh.line_mesh_1D(2, -0.5, 0.5)
    path = fd.mesh.line_mesh_cylindric(3, r=2.0, theta_min=0.0, theta_max=np.pi / 2)
    radial = path.nodes / np.linalg.norm(path.nodes, axis=1, keepdims=True)
    tangent = np.column_stack((-radial[:, 1], radial[:, 0]))
    path_frames = np.stack((-tangent, radial), axis=1)

    extruded = fd.mesh.extrude(profile, path, local_frame=path_frames)

    expected = np.vstack(
        [
            path.nodes - 0.5 * radial,
            path.nodes + 0.5 * radial,
        ]
    )
    np.testing.assert_allclose(extruded.nodes, expected)
    assert not hasattr(path, "local_frame")
    assert not hasattr(extruded, "local_frame")


def test_extrude_rejects_incompatible_path_or_frame_dimensions():
    profile = fd.mesh.line_mesh_1D(2, -1.0, 1.0)
    path = fd.mesh.rectangle_mesh(2, 2, elm_type="quad4")

    with pytest.raises(ValueError, match="line elements"):
        fd.mesh.extrude(profile, path)

    path = fd.mesh.line_mesh(2, ndim=3)
    with pytest.raises(ValueError, match="ndim=2.*3D local frames"):
        fd.mesh.extrude(profile, path, ndim=2, local_frame=np.eye(3))

    path = fd.Mesh(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
        np.array([[0, 1]]),
        "lin2",
    )
    with pytest.raises(ValueError, match="2D local frames.*3D extrusion path"):
        fd.mesh.extrude(profile, path, local_frame=np.eye(2))
