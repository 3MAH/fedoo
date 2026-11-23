"""Reuse of the material frames resolved at the Gauss points of an assembly.

Mechanical3D resolves ``get_local_frame(assembly)`` once and keeps the result on the
assembly while the law, its frames, their location, the connectivity and the number of
Gauss points are unchanged. These tests check that the reused frames are always the
ones ``get_local_frame`` returns, and that public methods still return writable arrays.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as ScipyRotation

import fedoo as fd
from fedoo.util.voigt_tensors import StrainTensorList, StressTensorList


def _assembly(mesh, n_elm_gp, nlgeom=False):
    return SimpleNamespace(
        mesh=mesh,
        n_elm_gp=n_elm_gp,
        n_gauss_points=mesh.n_elements * n_elm_gp,
        _nlgeom=nlgeom,
        sv={},
        sv_start={},
    )


def _z_frames(angles):
    rotvec = np.zeros((len(angles), 3))
    rotvec[:, 2] = angles
    return ScipyRotation.from_rotvec(rotvec).as_matrix()


@pytest.fixture
def mesh():
    return fd.mesh.rectangle_mesh(4, 3, elm_type="quad4", ndim=3)


@pytest.mark.parametrize("location", ["Node", "Element", "GaussPoint", None])
def test_reused_frames_are_the_resolved_frames(mesh, location):
    material = fd.constitutivelaw.ElasticAnisotropic(np.eye(6))
    count = {"Node": mesh.n_nodes, "Element": mesh.n_elements, None: 1}.get(
        location, mesh.n_elements * 4
    )
    material.set_local_frame(_z_frames(np.linspace(0.1, 0.9, count)), location)
    assembly = _assembly(mesh, 4)

    for _ in range(2):  # resolution, then reuse
        frames = material._material_frames(assembly)
        expected = np.asarray(material.get_local_frame(assembly)).reshape(-1, 3, 3)
        assert np.array_equal(frames, expected)
        assert not frames.flags.writeable
    assert assembly._material_frame_cache["frames"] is frames


def test_frames_follow_changes_of_the_law_and_assembly(mesh):
    material = fd.constitutivelaw.ElasticAnisotropic(np.eye(6))
    material.set_local_frame(_z_frames(np.full(mesh.n_nodes, 0.3)), "Node")
    assembly = _assembly(mesh, 4)
    first = material._material_frames(assembly)

    def check():
        expected = np.asarray(material.get_local_frame(assembly)).reshape(-1, 3, 3)
        assert np.array_equal(material._material_frames(assembly), expected)

    # frames modified in place, as LocalFrame.rotate does
    material.local_frame.rotate(20.0)
    check()
    assert not np.array_equal(material._material_frames(assembly), first)
    # new frames
    material.set_local_frame(_z_frames(np.linspace(0, 1, mesh.n_nodes)), "Node")
    check()
    # another number of Gauss points per element
    assembly.n_elm_gp = 1
    assembly.n_gauss_points = mesh.n_elements
    check()
    # connectivity changed in place
    mesh.elements[:] = mesh.elements[:, ::-1]
    check()
    # another law on the same assembly
    other = fd.constitutivelaw.ElasticAnisotropic(np.eye(6))
    other.set_local_frame(_z_frames(np.full(mesh.n_nodes, -0.4)), "Node")
    expected = np.asarray(other.get_local_frame(assembly)).reshape(-1, 3, 3)
    assert np.array_equal(other._material_frames(assembly), expected)
    check()
    # frames removed
    material.set_local_frame(None)
    assert material._material_frames(assembly) is None


def test_added_nodes_invalidate_the_frames(mesh):
    """Nodal frames inferred from their count: a new node makes them invalid."""
    material = fd.constitutivelaw.ElasticAnisotropic(np.eye(6))
    material.set_local_frame(_z_frames(np.full(mesh.n_nodes, 0.3)))
    assembly = _assembly(mesh, 4)
    expected = np.asarray(material.get_local_frame(assembly)).reshape(-1, 3, 3)
    assert np.array_equal(material._material_frames(assembly), expected)

    mesh.add_nodes(np.array([[10.0, 10.0, 0.0]]))
    with pytest.raises(ValueError):
        material.get_local_frame(assembly)
    with pytest.raises(ValueError):
        material._material_frames(assembly)


def test_public_frames_stay_writable(mesh):
    material = fd.constitutivelaw.Simcoon("ELORT", np.zeros(12))
    material.set_local_frame(_z_frames(np.full(mesh.n_elements, 0.5)), "Element")
    assembly = _assembly(mesh, 4)

    small_strain = material.get_current_local_frame(assembly)
    assert isinstance(small_strain, fd.util.localframe.LocalFrame)
    assert small_strain.flags.writeable
    assert material._initial_material_frame(assembly).flags.writeable

    assembly._nlgeom = "UL"
    current = material.get_current_local_frame(assembly)
    assert current.flags.writeable
    assert assembly.sv["_MaterialFrame"].flags.writeable
    np.testing.assert_array_equal(current, material.get_local_frame(assembly))


def test_umat_update_with_reused_frames_is_unchanged(mesh):
    """Two successive updates (frames reused) equal an update with resolved frames."""
    props = np.array([150e3, 12e3, 8e3, 0.25, 0.22, 0.30, 5e3, 4e3, 3e3, 0.0, 0.0, 0.0])
    material = fd.constitutivelaw.Simcoon("ELORT", props)
    material.set_local_frame(_z_frames(np.linspace(0, 1.2, mesh.n_nodes)), "Node")
    n = mesh.n_elements * 4
    rng = np.random.default_rng(0)
    strain = StrainTensorList(1e-3 * rng.standard_normal((6, n)))
    statev = np.zeros((material.n_statev, n), order="F")
    wm = np.zeros((4, n), order="F")

    def assembly():
        return SimpleNamespace(
            mesh=mesh,
            n_elm_gp=4,
            n_gauss_points=n,
            _nlgeom=False,
            space=SimpleNamespace(
                get_dimension=lambda: "3D", list_variables=lambda: []
            ),
            sv={"Strain": strain, "DR": np.eye(3)[:, :, None], "Statev": statev},
            sv_start={
                "Strain": StrainTensorList(np.zeros((6, n))),
                "Stress": StressTensorList(np.zeros((6, n))),
                "Statev": statev,
                "Wm": wm,
            },
        )

    problem = SimpleNamespace(time=0.0, dtime=1.0)
    reused = assembly()
    material.update(reused, problem)
    material.update(reused, problem)
    fresh = assembly()
    material.update(fresh, problem)
    for key in ("Stress", "TangentMatrix", "Statev"):
        a, b = np.asarray(reused.sv[key]), np.asarray(fresh.sv[key])
        assert a.tobytes() == b.tobytes()


def test_overridden_frame_getters_are_still_called(mesh):
    calls = []

    class Law(fd.constitutivelaw.ElasticAnisotropic):
        def get_local_frame(self, assembly=None):
            calls.append("initial")
            return super().get_local_frame(assembly)

        def get_current_local_frame(self, assembly=None):
            calls.append("current")
            return super().get_current_local_frame(assembly)

    material = Law(np.eye(6))
    material.set_local_frame(_z_frames(np.full(mesh.n_nodes, 0.3)), "Node")
    assembly = _assembly(mesh, 4)
    stress = np.ones((6, mesh.n_elements * 4))
    for current in (True, False):
        for _ in range(2):
            calls.clear()
            material.local2global_stress(stress, assembly, current=current)
            assert ("current" if current else "initial") in calls
    assert not hasattr(assembly, "_material_frame_cache")


@pytest.mark.parametrize("nlgeom", [False, "UL"])
def test_solution_with_reused_frames_is_unchanged(monkeypatch, nlgeom):
    def solve():
        fd.ModelingSpace("3D")
        mesh = fd.mesh.box_mesh(3, 3, 3, elm_type="hex8")
        material = fd.constitutivelaw.ElasticOrthotropic(
            150e3, 12e3, 8e3, 3e3, 4e3, 5e3, 0.30, 0.22, 0.25
        )
        rotvec = 0.6 * np.random.default_rng(1).standard_normal((mesh.n_nodes, 3))
        frames = ScipyRotation.from_rotvec(rotvec).as_matrix()
        material.set_local_frame(frames, "Node")
        wf = fd.weakform.StressEquilibrium(material, nlgeom=nlgeom)
        assembly = fd.Assembly.create(wf, mesh, "hex8")
        pb = fd.problem.NonLinear(assembly)
        right = mesh.find_nodes("X", mesh.bounding_box.xmax)
        pb.bc.add("Dirichlet", mesh.find_nodes("X", 0), "Disp", 0)
        pb.bc.add("Dirichlet", right, "DispX", 0.08)
        pb.bc.add("Dirichlet", right, "DispY", 0.25)
        pb.nlsolve(dt=0.5, update_dt=False, tmax=1.0, print_info=0)
        return assembly, np.array(pb.get_disp()), assembly.sv["Stress"].asarray()

    assembly, disp, stress = solve()
    assert hasattr(assembly, "_material_frame_cache")

    resolve = fd.core.mechanical3d.Mechanical3D._material_frames

    def resolve_again(self, assembly):
        vars(assembly).pop("_material_frame_cache", None)
        return resolve(self, assembly)

    monkeypatch.setattr(
        fd.core.mechanical3d.Mechanical3D, "_material_frames", resolve_again
    )
    _, disp_ref, stress_ref = solve()
    assert np.array_equal(disp, disp_ref)
    assert np.array_equal(stress, stress_ref)
