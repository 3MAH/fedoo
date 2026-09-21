from types import SimpleNamespace

import numpy as np

import fedoo as fd
from fedoo.core.mechanical3d import MechanicalUMAT
from fedoo.util.voigt_tensors import StrainTensorList, StressTensorList


def _assembly(n_points=1, nlgeom=False):
    return SimpleNamespace(
        mesh=fd.mesh.rectangle_mesh(2, 2, elm_type="quad4", ndim=3),
        n_elm_gp=n_points,
        n_gauss_points=n_points,
        _nlgeom=nlgeom,
        space=SimpleNamespace(get_dimension=lambda: "3D", list_variables=lambda: []),
        sv={},
        sv_component={},
    )


def test_generic_mechanical_umat_callback_and_metadata():
    calls = []
    tangent = np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])[:, :, None]

    def umat(
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
        temperature,
        *,
        ndi,
        tangent_mode,
    ):
        calls.append(
            {
                "strain": np.asarray(strain),
                "dstrain": np.asarray(dstrain),
                "props": props,
                "time": time,
                "dtime": dtime,
                "ndi": ndi,
                "tangent_mode": tangent_mode,
            }
        )
        statev = statev_start.copy()
        statev[0] += 1.0
        wm = wm_start.copy()
        wm[0] += 2.0
        return stress + dstrain, statev, wm, tangent

    material = MechanicalUMAT(
        umat,
        [10.0, 20.0],
        n_statev=2,
        props_label={"first": 0, "second": 1},
        statev_label={"internal": 0},
        tangent_mode=2,
    )
    assembly = _assembly()
    problem = SimpleNamespace(time=1.5, dtime=0.25)

    material.initialize(assembly, problem)

    assert material.props_label == {"first": 0, "second": 1}
    assert material.statev_label == {"internal": 0}
    assert assembly.sv_component["internal"] == ("Statev", 0)
    assert assembly.sv["Statev"].shape == (2, 1)
    assert calls[0]["time"] == 0
    assert calls[0]["dtime"] == 0
    assert calls[0]["ndi"] == 3
    assert calls[0]["tangent_mode"] == 2
    np.testing.assert_allclose(calls[0]["props"], [[10.0], [20.0]])

    strain_start = StrainTensorList(np.zeros((6, 1)))
    stress_start = StressTensorList(np.zeros((6, 1)))
    dstrain = StrainTensorList(np.arange(1.0, 7.0)[:, None] * 1e-3)
    assembly.sv.update(
        Strain=strain_start,
        Stress=stress_start,
        DStrain=dstrain,
    )
    assembly.sv_start = dict(assembly.sv)
    material.update(assembly, problem)

    assert calls[1]["time"] == 1.5
    assert calls[1]["dtime"] == 0.25
    np.testing.assert_allclose(calls[1]["dstrain"], dstrain.asarray())
    np.testing.assert_allclose(assembly.sv["Stress"].asarray(), dstrain.asarray())
    np.testing.assert_allclose(assembly.sv["Statev"], [[1.0], [0.0]])
    np.testing.assert_allclose(assembly.sv["Wm"][0], 2.0)
    np.testing.assert_allclose(assembly.sv["TangentMatrix"], tangent)


def test_initial_statev_is_stored_on_the_assembly_and_broadcast_by_label():
    initial_calls = []

    def umat(
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
        temperature,
        *,
        ndi,
        tangent_mode,
    ):
        initial_calls.append(statev_start.copy())
        return stress, statev_start.copy(), wm_start.copy(), np.eye(6)

    material = MechanicalUMAT(
        umat,
        n_statev=3,
        statev_label={"P": 0, "EP": slice(1, 3)},
    )
    assembly = _assembly(n_points=3)

    material.set_initial_statev(assembly, "P", [1.0, 2.0, 3.0])
    material.set_initial_statev(assembly, "EP", [4.0, 5.0])

    expected = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 4.0, 4.0],
            [5.0, 5.0, 5.0],
        ]
    )
    np.testing.assert_allclose(assembly.sv["Statev"], expected)

    material.initialize(assembly, SimpleNamespace())
    np.testing.assert_allclose(assembly.sv["Statev"], expected)
    np.testing.assert_allclose(initial_calls[0], expected)
    with np.testing.assert_raises_regex(RuntimeError, "before assembly initialization"):
        material.set_initial_statev(assembly, "P", 0.0)

    second_assembly = _assembly(n_points=2)
    material.set_initial_statev(second_assembly, "P", 9.0)
    material.initialize(second_assembly, SimpleNamespace())
    np.testing.assert_allclose(second_assembly.sv["Statev"][0], [9.0, 9.0])
    np.testing.assert_allclose(second_assembly.sv["Statev"][1:], 0.0)
    np.testing.assert_allclose(initial_calls[1], second_assembly.sv["Statev"])


def test_initialize_preserves_a_valid_manual_statev_array():
    seen = []

    def umat(*args, ndi, tangent_mode):
        seen.append(args[7].copy())
        return args[4], args[7].copy(), args[10].copy(), np.eye(6)

    material = MechanicalUMAT(umat, n_statev=2)
    assembly = _assembly(n_points=3)
    initial = np.arange(6.0).reshape(2, 3)
    assembly.sv["Statev"] = initial.copy()

    material.initialize(assembly, SimpleNamespace())

    np.testing.assert_allclose(assembly.sv["Statev"], initial)
    np.testing.assert_allclose(seen[0], initial)


def test_initialize_consumes_weakform_deformation_gradient_without_overwriting_it():
    seen = []

    def umat(*args, ndi, tangent_mode):
        seen.append((args[2].copy(), args[3].copy()))
        return args[4], args[7].copy(), args[10].copy(), np.eye(6)

    material = MechanicalUMAT(umat)
    assembly = _assembly(n_points=2, nlgeom="UL")
    deformation_gradient = np.repeat(np.eye(3)[:, :, None], 2, axis=2)
    deformation_gradient[0, 0] = [1.1, 1.2]
    assembly.sv["F"] = deformation_gradient

    material.initialize(assembly, SimpleNamespace())

    assert assembly.sv["F"] is deformation_gradient
    np.testing.assert_allclose(seen[0][0], deformation_gradient)
    np.testing.assert_allclose(seen[0][1], deformation_gradient)


def test_initialize_rejects_an_invalid_manual_statev_shape():
    material = MechanicalUMAT(lambda *args, **kwargs: None, n_statev=2)
    assembly = _assembly(n_points=3)
    assembly.sv["Statev"] = np.zeros((2, 1))

    with np.testing.assert_raises_regex(ValueError, "expected \\(2, 3\\)"):
        material.initialize(assembly, SimpleNamespace())
