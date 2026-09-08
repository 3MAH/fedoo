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
