"""simcoon capabilities are queried by feature, with a fallback when absent."""

import numpy as np
import pytest

import fedoo as fd
from fedoo.constitutivelaw import _simcoon_features as features

sim = pytest.importorskip("simcoon")


def _condenses_finite_strain_f():
    """Whether simcoon solves the plane-stress state of an F-based law."""
    F0 = np.eye(3)[:, :, None]
    F1 = F0.copy()
    F1[0, 0] = 1.2
    strain, dstrain = np.zeros((6, 1)), np.zeros((6, 1))
    dstrain[0] = np.log(1.2)
    try:
        stress = sim.umat(
            "NEOHC",
            strain,
            dstrain,
            F0,
            F1,
            np.zeros((6, 1)),
            F0.copy(),
            np.array([1.0, 30.0]),
            np.zeros((1, 1)),
            0.0,
            1.0,
            np.zeros((4, 1)),
            ndi=2,
        )[0]
    except Exception:  # noqa: BLE001 - any failure means "not served"
        return False
    return abs(np.ravel(stress)[2]) < 1e-10


def _uniaxial_bar(dim):
    fd.ModelingSpace(dim)
    if dim == "3D":
        mesh = fd.mesh.box_mesh(2, 2, 2, 0, 1, 0, 1, 0, 1, "hex8")
    else:
        mesh = fd.mesh.rectangle_mesh(2, 2, 0, 1, 0, 1, "quad4")
    law = fd.constitutivelaw.Simcoon("NEOHC", [1.0, 30.0])
    wf = fd.weakform.StressEquilibrium(law, nlgeom="UL")
    pb = fd.problem.NonLinear(fd.Assembly.create(wf, mesh))
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0), "DispX", 0)
    pb.bc.add("Dirichlet", mesh.find_nodes("Y", 0), "DispY", 0)
    if dim == "3D":
        pb.bc.add("Dirichlet", mesh.find_nodes("Z", 0), "DispZ", 0)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 1), "DispX", 0.5)
    pb.set_nr_criterion("Displacement", tol=1e-10, max_subiter=20)
    return pb


def test_reduced_state_needs_the_capability(monkeypatch):
    monkeypatch.delattr(sim, "condensation_serves", raising=False)
    assert not features.serves_condensed("NEOHC", 2)
    assert features.serves_condensed("NEOHC", 3)
    pb = _uniaxial_bar("2Dstress")
    with pytest.raises(NotImplementedError, match="condensation_serves"):
        pb.nlsolve(dt=0.5, update_dt=False, print_info=0)


def test_query_arguments_are_forwarded(monkeypatch):
    calls = []

    def serves(name, ndi, **kwargs):
        calls.append((name, ndi, kwargs))
        return name == "NEOHC"

    monkeypatch.setattr(sim, "condensation_serves", serves, raising=False)
    assert features.serves_condensed("NEOHC", 1, nshr=0, kinematic_form="total")
    assert not features.serves_condensed("ELISO", 2)
    assert calls[0] == ("NEOHC", 1, {"nshr": 0, "kinematic_form": "total"})


@pytest.mark.skipif(
    not _condenses_finite_strain_f(),
    reason="simcoon does not condense finite-strain F-based laws",
)
def test_served_plane_stress_hyperelastic_matches_uniaxial_3d(monkeypatch):
    if not hasattr(sim, "condensation_serves"):
        monkeypatch.setattr(
            sim, "condensation_serves", lambda name, ndi, **k: True, raising=False
        )
    results = []
    for dim in ("3D", "2Dstress"):
        pb = _uniaxial_bar(dim)
        pb.nlsolve(dt=0.1, update_dt=True, print_info=0)
        stress = pb.get_results(pb.assembly, ["Stress"], "GaussPoint")["Stress"]
        results.append(np.asarray(stress[0]).mean())
        np.testing.assert_allclose(stress[1], 0, atol=1e-9)
    assert results[1] == pytest.approx(results[0], rel=1e-9)
