"""Per-law Simcoon capabilities and reduced-state numerical regressions."""

from types import SimpleNamespace

import numpy as np
import pytest

import fedoo as fd
import simcoon as sim


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
    monkeypatch.setattr(sim, "condensation_serves", lambda *args, **kwargs: False)
    pb = _uniaxial_bar("2Dstress")
    with pytest.raises(NotImplementedError, match="reduced stress state"):
        pb.nlsolve(dt=0.5, update_dt=False, print_info=0)


def test_query_arguments_are_forwarded(monkeypatch):
    calls = []

    def serves(name, ndi, **kwargs):
        calls.append((name, ndi, kwargs))
        return name == "NEOHC"

    monkeypatch.setattr(sim, "condensation_serves", serves)
    law = fd.constitutivelaw.Simcoon("NEOHC", [1.0, 30.0])
    total = SimpleNamespace(_nlgeom="TL", _corate=None)
    assert law._serves_reduced_state(1, total)
    assert law._serves_reduced_state(2)
    assert law._serves_reduced_state(3)
    assert calls == [("NEOHC", 1, {"kinematic_form": "total"}), ("NEOHC", 2, {})]


def test_served_plane_stress_hyperelastic_matches_uniaxial_3d():
    results = []
    for dim in ("3D", "2Dstress"):
        pb = _uniaxial_bar(dim)
        pb.nlsolve(dt=0.1, update_dt=True, print_info=0)
        stress = pb.get_results(pb.assembly, ["Stress"], "GaussPoint")["Stress"]
        results.append(np.asarray(stress[0]).mean())
        np.testing.assert_allclose(stress[1], 0, atol=1e-9)
    assert results[1] == pytest.approx(results[0], rel=1e-9)


def test_reduced_state_is_asked_for_the_formulation_the_law_runs_with(monkeypatch):
    # Capability sets may change; ensure the adapter asks for the actual form.
    def serves(name, ndi, **kwargs):
        return kwargs.get("kinematic_form") != "total"

    monkeypatch.setattr(sim, "condensation_serves", serves)
    law = fd.constitutivelaw.Simcoon.__new__(fd.constitutivelaw.Simcoon)
    law.umat_name = "ZENER"
    rate = SimpleNamespace(_nlgeom="UL", _corate="log_r")
    total = SimpleNamespace(_nlgeom="TL", _corate=None)
    assert law._serves_reduced_state(2, rate)
    assert not law._serves_reduced_state(2, total)
