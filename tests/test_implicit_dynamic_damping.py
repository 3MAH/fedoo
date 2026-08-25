"""Rayleigh damping validation for the ImplicitDynamic weak form.

``ImplicitDynamic`` (a Newmark weak form summed onto a standard ``NonLinear``
problem) is fedoo's current implicit-dynamics path — the same
dynamics-as-assembly architecture as this PR's ``RigidBodyAssembly`` (and
unlike the legacy ``NonLinearNewmark`` problem class). The existing
``test_2DDynamicPlasticBending_v2`` sets ``rayleigh_damping`` but asserts
nothing about it; these tests pin the actual behaviour.
"""

import numpy as np
import pytest

import fedoo as fd


def _build_axial_bar(rayleigh=None):
    """Axial bar (nu=0), fixed at x=0, constant tip load in X, ImplicitDynamic."""
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dplane")
    E, nu, rho = 1.0, 0.0, 1.0
    fd.mesh.rectangle_mesh(
        nx=9,
        ny=3,
        x_min=0,
        x_max=1.0,
        y_min=0,
        y_max=0.2,
        elm_type="quad4",
        name="Domain",
    )
    mesh = fd.Mesh["Domain"]
    fd.constitutivelaw.ElasticIsotrop(E, nu, name="law")
    wf = fd.weakform.ImplicitDynamic("law", rho, 0.25, 0.5)
    if rayleigh is not None:
        wf.rayleigh_damping = rayleigh  # [alpha (mass), beta (stiffness)]
    fd.Assembly.create(wf, "Domain", "quad4", name="asm")
    pb = fd.problem.NonLinear("asm")
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0.0), "Disp", 0)
    pb.bc.add("Neumann", mesh.find_nodes("X", 1.0), "DispX", 0.05)
    return pb, mesh


def test_rayleigh_damping_property_roundtrip():
    """``rayleigh_damping = [a, b]`` must read back as ``[a, b]`` (mass, stiff)."""
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Dplane")
    fd.constitutivelaw.ElasticIsotrop(1.0, 0.0, name="law")
    wf = fd.weakform.ImplicitDynamic("law", 1.0, 0.25, 0.5)
    assert wf.rayleigh_damping is None
    wf.rayleigh_damping = [0.7, 0.2]
    assert wf.rayleigh_damping == [0.7, 0.2]


def _max_speed_over_run(rayleigh):
    pb, mesh = _build_axial_bar(rayleigh=rayleigh)
    speeds = []

    def cb(p):
        v = p.get_results("asm", ["Velocity"], "Node").node_data["Velocity"]
        speeds.append(float(np.abs(v).max()))

    pb.nlsolve(
        dt=0.05,
        tmax=8.0,
        update_dt=False,
        print_info=0,
        callback=cb,
        exec_callback_at_each_iter=True,
    )
    return max(speeds) if speeds else 0.0


def test_implicit_dynamic_rayleigh_damping_reduces_peak_velocity():
    """Heavy mass-proportional Rayleigh damping must remove kinetic energy.

    A wrong-sign damping term would amplify the response, so this also pins
    the sign of the damping contribution.
    """
    v_undamped = _max_speed_over_run(None)
    v_damped = _max_speed_over_run([20.0, 0.0])
    assert v_undamped > 0.0
    assert (
        v_damped < 0.5 * v_undamped
    ), f"damped peak speed {v_damped:.3e} not < 0.5 * undamped {v_undamped:.3e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
