"""The simcoon integrator and complementarity options through an assembly (simcoon >= 2.4).

``integrator`` and ``ncp`` are forwarded to ``sim.umat`` at every call: the two integrators land
on the same plastic state of a uniaxially stretched box, ``ncp="min"`` runs under both, and a law
without a closest-point update is refused by simcoon, not silently run with the cutting plane.
"""

import numpy as np
import pytest
import fedoo as fd

EPICP = np.array([200e3, 0.3, 1e-5, 300.0, 1000.0, 0.5])  # yields at 0.15 % strain


def _stretch(tag, umat="EPICP", props=EPICP, strain=0.01, **options):
    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(2, 2, 2, 0, 1, 0, 1, 0, 1, "hex8", name="m_int_" + tag)
    law = fd.constitutivelaw.Simcoon(umat, props, name="law_int_" + tag, **options)
    wf = fd.weakform.StressEquilibrium(law, name="wf_int_" + tag)
    assembly = fd.Assembly.create(wf, mesh, name="asm_int_" + tag)
    pb = fd.problem.NonLinear(assembly, name="pb_int_" + tag)
    left, right = mesh.find_nodes("X", 0.0), mesh.find_nodes("X", 1.0)
    pb.bc.add("Dirichlet", left, "DispX", 0.0)
    pb.bc.add("Dirichlet", mesh.find_nodes("Y", 0.0), "DispY", 0.0)
    pb.bc.add("Dirichlet", mesh.find_nodes("Z", 0.0), "DispZ", 0.0)
    pb.bc.add("Dirichlet", right, "DispX", strain)
    pb.nlsolve(dt=0.25, tmax=1.0, update_dt=False, print_info=0)
    return np.asarray(assembly.sv["Stress"]), np.asarray(assembly.sv["Statev"])


def test_options_reach_the_law():
    law = fd.constitutivelaw.Simcoon(
        "EPICP", EPICP, name="law_opt_attr", ncp="min", integrator="closest_point"
    )
    assert (law.ncp, law.integrator, law.tangent_mode) == ("min", "closest_point", 2)
    assert (
        fd.constitutivelaw.Simcoon("EPICP", EPICP, name="law_opt_def").tangent_mode == 1
    )


def test_closest_point_lands_on_the_cutting_plane_state():
    """Both integrators converge to the same plastic state (p > 0 everywhere); they differ by the
    integration error of the four increments, far below the plastic strain reached."""
    s_cp, sv_cp = _stretch("cp")
    s_cpp, sv_cpp = _stretch("cpp", integrator="closest_point")
    assert sv_cp[1].min() > 1e-3  # p: the box has yielded
    np.testing.assert_allclose(s_cpp, s_cp, rtol=0, atol=1e-3 * np.abs(s_cp).max())
    np.testing.assert_allclose(sv_cpp[1], sv_cp[1], rtol=1e-3)


@pytest.mark.parametrize("integrator", ["cutting_plane", "closest_point"])
def test_min_complementarity_runs_under_both_integrators(integrator):
    """ncp="min" is the same return mapping with another complementarity function: the same
    state as Fischer-Burmeister to the local tolerance."""
    s_fb, sv_fb = _stretch("fb_" + integrator, integrator=integrator)
    s_min, sv_min = _stretch("min_" + integrator, integrator=integrator, ncp="min")
    np.testing.assert_allclose(s_min, s_fb, rtol=0, atol=1e-6 * np.abs(s_fb).max())
    np.testing.assert_allclose(sv_min[1], sv_fb[1], rtol=1e-6)


def test_closed_names_share_the_layout_of_their_plain_name():
    """EPICP_closed is simcoon's dedicated EPICP kernel: same props, same statev, same state."""
    law = fd.constitutivelaw.Simcoon("EPICP_closed", EPICP, name="law_closed_attr")
    assert (law.umat_name, law.n_statev, law.is_isotropic) == ("EPICP_closed", 8, True)
    s_closed, sv_closed = _stretch("closed", umat="EPICP_closed")
    s_plain, sv_plain = _stretch("plain")
    np.testing.assert_allclose(
        s_closed, s_plain, rtol=0, atol=1e-9 * np.abs(s_plain).max()
    )
    np.testing.assert_allclose(sv_closed[1], sv_plain[1], rtol=1e-9)


def test_a_law_without_closest_point_is_refused():
    """The names served by the modular engine all have one (an elastic law trivially); a dedicated
    kernel that keeps its own algorithm, such as the Kelvin unit ZENER_closed, is refused."""
    zener = np.array([3000.0, 0.35, 0.0, 1500.0, 0.35, 3000.0, 1200.0])
    with pytest.raises(ValueError, match="closest_point"):
        _stretch(
            "refused",
            umat="ZENER_closed",
            props=zener,
            strain=0.001,
            integrator="closest_point",
        )
