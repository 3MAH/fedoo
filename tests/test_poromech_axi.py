"""Axisymmetric (``2Daxi``) poromechanics.

Unconfined compression of a saturated cylinder between frictionless plates.

* With an impermeable lateral face the response is homogeneous and
  undrained at all times, with a closed-form solution: this checks the hoop
  strain in the Biot coupling and in the storage term, and the ``2*pi*r``
  measure of the momentum balance, for both formulations.
* With a draining lateral face the fluid flows radially out of the
  cylinder and the force relaxes to the drained response, in small strain
  and in finite strain (UL).

Steady radial flow through a hollow cylinder with a fixed skeleton checks
the ``2*pi*r`` measure of the Darcy term: the pressure is logarithmic in r
(it would be linear without the measure).

The draining case was also compared once with a 3D model of the full
cylinder (13651 nodes, too slow for the test suite): the force histories
agreed within 1e-3 in small strain and in UL with a Neo-Hookean skeleton at
5 % compression.
"""

import numpy as np
import pytest

import fedoo as fd

try:
    from simcoon import simmit  # noqa: F401

    USE_SIMCOON = True
except ImportError:
    USE_SIMCOON = False

R, HC = 1.0, 1.0  # cylinder radius and height
E, NU = 1.0, 0.3
M, ALPHA, K = 10.0, 1.0, 0.05


def _skeleton(law):
    if law == "elastic":
        return fd.constitutivelaw.ElasticIsotrop(E, NU)
    mu = E / (2.0 * (1.0 + NU))
    kappa = E / (3.0 * (1.0 - 2.0 * NU))
    return fd.constitutivelaw.Simcoon("NEOHC", np.array([mu, kappa]))


def _fluid():
    return fd.constitutivelaw.PoroFluidProperties(
        permeability=K,
        fluid_viscosity=1.0,
        biot_coefficient=ALPHA,
        biot_modulus=M,
        initial_porosity=0.5,
    )


def _solve(pb, n_ramp, dt_ramp, dt_hold, n_hold, force_fn):
    """Ramp the top displacement then hold.

    Return the axial force history and the largest number of Newton-Raphson
    iterations of an increment.
    """
    t_ramp = n_ramp * dt_ramp
    pb.initialize()
    pb.tmax = t_ramp + n_hold * dt_hold
    pb.set_start()
    t = 0.0
    force = []
    max_iter = 0
    for dt in [dt_ramp] * n_ramp + [dt_hold] * n_hold:
        pb.time = t
        pb.dtime = dt
        convergence, n_iter, res = pb.solve_time_increment()
        assert convergence, f"t={t:g}: Newton failed (res={res:g})"
        max_iter = max(max_iter, n_iter)
        pb.set_start()
        t += dt
        force.append(force_fn())
    return np.array(force), max_iter


def _ramp(t_ramp, tmax):
    return lambda tf: min(1.0, tf * tmax / t_ramp)


def _weakform(formulation, law, nlgeom):
    if formulation == "mixed":
        lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
        bulk = lam + 2.0 * E / (6.0 * (1.0 + NU))
        return fd.weakform.PoroMechanics(
            _skeleton(law), _fluid(), bulk_modulus=bulk, nlgeom=nlgeom
        )
    return fd.weakform.PoroMechanicsSimple(_skeleton(law), _fluid(), nlgeom=nlgeom)


def _axi_cylinder(
    law, nlgeom, eps_z, drained_side, n_ramp=4, n_hold=16, formulation="simple"
):
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Daxi")
    mesh = fd.mesh.rectangle_mesh(
        nx=11, ny=11, x_min=0, x_max=R, y_min=0, y_max=HC, elm_type="quad4"
    )
    wf = _weakform(formulation, law, nlgeom)
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.set_nr_criterion("Displacement", tol=1e-8, max_subiter=30)

    top = mesh.find_nodes("Y", HC)
    dt_ramp, dt_hold = 0.25, 1.0
    tmax = n_ramp * dt_ramp + n_hold * dt_hold
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0.0), "DispX", 0.0)  # axis
    pb.bc.add("Dirichlet", mesh.find_nodes("Y", 0.0), "DispY", 0.0)
    pb.bc.add(
        "Dirichlet",
        top,
        "DispY",
        eps_z * HC,
        time_func=_ramp(n_ramp * dt_ramp, tmax),
    )
    if drained_side:
        pb.bc.add("Dirichlet", mesh.find_nodes("X", R), "PorePressure", 0.0)

    force, max_iter = _solve(
        pb,
        n_ramp,
        dt_ramp,
        dt_hold,
        n_hold,
        lambda: -np.sum(pb.get_ext_forces("DispY")[top]),
    )
    return pb, assembly, force, max_iter


@pytest.mark.parametrize("formulation", ["simple", "mixed"])
def test_axi_sealed_unconfined_compression_is_homogeneous_undrained(formulation):
    eps_z = -1.0e-3
    pb, assembly, force, max_iter = _axi_cylinder(
        "elastic", False, eps_z, drained_side=False, formulation=formulation
    )
    # linear problem with an exact tangent: the displacement criterion
    # converges at the third iteration, as in plane strain
    assert max_iter <= 3

    lam = E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
    mu = E / (2.0 * (1.0 + NU))
    lam_u = lam + ALPHA**2 * M  # undrained Lame coefficient
    e_r = -lam_u * eps_z / (2.0 * (lam_u + mu))
    tr_eps = 2.0 * e_r + eps_z
    p_exact = -ALPHA * M * tr_eps
    sigma_zz = lam_u * tr_eps + 2.0 * mu * eps_z

    u_r = pb.get_dof_solution("DispX")
    r = assembly.mesh.nodes[:, 0]
    assert np.allclose(u_r, e_r * r, rtol=0, atol=1e-9)
    p = pb.get_dof_solution("PorePressure")
    assert np.allclose(p, p_exact, rtol=1e-6)
    # total force on the full circle (the 2Daxi weak form carries 2*pi*r)
    assert force[-1] == pytest.approx(-np.pi * R**2 * sigma_zz, rel=1e-6)


def test_axi_drained_unconfined_compression_relaxes():
    eps_z = -1.0e-3
    _, _, force, max_iter = _axi_cylinder("elastic", False, eps_z, drained_side=True)
    assert max_iter <= 3
    # undrained peak at the end of the ramp, then relaxation to the drained
    # response E * eps_z on the full section
    assert force[3] > 1.05 * force[-1]
    assert np.all(np.diff(force[3:]) < 0)
    assert force[-1] == pytest.approx(-np.pi * R**2 * E * eps_z, rel=0.02)


@pytest.mark.skipif(not USE_SIMCOON, reason="simcoon is required for NEOHC")
def test_axi_finite_strain_unconfined_compression_converges():
    eps_z = -0.05
    _, assembly, force, _ = _axi_cylinder("neohc", "UL", eps_z, drained_side=True)
    assert force[3] > force[-1] > 0.0
    # the drained state is nearly uniaxial: the volume change follows the
    # drained Poisson effect, not the undrained one (lnJ = 0)
    lnJ = assembly.sv["lnJ"]
    assert np.all(lnJ < 0.5 * eps_z * (1.0 - 2.0 * NU))


def test_axi_steady_radial_darcy_flow_is_logarithmic():
    r1, r2, p1 = 1.0, 2.0, 1.0e-3
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Daxi")
    mesh = fd.mesh.rectangle_mesh(
        nx=41, ny=2, x_min=r1, x_max=r2, y_min=0, y_max=0.1, elm_type="quad4"
    )
    wf = fd.weakform.PoroMechanicsSimple(_skeleton("elastic"), _fluid())
    pb = fd.problem.NonLinear(fd.Assembly.create(wf, mesh))
    pb.set_nr_criterion("Displacement", tol=1e-8, max_subiter=10)
    pb.bc.add("Dirichlet", np.arange(mesh.n_nodes), "Disp", 0.0)  # rigid skeleton
    pb.bc.add("Dirichlet", mesh.find_nodes("X", r1), "PorePressure", p1)
    pb.bc.add("Dirichlet", mesh.find_nodes("X", r2), "PorePressure", 0.0)
    # one long backward-Euler step reaches the steady state
    pb.nlsolve(dt=1.0e6, tmax=1.0e6, print_info=0)

    r = mesh.nodes[:, 0]
    p = pb.get_dof_solution("PorePressure")
    p_exact = p1 * np.log(r2 / r) / np.log(r2 / r1)
    assert np.allclose(p, p_exact, rtol=0, atol=1e-4 * p1)
    # without the 2*pi*r measure the profile would be linear
    p_linear = p1 * (r2 - r) / (r2 - r1)
    assert np.abs(p_linear - p_exact).max() > 50 * np.abs(p - p_exact).max()


@pytest.mark.parametrize("formulation", ["simple", "mixed"])
def test_axi_total_lagrangian_is_rejected(formulation):
    fd.Assembly.delete_memory()
    fd.ModelingSpace("2Daxi")
    mesh = fd.mesh.rectangle_mesh(nx=3, ny=3, elm_type="quad4")
    wf = _weakform(formulation, "elastic", "TL")
    pb = fd.problem.NonLinear(fd.Assembly.create(wf, mesh))
    with pytest.raises(NotImplementedError, match="2Daxi"):
        pb.nlsolve(dt=1.0, tmax=1.0, print_info=0)
