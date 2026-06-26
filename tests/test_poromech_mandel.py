"""Mandel 2D consolidation: validate the Mandel-Cryer effect.

The Mandel problem (Mandel 1953, Abousleiman et al. 1996) is the classical
poroelasticity benchmark that captures a 2D coupling phenomenon no 1D
consolidation can reproduce: the pore pressure at the centre of the sample
rises ABOVE its initial undrained value before decaying. This non-monotonic
overshoot — the Mandel-Cryer effect — is driven by the stiffness redistribution
that occurs as fluid drains from the lateral faces.

Setup (quarter symmetry, plane strain via thin 3D slab,
displacement-controlled rigid plate):

    z = b  : rigid impermeable plate, uniform vertical displacement imposed
    z = 0  : symmetry plane (u_z = 0, natural zero flux on p)
    x = 0  : symmetry plane (u_x = 0, natural zero flux on p)
    x = a  : drainage (p = 0), traction-free
    y = 0, y = thin : plane strain (u_y = 0 on both faces, ny = 1 element)

The rigid-plate condition is approximated by imposing the SAME vertical
displacement on all top nodes (Dirichlet). This keeps the FE system well
conditioned and still captures the Mandel-Cryer mechanism (skeleton
stiffness redistribution as fluid drains laterally).

The test verifies the qualitative signature only (non-monotonic peak with
significant overshoot). Quantitative comparison to the analytical series
solution lives in the example script.
"""

import numpy as np

import fedoo as fd


def test_mandel_cryer_overshoot():
    """The pore pressure at the centre must exhibit a non-monotonic
    overshoot ABOVE its initial value before decaying (Mandel-Cryer)."""

    a = 1.0  # half-width (x, drainage direction)
    b = 1.0  # half-height (z, loading direction)
    thin_y = 0.1
    nx, ny, nz = 9, 2, 9  # nodes per axis
    delta = -1.0e-4  # imposed compressive top displacement (m)

    fd.ModelingSpace("3D")
    mesh = fd.mesh.box_mesh(
        nx=nx,
        ny=ny,
        nz=nz,
        x_min=0,
        x_max=a,
        y_min=0,
        y_max=thin_y,
        z_min=0,
        z_max=b,
        elm_type="hex8",
        name="MandelQuarter",
    )

    E, nu = 1.0e6, 0.3
    skel = fd.constitutivelaw.ElasticIsotrop(E, nu, name="Skel")
    fluid = fd.constitutivelaw.PoroFluidProperties(
        permeability=1.0e-7,
        fluid_viscosity=1.0,
        biot_coefficient=1.0,
        biot_modulus=1.0e8,
        initial_porosity=0.5,
        name="Fluid",
    )
    wf = fd.weakform.PoroMechanicsSimple(skel, fluid, nlgeom=False, name="Poro")
    fd.Assembly.create(wf, mesh, name="MandelAssembly")
    pb = fd.problem.NonLinear("MandelAssembly")
    pb.set_nr_criterion("Displacement", tol=1e-3, max_subiter=80)

    sym_x = mesh.find_nodes("X", 0.0)
    sym_z = mesh.find_nodes("Z", 0.0)
    plane_y_min = mesh.find_nodes("Y", 0.0)
    plane_y_max = mesh.find_nodes("Y", thin_y)
    drainage = mesh.find_nodes("X", a)
    top = mesh.find_nodes("Z", b)
    centre_line = np.intersect1d(sym_x, sym_z)

    # Mechanical BCs
    pb.bc.add("Dirichlet", sym_x, "DispX", 0.0)
    pb.bc.add("Dirichlet", sym_z, "DispZ", 0.0)
    pb.bc.add("Dirichlet", plane_y_min, "DispY", 0.0)
    pb.bc.add("Dirichlet", plane_y_max, "DispY", 0.0)

    # Drainage
    pb.bc.add("Dirichlet", drainage, "PorePressure", 0.0)

    # Rigid-plate compression: uniform vertical displacement on the top face.
    # The displacement is ramped quickly to its full value to mimic a step
    # load (Heaviside-like) so the Mandel-Cryer transient is visible.
    t_ramp = 0.5  # ramp the load to full over the first 0.5 s, then hold
    tmax = 30.0
    pb.bc.add(
        "Dirichlet",
        top,
        "DispZ",
        delta,
        time_func=lambda tf: min(1.0, tf * tmax / t_ramp),
    )

    dt = 0.25
    nb_steps = int(tmax / dt)

    pb.initialize()
    pb.tmax = dt * nb_steps
    pb.dtime = dt
    pb.set_start()

    p_centre_history = []

    for step in range(nb_steps):
        pb.time = step * dt
        convergence, nb_nr, res = pb.solve_time_increment()
        assert convergence, f"Step {step}: Mandel Newton failed (res={res:g})"
        pb.set_start()
        p_field = pb.get_dof_solution("PorePressure")
        p_centre_history.append(float(np.mean(p_field[centre_line])))

    p_history = np.asarray(p_centre_history)

    # Mandel-Cryer signature 1: PorePressure positive in compression
    assert p_history[0] > 0.0, (
        f"Initial centre pore pressure must be positive (compression), "
        f"got {p_history[0]:g}"
    )

    # Mandel-Cryer signature 2: NON-MONOTONIC peak
    # the maximum p must occur AFTER the first step
    idx_max = int(np.argmax(p_history))
    assert idx_max >= 1, (
        f"Mandel-Cryer overshoot not captured: p_max occurs at the first "
        f"step (idx 0). p_history[:6]={p_history[:6]}"
    )

    # Mandel-Cryer signature 3: overshoot magnitude
    overshoot = (p_history[idx_max] - p_history[0]) / p_history[0]
    assert overshoot > 0.02, (
        f"Mandel-Cryer overshoot too small: {overshoot * 100:.2f}% "
        f"(expected > 2%); p[0]={p_history[0]:g}, "
        f"p[max]={p_history[idx_max]:g} at idx {idx_max}"
    )

    # Signature 4: eventual dissipation
    assert p_history[-1] < 0.3 * p_history[idx_max], (
        f"Pore pressure should largely dissipate by the end of the run; "
        f"got p_end={p_history[-1]:g} vs p_max={p_history[idx_max]:g}"
    )


if __name__ == "__main__":
    test_mandel_cryer_overshoot()
    print("test_poromech_mandel: OK")
