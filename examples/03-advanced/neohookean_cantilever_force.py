"""
Force-driven Neo-Hookean cantilever (massive rigid cap, implicit dynamics)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Companion of :ref:`the displacement-driven cantilever example
<neohookean_cantilever>`: the same slender, nearly-incompressible NEOHC
cylinder, but loaded by a **transverse force** applied to its rigid cap —
the natural protocol when mirroring a dynamics engine such as ArtiSynth.

Static force control is ill-posed here: the transverse stiffness of the
cap is tiny (~40 N/m at the origin, rotational ~2.5e-2 N.m/rad), so a
Newton step towards the distant equilibrium leaves any convergence basin
(the classical remedies, arc-length continuation or Tikhonov
stabilization, are not used in this example). The robust protocol is
**quasi-static-by-dynamics**:

* the cap is a true rigid body — a :class:`fedoo.constraint.RigidTie` for
  the kinematics plus a :class:`~fedoo.constraint.rigid_body.RigidBodyAssembly`
  carrying its 6x6 mass/inertia, so the Newmark term ``M/(beta*dt^2)``
  regularizes the soft cap DOFs (this is exactly what makes the ArtiSynth
  mirror robust);
* a Newmark integrator attached at the problem level
  (``pb.set_time_integrator``) with an **unconditionally stable** pair:
  ``gamma = 0.6`` for high-frequency damping requires
  ``beta >= gamma/2 = 0.30`` — a violating pair (e.g. the
  default-looking ``beta = 0.25``) grows the high-frequency modes
  geometrically and collapses after a few tens of increments, whatever dt
  (fedoo now warns in that case);
* a slow force ramp (quadratic, gentle start) followed by a hold period so
  the response settles to the static equilibrium;
* the safeguard line search (``mode="safeguard"``), which rejects only
  trial states with inverted elements and never throttles legitimate large
  soft-mode steps. The default ``mode="natural"`` (affine-invariant test on
  the simplified Newton correction) handles them as well; a pure
  residual-descent line search (``mode="minimize"``) would strangle them.

Cross-check: the displacement-driven curve gives ``ux(F = 20 N) = 45.5 mm
= 0.911 L``; this run settles within ~2% of it (the difference is the
residual numerical damping at the end of the hold).

The tie is registered automatically when the ``RigidBodyAssembly`` is part
of the problem assembly — do not add it again (fedoo now ignores the
duplicate, which used to corrupt the MPC elimination).
"""

import numpy as np

import fedoo as fd
from fedoo.constraint.rigid_body import RigidBodyAssembly

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
L, R = 0.05, 0.0025  # cylinder length / radius [m]
E, nu = 60e6, 0.49  # Young's modulus [Pa], Poisson's ratio
mu = E / (2 * (1 + nu))
kappa = E / (3 * (1 - 2 * nu))
rho = 1000.0  # cylinder density [kg/m^3]

F_CAP = 20.0  # transverse force on the cap [N]
RAMP = 1.0  # force ramp duration [s] (quadratic)
HOLD = 1.0  # settling period at constant force [s]
DT = 0.01  # time step [s] (kept constant: dt_max=DT)

# a small physical cap: ~10 g, I ~ 1e-6 kg.m^2 — enough for the Newmark
# inertia to regularize the soft cap DOFs at this time step
M_CAP = 0.01
I_CAP = 1e-6 * np.eye(3)

# same Abaqus deck as the displacement-driven example (and as the
# ArtiSynth mirror model)
MESH_FILE = "../../util/meshes/cyl08_hexa_lin.inp"

fd.ModelingSpace("3D")
mesh = fd.Mesh.read(MESH_FILE)
z = mesh.nodes[:, 2]
bottom = mesh.find_nodes("Z", z.min())  # clamped base
top = mesh.find_nodes("Z", z.max())  # tied to the rigid cap

# --------------------------------------------------------------------------
# Material, weak form, assemblies (FE + rigid cap inertia)
# --------------------------------------------------------------------------
material = fd.constitutivelaw.Simcoon("NEOHC", [mu, kappa], name="neohookean")
material.set_density(rho)

wf = fd.weakform.StressEquilibriumRI(material, nlgeom="UL")
wf.list_weakform[0].geometric_stiffness = True
assembly_fe = fd.Assembly.create(wf, mesh, name="cylinder")

tie = fd.constraint.RigidTie(top)
cap = RigidBodyAssembly(
    mass=M_CAP,
    inertia_tensor=I_CAP,
    rigid_tie=tie,
    mesh=mesh,
    name="cap_inertia",
)
assembly = assembly_fe + cap

# --------------------------------------------------------------------------
# Problem: Newmark at the problem level, force ramp, safeguard line search
# --------------------------------------------------------------------------
pb = fd.problem.NonLinear(assembly)
pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark(beta=0.3025, gamma=0.6))
pb.set_nr_criterion("Displacement", err0=1.0, tol=1e-3, max_subiter=20)
pb.add_line_search(mode="safeguard")  # validity filter, never throttles

results = pb.add_output(
    "neohookean_cantilever_force", assembly_fe, ["Disp", "Stress", "Strain"]
)

# NB: the tie is auto-registered by the RigidBodyAssembly — no pb.bc.add(tie)
pb.bc.add("Dirichlet", bottom, "Disp", 0)  # clamp the base
TMAX = RAMP + HOLD
pb.bc.add(
    "Neumann",
    "RigidDispX",
    F_CAP,
    # quadratic ramp on [0, RAMP], hold at F_CAP on [RAMP, TMAX]
    time_func=lambda tf: min(tf * TMAX / RAMP, 1.0) ** 2,
)

history = []


def record(problem):
    ux = float(np.ravel(problem.get_dof_solution("RigidDispX"))[0])
    history.append((problem.time, ux))


pb.nlsolve(
    dt=DT,
    tmax=TMAX,
    update_dt=True,
    dt_max=DT,
    print_info=1,
    interval_output=0.05,
    callback=record,
)

# --------------------------------------------------------------------------
# Post-processing: settlement and cross-check
# --------------------------------------------------------------------------
hist = np.array(history)
ux_final = hist[-1, 1]
drift = abs(hist[-1, 1] - hist[-11, 1]) if len(hist) > 11 else float("nan")
rot_y = float(np.ravel(pb.get_dof_solution("RigidRotY"))[0])

print(
    f"\nF = {F_CAP} N -> ux = {ux_final * 1000:.2f} mm ({ux_final / L:.3f} L), "
    f"rotY = {np.degrees(rot_y):.1f} deg"
)
print(f"settlement drift over the last 10 steps: {drift * 1000:.4f} mm")
print("displacement-control cross-check: 45.5 mm (0.911 L)")

np.savetxt(
    "neohookean_cantilever_force_history.csv",
    hist,
    delimiter=",",
    header="time[s],ux_cap[m]",
)

results.load(results.n_iter - 1)
results.plot("Stress", component="vm", data_type="Node", show=True)
