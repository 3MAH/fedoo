"""
Finite-strain Neo-Hookean cantilever (rigid cap, force and displacement control)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A slender, nearly-incompressible cylinder is clamped at its base and bent by a
transverse load applied to a rigid cap tied to its top face. It is loaded well
into the large-deflection, finite-strain regime (tip deflection of order the
cylinder length, cap rotated by ~90 degrees).

This exercises the updated-lagrangian hyperelastic path of
:class:`fedoo.weakform.StressEquilibrium` with a simcoon ``NEOHC`` (compressible
Neo-Hookean) law. The strain energy is

.. math::
    W = \\tfrac{\\mu}{2}(\\bar I_1 - 3) + \\kappa (J \\ln J - J + 1),

with :math:`\\mu = E/2(1+\\nu)` the shear modulus and
:math:`\\kappa = E/3(1-2\\nu)` the bulk modulus.

The rigid cap is a :class:`fedoo.constraint.RigidTie`: it ties the whole top face
to six global rigid-body DOFs (``RigidDispX/Y/Z``, ``RigidRotX/Y/Z``). The same
static problem is solved with the two loading protocols:

* **force control**: a transverse force is applied on ``RigidDispX``
  (Neumann condition on the cap DOF);
* **displacement control**: ``RigidDispX`` is prescribed up to the deflection
  reached under force control, and the reaction force is recovered with
  :meth:`get_ext_forces`.

Both protocols must follow the same force-deflection curve, which is the
cross-check plotted at the end.

Force control is the demanding one: the transverse stiffness of the cap is tiny
(~40 N/m at the origin) and the response stiffens strongly as the cylinder
aligns with the load, so each Newton step is a large move along a soft bending
mode. The equilibrium path is nevertheless monotonic (no limit point), so a plain
static analysis follows it without any inertial or viscous regularization,
provided that:

* the force increments are small while the structure is soft. From the
  undeformed state, the linear prediction of a 2 N increment is already a
  transverse translation of 0.9 L, far outside the convergence basin of
  Newton's method. The force is therefore ramped quadratically in time (gentle
  start, larger increments once the cylinder has stiffened); the automatic
  time stepping (``update_dt=True``) remains as a safety net and would cut an
  increment that fails;
* the ``"Force"`` convergence criterion (residual relative to the applied
  load), which measures equilibrium directly. On a soft mode the norm of the
  displacement correction says little about the remaining out-of-balance
  force, so a loose absolute displacement tolerance can accept unbalanced
  states;
* the safeguard line search (``mode="safeguard"``), which only rejects trial
  states with inverted elements and never throttles legitimate large soft-mode
  steps. A pure residual-descent line search (``mode="minimize"``) would
  strangle them.

The mesh is read from an Abaqus ``.inp`` deck. Two are provided and give the same
result: a **linear** hex8 mesh solved with reduced integration
(:class:`fedoo.weakform.StressEquilibriumRI`, which avoids volumetric locking at
:math:`\\nu = 0.49`), and a **quadratic** hex20 mesh solved with full integration
(:class:`fedoo.weakform.StressEquilibrium`).
"""

import matplotlib.pyplot as plt
import numpy as np

import fedoo as fd

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
L, R = 0.05, 0.0025  # cylinder length / radius [m]
E, nu = 60e6, 0.49  # Young's modulus [Pa], Poisson's ratio
mu = E / (2 * (1 + nu))
kappa = E / (3 * (1 - 2 * nu))

F_CAP = 20.0  # transverse force on the cap [N]

# "linear"    -> hex8,  reduced integration (StressEquilibriumRI); faster
# "quadratic" -> hex20, full integration    (StressEquilibrium)
MESH = "linear"

# path relative to this example's directory (as run by sphinx-gallery)
MESH_FILE = (
    "../../util/meshes/cyl08_hexa_lin.inp"
    if MESH == "linear"
    else "../../util/meshes/cyl08_hexa_quad.inp"
)

fd.ModelingSpace("3D")

# Read the C3D8/C3D20 cylinder mesh from its Abaqus .inp file.
mesh = fd.Mesh.read(MESH_FILE)
z = mesh.nodes[:, 2]
bottom = mesh.find_nodes("Z", z.min())  # clamped base
top = mesh.find_nodes("Z", z.max())  # tied to the rigid cap

# --------------------------------------------------------------------------
# Material and weak form
# --------------------------------------------------------------------------
material = fd.constitutivelaw.Simcoon("NEOHC", [mu, kappa], name="neohookean")

if mesh.elm_type == "hex8":
    # reduced integration + hourglass control avoids volumetric locking
    wf = fd.weakform.StressEquilibriumRI(material, nlgeom="UL")
else:  # hex20: full integration
    wf = fd.weakform.StressEquilibrium(material, nlgeom="UL")
# the initial-stress stiffness is part of the tangent under large rotations;
# it is assembled by default in finite strain (geometric_stiffness=None)


# --------------------------------------------------------------------------
# Static solve, either protocol
# --------------------------------------------------------------------------
def solve(control, value, output=None):
    """Load the cap by a force [N] or a displacement [m] along X.

    Returns the problem, its output and the (deflection, force) history of
    the cap, one row per converged increment.
    """
    assembly = fd.Assembly.create(wf, mesh)
    pb = fd.problem.NonLinear(assembly)
    pb.set_nr_criterion("Force", tol=1e-4, max_subiter=20)
    pb.add_line_search(mode="safeguard")  # validity filter, never throttles

    results = None
    if output is not None:
        results = pb.add_output(output, assembly, ["Disp", "Stress", "Strain"])

    pb.bc.add(fd.constraint.RigidTie(top))  # rigid cap on the top face
    pb.bc.add("Dirichlet", bottom, "Disp", 0)  # clamp the base
    if control == "force":
        # quadratic ramp: small force increments while the structure is soft
        pb.bc.add("Neumann", "RigidDispX", value, time_func=lambda t: t**2)
    else:
        pb.bc.add("Dirichlet", "RigidDispX", value)

    history = [(0.0, 0.0)]

    def record(problem):
        ux = float(np.ravel(problem.get_dof_solution("RigidDispX"))[0])
        fx = float(np.ravel(problem.get_ext_forces("RigidDispX"))[0])
        history.append((ux, fx))

    pb.nlsolve(
        dt=0.02,
        tmax=1.0,
        update_dt=True,
        dt_max=0.02,
        print_info=1,
        interval_output=0.02,
        callback=record,
    )
    return pb, results, np.array(history)


pb, results, curve_force = solve("force", F_CAP, output="neohookean_cantilever")
ux_final = curve_force[-1, 0]
rot_y = float(np.ravel(pb.get_dof_solution("RigidRotY"))[0])

# displacement control up to the same deflection: the reaction must match
_, _, curve_disp = solve("displacement", ux_final)

# --------------------------------------------------------------------------
# Post-processing
# --------------------------------------------------------------------------
print(
    f"\nforce control       : F = {F_CAP:.2f} N -> ux = {ux_final * 1000:.2f} mm "
    f"({ux_final / L:.3f} L), rotY = {np.degrees(rot_y):.1f} deg"
)
print(
    f"displacement control: ux = {curve_disp[-1, 0] * 1000:.2f} mm "
    f"-> F = {curve_disp[-1, 1]:.2f} N"
)

results.load(results.n_iter - 1)  # last (fully-loaded) increment
print(f"max von Mises stress: {results.get_data('Stress', 'vm', 'Node').max():.4e} Pa")

fig, ax = plt.subplots()
ax.plot(curve_disp[:, 0] * 1000, curve_disp[:, 1], "-", label="displacement control")
ax.plot(curve_force[:, 0] * 1000, curve_force[:, 1], "o", label="force control")
ax.set_xlabel("cap deflection [mm]")
ax.set_ylabel("transverse force [N]")
ax.legend()
plt.show()

results.plot("Stress", component="vm", data_type="Node", show=True)
