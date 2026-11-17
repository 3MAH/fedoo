r"""
Rigid-body free fall: validation of the time integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A rigid sphere falls under gravity and its final height is compared with
the analytical trajectory :math:`z(t) = z_0 - \frac{1}{2} g t^2`. The
``fedoo.constraint.RigidBody`` is solved by fedoo's standard
:class:`fedoo.problem.NonLinear` solver through ``body.solve``, which attaches
a Newmark integrator. Any Newmark scheme (whatever :math:`\beta` and
:math:`\gamma`) integrates a constant acceleration exactly, so the error is
round-off. No contact is involved, so ``ipctk`` is not needed.
"""

import time
import numpy as np
import pyvista as pv

import fedoo as fd

g = 9.81
mass = 1.0
radius = 0.1
z0 = 1.0
dt = 1e-3
t_end = 0.6

print("=" * 60)
print("RIGID BODY FREE FALL — Fedoo validation")
print(f"  m={mass}kg, z0={z0}m, dt={dt}s")
print("=" * 60)

space = fd.ModelingSpace("3D")
space.new_variable("DispX")
space.new_variable("DispY")
space.new_variable("DispZ")
space.new_vector("Disp", ("DispX", "DispY", "DispZ"))

mesh = fd.Mesh.from_pyvista(
    pv.Sphere(radius=radius, center=(0, 0, z0), theta_resolution=8, phi_resolution=8)
)
body = fd.constraint.RigidBody(
    mesh,
    mass=mass,
    inertia_tensor=0.004 * np.eye(3),
    center_of_mass=np.array([0, 0, z0]),
)
body.set_force([0, 0, -mass * g])

pb = body.solve(dt=dt, tmax=t_end, print_info=0)

# Read trajectory from final state
idx = body.assembly.dof_indices
dof = pb.get_dof_solution()
dz_final = dof[idx[2]]
z_final = z0 + dz_final
z_analytical = z0 - 0.5 * g * t_end**2

print(f"  z_final = {z_final:.4f}m (analytical: {z_analytical:.4f}m)")
print(f"  error = {abs(z_final - z_analytical):.2e}m")
print(f"  {'PASS' if abs(z_final - z_analytical) < 0.01 else 'FAIL'}")
