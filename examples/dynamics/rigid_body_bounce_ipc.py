"""
Rigid sphere bouncing on an analytic plane (IPC, Rayleigh damping)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A rigid sphere (``fedoo.constraint.RigidBody``) falls under gravity and
bounces on a floor. The body is solved by fedoo's standard
:class:`fedoo.problem.NonLinear` solver, with a Newmark time integrator
attached to the problem (``pb.set_time_integrator``), Rayleigh damping on the
rigid-body motion, and an analytic floor plane
(``RigidBody.set_static_plane``) for the IPC contact. This is the longer,
higher-resolution version of
:ref:`sphx_glr_examples_dynamics_rigid_body_bounce.py`; it writes the
animation ``rigid_body_bounce_ipc.gif``.

This example requires ``ipctk``.
"""

import os
import time
import numpy as np
import pyvista as pv

import fedoo as fd

g = 9.81
mass = 1.0
radius = 0.1
z0 = 0.5
dt = 5e-4
t_end = 2.0

print("=" * 60)
print("SPHERE BOUNCE — NonLinear + IPC + Rayleigh")
print(f"  m={mass}kg, z0={z0}m, r={radius}m, dt={dt}s")
print("=" * 60)

space = fd.ModelingSpace("3D")
space.new_variable("DispX")
space.new_variable("DispY")
space.new_variable("DispZ")
space.new_vector("Disp", ("DispX", "DispY", "DispZ"))

ball_mesh = fd.Mesh.from_pyvista(
    pv.Sphere(radius=radius, center=(0, 0, z0), theta_resolution=10, phi_resolution=10)
)
body = fd.constraint.RigidBody(
    ball_mesh,
    mass=mass,
    inertia_tensor=(2 / 5) * mass * radius**2 * np.eye(3),
    center_of_mass=np.array([0, 0, z0]),
)
body.set_force([0, 0, -mass * g])
body.set_rayleigh_damping(1.0)
# The floor is an analytic plane: the contact force only depends on the
# distance to the floor, not on the layout of a floor mesh.
body.set_static_plane(normal=(0, 0, 1), point=(0, 0, 0), dhat=0.01, kappa=1e8)

# Solve with NonLinear and save the trajectory through Fedoo's output system.
pb = fd.problem.NonLinear(body.assembly)
pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())

_results_dir = os.path.join(
    os.path.dirname(__file__) if "__file__" in globals() else os.getcwd(), "results"
)
os.makedirs(_results_dir, exist_ok=True)
results = pb.add_output(
    os.path.join(_results_dir, "rigid_body_bounce"), ["Disp", "RigidDisp", "RigidRot"]
)

t0 = time.time()
pb.nlsolve(
    dt=dt,
    dt_max=dt,
    tmax=t_end,
    update_dt=True,
    print_info=1,
    interval_output=dt * 80,
)

# Results can be visualized with ``fd.viewer(results)``.

elapsed = time.time() - t0
# The initial state is prepended because output frames start after the first
# requested output interval.
history = results.get_history(
    ["RigidDisp", "Time"],
    component=["Z", 0],
)
t_hist = np.r_[0.0, history["Time"]]
z_hist = z0 + np.r_[0.0, np.asarray(history["RigidDisp"]).reshape(-1)]
print(f"\n  {results.n_iter} saved frames in {elapsed:.1f}s")
print(f"  z_min={z_hist.min():.4f}m, z_max={z_hist.max():.4f}m")

# Animation
_here = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
gif_path = os.path.join(_here, "rigid_body_bounce_ipc.gif")
fps = 25
frame_indices = np.arange(len(t_hist))

sphere = pv.Sphere(
    radius=radius, center=(0, 0, z0), theta_resolution=20, phi_resolution=20
)
pts_ref = sphere.points.copy()
vis_plane = pv.Plane(
    center=(0, 0, 0),
    direction=(0, 0, 1),
    i_size=1.5,
    j_size=1.5,
    i_resolution=10,
    j_resolution=10,
)

pl = pv.Plotter(window_size=[800, 600], off_screen=True)
pl.set_background("white")
pl.add_mesh(vis_plane, color="lightgrey", opacity=0.8, show_edges=True)
pl.add_mesh(sphere, color="steelblue", smooth_shading=True)
pl.camera_position = [(1.2, -1.2, 0.8), (0, 0, 0.25), (0, 0, 1)]
pl.open_gif(gif_path, fps=fps)

for i in frame_indices:
    sphere.points[:] = pts_ref + np.array([[0, 0, z_hist[i] - z0]])
    sphere.GetPoints().Modified()
    pl.add_text(
        f"t={t_hist[i]:.2f}s  z={z_hist[i]:.3f}m  NonLinear+IPC",
        position="upper_edge",
        font_size=11,
        color="black",
        name="title",
    )
    pl.render()
    pl.write_frame()

pl.close()
print(f"  Saved: {gif_path}")
