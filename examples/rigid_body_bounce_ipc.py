"""Sphere bouncing on a plane — NonLinear + IPC contact + Rayleigh damping.

Uses RigidBody with Newmark integration built into the assembly,
solved by Fedoo's standard NonLinear solver.
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
plane_mesh = fd.Mesh.from_pyvista(
    pv.Plane(
        center=(0, 0, 0),
        direction=(0, 0, 1),
        i_size=1.5,
        j_size=1.5,
        i_resolution=6,
        j_resolution=6,
    ).triangulate(),
    name="Floor",
)

body = fd.constraint.RigidBody(
    ball_mesh,
    mass=mass,
    inertia_tensor=(2 / 5) * mass * radius**2 * np.eye(3),
    center_of_mass=np.array([0, 0, z0]),
)
body.set_force([0, 0, -mass * g])
body.set_rayleigh_damping(1.0)
body.set_static_obstacle(plane_mesh, dhat=0.01, kappa=1e8)

# Solve with NonLinear and save the trajectory through Fedoo's output system.
pb = fd.problem.NonLinear(body.assembly)
pb.set_time_integrator(fd.time.SECOND_ORDER, fd.time.Newmark())

results = pb.add_output(
    "rigid_body_bounce",
    ["Disp", "RigidDisp", "RigidRot"],
    include_static_obstacles=True,  # include the plane_mesh in the results field
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
