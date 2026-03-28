"""Rigid body free fall — validation of RigidBody + NonLinearNewmark.

A rigid sphere falls under gravity. The result is compared with the
analytical solution z(t) = z0 - 0.5*g*t^2.

Demonstrates the RigidBody API:
- fd.constraint.RigidBody(mesh, mass, inertia_tensor)
- body.set_force([0, 0, -m*g])
- body.assembly as both Stiffness and Mass in NonLinearNewmark
- PyVista animation of the result
"""

import sys
import numpy as np
import pyvista as pv

sys.path.insert(0, "/Users/ychemisky/Documents/GitHub/fedoo")
import fedoo as fd

# ==============================================================================
# Parameters
# ==============================================================================
g = 9.81
mass = 1.0
radius = 0.1
z0 = 1.0
dt = 1e-3
t_end = 0.6

print("=" * 60)
print("RIGID BODY FREE FALL — Fedoo validation")
print(f"  Fedoo {fd.__version__}, m={mass}kg, z0={z0}m, r={radius}m, dt={dt}s")
print("=" * 60)

# ==============================================================================
# Setup
# ==============================================================================
space = fd.ModelingSpace("3D")
space.new_variable("DispX")
space.new_variable("DispY")
space.new_variable("DispZ")
space.new_vector("Disp", ("DispX", "DispY", "DispZ"))

# Sphere mesh via PyVista
pv_sphere = pv.Sphere(
    radius=radius, center=(0, 0, z0), theta_resolution=12, phi_resolution=12
)
mesh = fd.Mesh.from_pyvista(pv_sphere)
print(f"  Mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements")

# Rigid body
inertia = (2.0 / 5.0) * mass * radius**2 * np.eye(3)
body = fd.constraint.RigidBody(mesh, mass=mass, inertia_tensor=inertia, name="Ball")
body.set_force([0, 0, -mass * g])

# Problem
pb = fd.problem.NonLinearNewmark(
    body.assembly, body.assembly, 0.25, 0.5, name="FreeFall"
)
body.add_to_problem(pb)

# ==============================================================================
# Solve step by step
# ==============================================================================
print("  Solving...")
n_steps = int(t_end / dt)
z_history = [z0]
t_history = [0.0]

pb.initialize()

idx_z = (
    pb.n_node_dof
    + pb._global_dof.indice_start("RigidDispZ")
    + body.constraint.node_cd[2]
)

for step in range(n_steps):
    t = (step + 1) * dt
    pb.dtime = dt
    pb.solve_time_increment()
    pb.set_start()

    dz = pb.get_dof_solution()[idx_z]
    z_history.append(z0 + dz)
    t_history.append(t)

    if step % 100 == 0:
        z_ana = z0 - 0.5 * g * t**2
        print(f"    t={t:.3f}s  z={z0+dz:.4f}m  (analytical: {z_ana:.4f}m)")

t_history = np.array(t_history)
z_history = np.array(z_history)
z_analytical = z0 - 0.5 * g * t_history**2
error = np.max(np.abs(z_history - z_analytical))
print(f"\n  Max error |z_num - z_analytical| = {error:.2e} m")
print(f"  {'PASS' if error < 0.01 else 'FAIL'}")

# ==============================================================================
# PyVista animation
# ==============================================================================
print("\nGenerating PyVista animation...")
out_dir = "/Users/ychemisky/Documents/GitHub/fedoo/examples"
gif_path = f"{out_dir}/rigid_body_freefall.gif"

fps = 25
frame_skip = max(1, int(1.0 / (fps * dt)))
frame_indices = np.arange(0, len(t_history), frame_skip)
print(f"  {len(frame_indices)} frames at {fps} fps")

sphere = pv.Sphere(
    radius=radius, center=(0, 0, z0), theta_resolution=20, phi_resolution=20
)
pts_ref = sphere.points.copy()
plane = pv.Plane(
    center=(0, 0, 0),
    direction=(0, 0, 1),
    i_size=2.0,
    j_size=2.0,
    i_resolution=10,
    j_resolution=10,
)

pl = pv.Plotter(window_size=[800, 600], off_screen=True)
pl.set_background("white")
pl.add_mesh(plane, color="lightgrey", opacity=0.7, show_edges=True)
pl.add_mesh(sphere, color="steelblue", smooth_shading=True)
pl.camera_position = [(2.0, -2.0, 1.5), (0, 0, 0.5), (0, 0, 1)]
pl.open_gif(gif_path, fps=fps)

for i in frame_indices:
    z = z_history[i]
    t = t_history[i]
    sphere.points[:] = pts_ref + np.array([[0, 0, z - z0]])
    sphere.GetPoints().Modified()
    pl.add_text(
        f"t = {t:.3f}s  |  z = {z:.3f}m  |  Fedoo RigidBody",
        position="upper_edge",
        font_size=11,
        color="black",
        name="title",
    )
    pl.render()
    pl.write_frame()

pl.close()
print(f"  Saved: {gif_path}")

# Trajectory plot
try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_history, z_history, "b-", linewidth=1.5, label="Fedoo Newmark")
    ax.plot(t_history, z_analytical, "r--", linewidth=1, label="Analytical")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("z (m)")
    ax.set_title("Rigid body free fall — Fedoo validation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    png_path = f"{out_dir}/rigid_body_freefall.png"
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"  Saved: {png_path}")
except ImportError:
    pass
