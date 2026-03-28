"""Rigid body bouncing on a plane — 6x6 solver + IPC contact + damping.

Uses RigidBody.solve() for direct 6-DOF Newmark integration.
Comparison with analytical elastic bounce.
"""

import sys
import time
import numpy as np
import pyvista as pv

sys.path.insert(0, "/Users/ychemisky/Documents/GitHub/fedoo")
import fedoo as fd

g = 9.81
mass = 1.0
radius = 0.1
z0 = 0.5
dt = 5e-4
t_end = 2.0

print("=" * 60)
print("SPHERE BOUNCE — 6x6 Newmark + IPC + Rayleigh damping")
print(f"  m={mass}kg, z0={z0}m, r={radius}m, dt={dt}s")
print("=" * 60)

# Modeling space
space = fd.ModelingSpace("3D")
space.new_variable("DispX")
space.new_variable("DispY")
space.new_variable("DispZ")
space.new_vector("Disp", ("DispX", "DispY", "DispZ"))

# Meshes
pv_ball = pv.Sphere(
    radius=radius, center=(0, 0, z0), theta_resolution=10, phi_resolution=10
)
ball_mesh = fd.Mesh.from_pyvista(pv_ball)
pv_plane = pv.Plane(
    center=(0, 0, 0),
    direction=(0, 0, 1),
    i_size=1.5,
    j_size=1.5,
    i_resolution=6,
    j_resolution=6,
)
plane_mesh = fd.Mesh.from_pyvista(pv_plane.triangulate())

# Rigid body
inertia = (2 / 5) * mass * radius**2 * np.eye(3)
body = fd.constraint.RigidBody(
    ball_mesh, mass=mass, inertia_tensor=inertia, center_of_mass=np.array([0, 0, z0])
)
body.set_force([0, 0, -mass * g])
body.set_rayleigh_damping(1.0)
body.enable_ipc_contact(plane_mesh, dhat=0.01, kappa=1e8)

print(f"  Ball: {ball_mesh.n_nodes} nodes, IPC kappa=1e8, Rayleigh alpha=1.0")

# Solve with 6x6 direct solver
z_hist = [z0]
t_hist = [0.0]


def collect(t, q, v):
    z_hist.append(z0 + q[2])
    t_hist.append(t)


t0 = time.time()
q, v, a = body.solve(dt=dt, tmax=t_end, callback=collect, print_info=True)
elapsed = time.time() - t0

t_hist = np.array(t_hist)
z_hist = np.array(z_hist)
print(
    f"\n  {len(t_hist)} steps in {elapsed:.1f}s ({elapsed/len(t_hist)*1000:.1f}ms/step)"
)
print(f"  z_min={z_hist.min():.4f}m, z_max={z_hist.max():.4f}m")


# Analytical elastic bounce
def analytical_bounce(t_arr, z0, g, r):
    z = np.empty_like(t_arr)
    zc, vc, tc = z0, 0.0, 0.0
    for i, t in enumerate(t_arr):
        dt_l = t - tc
        z_val = zc + vc * dt_l - 0.5 * g * dt_l**2
        v_val = vc - g * dt_l
        if z_val <= r and v_val < 0:
            disc = vc**2 + 2 * g * (zc - r)
            if disc >= 0:
                dt_c = (vc + np.sqrt(disc)) / g
                vc = -(vc - g * dt_c)
                zc, tc = r, tc + dt_c
                dt_l = t - tc
                z_val = zc + vc * dt_l - 0.5 * g * dt_l**2
        z[i] = z_val
    return z


z_analytical = analytical_bounce(t_hist, z0, g, radius)

# PyVista animation
print("\nGenerating PyVista animation...")
out_dir = "/Users/ychemisky/Documents/GitHub/fedoo/examples"
gif_path = f"{out_dir}/rigid_body_bounce_ipc.gif"

fps = 25
frame_skip = max(1, int(1.0 / (fps * dt)))
frame_indices = np.arange(0, len(t_hist), frame_skip)

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
        f"t={t_hist[i]:.2f}s  z={z_hist[i]:.3f}m  6x6 Newmark+IPC",
        position="upper_edge",
        font_size=11,
        color="black",
        name="title",
    )
    pl.render()
    pl.write_frame()

pl.close()
print(f"  Saved: {gif_path}")

# Plot
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.plot(t_hist, z_hist, "b-", linewidth=1, label="Fedoo 6x6 (IPC+Rayleigh)")
    ax1.plot(
        t_hist,
        z_analytical,
        "r--",
        linewidth=0.8,
        alpha=0.6,
        label="Analytical elastic",
    )
    ax1.axhline(
        y=radius, color="grey", linestyle=":", alpha=0.5, label=f"contact z={radius}m"
    )
    ax1.set_ylabel("z (m)")
    ax1.set_title(
        f"Sphere bounce — {elapsed:.1f}s ({elapsed/len(t_hist)*1000:.1f}ms/step)"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(t_hist, z_hist - z_analytical, "g-", linewidth=0.8)
    ax2.set_ylabel("z_fedoo - z_analytical (m)")
    ax2.set_xlabel("t (s)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/rigid_body_bounce_ipc.png", dpi=150)
    plt.close()
    print(f"  Saved: {out_dir}/rigid_body_bounce_ipc.png")
except ImportError:
    pass
