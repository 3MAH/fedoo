"""Cow bouncing on a plane — 6x6 Newmark + IPC contact.

The cow mesh from PyVista is watertight and manifold (2903 nodes).
"""

import sys
import time
import numpy as np
import pyvista as pv

sys.path.insert(0, "/Users/ychemisky/Documents/GitHub/fedoo")
import fedoo as fd

g = 9.81
dt = 5e-4
t_end = 3.0

print("=" * 60)
print("COW BOUNCE — 6x6 Newmark + IPC contact")
print("=" * 60)

space = fd.ModelingSpace("3D")
space.new_variable("DispX")
space.new_variable("DispY")
space.new_variable("DispZ")
space.new_vector("Disp", ("DispX", "DispY", "DispZ"))

# Cow mesh — watertight, manifold
pv_cow = pv.examples.download_cow().triangulate().clean()
print(
    f"  Raw cow: {pv_cow.n_points} pts, manifold={pv_cow.is_manifold}, open_edges={pv_cow.n_open_edges}"
)

# Scale to ~0.3m and position above ground
bounds = pv_cow.bounds
size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
pv_cow = pv_cow.scale(0.3 / size, inplace=False)
pv_cow = pv_cow.translate(
    [-pv_cow.center[0], -pv_cow.center[1], -pv_cow.bounds[4] + 0.5],
    inplace=False,
)
# Decimate for speed
pv_cow = pv_cow.decimate(0.9).triangulate().clean()
pv_cow = pv_cow.compute_normals(consistent_normals=True, auto_orient_normals=True)

cow_mesh = fd.Mesh.from_pyvista(pv_cow)
print(f"  Decimated cow: {cow_mesh.n_nodes} nodes, {cow_mesh.n_elements} faces")

# Plane
pv_plane = pv.Plane(
    center=(0, 0, 0),
    direction=(0, 0, 1),
    i_size=2.0,
    j_size=2.0,
    i_resolution=8,
    j_resolution=8,
)
plane_mesh = fd.Mesh.from_pyvista(pv_plane.triangulate())

# Rigid body with box-approximate inertia
mass = 1.0
bb = pv_cow.bounds
lx, ly, lz = bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4]
I_approx = (mass / 12) * np.diag([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])

body = fd.constraint.RigidBody(
    cow_mesh,
    mass=mass,
    inertia_tensor=I_approx,
    center_of_mass=np.array(pv_cow.center),
    name="Cow",
)
body.set_force([0, 0, -mass * g])
body.set_rayleigh_damping(1.5)
body.enable_ipc_contact(plane_mesh, dhat=0.01)  # kappa auto-tuned

print(f"  mass={mass}kg, center_z={body.center_of_mass[2]:.3f}m")
print(f"  inertia diag: {np.diag(I_approx)}")
print(f"  dt={dt}s → {int(t_end/dt)} steps")

# Solve
q_hist = [np.zeros(6)]
t_hist = [0.0]


def collect(t, q, v):
    q_hist.append(q.copy())
    t_hist.append(t)


t0 = time.time()
q, v, a = body.solve(dt=dt, tmax=t_end, callback=collect, print_info=True)
elapsed = time.time() - t0

t_hist = np.array(t_hist)
q_hist = np.array(q_hist)
z_hist = body.center_of_mass[2] + q_hist[:, 2]

print(
    f"\n  {len(t_hist)} steps in {elapsed:.1f}s ({elapsed/len(t_hist)*1000:.1f}ms/step)"
)
print(f"  z_min={z_hist.min():.4f}m")
print(
    f"  max rotation: rx={np.degrees(q_hist[:,3].max()):.0f}° ry={np.degrees(q_hist[:,4].max()):.0f}° rz={np.degrees(q_hist[:,5].max()):.0f}°"
)

# Animation
print("\nGenerating PyVista animation...")
out_dir = "/Users/ychemisky/Documents/GitHub/fedoo/examples"
gif_path = f"{out_dir}/rigid_body_cow_bounce.gif"

fps = 25
frame_skip = max(1, int(1.0 / (fps * dt)))
frame_indices = np.arange(0, len(t_hist), frame_skip)
print(f"  {len(frame_indices)} frames")

vis_cow = pv_cow.copy()
pts_ref = vis_cow.points.copy()
center = body.center_of_mass
vis_plane = pv.Plane(
    center=(0, 0, 0),
    direction=(0, 0, 1),
    i_size=2.0,
    j_size=2.0,
    i_resolution=10,
    j_resolution=10,
)

pl = pv.Plotter(window_size=[900, 600], off_screen=True)
pl.set_background("white")
pl.add_mesh(vis_plane, color="lightgrey", opacity=0.8, show_edges=True)
pl.add_mesh(vis_cow, color="sandybrown", smooth_shading=True)
pl.camera_position = [(1.5, -1.5, 1.0), (0, 0, 0.25), (0, 0, 1)]
pl.open_gif(gif_path, fps=fps)

try:
    from simcoon import Rotation
except ImportError:
    from scipy.spatial.transform import Rotation

for i in frame_indices:
    qi = q_hist[i]
    R = Rotation.from_rotvec(qi[3:]).as_matrix()
    vis_cow.points[:] = (pts_ref - center) @ R.T + center + qi[:3]
    vis_cow.GetPoints().Modified()
    pl.add_text(
        f"t={t_hist[i]:.2f}s  z={z_hist[i]:.3f}m",
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

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t_hist, z_hist, "b-", linewidth=1)
    ax.axhline(y=0, color="grey", linestyle=":", alpha=0.5)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("z (m)")
    ax.set_title(
        f"Cow bounce — {cow_mesh.n_nodes} nodes, {elapsed:.1f}s ({elapsed/len(t_hist)*1000:.1f}ms/step)"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/rigid_body_cow_bounce.png", dpi=150)
    plt.close()
    print(f"  Saved: {out_dir}/rigid_body_cow_bounce.png")
except ImportError:
    pass
