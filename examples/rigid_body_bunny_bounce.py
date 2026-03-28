"""Stanford bunny (convex hull) bouncing on a plane — 6x6 solver + IPC.

The bunny_coarse mesh has 658 open edges, so we use its convex hull
(watertight, manifold). The shape is simplified but still asymmetric,
producing interesting tumbling on impact.
"""

import sys
import time
import numpy as np
import pyvista as pv

sys.path.insert(0, "/Users/ychemisky/Documents/GitHub/fedoo")
import fedoo as fd

g = 9.81
dt = 5e-4
t_end = 2.0

print("=" * 60)
print("BUNNY BOUNCE — 6x6 Newmark + IPC contact")
print("=" * 60)

space = fd.ModelingSpace("3D")
space.new_variable("DispX")
space.new_variable("DispY")
space.new_variable("DispZ")
space.new_vector("Disp", ("DispX", "DispY", "DispZ"))

# Build watertight bunny from convex hull of decimated full bunny
pv_raw = pv.examples.download_bunny().decimate(0.95)
pv_bunny = pv_raw.delaunay_3d().extract_surface().triangulate().clean()

# Scale and position
bounds = pv_bunny.bounds
size = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
pv_bunny = pv_bunny.scale(0.25 / size, inplace=False)
pv_bunny = pv_bunny.translate(
    [-pv_bunny.center[0], -pv_bunny.center[1], -pv_bunny.bounds[4] + 0.4],
    inplace=False,
)
pv_bunny = pv_bunny.compute_normals(consistent_normals=True, auto_orient_normals=True)

bunny_mesh = fd.Mesh.from_pyvista(pv_bunny)
print(f"  Bunny: {bunny_mesh.n_nodes} nodes, {bunny_mesh.n_elements} faces")
print(f"  Manifold: {pv_bunny.is_manifold}, Open edges: {pv_bunny.n_open_edges}")

# Plane
pv_plane = pv.Plane(
    center=(0, 0, 0),
    direction=(0, 0, 1),
    i_size=1.5,
    j_size=1.5,
    i_resolution=6,
    j_resolution=6,
)
plane_mesh = fd.Mesh.from_pyvista(pv_plane.triangulate())

# Rigid body — box inertia approximation
mass = 0.5
bb = pv_bunny.bounds
lx, ly, lz = bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4]
I_approx = (mass / 12) * np.diag([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])

body = fd.constraint.RigidBody(
    bunny_mesh,
    mass=mass,
    inertia_tensor=I_approx,
    center_of_mass=np.array(pv_bunny.center),
    name="Bunny",
)
body.set_force([0, 0, -mass * g])
body.set_rayleigh_damping(1.0)
body.enable_ipc_contact(plane_mesh, dhat=0.008, kappa=1e8)

print(f"  mass={mass}kg, center_z={body.center_of_mass[2]:.3f}m")
print(f"  dt={dt}s → {int(t_end/dt)} steps")

# Solve — store full q (6 DOFs) for rotation animation
q_hist = [np.zeros(6)]
t_hist = [0.0]


def collect(t, q, v):
    q_hist.append(q.copy())
    t_hist.append(t)


t0 = time.time()
q_final, v, a = body.solve(dt=dt, tmax=t_end, callback=collect, print_info=True)
elapsed = time.time() - t0

t_hist = np.array(t_hist)
q_hist = np.array(q_hist)
z_hist = body.center_of_mass[2] + q_hist[:, 2]
print(
    f"\n  {len(t_hist)} steps in {elapsed:.1f}s ({elapsed/len(t_hist)*1000:.1f}ms/step)"
)
print(f"  z_min={z_hist.min():.4f}m")
print(
    f"  max rotation (deg): rx={np.degrees(q_hist[:,3].max()):.1f} ry={np.degrees(q_hist[:,4].max()):.1f} rz={np.degrees(q_hist[:,5].max()):.1f}"
)

# Animation
print("\nGenerating PyVista animation...")
out_dir = "/Users/ychemisky/Documents/GitHub/fedoo/examples"
gif_path = f"{out_dir}/rigid_body_bunny_bounce.gif"

fps = 25
frame_skip = max(1, int(1.0 / (fps * dt)))
frame_indices = np.arange(0, len(t_hist), frame_skip)

vis = pv_bunny.copy()
pts_ref = vis.points.copy()
vis_plane = pv.Plane(
    center=(0, 0, 0),
    direction=(0, 0, 1),
    i_size=1.5,
    j_size=1.5,
    i_resolution=10,
    j_resolution=10,
)

pl = pv.Plotter(window_size=[900, 600], off_screen=True)
pl.set_background("white")
pl.add_mesh(vis_plane, color="lightgrey", opacity=0.8, show_edges=True)
pl.add_mesh(vis, color="sandybrown", smooth_shading=True)
pl.camera_position = [(1.0, -1.0, 0.7), (0, 0, 0.2), (0, 0, 1)]
pl.open_gif(gif_path, fps=fps)

try:
    from simcoon import Rotation
except ImportError:
    from scipy.spatial.transform import Rotation

center = body.center_of_mass
for i in frame_indices:
    q_i = q_hist[i]
    disp = q_i[:3]
    angles = q_i[3:]

    # Apply full rigid body transform: rotate around center, then translate
    R = Rotation.from_euler("XYZ", angles).as_matrix()
    pts_rotated = (pts_ref - center) @ R.T + center + disp
    vis.points[:] = pts_rotated
    vis.GetPoints().Modified()

    pl.add_text(
        f"t={t_hist[i]:.2f}s  z={z_hist[i]:.3f}m  rx={np.degrees(angles[0]):.1f}°",
        position="upper_edge",
        font_size=10,
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
        f"Bunny bounce — {bunny_mesh.n_nodes} nodes, {elapsed:.1f}s ({elapsed/len(t_hist)*1000:.1f}ms/step)"
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/rigid_body_bunny_bounce.png", dpi=150)
    plt.close()
    print(f"  Saved: {out_dir}/rigid_body_bunny_bounce.png")
except ImportError:
    pass
