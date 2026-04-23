"""Bunny bouncing on a plane — NonLinear + IPC contact.

Convex hull of Stanford bunny (watertight). Demonstrates tumbling
on impact with asymmetric geometry.
"""

import os
import time
import numpy as np
import pyvista as pv

import fedoo as fd

try:
    from simcoon import Rotation
except ImportError:
    from scipy.spatial.transform import Rotation

g = 9.81
dt = 5e-4
t_end = 2.0

print("=" * 60)
print("BUNNY BOUNCE — NonLinear + IPC contact")
print("=" * 60)

space = fd.ModelingSpace("3D")
space.new_variable("DispX")
space.new_variable("DispY")
space.new_variable("DispZ")
space.new_vector("Disp", ("DispX", "DispY", "DispZ"))

# Watertight bunny from convex hull
pv_raw = pv.examples.download_bunny().decimate(0.97)
pv_bunny = pv_raw.delaunay_3d().extract_surface().triangulate().clean()
s = max(
    pv_bunny.bounds[1] - pv_bunny.bounds[0],
    pv_bunny.bounds[3] - pv_bunny.bounds[2],
    pv_bunny.bounds[5] - pv_bunny.bounds[4],
)
pv_bunny = pv_bunny.scale(0.25 / s, inplace=False)
pv_bunny = pv_bunny.translate(
    [-pv_bunny.center[0], -pv_bunny.center[1], -pv_bunny.bounds[4] + 0.4], inplace=False
)
pv_bunny = pv_bunny.compute_normals(consistent_normals=True, auto_orient_normals=True)

bunny_mesh = fd.Mesh.from_pyvista(pv_bunny)
plane_mesh = fd.Mesh.from_pyvista(
    pv.Plane(
        center=(0, 0, 0),
        direction=(0, 0, 1),
        i_size=1.5,
        j_size=1.5,
        i_resolution=6,
        j_resolution=6,
    ).triangulate()
)

mass = 0.5
bb = pv_bunny.bounds
lx, ly, lz = bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4]
I = (mass / 12) * np.diag([ly**2 + lz**2, lx**2 + lz**2, lx**2 + ly**2])

body = fd.constraint.RigidBody(
    bunny_mesh,
    mass=mass,
    inertia_tensor=I,
    center_of_mass=np.array(pv_bunny.center),
    name="Bunny",
)
body.set_force([0, 0, -mass * g])
body.set_rayleigh_damping(1.0)
body.enable_ipc_contact(plane_mesh, dhat=0.008, kappa=1e8)

print(f"  Bunny: {bunny_mesh.n_nodes} nodes, mass={mass}kg")

# Solve with manual loop for trajectory
pb = fd.problem.NonLinear(body.assembly)
body.add_to_problem(pb)
pb.initialize()

idx = body.assembly._dof_indices
q_hist = [np.zeros(6)]
t_hist = [0.0]

t0 = time.time()
n_steps = int(round(t_end / dt))
for step in range(n_steps):
    pb.dtime = dt
    pb.solve_time_increment()
    pb.set_start()
    dof = pb.get_dof_solution()
    q_hist.append(dof[idx].copy())
    t_hist.append((step + 1) * dt)

elapsed = time.time() - t0
t_hist = np.array(t_hist)
q_hist = np.array(q_hist)
z_hist = body.center_of_mass[2] + q_hist[:, 2]
print(
    f"  {len(t_hist)} steps in {elapsed:.1f}s ({elapsed / len(t_hist) * 1000:.1f}ms/step)"
)
print(f"  z_min={z_hist.min():.4f}m")

# Animation (low res for file size)
_here = os.path.dirname(__file__) if "__file__" in globals() else os.getcwd()
out = os.path.join(_here, "rigid_body_bunny_bounce.gif")
fps = 15
frame_skip = max(1, int(1.0 / (fps * dt)))
frame_indices = np.arange(0, len(t_hist), frame_skip)

vis = pv_bunny.copy()
pts_ref = vis.points.copy()
center = body.center_of_mass

pl = pv.Plotter(window_size=[600, 400], off_screen=True)
pl.set_background("white")
pl.add_mesh(
    pv.Plane(
        center=(0, 0, 0),
        direction=(0, 0, 1),
        i_size=1.5,
        j_size=1.5,
        i_resolution=8,
        j_resolution=8,
    ),
    color="lightgrey",
    opacity=0.8,
    show_edges=True,
)
pl.add_mesh(vis, color="sandybrown", smooth_shading=True)
pl.camera_position = [(1.0, -1.0, 0.7), (0, 0, 0.2), (0, 0, 1)]
pl.open_gif(out, fps=fps)

for i in frame_indices:
    qi = q_hist[i]
    R = Rotation.from_rotvec(qi[3:]).as_matrix()
    vis.points[:] = (pts_ref - center) @ R.T + center + qi[:3]
    vis.GetPoints().Modified()
    pl.add_text(
        f"t={t_hist[i]:.2f}s z={z_hist[i]:.3f}m",
        position="upper_edge",
        font_size=10,
        color="black",
        name="t",
    )
    pl.render()
    pl.write_frame()

pl.close()
print(f"  Saved: {out}")
