"""
Disk-rectangle contact: penalty vs IPC comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example compares the two contact approaches available in fedoo
on a 2D problem: a stiff disk pushed into an elastic rectangle.

  - **Penalty method** (``fd.constraint.Contact``): node-to-surface
    formulation with a user-tuned penalty parameter ``eps_n``.
  - **IPC method** (``fd.constraint.IPCContact``): barrier-potential
    formulation from the ipctk library guaranteeing intersection-free
    configurations, with optional friction and CCD line search.

Both simulations use the same geometry and boundary conditions.
Per-increment metrics are tracked and a comparison summary is printed.

.. note::
   The IPC method requires the ``ipctk`` package:
   ``pip install ipctk``  or  ``pip install fedoo[ipc]``.
"""

import fedoo as fd
import numpy as np
import os
from time import time

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")


def build_mesh_and_material():
    """Build the shared geometry and material (fresh ModelingSpace each call)."""
    fd.ModelingSpace("2D")

    # --- Rectangle mesh ---
    mesh_rect = fd.mesh.rectangle_mesh(
        nx=11, ny=21,
        x_min=0, x_max=1, y_min=0, y_max=1,
        elm_type="quad4",
    )
    mesh_rect.element_sets["rect"] = np.arange(mesh_rect.n_elements)

    # --- Disk mesh ---
    mesh_disk = fd.mesh.disk_mesh(radius=0.5, nr=6, nt=6, elm_type="quad4")
    mesh_disk.nodes += np.array([1.5, 0.48])
    mesh_disk.element_sets["disk"] = np.arange(mesh_disk.n_elements)

    # --- Stack into one mesh ---
    mesh = fd.Mesh.stack(mesh_rect, mesh_disk)

    # --- Material ---
    E, nu = 200e3, 0.3
    material_rect = fd.constitutivelaw.ElasticIsotrop(E, nu)
    material_disk = fd.constitutivelaw.ElasticIsotrop(50e3, nu)
    material = fd.constitutivelaw.Heterogeneous(
        (material_rect, material_disk), ("rect", "disk")
    )

    return mesh, material


# =========================================================================
# Approach 1 : Penalty contact
# =========================================================================

print("=" * 60)
print("PENALTY CONTACT")
print("=" * 60)

mesh, material = build_mesh_and_material()

nodes_left = mesh.find_nodes("X", 0)
nodes_bc = mesh.find_nodes("X>1.5")
nodes_bc = list(set(nodes_bc).intersection(mesh.node_sets["boundary"]))

# Slave nodes = right face of rectangle (exclude disk nodes at X=1)
rect_nodes = set(np.unique(mesh.extract_elements("rect").elements).tolist())
nodes_contact = np.array([n for n in mesh.find_nodes("X", 1) if n in rect_nodes])
surf = fd.mesh.extract_surface(mesh.extract_elements("disk"))

penalty_contact = fd.constraint.Contact(nodes_contact, surf)
penalty_contact.contact_search_once = True
penalty_contact.eps_n = 5e5
penalty_contact.max_dist = 1.0

wf = fd.weakform.StressEquilibrium(material, nlgeom=True)
solid_assembly = fd.Assembly.create(wf, mesh)
assembly = fd.Assembly.sum(solid_assembly, penalty_contact)

pb_penalty = fd.problem.NonLinear(assembly)

if not os.path.isdir("results"):
    os.mkdir("results")
res_penalty = pb_penalty.add_output(
    "results/penalty_contact", solid_assembly, ["Disp", "Stress"]
)

pb_penalty.bc.add("Dirichlet", nodes_left, "Disp", 0)
pb_penalty.bc.add("Dirichlet", nodes_bc, "Disp", [-0.05, 0.025])
pb_penalty.set_nr_criterion("Displacement", tol=5e-3, max_subiter=5)

# --- Tracking callback ---
history_penalty = {
    "time": [],
    "nr_iters": [],
    "reaction_x": [],
    "contact_force_norm": [],
    "max_disp_x": [],
}
_penalty_prev_niter = [0]  # mutable container for closure


def track_penalty(pb):
    history_penalty["time"].append(pb.time)
    # NR iterations for this increment
    nr_this = pb.n_iter - _penalty_prev_niter[0]
    _penalty_prev_niter[0] = pb.n_iter
    history_penalty["nr_iters"].append(nr_this)
    # Reaction force at left boundary (X component)
    F = pb.get_ext_forces("Disp")
    history_penalty["reaction_x"].append(np.sum(F[0, nodes_left]))
    # Contact force norm (use .current which holds the updated state)
    gv = penalty_contact.current.global_vector
    history_penalty["contact_force_norm"].append(
        np.linalg.norm(gv) if not np.isscalar(gv) else 0.0
    )
    # Max X-displacement at contact interface
    disp = pb.get_disp()
    history_penalty["max_disp_x"].append(np.max(np.abs(disp[0, nodes_contact])))


t0 = time()
pb_penalty.nlsolve(
    dt=0.05, tmax=1, update_dt=True, print_info=1, interval_output=0.1,
    callback=track_penalty, exec_callback_at_each_iter=True,
)
penalty_solve_time = time() - t0
print(f"Penalty solve time: {penalty_solve_time:.2f} s")


# =========================================================================
# Approach 2 : IPC contact
# =========================================================================

print("\n" + "=" * 60)
print("IPC CONTACT")
print("=" * 60)

mesh2, material2 = build_mesh_and_material()

nodes_left2 = mesh2.find_nodes("X", 0)
nodes_bc2 = mesh2.find_nodes("X>1.5")
nodes_bc2 = list(set(nodes_bc2).intersection(mesh2.node_sets["boundary"]))

# Nodes at rectangle right face only (exclude disk nodes at X=1)
rect_nodes2 = set(np.unique(mesh2.extract_elements("rect").elements).tolist())
nodes_contact2 = np.array([n for n in mesh2.find_nodes("X", 1) if n in rect_nodes2])

# IPC contact: extract the whole surface -- no slave/master distinction needed
surf_ipc = fd.mesh.extract_surface(mesh2)

ipc_contact = fd.constraint.IPCContact(
    mesh2,
    surface_mesh=surf_ipc,
    dhat=1e-3,                     # relative to bbox diagonal
    dhat_is_relative=True,
    friction_coefficient=0.0,      # frictionless (same as penalty)
    use_ccd=True,                  # CCD line search for robustness
)

wf2 = fd.weakform.StressEquilibrium(material2, nlgeom=True)
solid_assembly2 = fd.Assembly.create(wf2, mesh2)
assembly2 = fd.Assembly.sum(solid_assembly2, ipc_contact)

pb_ipc = fd.problem.NonLinear(assembly2)

res_ipc = pb_ipc.add_output(
    "results/ipc_contact", solid_assembly2, ["Disp", "Stress"]
)

pb_ipc.bc.add("Dirichlet", nodes_left2, "Disp", 0)
pb_ipc.bc.add("Dirichlet", nodes_bc2, "Disp", [-0.05, 0.025])
pb_ipc.set_nr_criterion("Displacement", tol=5e-3, max_subiter=5)

# --- Tracking callback ---
history_ipc = {
    "time": [],
    "nr_iters": [],
    "reaction_x": [],
    "contact_force_norm": [],
    "max_disp_x": [],
    "n_collisions": [],
    "kappa": [],
    "min_distance": [],
}
_ipc_prev_niter = [0]


def track_ipc(pb):
    history_ipc["time"].append(pb.time)
    # NR iterations for this increment
    nr_this = pb.n_iter - _ipc_prev_niter[0]
    _ipc_prev_niter[0] = pb.n_iter
    history_ipc["nr_iters"].append(nr_this)
    # Reaction force at left boundary (X component)
    F = pb.get_ext_forces("Disp")
    history_ipc["reaction_x"].append(np.sum(F[0, nodes_left2]))
    # Contact force norm
    history_ipc["contact_force_norm"].append(
        np.linalg.norm(ipc_contact.global_vector)
    )
    # Max X-displacement at contact interface (rectangle right face)
    disp = pb.get_disp()
    history_ipc["max_disp_x"].append(np.max(np.abs(disp[0, nodes_contact2])))
    # IPC-specific metrics
    history_ipc["n_collisions"].append(len(ipc_contact._collisions))
    history_ipc["kappa"].append(ipc_contact._kappa)
    # Minimum distance (gap)
    verts = ipc_contact._get_current_vertices(pb)
    if len(ipc_contact._collisions) > 0:
        min_d = ipc_contact._collisions.compute_minimum_distance(
            ipc_contact._collision_mesh, verts
        )
    else:
        min_d = float("inf")
    history_ipc["min_distance"].append(min_d)


t0 = time()
pb_ipc.nlsolve(
    dt=0.05, tmax=1, update_dt=True, print_info=1, interval_output=0.1,
    callback=track_ipc, exec_callback_at_each_iter=True,
)
ipc_solve_time = time() - t0
print(f"IPC solve time: {ipc_solve_time:.2f} s")


# =========================================================================
# Comparison summary
# =========================================================================

print("\n")
print("=" * 62)
print("  PERFORMANCE COMPARISON: Penalty vs IPC Contact")
print("=" * 62)

total_nr_penalty = sum(history_penalty["nr_iters"])
total_nr_ipc = sum(history_ipc["nr_iters"])

# Build rows: (label, penalty_value, ipc_value)
rows = [
    ("Total solve time",
     f"{penalty_solve_time:.2f} s",
     f"{ipc_solve_time:.2f} s"),
    ("Total increments",
     str(len(history_penalty["time"])),
     str(len(history_ipc["time"]))),
    ("Total NR iterations",
     str(total_nr_penalty),
     str(total_nr_ipc)),
    ("Avg NR iter / increment",
     f"{total_nr_penalty / max(len(history_penalty['time']), 1):.1f}",
     f"{total_nr_ipc / max(len(history_ipc['time']), 1):.1f}"),
    ("Final reaction Fx",
     f"{history_penalty['reaction_x'][-1]:.1f}" if history_penalty["reaction_x"] else "N/A",
     f"{history_ipc['reaction_x'][-1]:.1f}" if history_ipc["reaction_x"] else "N/A"),
    ("Max contact force norm",
     f"{max(history_penalty['contact_force_norm']):.1f}" if history_penalty["contact_force_norm"] else "N/A",
     f"{max(history_ipc['contact_force_norm']):.1f}" if history_ipc["contact_force_norm"] else "N/A"),
    ("Final contact force norm",
     f"{history_penalty['contact_force_norm'][-1]:.1f}" if history_penalty["contact_force_norm"] else "N/A",
     f"{history_ipc['contact_force_norm'][-1]:.1f}" if history_ipc["contact_force_norm"] else "N/A"),
    ("Max |disp_x|",
     f"{max(history_penalty['max_disp_x']):.6f}" if history_penalty["max_disp_x"] else "N/A",
     f"{max(history_ipc['max_disp_x']):.6f}" if history_ipc["max_disp_x"] else "N/A"),
    ("IPC min gap distance",
     "N/A",
     f"{min(d for d in history_ipc['min_distance'] if np.isfinite(d)):.2e}" if any(np.isfinite(d) for d in history_ipc["min_distance"]) else "inf"),
    ("IPC final barrier kappa",
     "N/A",
     f"{history_ipc['kappa'][-1]:.2e}" if history_ipc["kappa"] else "N/A"),
    ("IPC max active collisions",
     "N/A",
     str(max(history_ipc["n_collisions"])) if history_ipc["n_collisions"] else "N/A"),
]

# Print formatted table
w_label = 28
w_val = 16
print(f"{'Metric':<{w_label}} {'Penalty':>{w_val}} {'IPC':>{w_val}}")
print("-" * (w_label + 2 * w_val + 2))
for label, v_pen, v_ipc in rows:
    print(f"{label:<{w_label}} {v_pen:>{w_val}} {v_ipc:>{w_val}}")
print()


# =========================================================================
# Per-increment detail tables
# =========================================================================

print("-" * 62)
print("  Per-increment detail: PENALTY")
print("-" * 62)
print(f"{'Inc':>4} {'Time':>8} {'NR iter':>8} {'Reaction Fx':>14} {'|F_contact|':>14}")
for i in range(len(history_penalty["time"])):
    print(
        f"{i+1:4d} {history_penalty['time'][i]:8.4f} "
        f"{history_penalty['nr_iters'][i]:8d} "
        f"{history_penalty['reaction_x'][i]:14.2f} "
        f"{history_penalty['contact_force_norm'][i]:14.2f}"
    )

print()
print("-" * 62)
print("  Per-increment detail: IPC")
print("-" * 62)
print(
    f"{'Inc':>4} {'Time':>8} {'NR iter':>8} {'Reaction Fx':>14} "
    f"{'|F_contact|':>14} {'Collisions':>11} {'Min gap':>12} {'Kappa':>12}"
)
for i in range(len(history_ipc["time"])):
    min_d = history_ipc["min_distance"][i]
    min_d_str = f"{min_d:.2e}" if np.isfinite(min_d) else "inf"
    print(
        f"{i+1:4d} {history_ipc['time'][i]:8.4f} "
        f"{history_ipc['nr_iters'][i]:8d} "
        f"{history_ipc['reaction_x'][i]:14.2f} "
        f"{history_ipc['contact_force_norm'][i]:14.2f} "
        f"{history_ipc['n_collisions'][i]:11d} "
        f"{min_d_str:>12} "
        f"{history_ipc['kappa'][i]:12.2e}"
    )

print()

# =========================================================================
# Optional plots (requires matplotlib)
# =========================================================================

try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Penalty vs IPC Contact Comparison", fontsize=14)

    # Reaction force vs time
    ax = axes[0, 0]
    ax.plot(history_penalty["time"], history_penalty["reaction_x"],
            "o-", label="Penalty", markersize=3)
    ax.plot(history_ipc["time"], history_ipc["reaction_x"],
            "s-", label="IPC", markersize=3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Reaction Fx")
    ax.set_title("Reaction Force (X) at Left Boundary")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Contact force norm vs time
    ax = axes[0, 1]
    ax.plot(history_penalty["time"], history_penalty["contact_force_norm"],
            "o-", label="Penalty", markersize=3)
    ax.plot(history_ipc["time"], history_ipc["contact_force_norm"],
            "s-", label="IPC", markersize=3)
    ax.set_xlabel("Time")
    ax.set_ylabel("||F_contact||")
    ax.set_title("Contact Force Norm")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # NR iterations per increment
    ax = axes[1, 0]
    ax.bar(np.arange(len(history_penalty["nr_iters"])) - 0.2,
           history_penalty["nr_iters"], width=0.4, label="Penalty")
    ax.bar(np.arange(len(history_ipc["nr_iters"])) + 0.2,
           history_ipc["nr_iters"], width=0.4, label="IPC")
    ax.set_xlabel("Increment")
    ax.set_ylabel("NR Iterations")
    ax.set_title("Newton-Raphson Iterations per Increment")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # IPC-specific: min gap and kappa
    ax = axes[1, 1]
    finite_gaps = [(t, d) for t, d in zip(history_ipc["time"], history_ipc["min_distance"])
                   if np.isfinite(d)]
    if finite_gaps:
        t_gap, d_gap = zip(*finite_gaps)
        ax.plot(t_gap, d_gap, "s-", color="tab:green", markersize=3, label="Min gap")
    ax.set_xlabel("Time")
    ax.set_ylabel("Min Gap Distance")
    ax.set_title("IPC Minimum Gap Distance")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Zero (penetration)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/penalty_vs_ipc_comparison.png", dpi=150)
    print("Comparison plot saved to results/penalty_vs_ipc_comparison.png")
    plt.show()

except ImportError:
    print("matplotlib not available -- skipping plots.")


# =========================================================================
# Post-processing (requires pyvista)
# =========================================================================
# Uncomment the lines below to visualise and compare the results.

# res_penalty.plot("Stress", "vm", "Node", show=True, scale=1, show_nodes=True)
# res_ipc.plot("Stress", "vm", "Node", show=True, scale=1, show_nodes=True)
