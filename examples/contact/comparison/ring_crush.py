"""
Hole plate self-contact: penalty vs IPC comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A plate with a large circular hole (radius 45 mm in a 100 x 100 mm plate)
is compressed vertically under plane strain until the hole walls come into
self-contact during buckling.

Two contact methods are compared:

  - **Penalty method** (``fd.constraint.SelfContact``): node-to-surface
    formulation with a user-tuned penalty parameter.
  - **IPC method** (``fd.constraint.IPCSelfContact``): barrier-potential
    formulation from the ipctk library guaranteeing intersection-free
    configurations, with CCD line search.

Both simulations use the same geometry, material and boundary conditions.
Per-increment reaction forces and IPC-specific metrics are tracked and
a comparison summary is printed.

Expected behaviour
------------------
Before self-contact occurs (roughly the first half of the loading), both
methods produce identical results.  Once the hole walls come into contact:

- The **penalty method** allows some interpenetration and the contact forces
  may grow rapidly if the penalty parameter is too large relative to the
  stiffness.
- The **IPC method** prevents all interpenetration (minimum gap > 0) but the
  adaptive barrier stiffness (kappa) may grow aggressively, especially
  when sustained contacts with decreasing gap are present.

.. note::
   This example requires the ``simcoon`` package for the elasto-plastic
   material and the ``ipctk`` package for the IPC method.
"""

import fedoo as fd
import numpy as np
import os
from time import time

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")


# =========================================================================
# Shared geometry and material builder
# =========================================================================

def build_mesh_and_material():
    """Build the hole plate geometry and elasto-plastic material.

    Returns fresh objects each call (new ModelingSpace).
    """
    fd.ModelingSpace("2D")

    mesh = fd.mesh.hole_plate_mesh(
        nr=15, nt=15, length=100, height=100, radius=45,
    )

    # Material: elasto-plastic (EPICP) — same as tube_compression example
    E, nu = 200e3, 0.3
    sigma_y = 300
    k, m = 1000, 0.3
    props = np.array([E, nu, 1e-5, sigma_y, k, m])
    material = fd.constitutivelaw.Simcoon("EPICP", props)

    return mesh, material


def get_bc_nodes(mesh):
    """Return (bottom, top) node arrays."""
    nodes_bot = mesh.find_nodes("Y", mesh.bounding_box.ymin)
    nodes_top = mesh.find_nodes("Y", mesh.bounding_box.ymax)
    return nodes_bot, nodes_top


# =========================================================================
# Approach 1 : Penalty self-contact
# =========================================================================

print("=" * 62)
print("  PENALTY SELF-CONTACT  (hole plate crushing)")
print("=" * 62)

mesh, material = build_mesh_and_material()
nodes_bot, nodes_top = get_bc_nodes(mesh)

surf = fd.mesh.extract_surface(mesh)
penalty_contact = fd.constraint.SelfContact(
    surf, "linear", search_algorithm="bucket",
)
penalty_contact.contact_search_once = True
penalty_contact.eps_n = 1e6
penalty_contact.max_dist = 1.5

wf = fd.weakform.StressEquilibrium(material, nlgeom="UL")
solid_assembly = fd.Assembly.create(wf, mesh)
assembly = fd.Assembly.sum(solid_assembly, penalty_contact)

pb_penalty = fd.problem.NonLinear(assembly)

if not os.path.isdir("results"):
    os.mkdir("results")
res_penalty = pb_penalty.add_output(
    "results/ring_crush_penalty", solid_assembly, ["Disp", "Stress"],
)

pb_penalty.bc.add("Dirichlet", nodes_bot, "Disp", 0)
pb_penalty.bc.add("Dirichlet", nodes_top, "Disp", [0, -40])
pb_penalty.set_nr_criterion("Displacement", tol=5e-3, max_subiter=5)

# --- Tracking callback ---
history_penalty = {"time": [], "reaction_y": []}


def track_penalty(pb):
    history_penalty["time"].append(pb.time)
    F = pb.get_ext_forces("Disp")
    history_penalty["reaction_y"].append(np.sum(F[1, nodes_top]))


t0_penalty = time()
pb_penalty.nlsolve(
    dt=0.01, tmax=1, update_dt=True, print_info=1,
    callback=track_penalty, exec_callback_at_each_iter=True,
)
penalty_time = time() - t0_penalty
print(f"\nPenalty solve time: {penalty_time:.2f} s")


# =========================================================================
# Approach 2 : IPC contact
# =========================================================================

print("\n" + "=" * 62)
print("  IPC CONTACT  (hole plate crushing)")
print("=" * 62)

mesh2, material2 = build_mesh_and_material()
nodes_bot2, nodes_top2 = get_bc_nodes(mesh2)

ipc_contact = fd.constraint.IPCSelfContact(
    mesh2,
    dhat=5e-3,
    dhat_is_relative=True,
    friction_coefficient=0.0,
    use_ccd=True,
)

wf2 = fd.weakform.StressEquilibrium(material2, nlgeom="UL")
solid_assembly2 = fd.Assembly.create(wf2, mesh2)
assembly2 = fd.Assembly.sum(solid_assembly2, ipc_contact)

pb_ipc = fd.problem.NonLinear(assembly2)

res_ipc = pb_ipc.add_output(
    "results/ring_crush_ipc", solid_assembly2, ["Disp", "Stress"],
)

pb_ipc.bc.add("Dirichlet", nodes_bot2, "Disp", 0)
pb_ipc.bc.add("Dirichlet", nodes_top2, "Disp", [0, -40])
pb_ipc.set_nr_criterion("Displacement", tol=5e-3, max_subiter=5)

# --- Tracking callback ---
history_ipc = {
    "time": [], "reaction_y": [],
    "n_collisions": [], "kappa": [], "min_distance": [],
}


def track_ipc(pb):
    history_ipc["time"].append(pb.time)
    F = pb.get_ext_forces("Disp")
    history_ipc["reaction_y"].append(np.sum(F[1, nodes_top2]))
    history_ipc["n_collisions"].append(len(ipc_contact._collisions))
    history_ipc["kappa"].append(ipc_contact._kappa)
    verts = ipc_contact._get_current_vertices(pb)
    if len(ipc_contact._collisions) > 0:
        min_d = ipc_contact._collisions.compute_minimum_distance(
            ipc_contact._collision_mesh, verts,
        )
    else:
        min_d = float("inf")
    history_ipc["min_distance"].append(min_d)


t0_ipc = time()
pb_ipc.nlsolve(
    dt=0.01, tmax=1, update_dt=True, print_info=1,
    callback=track_ipc, exec_callback_at_each_iter=True,
)
ipc_time = time() - t0_ipc
print(f"\nIPC solve time: {ipc_time:.2f} s")


# =========================================================================
# Comparison summary
# =========================================================================

print("\n")
print("=" * 62)
print("  PERFORMANCE COMPARISON: Penalty vs IPC  (hole plate)")
print("=" * 62)

n_inc_pen = len(history_penalty["time"])
n_inc_ipc = len(history_ipc["time"])

rows = [
    ("Total solve time",
     f"{penalty_time:.2f} s",
     f"{ipc_time:.2f} s"),
    ("Total increments",
     str(n_inc_pen), str(n_inc_ipc)),
    ("Final reaction Fy (top)",
     f"{history_penalty['reaction_y'][-1]:.1f}" if history_penalty["reaction_y"] else "N/A",
     f"{history_ipc['reaction_y'][-1]:.1f}" if history_ipc["reaction_y"] else "N/A"),
    ("IPC min gap distance",
     "N/A",
     f"{min(d for d in history_ipc['min_distance'] if np.isfinite(d)):.2e}"
     if any(np.isfinite(d) for d in history_ipc["min_distance"]) else "inf"),
    ("IPC final barrier kappa",
     "N/A",
     f"{history_ipc['kappa'][-1]:.2e}" if history_ipc["kappa"] else "N/A"),
    ("IPC max active collisions",
     "N/A",
     str(max(history_ipc["n_collisions"])) if history_ipc["n_collisions"] else "N/A"),
]

w_label, w_val = 28, 16
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
print(f"{'Inc':>4} {'Time':>8} {'Reaction Fy':>14}")
for i in range(n_inc_pen):
    print(
        f"{i+1:4d} {history_penalty['time'][i]:8.4f} "
        f"{history_penalty['reaction_y'][i]:14.2f}"
    )

print()
print("-" * 62)
print("  Per-increment detail: IPC")
print("-" * 62)
print(
    f"{'Inc':>4} {'Time':>8} {'Reaction Fy':>14} "
    f"{'Collisions':>11} {'Min gap':>12} {'Kappa':>12}"
)
for i in range(n_inc_ipc):
    min_d = history_ipc["min_distance"][i]
    min_d_str = f"{min_d:.2e}" if np.isfinite(min_d) else "inf"
    print(
        f"{i+1:4d} {history_ipc['time'][i]:8.4f} "
        f"{history_ipc['reaction_y'][i]:14.2f} "
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
    fig.suptitle("Hole Plate Crushing: Penalty vs IPC Comparison", fontsize=14)

    # --- Force-displacement ---
    disp_pen = [t * 40 for t in history_penalty["time"]]
    disp_ipc = [t * 40 for t in history_ipc["time"]]

    ax = axes[0, 0]
    ax.plot(disp_pen, [-f for f in history_penalty["reaction_y"]],
            "o-", label="Penalty", markersize=3)
    ax.plot(disp_ipc, [-f for f in history_ipc["reaction_y"]],
            "s-", label="IPC", markersize=3)
    ax.set_xlabel("Top displacement (mm)")
    ax.set_ylabel("Reaction force Fy (N/mm)")
    ax.set_title("Force-Displacement Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- IPC collisions ---
    ax = axes[0, 1]
    ax.plot(history_ipc["time"], history_ipc["n_collisions"],
            "s-", color="tab:red", markersize=3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Active collisions")
    ax.set_title("IPC Active Collision Count")
    ax.grid(True, alpha=0.3)

    # --- IPC kappa evolution ---
    ax = axes[1, 0]
    ax.plot(history_ipc["time"], history_ipc["kappa"],
            "s-", color="tab:orange", markersize=3)
    ax.set_xlabel("Time")
    ax.set_ylabel("Barrier stiffness kappa")
    ax.set_title("IPC Barrier Stiffness")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # --- IPC min gap ---
    ax = axes[1, 1]
    finite_gaps = [(t, d) for t, d in zip(history_ipc["time"],
                   history_ipc["min_distance"]) if np.isfinite(d)]
    if finite_gaps:
        t_gap, d_gap = zip(*finite_gaps)
        ax.plot(t_gap, d_gap, "s-", color="tab:green", markersize=3,
                label="Min gap")
    if ipc_contact._actual_dhat is not None:
        ax.axhline(y=ipc_contact._actual_dhat, color="blue",
                   linestyle="--", alpha=0.5, label="dhat")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5,
               label="Zero (penetration)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Min gap distance")
    ax.set_title("IPC Minimum Gap Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/ring_crush_comparison.png", dpi=150)
    print("Comparison plot saved to results/ring_crush_comparison.png")
    plt.show()

except ImportError:
    print("matplotlib not available -- skipping plots.")


# =========================================================================
# Final deformed shape with von Mises stress (requires pyvista)
# =========================================================================
# Uncomment the lines below to visualise the final deformed configuration.

# res_penalty.plot("Stress", "vm", "Node", show=True, scale=1, show_nodes=True)
# res_ipc.plot("Stress", "vm", "Node", show=True, scale=1, show_nodes=True)
