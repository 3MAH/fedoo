"""
Westergaard sinusoidal contact benchmark: IPC vs penalty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validates both IPC and penalty contact against the **exact** Westergaard
analytical solution for periodic sinusoidal contact (plane strain).

A rigid flat plate is pressed onto an elastic body whose top surface has
a sinusoidal profile z(x) = A*cos(2*pi*x/lam). Periodic BCs in x make the
domain naturally finite — no infinite half-space approximation error.

Analytical formulas (Westergaard 1939, Johnson 1985):
  - Complete contact pressure: p* = pi * E* * A / lam
  - Mean pressure vs contact half-width: p_bar = p* sin^2(pi*a/lam)
  - Mean displacement (from ContactMechanics):
      d(a) = 2A * [-0.5 + 0.5*cos^2(pi*a/lam)
                    + sin^2(pi*a/lam) * ln(sin(pi*a/lam))]

.. note::
   The IPC method requires the ``ipctk`` package:
   ``pip install ipctk``  or  ``pip install fedoo[ipc]``.
"""

import fedoo as fd
import numpy as np
import os
from time import time

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

# =========================================================================
# Parameters
# =========================================================================
lam = 1.0          # wavelength
A = 0.02           # amplitude (2% of wavelength)
H = 2.0            # elastic block height (2*lam)
H_rigid = 0.1      # rigid plate thickness
gap = 0.005        # initial gap between peak and plate

E = 1000.0
nu = 0.3
E_star = E / (1.0 - nu**2)      # plane strain modulus
E_rigid = 1e6                    # quasi-rigid
nu_rigid = 0.3

p_star = np.pi * E_star * A / lam   # complete contact pressure

# Mesh resolution
nx = 80
ny_block = 80
ny_plate = 2

# Prescribed displacement at tmax=1
delta_max = 0.06   # total descent of rigid plate top


# =========================================================================
# Analytical solution (parametric in contact half-width a)
# =========================================================================
def westergaard_analytical(lam, A, gap, E_star, H_block=None):
    """Return (delta_ana, p_bar_ana) arrays parametric in a.

    If H_block is given, adds the finite-block bulk compression correction:
    delta_total = gap - d_mean + p_bar * H / E_star
    (accounts for the laterally-constrained block compliance that the
    semi-infinite half-space solution does not include).
    """
    p_star_val = np.pi * E_star * A / lam
    a = np.linspace(0.001, 0.499, 500) * lam
    s = np.sin(np.pi * a / lam)
    c = np.cos(np.pi * a / lam)
    p_bar = p_star_val * s**2
    # Mean surface displacement (compression into the block)
    d_mean = 2.0 * A * (-0.5 + 0.5 * c**2 + s**2 * np.log(s))
    # Rigid plate descent = gap closure + elastic compression + bulk
    delta = gap - d_mean
    if H_block is not None:
        delta += p_bar * H_block / E_star
    return delta, p_bar, p_star_val


# =========================================================================
# Helpers
# =========================================================================
def add_symmetry_bc(pb, mesh):
    """Apply DispX=0 on left/right faces — exact for the cosine symmetry.

    The cosine surface z = A*cos(2*pi*x/lam) has mirror symmetry about
    x=0 and x=lam/2, so u_x = 0 at x=0 and x=lam.  This is equivalent
    to periodic BC with E_xx=0 for this specific geometry.
    """
    nodes_left = mesh.find_nodes("X", 0)
    nodes_right = mesh.find_nodes("X", lam)
    pb.bc.add("Dirichlet", nodes_left, "DispX", 0)
    pb.bc.add("Dirichlet", nodes_right, "DispX", 0)


def filter_surface_no_lateral(surf_mesh, vol_mesh, tol=1e-8):
    """Remove surface edges on left/right boundaries (x=0 or x=lam).

    This prevents IPC from detecting spurious contacts between vertical
    edges of the stacked bodies.
    """
    nodes = vol_mesh.nodes
    keep = []
    for i, elem in enumerate(surf_mesh.elements):
        xs = nodes[elem, 0]
        # Drop edge if all nodes are on left OR all on right boundary
        if np.all(np.abs(xs) < tol) or np.all(np.abs(xs - lam) < tol):
            continue
        keep.append(i)
    if len(keep) == len(surf_mesh.elements):
        return surf_mesh  # nothing to filter
    from fedoo.core.mesh import Mesh
    new_elements = surf_mesh.elements[keep]
    return Mesh(nodes, new_elements, surf_mesh.elm_type, name="filtered_surface")


# =========================================================================
# Mesh and material builder
# =========================================================================
def build_mesh_and_material():
    """Build two-body mesh (elastic block + rigid plate) with sinusoidal top.

    Returns (mesh, material, nodes_bottom, nodes_top_plate, n_block).
    """
    fd.ModelingSpace("2D")

    # --- Elastic block ---
    mesh_block = fd.mesh.rectangle_mesh(
        nx=nx + 1, ny=ny_block + 1,
        x_min=0, x_max=lam,
        y_min=0, y_max=H,
        elm_type="tri3",
    )
    # Sinusoidal deformation of top surface (blended through height)
    nodes = mesh_block.nodes
    nodes[:, 1] += A * np.cos(2 * np.pi * nodes[:, 0] / lam) * (nodes[:, 1] / H)
    mesh_block.element_sets["block"] = np.arange(mesh_block.n_elements)

    # --- Rigid flat plate ---
    y_plate_bot = H + A + gap
    y_plate_top = y_plate_bot + H_rigid
    mesh_plate = fd.mesh.rectangle_mesh(
        nx=nx + 1, ny=ny_plate + 1,
        x_min=0, x_max=lam,
        y_min=y_plate_bot, y_max=y_plate_top,
        elm_type="tri3",
    )
    mesh_plate.element_sets["plate"] = np.arange(mesh_plate.n_elements)

    # --- Stack ---
    n_block = mesh_block.n_nodes
    mesh = fd.Mesh.stack(mesh_block, mesh_plate)

    # Node sets
    nodes_bottom = mesh.find_nodes("Y", 0)
    nodes_top_plate = mesh.find_nodes("Y", y_plate_top)

    # --- Material (heterogeneous) ---
    mat_block = fd.constitutivelaw.ElasticIsotrop(E, nu)
    mat_plate = fd.constitutivelaw.ElasticIsotrop(E_rigid, nu_rigid)
    material = fd.constitutivelaw.Heterogeneous(
        (mat_block, mat_plate), ("block", "plate"),
    )

    return mesh, material, nodes_bottom, nodes_top_plate, n_block


# =========================================================================
# Approach 1: IPC contact
# =========================================================================
print("=" * 62)
print("  IPC CONTACT  (Westergaard sinusoidal benchmark)")
print("=" * 62)

mesh, material, nodes_bottom, nodes_top_plate, n_block = (
    build_mesh_and_material()
)

# Extract surface and remove left/right edges to avoid spurious contacts
surf_raw = fd.mesh.extract_surface(mesh)
surf_ipc = filter_surface_no_lateral(surf_raw, mesh)

ipc_contact = fd.constraint.IPCContact(
    mesh,
    surface_mesh=surf_ipc,
    dhat=0.003,
    dhat_is_relative=False,
    friction_coefficient=0.0,
    use_ccd=True,
)

wf = fd.weakform.StressEquilibrium(material, nlgeom=False)
solid = fd.Assembly.create(wf, mesh)
assembly = fd.Assembly.sum(solid, ipc_contact)

pb_ipc = fd.problem.NonLinear(assembly)

if not os.path.isdir("results"):
    os.mkdir("results")
pb_ipc.add_output("results/westergaard_ipc", solid, ["Disp", "Stress"])

# Fix bottom of elastic block
pb_ipc.bc.add("Dirichlet", nodes_bottom, "Disp", 0)

# Prescribe vertical displacement on top of rigid plate
pb_ipc.bc.add("Dirichlet", nodes_top_plate, "DispY", -delta_max)

# Symmetry: DispX=0 on left/right faces (exact for cosine geometry)
add_symmetry_bc(pb_ipc, mesh)

pb_ipc.set_nr_criterion("Displacement", tol=5e-3, max_subiter=15)

# --- Tracking ---
history_ipc = {"time": [], "reaction_y": [], "n_collisions": [],
               "kappa": [], "min_distance": []}


def track_ipc(pb):
    history_ipc["time"].append(pb.time)
    F = pb.get_ext_forces("Disp")
    history_ipc["reaction_y"].append(np.sum(F[1, nodes_top_plate]))
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


t0 = time()
pb_ipc.nlsolve(
    dt=0.05, tmax=1, update_dt=True, print_info=1, interval_output=0.2,
    callback=track_ipc, exec_callback_at_each_iter=True,
)
ipc_time = time() - t0
print(f"\nIPC solve time: {ipc_time:.2f} s")


# =========================================================================
# Approach 2: Penalty contact
# =========================================================================
print("\n" + "=" * 62)
print("  PENALTY CONTACT  (Westergaard sinusoidal benchmark)")
print("=" * 62)

mesh2, material2, nodes_bottom2, nodes_top_plate2, n_block2 = (
    build_mesh_and_material()
)

# Slave nodes: top of elastic block (sinusoidal surface)
surf_block = fd.mesh.extract_surface(mesh2.extract_elements("block"))
block_boundary = set(np.unique(surf_block.elements).tolist())
nodes_block_top = np.array([
    n for n in block_boundary
    if mesh2.nodes[n, 1] > H * 0.9  # top ~10% of block
])

# Master surface: bottom of rigid plate
surf_plate = fd.mesh.extract_surface(mesh2.extract_elements("plate"))

penalty_contact = fd.constraint.Contact(
    nodes_block_top, surf_plate, search_algorithm="bucket",
)
penalty_contact.contact_search_once = False
penalty_contact.eps_n = 1e5
penalty_contact.max_dist = gap + A + 0.01

wf2 = fd.weakform.StressEquilibrium(material2, nlgeom=False)
solid2 = fd.Assembly.create(wf2, mesh2)
assembly2 = fd.Assembly.sum(solid2, penalty_contact)

pb_penalty = fd.problem.NonLinear(assembly2)
pb_penalty.add_output("results/westergaard_penalty", solid2, ["Disp", "Stress"])

# Fix bottom of elastic block
pb_penalty.bc.add("Dirichlet", nodes_bottom2, "Disp", 0)

# Prescribe vertical displacement on top of rigid plate
pb_penalty.bc.add("Dirichlet", nodes_top_plate2, "DispY", -delta_max)

# Symmetry: DispX=0 on left/right faces
add_symmetry_bc(pb_penalty, mesh2)

pb_penalty.set_nr_criterion("Displacement", tol=5e-3, max_subiter=15)

# --- Tracking ---
history_penalty = {"time": [], "reaction_y": []}


def track_penalty(pb):
    history_penalty["time"].append(pb.time)
    F = pb.get_ext_forces("Disp")
    history_penalty["reaction_y"].append(np.sum(F[1, nodes_top_plate2]))


t0 = time()
pb_penalty.nlsolve(
    dt=0.05, tmax=1, update_dt=True, print_info=1, interval_output=0.2,
    callback=track_penalty, exec_callback_at_each_iter=True,
)
penalty_time = time() - t0
print(f"\nPenalty solve time: {penalty_time:.2f} s")


# =========================================================================
# Comparison summary
# =========================================================================
print("\n")
print("=" * 62)
print("  PERFORMANCE COMPARISON: IPC vs Penalty")
print("=" * 62)

rows = [
    ("Total solve time",
     f"{ipc_time:.2f} s",
     f"{penalty_time:.2f} s"),
    ("Total increments",
     str(len(history_ipc["time"])),
     str(len(history_penalty["time"]))),
    ("Final reaction Fy",
     f"{history_ipc['reaction_y'][-1]:.4f}" if history_ipc["reaction_y"] else "N/A",
     f"{history_penalty['reaction_y'][-1]:.4f}" if history_penalty["reaction_y"] else "N/A"),
    ("p* (analytical)",
     f"{p_star:.4f}", f"{p_star:.4f}"),
    ("IPC min gap distance",
     f"{min(d for d in history_ipc['min_distance'] if np.isfinite(d)):.2e}"
     if any(np.isfinite(d) for d in history_ipc["min_distance"]) else "inf",
     "N/A"),
    ("IPC final kappa",
     f"{history_ipc['kappa'][-1]:.2e}" if history_ipc["kappa"] else "N/A",
     "N/A"),
]

w_label, w_val = 28, 18
print(f"{'Metric':<{w_label}} {'IPC':>{w_val}} {'Penalty':>{w_val}}")
print("-" * (w_label + 2 * w_val + 2))
for label, v_ipc, v_pen in rows:
    print(f"{label:<{w_label}} {v_ipc:>{w_val}} {v_pen:>{w_val}}")
print()


# =========================================================================
# Per-increment detail: IPC
# =========================================================================
print("-" * 62)
print("  Per-increment detail: IPC")
print("-" * 62)
print(
    f"{'Inc':>4} {'Time':>8} {'Reaction Fy':>14} "
    f"{'Collisions':>11} {'Min gap':>12} {'Kappa':>12}"
)
for i in range(len(history_ipc["time"])):
    min_d = history_ipc["min_distance"][i]
    min_d_str = f"{min_d:.2e}" if np.isfinite(min_d) else "inf"
    print(
        f"{i+1:4d} {history_ipc['time'][i]:8.4f} "
        f"{history_ipc['reaction_y'][i]:14.4f} "
        f"{history_ipc['n_collisions'][i]:11d} "
        f"{min_d_str:>12} "
        f"{history_ipc['kappa'][i]:12.2e}"
    )
print()


# =========================================================================
# Per-increment detail: Penalty
# =========================================================================
print("-" * 62)
print("  Per-increment detail: PENALTY")
print("-" * 62)
print(f"{'Inc':>4} {'Time':>8} {'Reaction Fy':>14}")
for i in range(len(history_penalty["time"])):
    print(
        f"{i+1:4d} {history_penalty['time'][i]:8.4f} "
        f"{history_penalty['reaction_y'][i]:14.4f}"
    )
print()


# =========================================================================
# Plots: numerical vs analytical
# =========================================================================
try:
    import matplotlib.pyplot as plt

    delta_ana, p_bar_ana, _ = westergaard_analytical(lam, A, gap, E_star, H_block=H)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Westergaard Sinusoidal Contact Benchmark", fontsize=14)

    # --- Normalized pressure vs displacement ---
    ax = axes[0]
    ax.plot(delta_ana / A, p_bar_ana / p_star, "k-", linewidth=2,
            label="Analytical (corrected)")

    # IPC results
    if history_ipc["reaction_y"]:
        delta_ipc = np.array(history_ipc["time"]) * delta_max
        # Mean pressure = -reaction_y / lam (reaction is negative for compression)
        p_bar_ipc = -np.array(history_ipc["reaction_y"]) / lam
        ax.plot(delta_ipc / A, p_bar_ipc / p_star, "s-",
                color="tab:blue", markersize=4, label="IPC")

    # Penalty results
    if history_penalty["reaction_y"]:
        delta_pen = np.array(history_penalty["time"]) * delta_max
        p_bar_pen = -np.array(history_penalty["reaction_y"]) / lam
        ax.plot(delta_pen / A, p_bar_pen / p_star, "o-",
                color="tab:orange", markersize=4, label="Penalty")

    ax.set_xlabel(r"$\delta / A$")
    ax.set_ylabel(r"$\bar{p} / p^*$")
    ax.set_title("Mean Pressure vs Displacement (normalized)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Clip to numerical data range
    max_delta_A = delta_max / A * 1.1
    ax.set_xlim(0, max_delta_A)
    ax.set_ylim(bottom=0)

    # --- IPC-specific: min gap and kappa ---
    ax = axes[1]
    finite_gaps = [(t, d) for t, d in zip(history_ipc["time"],
                   history_ipc["min_distance"]) if np.isfinite(d)]
    if finite_gaps:
        t_gap, d_gap = zip(*finite_gaps)
        ax.plot(t_gap, d_gap, "s-", color="tab:green", markersize=3,
                label="Min gap")
    if hasattr(ipc_contact, "_actual_dhat") and ipc_contact._actual_dhat is not None:
        ax.axhline(y=ipc_contact._actual_dhat, color="blue",
                   linestyle="--", alpha=0.5, label=f"dhat = {ipc_contact._actual_dhat:.4f}")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5,
               label="Zero (penetration)")
    ax.set_xlabel("Time")
    ax.set_ylabel("Min gap distance")
    ax.set_title("IPC Minimum Gap Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("results/westergaard_benchmark.png", dpi=150)
    print("Benchmark plot saved to results/westergaard_benchmark.png")
    plt.show()

except ImportError:
    print("matplotlib not available -- skipping plots.")


# =========================================================================
# Post-processing (requires pyvista)
# =========================================================================
# Uncomment the lines below to visualise the final deformed configuration.
# res_ipc.plot("Stress", "vm", "Node", show=True, scale=1, show_nodes=True)
# res_penalty.plot("Stress", "vm", "Node", show=True, scale=1, show_nodes=True)
