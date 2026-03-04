"""
Hertz axisymmetric contact benchmark: rigid sphere on elastic half-space
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Validates penalty contact in fedoo's ``2Daxi`` modeling space against the
classical Hertz (1882) analytical solution for a rigid sphere pressed
onto an elastic half-space.

The 2D axisymmetric formulation makes the 3D problem computationally
cheap (2D mesh on the r-z plane) while producing the full 3D result
including the ``2*pi*r`` circumferential integration.

**Two-body setup (r-z plane):**
  - Elastic substrate (half-space approximation): large rectangle
  - Quasi-rigid parabolic indenter: thin body with bottom surface
    following z(r) = H_sub + gap + r^2/(2R)

Analytical solution (rigid indenter on elastic body):
  - Effective modulus: ``E* = E / (1 - nu^2)``
  - Force-displacement: ``F = 4/3 * E* * sqrt(R) * delta^(3/2)``
  - Contact radius: ``a = sqrt(R * delta)``
  - Max pressure: ``p0 = 2*E*/(pi) * sqrt(delta/R)``

.. note::
   This benchmark is penalty-only.  IPCContact does not support ``2Daxi``
   (no radial ``2*pi*r`` weighting in the IPC barrier formulation).
"""

import fedoo as fd
import numpy as np
import os
from time import time

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

# =========================================================================
# Parameters
# =========================================================================
R = 10.0  # sphere radius
E = 1000.0  # elastic modulus (substrate)
nu = 0.3  # Poisson's ratio
E_star = E / (1.0 - nu**2)  # effective modulus (rigid indenter)
E_rigid = 1e6  # quasi-rigid indenter
nu_rigid = 0.3

gap = 0.05  # initial gap at r=0 between substrate top and indenter
delta_max = 0.5  # max indentation at tmax=1

a_max = np.sqrt(R * delta_max)  # max contact radius ~ 2.236

R_domain = 25.0  # substrate radial extent (~11 * a_max)
H_sub = 25.0  # substrate height
R_indenter = 5.0  # indenter radial extent (~2.2 * a_max)
H_indenter = 2.0  # indenter thickness

# Mesh resolution (number of nodes along each axis)
nx_sub = 101
ny_sub = 101
nx_ind = 51
ny_ind = 5


# =========================================================================
# Analytical solution
# =========================================================================
def hertz_analytical(R, E_star, delta):
    """Hertz force-displacement for rigid sphere on elastic half-space."""
    delta = np.asarray(delta, dtype=float)
    d_pos = np.maximum(delta, 0.0)
    F = 4.0 / 3.0 * E_star * np.sqrt(R) * d_pos**1.5
    a = np.sqrt(R * d_pos)
    p0 = np.where(delta > 0, 2.0 * E_star / np.pi * np.sqrt(d_pos / R), 0.0)
    return F, a, p0


# =========================================================================
# Mesh
# =========================================================================
fd.ModelingSpace("2Daxi")

# --- Substrate (elastic half-space approximation) ---
mesh_sub = fd.mesh.rectangle_mesh(
    nx=nx_sub,
    ny=ny_sub,
    x_min=0,
    x_max=1,
    y_min=0,
    y_max=1,
    elm_type="quad4",
)
# Biased node placement: dense near r=0 and z=H_sub (contact zone)
nodes = mesh_sub.nodes
nodes[:, 0] = R_domain * nodes[:, 0] ** 1.5  # radial grading
nodes[:, 1] = H_sub * (1.0 - (1.0 - nodes[:, 1]) ** 1.5)  # axial: dense at top
mesh_sub.element_sets["substrate"] = np.arange(mesh_sub.n_elements)

# --- Indenter (quasi-rigid parabolic body) ---
mesh_ind = fd.mesh.rectangle_mesh(
    nx=nx_ind,
    ny=ny_ind,
    x_min=0,
    x_max=1,
    y_min=0,
    y_max=1,
    elm_type="quad4",
)
ind_nodes = mesh_ind.nodes
r_frac = ind_nodes[:, 0].copy()
z_frac = ind_nodes[:, 1].copy()

# Radial grading (dense near r=0)
r_ind = R_indenter * r_frac**1.5

# Bottom surface follows parabola: z(r) = H_sub + gap + r^2/(2R)
# Top surface: z = z_bottom + H_indenter
z_bottom = H_sub + gap + r_ind**2 / (2.0 * R)
ind_nodes[:, 0] = r_ind
ind_nodes[:, 1] = z_bottom + z_frac * H_indenter

mesh_ind.element_sets["indenter"] = np.arange(mesh_ind.n_elements)

# Store indenter bottom node indices before stacking
indenter_bottom_local = list(mesh_ind.node_sets["bottom"])

# --- Stack meshes ---
n_sub = mesh_sub.n_nodes
mesh = fd.Mesh.stack(mesh_sub, mesh_ind)

# Indenter bottom/top nodes in global numbering
indenter_bottom_global = set(n + n_sub for n in indenter_bottom_local)
indenter_top_local = list(mesh_ind.node_sets["top"])
nodes_ind_top = np.array([n + n_sub for n in indenter_top_local])


# =========================================================================
# Contact setup
# =========================================================================
# Slave nodes: top surface of substrate
nodes_sub_top = mesh.find_nodes("Y", H_sub)
nodes_sub_top = nodes_sub_top[nodes_sub_top < n_sub]  # only substrate nodes

# Master surface: bottom of indenter only (filter out top/side edges)
surf_ind_full = fd.mesh.extract_surface(mesh.extract_elements("indenter"))
keep = [
    i
    for i, elem in enumerate(surf_ind_full.elements)
    if all(int(n) in indenter_bottom_global for n in elem)
]
from fedoo.core.mesh import Mesh as _Mesh

surf_indenter = _Mesh(
    mesh.nodes,
    surf_ind_full.elements[keep],
    surf_ind_full.elm_type,
    name="indenter_bottom",
)

print(f"Substrate top nodes:  {len(nodes_sub_top)}")
print(f"Indenter bottom edges: {len(keep)}")
print(f"a_max = {a_max:.3f},  E* = {E_star:.1f},  gap = {gap}")

contact = fd.constraint.Contact(
    nodes_sub_top,
    surf_indenter,
    search_algorithm="bucket",
)
contact.contact_search_once = False
contact.eps_n = 1e5  # ~100*E, penalty stiffness
contact.max_dist = gap + delta_max + 0.5


# =========================================================================
# Material
# =========================================================================
mat_sub = fd.constitutivelaw.ElasticIsotrop(E, nu)
mat_ind = fd.constitutivelaw.ElasticIsotrop(E_rigid, nu_rigid)
material = fd.constitutivelaw.Heterogeneous(
    (mat_sub, mat_ind),
    ("substrate", "indenter"),
)


# =========================================================================
# Assembly and problem
# =========================================================================
wf = fd.weakform.StressEquilibrium(material, nlgeom=False)
solid = fd.Assembly.create(wf, mesh)
assembly = fd.Assembly.sum(solid, contact)

pb = fd.problem.NonLinear(assembly)

if not os.path.isdir("results"):
    os.mkdir("results")
pb.add_output("results/hertz_axi", solid, ["Disp", "Stress"])


# =========================================================================
# Boundary conditions
# =========================================================================
# Bottom of substrate: fix axial (Y), free radial
nodes_bottom = mesh.find_nodes("Y", 0)
pb.bc.add("Dirichlet", nodes_bottom, "DispY", 0)

# Symmetry axis (r=0): fix radial displacement
nodes_axis = mesh.find_nodes("X", 0)
pb.bc.add("Dirichlet", nodes_axis, "DispX", 0)

# Top of indenter: prescribe downward displacement
pb.bc.add("Dirichlet", nodes_ind_top, "DispY", -delta_max)


# =========================================================================
# Solver
# =========================================================================
pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=15)

# --- Tracking ---
r_top = mesh.nodes[nodes_sub_top, 0]
history = {"time": [], "reaction_y": [], "a_num": []}


def track(pb):
    history["time"].append(pb.time)
    F = pb.get_ext_forces("Disp")
    history["reaction_y"].append(np.sum(F[1, nodes_ind_top]))

    # Estimate contact radius from displacement profile.
    # In Hertz theory u_z(a) = delta/2, so the contact edge is where
    # the displacement drops below half the center value.
    disp = pb.get_disp()
    uy = disp[1, nodes_sub_top]
    uy_center = np.min(uy)  # most negative = max compression
    if uy_center < -1e-10:
        in_contact = uy < 0.5 * uy_center
        a = np.max(r_top[in_contact]) if np.any(in_contact) else 0.0
    else:
        a = 0.0
    history["a_num"].append(a)


print("\n" + "=" * 62)
print("  HERTZ AXISYMMETRIC CONTACT BENCHMARK (Penalty)")
print("=" * 62)

t0 = time()
pb.nlsolve(
    dt=0.05,
    tmax=1,
    update_dt=True,
    print_info=1,
    interval_output=0.2,
    callback=track,
    exec_callback_at_each_iter=True,
)
solve_time = time() - t0
print(f"\nSolve time: {solve_time:.2f} s")


# =========================================================================
# Comparison with analytical solution
# =========================================================================
t_num = np.array(history["time"])
F_num = -np.array(history["reaction_y"])  # reaction is negative (compression)
delta_num = delta_max * t_num - gap  # effective indentation

# Analytical curve
delta_ana = np.linspace(0, delta_max - gap, 200)
F_ana, a_ana, p0_ana = hertz_analytical(R, E_star, delta_ana)

# Numerical at final step
delta_final = delta_num[-1]
F_final = F_num[-1]
F_ana_final = hertz_analytical(R, E_star, delta_final)[0]

print("\n" + "-" * 62)
print("  RESULTS SUMMARY")
print("-" * 62)
print(f"{'Quantity':<30} {'Numerical':>15} {'Analytical':>15}")
print("-" * 62)
print(f"{'Final indentation':<30} {delta_final:15.4f} {delta_max - gap:15.4f}")
print(f"{'Final force F':<30} {F_final:15.2f} {float(F_ana_final):15.2f}")
err_pct = 100 * (F_final - float(F_ana_final)) / float(F_ana_final)
print(f"{'F error':<30} {err_pct:14.2f}%")

# Contact radius at final step
a_num_arr = np.array(history["a_num"])
a_num_final = a_num_arr[-1]
a_ana_val = np.sqrt(R * max(delta_final, 0))
print(f"{'Contact radius a':<30} {a_num_final:15.4f} {a_ana_val:15.4f}")
if a_ana_val > 0:
    a_err = 100 * (a_num_final - a_ana_val) / a_ana_val
    print(f"{'a error':<30} {a_err:14.2f}%")
print()


# =========================================================================
# Per-increment detail
# =========================================================================
print("-" * 62)
print("  Per-increment detail")
print("-" * 62)
print(
    f"{'Inc':>4} {'Time':>8} {'Delta':>10} {'F_num':>12} {'F_ana':>12} {'Err%':>8} {'a_num':>8} {'a_ana':>8}"
)
for i in range(len(history["time"])):
    d = delta_num[i]
    f_n = F_num[i]
    a_n = a_num_arr[i]
    if d > 0:
        f_a = 4.0 / 3.0 * E_star * np.sqrt(R) * d**1.5
        err = 100 * (f_n - f_a) / f_a if f_a > 1e-10 else 0
        a_a = np.sqrt(R * d)
    else:
        f_a = 0.0
        err = 0.0
        a_a = 0.0
    print(
        f"{i+1:4d} {t_num[i]:8.4f} {d:10.4f} {f_n:12.2f} {f_a:12.2f} {err:8.2f} {a_n:8.4f} {a_a:8.4f}"
    )
print()


# =========================================================================
# Plots
# =========================================================================
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Hertz Axisymmetric Contact Benchmark (Penalty)", fontsize=14)

    # --- Force vs indentation ---
    ax = axes[0]
    ax.plot(delta_ana, F_ana, "k-", linewidth=2, label="Hertz analytical")
    mask = delta_num > 0
    if np.any(mask):
        ax.plot(
            delta_num[mask],
            F_num[mask],
            "o-",
            color="tab:blue",
            markersize=4,
            label="FEM (penalty)",
        )
    ax.set_xlabel(r"Indentation $\delta$")
    ax.set_ylabel(r"Contact force $F$")
    ax.set_title(r"Force vs Indentation: $F = \frac{4}{3} E^* \sqrt{R}\, \delta^{3/2}$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # --- Contact radius vs indentation ---
    ax = axes[1]
    ax.plot(
        delta_ana, a_ana, "k-", linewidth=2, label=r"Analytical $a = \sqrt{R\,\delta}$"
    )
    # Plot FEM contact radius at each increment (only where delta > 0)
    mask_a = delta_num > 0
    if np.any(mask_a):
        ax.plot(
            delta_num[mask_a],
            a_num_arr[mask_a],
            "o-",
            color="tab:blue",
            markersize=4,
            label="FEM (penalty)",
        )
    ax.set_xlabel(r"Indentation $\delta$")
    ax.set_ylabel(r"Contact radius $a$")
    ax.set_title(r"Contact Radius: $a = \sqrt{R\,\delta}$")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig("results/hertz_axisymmetric_benchmark.png", dpi=150)
    print("Benchmark plot saved to results/hertz_axisymmetric_benchmark.png")
    plt.show()

except ImportError:
    print("matplotlib not available -- skipping plots.")
