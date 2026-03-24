"""Single Edge Notch Tension (SENT) — Phase-Field Fracture Benchmark.

This example reproduces the classic SENT benchmark from Miehe et al. (2010).

A square plate [0,1]x[0,1] mm has a horizontal notch from the left edge to
the center at y = 0.5. Vertical displacement is applied at the top while
the bottom is fixed. The crack propagates horizontally from the notch tip.

This uses the staggered (alternate minimization) scheme:
  1. Solve damage sub-problem (reaction-diffusion, linear)
  2. Solve displacement sub-problem (elasticity with degraded stiffness, linear)
  3. Check convergence and iterate

Each sub-problem is a *separate* fedoo Problem with its own ModelingSpace.
"""

import fedoo as fd
import numpy as np
import os

# ======================== Parameters ========================
E = 210e3  # Young's modulus (MPa)
nu = 0.3  # Poisson's ratio
Gc = 2.7e-3  # Critical energy release rate (MPa.mm)
l0 = 0.015  # Regularization length (mm) — should be ~2x element size

# Mesh
Lx, Ly = 1.0, 1.0  # Plate dimensions (mm)
nx, ny = 101, 101  # Number of nodes
notch_y = 0.5  # Notch y-position
notch_length = 0.5  # Notch extends from x=0 to x=notch_length

# Loading
u_max = 7e-3  # Maximum applied displacement (mm)
n_steps = 200  # Number of load steps

# Stagger
max_stagger = 100  # Max stagger iterations per step
stagger_tol = 1e-4  # Convergence tolerance on damage increment

# Phase-field model
model = "AT2"  # 'AT1' or 'AT2'
split = "miehe"  # 'bourdin', 'amor', or 'miehe'

# Output
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)


# ======================== Mesh ========================
mesh = fd.mesh.rectangle_mesh(
    nx=nx,
    ny=ny,
    x_min=0,
    x_max=Lx,
    y_min=0,
    y_max=Ly,
    elm_type="quad4",
    name="Domain",
)

crd = mesh.nodes
n_nodes = mesh.n_nodes

# Find node sets for boundary conditions
bottom = mesh.find_nodes("Y", 0.0)
top = mesh.find_nodes("Y", Ly)
left = mesh.find_nodes("X", 0.0)

# Find notch nodes: y ≈ notch_y and x <= notch_length
tol_notch = Ly / (ny - 1) / 4  # quarter of element size
notch_nodes = np.where(
    (np.abs(crd[:, 1] - notch_y) < tol_notch) & (crd[:, 0] <= notch_length + 1e-10)
)[0]


# ======================== Mechanical Problem ========================
mech_space = fd.ModelingSpace("2Dplane")

# Phase-field constitutive law (wraps elastic law)
elastic_cl = fd.constitutivelaw.ElasticIsotrop(E, nu, name="Elastic")
mat = fd.constitutivelaw.PhaseFieldDamage(
    elastic_cl, Gc=Gc, l0=l0, split=split, model=model, name="PFDamage"
)

wf_mech = fd.weakform.StressEquilibrium(mat, space=mech_space)
assembly_mech = fd.Assembly.create(wf_mech, "Domain", name="Assembly_Mech")

pb_mech = fd.problem.Linear("Assembly_Mech")

# Mechanical BCs
pb_mech.bc.add("Dirichlet", bottom, "DispY", 0)
pb_mech.bc.add("Dirichlet", bottom, "DispX", 0)
pb_mech.bc.add("Dirichlet", top, "DispY", 0)  # will be updated each step


# ======================== Damage Problem ========================
dam_space = fd.ModelingSpace("2Dplane")

wf_dam = fd.weakform.PhaseFieldEvolution(mat, space=dam_space)
assembly_dam = fd.Assembly.create(wf_dam, "Domain", name="Assembly_Dam")

pb_dam = fd.problem.Linear("Assembly_Dam")

# Initial damage at notch nodes = 1
d_init = np.zeros(n_nodes)
d_init[notch_nodes] = 1.0


# ======================== Staggered Solution ========================
print(f"Phase-field fracture: SENT benchmark")
print(f"Model: {model}, Split: {split}")
print(f"Mesh: {nx}x{ny} = {n_nodes} nodes")
print(f"l0 = {l0}, Gc = {Gc}, E = {E}, nu = {nu}")
print(f"Steps: {n_steps}, u_max: {u_max}")
print("=" * 60)

# Storage for load-displacement curve
disp_history = []
force_history = []

d_nodal = d_init.copy()

for step in range(n_steps):
    u_applied = u_max * (step + 1) / n_steps

    # Update displacement BC on top
    pb_mech.bc[-1].value = u_applied

    d_prev = d_nodal.copy()

    for stagger_iter in range(max_stagger):
        # ----- 1. Pass damage to mechanical problem -----
        d_gp = assembly_mech.convert_data(
            d_nodal, convert_from="Node", convert_to="GaussPoint"
        )
        assembly_mech.sv["Damage"] = d_gp

        # ----- 2. Solve mechanical problem -----
        # update() recomputes stress/tangent with current damage and reassembles
        pb_mech.update()
        pb_mech.solve()

        # ----- 3. Solve damage problem -----
        # The constitutive law update (called inside pb_mech.update/solve)
        # has already computed H (history variable) and stored it.
        pb_dam.update()
        pb_dam.solve()

        # ----- 4. Extract damage and enforce irreversibility -----
        d_new = pb_dam.get_dof_solution("Damage")
        if not np.isscalar(d_new):
            d_new = np.clip(d_new, d_nodal, 1.0)  # irreversibility
        d_nodal = d_new

        # ----- 5. Check stagger convergence -----
        d_change = np.max(np.abs(d_nodal - d_prev))
        if d_change < stagger_tol:
            break
        d_prev = d_nodal.copy()

    # Update history variable (lock for next step)
    mat.set_start(assembly_mech, pb_mech)

    # Compute reaction force on top boundary
    # (sum of internal forces at top nodes in Y direction)
    A = assembly_mech.get_global_matrix()
    X = pb_mech.get_dof_solution()
    if not np.isscalar(X):
        F_int = A @ X
        rank_y = mech_space.variable_rank("DispY")
        force_top = np.sum(F_int[rank_y * n_nodes + top])
    else:
        force_top = 0.0

    disp_history.append(u_applied)
    force_history.append(force_top)

    d_max = np.max(d_nodal)
    print(
        f"Step {step+1:4d}/{n_steps} | u = {u_applied:.5f} | "
        f"stagger iters: {stagger_iter+1:3d} | "
        f"max(d) = {d_max:.4f} | F_top = {force_top:.4f}"
    )


# ======================== Post-Processing ========================
print("\nSaving results...")

# Save load-displacement curve
np.savetxt(
    os.path.join(output_dir, "load_displacement.csv"),
    np.column_stack([disp_history, force_history]),
    header="displacement,force",
    delimiter=",",
    comments="",
)

# Try to plot if matplotlib is available
try:
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Load-displacement curve
    ax1.plot(disp_history, force_history, "b-", linewidth=1.5)
    ax1.set_xlabel("Applied displacement (mm)")
    ax1.set_ylabel("Reaction force (N/mm)")
    ax1.set_title("Load-Displacement Curve")
    ax1.grid(True)

    # Damage field
    x = crd[:, 0]
    y = crd[:, 1]
    sc = ax2.tricontourf(x, y, d_nodal, levels=50, cmap="hot_r")
    plt.colorbar(sc, ax=ax2, label="Damage d")
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    ax2.set_title("Damage Field")
    ax2.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sent_results.png"), dpi=150)
    plt.show()
    print("Plot saved to results/sent_results.png")

except ImportError:
    print("matplotlib not available — skipping plot")

print("Done!")
