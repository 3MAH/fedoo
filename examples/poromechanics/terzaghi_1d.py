"""1D Terzaghi consolidation of a saturated linear-elastic column.

A thin column of saturated linear-elastic porous material is compressed by an
imposed downward displacement of its top surface. The top surface drains
(``PorePressure = 0``), the bottom and lateral surfaces are sealed. Lateral
displacement is prevented to enforce 1D consolidation.

Runs the coupled (u, PorePressure, Pressure) system and plots:
  * the pore pressure profile along the column at several time instants;
  * the time history of pore pressure at the closed (bottom) end.
"""

import numpy as np

import fedoo as fd


# ---------------------- Geometry and material -----------------------
L = 1.0
nx = ny = 2
nz = 20

E = 1.0e6  # Pa
nu = 0.3
biot_M = 1.0e8  # Pa, near-incompressible fluid
k_intrinsic = 1.0e-7
mu_f = 1.0
delta = -1.0e-4  # imposed compressive top displacement (m)

dt = 0.5
nb_steps = 60
snapshot_steps = [0, 1, 3, 7, 15, 30, 59]

# ---------------------- Setup ----------------------------------------
fd.ModelingSpace("3D")

mesh = fd.mesh.box_mesh(
    nx=nx + 1,
    ny=ny + 1,
    nz=nz + 1,
    x_min=0,
    x_max=0.1,
    y_min=0,
    y_max=0.1,
    z_min=0,
    z_max=L,
    elm_type="hex8",
    name="Column",
)

skel = fd.constitutivelaw.ElasticIsotrop(E, nu, name="Skeleton")
fluid = fd.constitutivelaw.PoroFluidProperties(
    permeability=k_intrinsic,
    fluid_viscosity=mu_f,
    biot_coefficient=1.0,
    biot_modulus=biot_M,
    initial_porosity=0.5,
)
K_bulk = E / (3.0 * (1.0 - 2.0 * nu))
wf = fd.weakform.PoroMechanics(
    skel, fluid, bulk_modulus=K_bulk, nlgeom=False, name="Poro"
)
fd.Assembly.create(wf, mesh, name="PoroAssembly")

pb = fd.problem.NonLinear("PoroAssembly")
pb.set_nr_criterion("Displacement", tol=1e-3, max_subiter=10, err0=1.0)

# ---------------------- Boundary conditions --------------------------
top = mesh.find_nodes("Z", L)
bottom = mesh.find_nodes("Z", 0.0)
side_x = np.unique(
    np.concatenate([mesh.find_nodes("X", 0.0), mesh.find_nodes("X", 0.1)])
)
side_y = np.unique(
    np.concatenate([mesh.find_nodes("Y", 0.0), mesh.find_nodes("Y", 0.1)])
)

pb.bc.add("Dirichlet", side_x, "DispX", 0.0)
pb.bc.add("Dirichlet", side_y, "DispY", 0.0)
pb.bc.add("Dirichlet", bottom, "DispZ", 0.0)
# Near-step load: ramp to full over the first time step, then hold, so the
# consolidation transient (undrained pressure rise, then drainage decay) shows.
pb.bc.add(
    "Dirichlet", top, "DispZ", delta, time_func=lambda tf: min(1.0, tf * nb_steps)
)
pb.bc.add("Dirichlet", top, "PorePressure", 0.0)

# ---------------------- Solve ---------------------------------------
pb.initialize()
pb.tmax = dt * nb_steps
pb.dtime = dt
pb.set_start()

# Sample 'column' of nodes along z (use x=0, y=0 nodes if available)
col_mask = (mesh.nodes[:, 0] == 0.0) & (mesh.nodes[:, 1] == 0.0)
col_idx = np.where(col_mask)[0]
col_z = mesh.nodes[col_idx, 2]
sort_idx = np.argsort(col_z)
col_idx = col_idx[sort_idx]
col_z = col_z[sort_idx]

p_profiles = {}
p_bottom_hist = np.zeros(nb_steps)
uz_top_hist = np.zeros(nb_steps)

for step in range(nb_steps):
    pb.time = step * dt
    convergence, nb_nr, res = pb.solve_time_increment()
    if not convergence:
        print(f"Step {step}: not converged (res={res:g})")
    pb.set_start()

    p = pb.get_dof_solution("PorePressure")
    uz = pb.get_dof_solution("DispZ")
    p_bottom_hist[step] = np.mean(p[bottom])
    uz_top_hist[step] = np.mean(uz[top])

    if step in snapshot_steps:
        p_profiles[step] = p[col_idx].copy()

# ---------------------- Plot ----------------------------------------
try:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for step, p_z in p_profiles.items():
        ax.plot(p_z, col_z, label=f"t = {step * dt:.1f} s")
    ax.set_xlabel("PorePressure [Pa]")
    ax.set_ylabel("z [m]")
    ax.set_title("Pore pressure profile during consolidation")
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    t = np.arange(nb_steps) * dt
    ax.plot(t, p_bottom_hist, "b-", label="bottom (z=0)")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("PorePressure [Pa]")
    ax.set_title("Bottom pore pressure history")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    plt.show()
except ImportError:
    print("matplotlib not available, skipping plots.")
    for step, p_z in p_profiles.items():
        print(f"\nt = {step * dt:.1f} s — p(z):")
        for z, p_val in zip(col_z, p_z):
            print(f"  z = {z:.3f} m   p = {p_val:.4g} Pa")
