"""
CCD vs OGC line-search comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runs the 2D disk indentation benchmark twice — once with CCD
(Continuous Collision Detection) scalar line search and once with
OGC (Offset Geometric Contact) per-vertex trust-region filtering —
then compares convergence behaviour and accuracy.

Metrics compared:

  - **NR iterations per converged increment** (fewer is better)
  - **Total wall-clock solve time**
  - **Force–indentation curve** (both should agree with each other
    and with the Hertz analytical prediction)

.. note::
   Requires ``ipctk``, ``gmsh``, and ``matplotlib``.
"""

import fedoo as fd
import numpy as np
import os
import tempfile
from time import time

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

# =========================================================================
# Parameters (shared by both runs)
# =========================================================================

E_plate = 1e3
E_disk = 1e5
nu = 0.3
R = 5.0
plate_half = 30.0
plate_h = 40.0
gap = 0.1
imposed_disp = -2.0
dhat_abs = 0.05

DT = 0.05
TMAX = 1.0
TOL = 5e-3
MAX_SUBITER = 8


# =========================================================================
# Helper: build the full problem (mesh + material + BCs)
# =========================================================================

def build_problem(method):
    """Build and return (pb, solid, nodes_disk_top, history).

    Parameters
    ----------
    method : str
        ``"ccd"`` or ``"ogc"``.
    """
    fd.ModelingSpace("2D")

    # --- Plate mesh (gmsh) ---
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 1)

    plate_tag = gmsh.model.occ.addRectangle(
        -plate_half, 0, 0, 2 * plate_half, plate_h,
    )
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(2, [plate_tag], tag=1, name="plate")

    gmsh.model.mesh.field.add("Box", 1)
    gmsh.model.mesh.field.setNumber(1, "VIn", 0.3)
    gmsh.model.mesh.field.setNumber(1, "VOut", 3.0)
    gmsh.model.mesh.field.setNumber(1, "XMin", -6)
    gmsh.model.mesh.field.setNumber(1, "XMax", 6)
    gmsh.model.mesh.field.setNumber(1, "YMin", plate_h - 4)
    gmsh.model.mesh.field.setNumber(1, "YMax", plate_h)
    gmsh.model.mesh.field.setNumber(1, "Thickness", 4)
    gmsh.model.mesh.field.setAsBackgroundMesh(1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

    gmsh.model.mesh.generate(2)
    plate_msh = os.path.join(tempfile.gettempdir(), "plate_cmp.msh")
    gmsh.write(plate_msh)
    gmsh.finalize()

    mesh_plate = fd.mesh.import_msh(plate_msh, mesh_type="surface")
    if mesh_plate.nodes.shape[1] == 3:
        mesh_plate.nodes = mesh_plate.nodes[:, :2]
    mesh_plate.element_sets["plate"] = np.arange(mesh_plate.n_elements)

    # --- Disk mesh ---
    mesh_disk = fd.mesh.disk_mesh(radius=R, nr=12, nt=24, elm_type="tri3")
    mesh_disk.nodes += np.array([0, plate_h + R + gap])
    mesh_disk.element_sets["disk"] = np.arange(mesh_disk.n_elements)

    mesh = fd.Mesh.stack(mesh_plate, mesh_disk)

    # --- IPC contact ---
    surf = fd.mesh.extract_surface(mesh)
    ipc_contact = fd.constraint.IPCContact(
        mesh,
        surface_mesh=surf,
        dhat=dhat_abs,
        dhat_is_relative=False,
        use_ccd=(method == "ccd"),
        use_ogc=(method == "ogc"),
    )

    # --- Material & assembly ---
    mat = fd.constitutivelaw.Heterogeneous(
        (fd.constitutivelaw.ElasticIsotrop(E_plate, nu),
         fd.constitutivelaw.ElasticIsotrop(E_disk, nu)),
        ("plate", "disk"),
    )
    wf = fd.weakform.StressEquilibrium(mat, nlgeom=False)
    solid = fd.Assembly.create(wf, mesh)
    assembly = fd.Assembly.sum(solid, ipc_contact)

    pb = fd.problem.NonLinear(assembly)

    nodes_bottom = mesh.find_nodes("Y", 0)
    nodes_disk_top = mesh.find_nodes("Y", mesh.bounding_box.ymax)

    pb.bc.add("Dirichlet", nodes_bottom, "Disp", 0)
    pb.bc.add("Dirichlet", nodes_disk_top, "Disp", [0, imposed_disp])
    pb.set_nr_criterion("Displacement", tol=TOL, max_subiter=MAX_SUBITER)

    history = {"time": [], "reaction_y": [], "nr_iters": []}

    return pb, solid, nodes_disk_top, history


# =========================================================================
# Instrumented solve: capture NR iteration count per increment
# =========================================================================

def solve_instrumented(pb, solid, nodes_disk_top, history, label):
    """Run nlsolve while tracking per-increment NR iterations."""

    if not os.path.isdir("results"):
        os.mkdir("results")

    # Monkey-patch solve_time_increment to capture NR iteration counts
    _original = pb.solve_time_increment

    def _patched(max_subiter=None, tol_nr=None):
        convergence, nr_iter, err = _original(max_subiter, tol_nr)
        if convergence:
            history["nr_iters"].append(nr_iter)
        return convergence, nr_iter, err

    pb.solve_time_increment = _patched

    def track(pb):
        history["time"].append(pb.time)
        F = pb.get_ext_forces("Disp")
        history["reaction_y"].append(np.sum(F[1, nodes_disk_top]))

    print("=" * 60)
    print(f"2D DISK INDENTATION -- {label}")
    print("=" * 60)
    t0 = time()
    pb.nlsolve(
        dt=DT, tmax=TMAX, update_dt=True, print_info=1, callback=track,
    )
    wall_time = time() - t0
    print(f"{label} solve time: {wall_time:.2f} s")

    # Restore original
    pb.solve_time_increment = _original

    return wall_time


# =========================================================================
# Run both methods
# =========================================================================

pb_ccd, solid_ccd, ndt_ccd, hist_ccd = build_problem("ccd")
wt_ccd = solve_instrumented(pb_ccd, solid_ccd, ndt_ccd, hist_ccd, "CCD")

pb_ogc, solid_ogc, ndt_ogc, hist_ogc = build_problem("ogc")
wt_ogc = solve_instrumented(pb_ogc, solid_ogc, ndt_ogc, hist_ogc, "OGC")


# =========================================================================
# Summary table
# =========================================================================

def summary(label, hist, wt):
    iters = np.array(hist["nr_iters"])
    return {
        "label": label,
        "increments": len(iters),
        "total_nr": int(iters.sum()),
        "mean_nr": iters.mean(),
        "max_nr": int(iters.max()),
        "wall_time": wt,
    }

s_ccd = summary("CCD", hist_ccd, wt_ccd)
s_ogc = summary("OGC", hist_ogc, wt_ogc)

print("\n" + "=" * 60)
print("COMPARISON SUMMARY")
print("=" * 60)
header = f"{'':12s} {'Increments':>10s} {'Total NR':>10s} {'Mean NR':>10s} {'Max NR':>10s} {'Time (s)':>10s}"
print(header)
print("-" * len(header))
for s in (s_ccd, s_ogc):
    print(f"{s['label']:12s} {s['increments']:10d} {s['total_nr']:10d} "
          f"{s['mean_nr']:10.2f} {s['max_nr']:10d} {s['wall_time']:10.2f}")


# =========================================================================
# Plots
# =========================================================================

try:
    import matplotlib.pyplot as plt

    # --- Force–indentation comparison ---
    t_ccd = np.array(hist_ccd["time"])
    Fy_ccd = -np.array(hist_ccd["reaction_y"])
    delta_ccd = np.maximum(t_ccd * abs(imposed_disp) - gap, 0)

    t_ogc = np.array(hist_ogc["time"])
    Fy_ogc = -np.array(hist_ogc["reaction_y"])
    delta_ogc = np.maximum(t_ogc * abs(imposed_disp) - gap, 0)

    # Hertz
    E_star = 1.0 / ((1 - nu**2) / E_plate + (1 - nu**2) / E_disk)

    def hertz_2d(a):
        d = a**2 / (4 * R) * (2 * np.log(4 * R / a) - 1)
        P = np.pi * E_star / (4 * R) * a**2
        return d, P

    a_vals = np.linspace(0.01, R * 0.95, 500)
    delta_h, F_h = np.array([hertz_2d(a) for a in a_vals]).T
    dmax = max(delta_ccd.max(), delta_ogc.max())
    mask = delta_h <= dmax * 1.05
    delta_h, F_h = delta_h[mask], F_h[mask]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Force vs indentation
    ax = axes[0]
    ax.plot(delta_ccd, Fy_ccd, "o-", ms=4, label="CCD")
    ax.plot(delta_ogc, Fy_ogc, "s-", ms=4, label="OGC")
    ax.plot(delta_h, F_h, "--", lw=2, color="gray", label="Hertz (2D)")
    ax.set_xlabel("Indentation depth (mm)")
    ax.set_ylabel("Force per unit thickness (N/mm)")
    ax.set_title("Force--Indentation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: NR iterations per increment
    ax = axes[1]
    ax.bar(np.arange(len(hist_ccd["nr_iters"])) - 0.2,
           hist_ccd["nr_iters"], width=0.4, label="CCD", alpha=0.8)
    ax.bar(np.arange(len(hist_ogc["nr_iters"])) + 0.2,
           hist_ogc["nr_iters"], width=0.4, label="OGC", alpha=0.8)
    ax.set_xlabel("Increment")
    ax.set_ylabel("NR iterations")
    ax.set_title("Newton--Raphson iterations per increment")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("CCD vs OGC -- 2D Disk Indentation", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("results/ccd_vs_ogc.png", dpi=150, bbox_inches="tight")
    print("\nComparison plot saved to results/ccd_vs_ogc.png")
    plt.show()

except ImportError:
    print("matplotlib not available -- skipping comparison plots")
