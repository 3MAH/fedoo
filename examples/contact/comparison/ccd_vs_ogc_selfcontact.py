"""
CCD vs OGC self-contact comparison (hole plate compression)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runs the 2D hole-plate self-contact benchmark twice — once with CCD
and once with OGC — then compares convergence behaviour, timing, and
reaction-force curves.

.. note::
   Requires ``ipctk`` and ``simcoon``.
"""

import fedoo as fd
import numpy as np
import os
from time import time

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

# =========================================================================
# Parameters
# =========================================================================

DHAT_REL = 3e-3
IMPOSED_DISP = -70.0
DT = 0.01
TMAX = 1.0
TOL = 5e-3
MAX_SUBITER = 10


# =========================================================================
# Helper: build problem
# =========================================================================

def build_problem(method):
    """Build hole-plate self-contact problem with CCD or OGC."""
    fd.ModelingSpace("2D")

    mesh = fd.mesh.hole_plate_mesh(nr=15, nt=15, length=100, height=100, radius=45)

    contact = fd.constraint.IPCSelfContact(
        mesh,
        dhat=DHAT_REL,
        dhat_is_relative=True,
        use_ccd=(method == "ccd"),
        use_ogc=(method == "ogc"),
    )

    nodes_top = mesh.find_nodes("Y", mesh.bounding_box.ymax)
    nodes_bottom = mesh.find_nodes("Y", mesh.bounding_box.ymin)

    E, nu = 200e3, 0.3
    props = np.array([E, nu, 1e-5, 300, 1000, 0.3])
    material = fd.constitutivelaw.Simcoon("EPICP", props)

    wf = fd.weakform.StressEquilibrium(material, nlgeom="UL")
    solid = fd.Assembly.create(wf, mesh)
    assembly = fd.Assembly.sum(solid, contact)

    pb = fd.problem.NonLinear(assembly)

    if not os.path.isdir("results"):
        os.mkdir("results")

    pb.bc.add("Dirichlet", nodes_bottom, "Disp", 0)
    pb.bc.add("Dirichlet", nodes_top, "Disp", [0, IMPOSED_DISP])
    pb.set_nr_criterion("Displacement", tol=TOL, max_subiter=MAX_SUBITER)

    history = {"time": [], "reaction_y": [], "nr_iters": []}
    return pb, solid, nodes_top, history


# =========================================================================
# Instrumented solve
# =========================================================================

def solve_instrumented(pb, solid, nodes_top, history, label):
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
        history["reaction_y"].append(np.sum(F[1, nodes_top]))

    print("=" * 60)
    print(f"HOLE PLATE SELF-CONTACT -- {label}")
    print("=" * 60)
    t0 = time()
    pb.nlsolve(
        dt=DT, tmax=TMAX, update_dt=True, print_info=1, callback=track,
    )
    wall_time = time() - t0
    print(f"{label} solve time: {wall_time:.2f} s")

    pb.solve_time_increment = _original
    return wall_time


# =========================================================================
# Run both
# =========================================================================

pb_ccd, solid_ccd, nt_ccd, hist_ccd = build_problem("ccd")
wt_ccd = solve_instrumented(pb_ccd, solid_ccd, nt_ccd, hist_ccd, "CCD")

pb_ogc, solid_ogc, nt_ogc, hist_ogc = build_problem("ogc")
wt_ogc = solve_instrumented(pb_ogc, solid_ogc, nt_ogc, hist_ogc, "OGC")


# =========================================================================
# Summary
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

    t_ccd = np.array(hist_ccd["time"])
    Fy_ccd = -np.array(hist_ccd["reaction_y"])

    t_ogc = np.array(hist_ogc["time"])
    Fy_ogc = -np.array(hist_ogc["reaction_y"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(t_ccd, Fy_ccd, "o-", ms=3, label="CCD")
    ax.plot(t_ogc, Fy_ogc, "s-", ms=3, label="OGC")
    ax.set_xlabel("Normalized time")
    ax.set_ylabel("Reaction force (Y)")
    ax.set_title("Reaction Force vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.bar(np.arange(len(hist_ccd["nr_iters"])) - 0.2,
           hist_ccd["nr_iters"], width=0.4, label="CCD", alpha=0.8)
    ax.bar(np.arange(len(hist_ogc["nr_iters"])) + 0.2,
           hist_ogc["nr_iters"], width=0.4, label="OGC", alpha=0.8)
    ax.set_xlabel("Increment")
    ax.set_ylabel("NR iterations")
    ax.set_title("Newton-Raphson iterations per increment")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("CCD vs OGC -- Self-Contact (Hole Plate)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("results/ccd_vs_ogc_selfcontact.png", dpi=150, bbox_inches="tight")
    print("\nComparison plot saved to results/ccd_vs_ogc_selfcontact.png")
    plt.show()

except ImportError:
    print("matplotlib not available -- skipping plots")
