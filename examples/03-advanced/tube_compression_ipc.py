"""
Compression of a tube using 2D axisymmetric model — IPC self-contact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same problem as ``tube_compression.py`` but with ``IPCSelfContact``
instead of the penalty ``SelfContact``. Used to benchmark the
2π·r-weighted IPC formulation against the penalty reference.

Run after ``tube_compression.py`` so both result folders exist; this
script prints reduced metrics from each run for side-by-side
comparison.
"""

import os

import fedoo as fd
import numpy as np

fd.ModelingSpace("2Daxi")
mesh = fd.mesh.rectangle_mesh(5, 240, 23, 25, 0, 180)

sigma_y = 300
k = 1000
m = 0.3
E = 200e3
nu = 0.3
props = np.array([E, nu, 1e-5, sigma_y, k, m])
material = fd.constitutivelaw.Simcoon("EPICP", props)

NLGEOM = "UL"
wf = fd.weakform.StressEquilibrium(material, nlgeom=NLGEOM)
solid_assembly = fd.Assembly.create(wf, mesh)

# --- IPC self-contact (replaces penalty SelfContact) ---
contact = fd.constraint.IPCSelfContact(
    mesh,
    dhat=1e-3,
    dhat_is_relative=True,
    use_ccd=True,
)

assembly = fd.Assembly.sum(solid_assembly, contact)

pb = fd.problem.NonLinear(assembly, nlgeom=NLGEOM)
pb.set_nr_criterion(
    "Displacement",
    tol=1e-2,
    max_subiter=20,
    adaptive_stiffness=True,
)

if not os.path.isdir("results"):
    os.mkdir("results")
res_ipc = pb.add_output(
    "results/tube_compression_ipc",
    solid_assembly,
    ["Disp", "Stress", "Strain", "P"],
)

bottom = mesh.node_sets["bottom"]
top = mesh.node_sets["top"]

pb.bc.add("Dirichlet", bottom, "Disp", 0)
pb.bc.add("Dirichlet", top, "Disp", [0, -150])
pb.add_line_search()
pb.nlsolve(dt=0.01, tmax=1, update_dt=True, print_info=0, dt_min=1e-8)

###############################################################################
# Reduced numeric metrics — peak axial stress, peak equivalent plastic strain,
# and final top displacement at the last saved iteration.


def _summarise(res, label):
    res.load(-1)
    stress_yy = np.asarray(res.get_data("Stress", component="YY", data_type="Node"))
    p = np.asarray(res.get_data("P", data_type="Node"))
    disp_y = np.asarray(res.get_data("Disp", component="Y", data_type="Node"))
    print(f"--- {label} ---")
    print(f"  peak |Stress YY|     = {np.abs(stress_yy).max():.4e}")
    print(f"  peak P (eq. plastic) = {p.max():.4e}")
    print(f"  min Disp Y           = {disp_y.min():.4e}")
    print(f"  max Disp Y           = {disp_y.max():.4e}")


print()
_summarise(res_ipc, "IPC (this run)")

# Compare against the penalty reference if it has been run.
penalty_path = "results/tube_compressoin.fdz"  # original spelling kept as in ref script
if os.path.isfile(penalty_path):
    res_penalty = fd.DataSet.read(penalty_path)
    _summarise(res_penalty, "Penalty (reference)")
else:
    print(
        f"\n(Penalty reference not found at '{penalty_path}'. Run "
        "tube_compression.py first to populate it for comparison.)"
    )
