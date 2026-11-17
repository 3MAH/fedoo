r"""
Tube compression (2Daxi): IPC versus penalty self-contact
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same problem as :ref:`sphx_glr_examples_contact_tube_compression.py`, solved
twice: with ``IPCSelfContact`` (collisions weighted by :math:`2 \pi r`) and
with the penalty ``SelfContact``. For each run the script prints the solve
time, the axial reaction forces on the top and bottom faces (they should be
opposite), the peak axial stress and the peak equivalent plastic strain at
the last saved increment. The page has no figure: the deformed shape is
shown in :ref:`sphx_glr_examples_contact_tube_compression.py`, and the
results of both runs are written to ``results/``.

This example requires ``ipctk``.
"""

import os
from time import time

import fedoo as fd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
if not os.path.isdir("results"):
    os.mkdir("results")

NLGEOM = "UL"


def solve(method):
    fd.ModelingSpace("2Daxi")
    mesh = fd.mesh.rectangle_mesh(5, 240, 23, 25, 0, 180)

    sigma_y = 300
    k = 1000
    m = 0.3
    E = 200e3
    nu = 0.3
    props = np.array([E, nu, 1e-5, sigma_y, k, m])
    material = fd.constitutivelaw.Simcoon("EPICP", props)

    wf = fd.weakform.StressEquilibrium(material, nlgeom=NLGEOM)
    solid_assembly = fd.Assembly.create(wf, mesh)

    if method == "ipc":
        contact = fd.constraint.IPCSelfContact(
            mesh,
            dhat=1e-3,
            dhat_is_relative=True,
        )
    else:
        surf = fd.mesh.extract_surface(mesh)
        contact = fd.constraint.contact.SelfContact(surf)
        contact.contact_search_once = True
        contact.eps_n = 1e4  # contact penalty
        contact.max_dist = 1.0  # max distance for the contact search

    assembly = fd.Assembly.sum(solid_assembly, contact)

    pb = fd.problem.NonLinear(assembly, nlgeom=NLGEOM)
    pb.set_nr_criterion(
        "Displacement",
        tol=1e-2,
        max_subiter=20,
        adaptive_stiffness=True,
    )

    res = pb.add_output(
        f"results/tube_compression_{method}",
        solid_assembly,
        ["Disp", "Stress", "Strain", "P"],
    )

    bottom = mesh.node_sets["bottom"]
    top = mesh.node_sets["top"]

    pb.bc.add("Dirichlet", bottom, "Disp", 0)
    pb.bc.add("Dirichlet", top, "Disp", [0, -150])
    pb.add_line_search()

    t0 = time()
    pb.nlsolve(dt=0.01, tmax=1, update_dt=True, print_info=1, dt_min=1e-8)
    solve_time = time() - t0

    # axial reaction forces: they should be opposite
    F = pb.get_ext_forces("Disp")
    return res, F[1, top].sum(), F[1, bottom].sum(), solve_time


###############################################################################
# Reduced numeric metrics
# -----------------------
# Reaction forces, peak axial stress and peak equivalent plastic strain at the
# last saved increment.


def _summarise(res, f_top, f_bottom, solve_time, label):
    res.load(-1)
    stress_yy = np.asarray(res.get_data("Stress", component="YY", data_type="Node"))
    p = np.asarray(res.get_data("P", data_type="Node"))
    print(f"--- {label} ---")
    print(f"  solve time           = {solve_time:.1f} s")
    print(f"  reaction top/bottom  = {f_top:.4e} / {f_bottom:.4e}")
    print(f"  peak |Stress YY|     = {np.abs(stress_yy).max():.4e}")
    print(f"  peak P (eq. plastic) = {p.max():.4e}")


results = {method: solve(method) for method in ("ipc", "penalty")}
print()
_summarise(*results["ipc"], "IPC")
_summarise(*results["penalty"], "Penalty")
