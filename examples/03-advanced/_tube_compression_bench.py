"""Headless benchmark driver: runs tube_compression with penalty and IPC,
then prints reduced metrics. Generated for PR #31 IPC-axisymmetric review.
Skips all plot/gif generation."""

import os
import time

import fedoo as fd
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")
if not os.path.isdir("results"):
    os.mkdir("results")

# ---- Material & mesh shared by both runs ----
sigma_y, k_h, m_h = 300, 1000, 0.3
E, nu = 200e3, 0.3
props = np.array([E, nu, 1e-5, sigma_y, k_h, m_h])
NLGEOM = "UL"


def _setup():
    fd.ModelingSpace("2Daxi")
    mesh = fd.mesh.rectangle_mesh(5, 240, 23, 25, 0, 180)
    material = fd.constitutivelaw.Simcoon("EPICP", props)
    wf = fd.weakform.StressEquilibrium(material, nlgeom=NLGEOM)
    solid = fd.Assembly.create(wf, mesh)
    return mesh, solid


def run_penalty():
    mesh, solid = _setup()
    surf = fd.mesh.extract_surface(mesh)
    contact = fd.constraint.contact.SelfContact(surf)
    contact.contact_search_once = True
    contact.eps_n = 1e4
    contact.max_dist = 1.0

    pb = fd.problem.NonLinear(solid + contact, nlgeom=NLGEOM)
    pb.set_nr_criterion(
        "Displacement", tol=1e-2, max_subiter=20, adaptive_stiffness=True
    )
    res = pb.add_output(
        "results/tube_compressoin", solid, ["Disp", "Stress", "Strain", "P"]
    )
    pb.bc.add("Dirichlet", mesh.node_sets["bottom"], "Disp", 0)
    pb.bc.add("Dirichlet", mesh.node_sets["top"], "Disp", [0, -150])
    pb.add_line_search()
    t0 = time.time()
    pb.nlsolve(dt=0.01, tmax=1, update_dt=True, print_info=1, dt_min=1e-8)
    return res, time.time() - t0


def run_ipc():
    mesh, solid = _setup()
    contact = fd.constraint.IPCSelfContact(
        mesh, dhat=1e-3, dhat_is_relative=True, use_ccd=True
    )
    pb = fd.problem.NonLinear(fd.Assembly.sum(solid, contact), nlgeom=NLGEOM)
    pb.set_nr_criterion(
        "Displacement", tol=1e-2, max_subiter=20, adaptive_stiffness=True
    )
    res = pb.add_output(
        "results/tube_compression_ipc", solid, ["Disp", "Stress", "Strain", "P"]
    )
    pb.bc.add("Dirichlet", mesh.node_sets["bottom"], "Disp", 0)
    pb.bc.add("Dirichlet", mesh.node_sets["top"], "Disp", [0, -150])
    pb.add_line_search()
    t0 = time.time()
    pb.nlsolve(dt=0.01, tmax=1, update_dt=True, print_info=1, dt_min=1e-8)
    return res, time.time() - t0


def summarise(res, label, runtime):
    res.load(-1)
    s_yy = np.asarray(res.get_data("Stress", component="YY", data_type="Node"))
    p = np.asarray(res.get_data("P", data_type="Node"))
    d_y = np.asarray(res.get_data("Disp", component="Y", data_type="Node"))
    print(f"--- {label} (runtime {runtime:.1f}s) ---")
    print(f"  peak |Stress YY|     = {np.abs(s_yy).max():.4e}")
    print(f"  peak P (eq. plastic) = {p.max():.4e}")
    print(f"  min Disp Y           = {d_y.min():.4e}")
    print(f"  max Disp Y           = {d_y.max():.4e}")


if __name__ == "__main__":
    print(">> running PENALTY (tube_compression reference) ...", flush=True)
    res_p, dt_p = run_penalty()
    print(">> running IPC (tube_compression_ipc) ...", flush=True)
    res_i, dt_i = run_ipc()
    print()
    summarise(res_p, "Penalty", dt_p)
    summarise(res_i, "IPC", dt_i)
