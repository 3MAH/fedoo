"""
Gyroid self-contact under compression (3D, IPC method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compresses a 3D gyroid unit cell with IPC self-contact to prevent
wall intersections.  50% compression with Augmented Lagrangian
outer loop (``al_max_iter=5``) for robust convergence at high strain.

.. note::
   Requires ``ipctk``.
"""

import fedoo as fd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

fd.ModelingSpace("3D")

# --- Geometry ---
MESH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../gyroid_per.vtk"
)
mesh = fd.Mesh.read(MESH_PATH)
material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)

# --- IPC self-contact ---
contact = fd.constraint.IPCSelfContact(
    mesh,
    dhat=1.0e-2,
    dhat_is_relative=True,
    use_ccd=True,
)

wf = fd.weakform.StressEquilibrium(material, nlgeom="UL")
solid_assembly = fd.Assembly.create(wf, mesh)
assembly = fd.Assembly.sum(solid_assembly, contact)

pb = fd.problem.NonLinear(assembly)

if not os.path.isdir("results"):
    os.mkdir("results")
res = pb.add_output("results/gyroid_ipc", solid_assembly, ["Disp", "Stress"])

# --- BCs: compression 50% ---
nodes_top = mesh.find_nodes("Z", mesh.bounding_box.zmax)
nodes_bottom = mesh.find_nodes("Z", mesh.bounding_box.zmin)
pb.bc.add("Dirichlet", nodes_bottom, "Disp", 0)
pb.bc.add("Dirichlet", nodes_top, "Disp", [0, 0, -0.5])
pb.set_nr_criterion("Displacement", tol=5.0e-3, max_subiter=25)

pb.nlsolve(dt=0.05, tmax=1, update_dt=True, print_info=1, interval_output=0.05)

# --- Static plot ---
res.plot("Stress", "vm", "Node", show=False, scale=1)

# --- Video output (uncomment to export MP4) ---
# res.write_movie("results/gyroid_ipc", "Stress", "vm", "Node",
#                 framerate=10, quality=8)
# print("Movie saved to results/gyroid_ipc.mp4")
