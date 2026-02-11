"""
Gyroid self-contact under compression (3D, OGC method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same benchmark as ``ipc/gyroid_compression.py`` but using the OGC
(Offset Geometric Contact) trust-region method instead of CCD.

.. note::
   Requires ``ipctk``.
"""
import fedoo as fd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

fd.ModelingSpace("3D")

# --- Geometry ---
MESH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../util/meshes/gyroid_per.vtk")
mesh = fd.Mesh.read(MESH_PATH)
material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)

# --- IPC self-contact with OGC trust-region ---
contact = fd.constraint.IPCSelfContact(
    mesh, dhat=5e-3, dhat_is_relative=True, use_ogc=True,
)

wf = fd.weakform.StressEquilibrium(material, nlgeom="UL")
solid_assembly = fd.Assembly.create(wf, mesh)
assembly = fd.Assembly.sum(solid_assembly, contact)

pb = fd.problem.NonLinear(assembly)

if not os.path.isdir("results"):
    os.mkdir("results")
res = pb.add_output("results/gyroid_ogc", solid_assembly, ["Disp", "Stress"])

# --- BCs: simple compression 15% ---
nodes_top = mesh.find_nodes("Z", mesh.bounding_box.zmax)
nodes_bottom = mesh.find_nodes("Z", mesh.bounding_box.zmin)
pb.bc.add("Dirichlet", nodes_bottom, "Disp", 0)
pb.bc.add("Dirichlet", nodes_top, "Disp", [0, 0, -0.15])
pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=10)

pb.nlsolve(dt=0.05, tmax=1, update_dt=True, print_info=1, interval_output=0.1)

# --- Static plot ---
res.plot("Stress", "vm", "Node", show=False, scale=1)

# --- Video output (uncomment to export MP4) ---
# res.write_movie("results/gyroid_ogc", "Stress", "vm", "Node",
#                 framerate=10, quality=8)
# print("Movie saved to results/gyroid_ogc.mp4")
