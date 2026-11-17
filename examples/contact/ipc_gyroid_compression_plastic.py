"""
Elasto-plastic gyroid self-contact under compression (3D, IPC method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compresses a 3D gyroid unit cell (50 % axial strain) with IPC
self-contact to prevent wall intersections. Elasto-plastic material
(simcoon ``EPICP``, isotropic power-law hardening), updated Lagrangian
formulation.

.. note::
   Requires ``ipctk`` and ``simcoon``.
"""

import fedoo as fd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

fd.ModelingSpace("3D")

# --- Geometry ---
MESH_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../util/meshes/gyroid_per_clean.vtk"
)
mesh = fd.Mesh.read(MESH_PATH)

E, nu = 200e3, 0.3
props = np.array([E, nu, 1e-5, 300, 1000, 0.3])
material = fd.constitutivelaw.Simcoon("EPICP", props)

# --- IPC self-contact ---
# The cleaned gyroid mesh (``util/meshes/make_gyroid_mesh.py``, shortest
# surface edge ~8e-3) still has surface pairs closer than dhat at rest with
# ``excluded_rings=1`` or ``2``: they all touch a cell-face vertex, in the
# shallow wedge where the gyroid sheet meets a face of the cell (geometry of
# the cut cell, not a mesh defect). ``excluded_rings=3`` removes these
# intra-surface pairs; dhat is 5e-3 x bbox diagonal.
contact = fd.constraint.IPCSelfContact(
    mesh,
    dhat=5e-3,
    dhat_is_relative=True,
    use_ccd=True,
    excluded_rings=3,
)

wf = fd.weakform.StressEquilibrium(material, nlgeom="UL")
solid_assembly = fd.Assembly.create(wf, mesh)
assembly = fd.Assembly.sum(solid_assembly, contact)

pb = fd.problem.NonLinear(assembly)

if not os.path.isdir("results"):
    os.mkdir("results")
res = pb.add_output("results/gyroid_ipc_plastic", solid_assembly, ["Disp", "Stress"])

# --- BCs: compression 50% ---
nodes_top = mesh.find_nodes("Z", mesh.bounding_box.zmax)
nodes_bottom = mesh.find_nodes("Z", mesh.bounding_box.zmin)
pb.bc.add("Dirichlet", nodes_bottom, "Disp", 0)
pb.bc.add("Dirichlet", nodes_top, "Disp", [0, 0, -0.5])
pb.set_nr_criterion("Displacement", tol=5.0e-3, max_subiter=25)

pb.nlsolve(dt=0.05, tmax=1, update_dt=True, print_info=1, interval_output=0.05)

# --- Static plot ---
res.plot("Stress", "vm", "Node", show=False, scale=1)
