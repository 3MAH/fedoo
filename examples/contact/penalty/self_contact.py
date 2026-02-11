"""
Self-contact — hole plate compression (2D, penalty method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A plate with a large circular hole is compressed until self-contact
occurs on the hole's inner surface.  Uses the penalty self-contact
method (``fd.constraint.SelfContact``) with an elasto-plastic material.

Large strain (Updated Lagrangian with logarithmic corotational).

.. note::
   Requires ``simcoon`` for the EPICP elasto-plastic material.
"""
import fedoo as fd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

fd.ModelingSpace("2D")

# --- Geometry ---
mesh = fd.mesh.hole_plate_mesh(nr=15, nt=15, length=100, height=100, radius=45)
surf = fd.mesh.extract_surface(mesh)

# --- Contact (penalty self-contact) ---
contact = fd.constraint.SelfContact(surf, "linear", search_algorithm="bucket")
contact.contact_search_once = True
contact.eps_n = 1e6

nodes_top = mesh.find_nodes("Y", mesh.bounding_box.ymax)
nodes_bottom = mesh.find_nodes("Y", mesh.bounding_box.ymin)

# --- Material: EPICP (E, nu, alpha, sigmaY, H, beta) ---
E, nu = 200e3, 0.3
props = np.array([E, nu, 1e-5, 300, 1000, 0.3])
material = fd.constitutivelaw.Simcoon("EPICP", props)

# --- Large strain ---
wf = fd.weakform.StressEquilibrium(material, nlgeom="UL")
solid_assembly = fd.Assembly.create(wf, mesh)
assembly = fd.Assembly.sum(solid_assembly, contact)

pb = fd.problem.NonLinear(assembly)

if not os.path.isdir("results"):
    os.mkdir("results")
res = pb.add_output("results/self_contact_penalty", solid_assembly,
                    ["Disp", "Stress", "Strain"])

pb.bc.add("Dirichlet", nodes_bottom, "Disp", 0)
pb.bc.add("Dirichlet", nodes_top, "Disp", [0, -70])
pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=5)

pb.nlsolve(dt=0.01, tmax=1, update_dt=True, print_info=1, interval_output=0.01)

# --- Static plot ---
res.plot("Stress", "vm", "Node", show=False, scale=1)

# --- Video output (uncomment to export MP4) ---
# res.write_movie("results/self_contact_penalty", "Stress", "vm", "Node",
#                 framerate=24, quality=8, clim=[0, 1.5e3])
# print("Movie saved to results/self_contact_penalty.mp4")
