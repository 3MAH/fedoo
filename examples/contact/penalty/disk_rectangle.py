"""
Disk-rectangle contact (2D, penalty method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A stiff elastic disk is pushed into an elasto-plastic rectangle (quad4,
plane strain) using the penalty contact method, then released.

Demonstrates the legacy penalty-based contact approach
(``fd.constraint.Contact``) with node-to-surface formulation.

.. note::
   Requires ``simcoon`` for the EPICP elasto-plastic material.
"""
import fedoo as fd
import numpy as np
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

fd.ModelingSpace("2D")

# --- Geometry ---
mesh_rect = fd.mesh.rectangle_mesh(
    nx=11, ny=21, x_min=0, x_max=1, y_min=0, y_max=1, elm_type="quad4",
)
mesh_rect.element_sets["rect"] = np.arange(mesh_rect.n_elements)

mesh_disk = fd.mesh.disk_mesh(radius=0.5, nr=6, nt=6, elm_type="quad4")
mesh_disk.nodes += np.array([1.5, 0.48])
mesh_disk.element_sets["disk"] = np.arange(mesh_disk.n_elements)

mesh = fd.Mesh.stack(mesh_rect, mesh_disk)

nodes_left = mesh.find_nodes("X", 0)
nodes_right = mesh.find_nodes("X", 1)
nodes_bc = mesh.find_nodes("X>1.5")
nodes_bc = list(set(nodes_bc).intersection(mesh.node_sets["boundary"]))

# --- Contact (penalty) ---
surf = fd.mesh.extract_surface(mesh.extract_elements("disk"))
contact = fd.constraint.Contact(nodes_right, surf)
contact.contact_search_once = True
contact.eps_n = 5e5
contact.max_dist = 1

# --- Material ---
E, nu = 200e3, 0.3
props = np.array([E, nu, 1e-5, 300, 1000, 0.3])
material_rect = fd.constitutivelaw.Simcoon("EPICP", props)
material_disk = fd.constitutivelaw.ElasticIsotrop(50e3, nu)
material = fd.constitutivelaw.Heterogeneous(
    (material_rect, material_disk), ("rect", "disk")
)

wf = fd.weakform.StressEquilibrium(material, nlgeom="UL")
solid_assembly = fd.Assembly.create(wf, mesh)
assembly = fd.Assembly.sum(solid_assembly, contact)

pb = fd.problem.NonLinear(assembly)

if not os.path.isdir("results"):
    os.mkdir("results")
res = pb.add_output("results/disk_rectangle_contact", solid_assembly,
                    ["Disp", "Stress", "Strain"])

# --- Step 1: push ---
pb.bc.add("Dirichlet", nodes_left, "Disp", 0)
pb.bc.add("Dirichlet", nodes_bc, "Disp", [-0.4, 0.2])
pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=5)
pb.nlsolve(dt=0.005, tmax=1, update_dt=True, print_info=1, interval_output=0.01)

# --- Step 2: release ---
pb.bc.remove(-1)
pb.bc.add("Dirichlet", nodes_bc, "Disp", [0, 0])
pb.nlsolve(dt=0.005, tmax=1, update_dt=True, print_info=1, interval_output=0.01)

# --- Static plot ---
res.plot("Stress", "XX", "Node", show=False, scale=1)

# --- Video output (uncomment to export MP4) ---
# res.write_movie("results/disk_rectangle_contact", "Stress", "XX", "Node",
#                 framerate=24, quality=5, clim=[-3e3, 3e3])
# print("Movie saved to results/disk_rectangle_contact.mp4")
