"""
Self-contact: penalty vs IPC comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example compares the two self-contact approaches available in fedoo
on a 2D hole plate being crushed:

  - **Penalty method** (``fd.constraint.SelfContact``): node-to-surface
    formulation with a user-tuned penalty parameter.
  - **IPC method** (``fd.constraint.IPCSelfContact``): barrier-potential
    formulation from the ipctk library guaranteeing intersection-free
    configurations.

Both simulations use the same geometry (a plate with a large hole)
and boundary conditions (vertical compression until self-contact occurs).

.. note::
   This example requires the ``simcoon`` package for the elasto-plastic
   material and the ``ipctk`` package for the IPC method.
"""

import fedoo as fd
import numpy as np
import os
from time import time

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")


def build_mesh_and_material():
    """Build the shared geometry and material (fresh ModelingSpace each call)."""
    fd.ModelingSpace("2D")

    # Plate with a large circular hole -- will self-contact under compression
    mesh = fd.mesh.hole_plate_mesh(nr=15, nt=15, length=100, height=100, radius=45)

    # Material: elasto-plastic (EPICP)
    E, nu = 200e3, 0.3
    alpha = 1e-5
    Re = 300  # yield stress
    k = 1000  # hardening modulus
    m = 0.3  # hardening exponent
    props = np.array([E, nu, alpha, Re, k, m])
    material = fd.constitutivelaw.Simcoon("EPICP", props)

    return mesh, material


# =========================================================================
# Approach 1 : Penalty self-contact
# =========================================================================

print("=" * 60)
print("PENALTY SELF-CONTACT")
print("=" * 60)

mesh, material = build_mesh_and_material()

nodes_top = mesh.find_nodes("Y", mesh.bounding_box.ymax)
nodes_bottom = mesh.find_nodes("Y", mesh.bounding_box.ymin)

surf = fd.mesh.extract_surface(mesh)
penalty_contact = fd.constraint.SelfContact(surf, "linear", search_algorithm="bucket")
penalty_contact.contact_search_once = True
penalty_contact.eps_n = 1e6

wf = fd.weakform.StressEquilibrium(material, nlgeom="UL")
solid_assembly = fd.Assembly.create(wf, mesh)
assembly = fd.Assembly.sum(solid_assembly, penalty_contact)

pb_penalty = fd.problem.NonLinear(assembly)

if not os.path.isdir("results"):
    os.mkdir("results")
res_penalty = pb_penalty.add_output(
    "results/self_contact_penalty", solid_assembly, ["Disp", "Stress"]
)

pb_penalty.bc.add("Dirichlet", nodes_bottom, "Disp", 0)
pb_penalty.bc.add("Dirichlet", nodes_top, "Disp", [0, -70])
pb_penalty.set_nr_criterion("Displacement", tol=5e-3, max_subiter=5)

t0 = time()
pb_penalty.nlsolve(dt=0.01, tmax=1, update_dt=True, print_info=1, interval_output=0.1)
print(f"Penalty self-contact solve time: {time() - t0:.2f} s")


# =========================================================================
# Approach 2 : IPC self-contact
# =========================================================================

print("\n" + "=" * 60)
print("IPC SELF-CONTACT")
print("=" * 60)

mesh2, material2 = build_mesh_and_material()

nodes_top2 = mesh2.find_nodes("Y", mesh2.bounding_box.ymax)
nodes_bottom2 = mesh2.find_nodes("Y", mesh2.bounding_box.ymin)

# IPC self-contact: auto-extracts surface from the mesh
ipc_contact = fd.constraint.IPCSelfContact(
    mesh2,
    dhat=5e-3,
    dhat_is_relative=True,
    friction_coefficient=0.0,
    use_ccd=True,
)

wf2 = fd.weakform.StressEquilibrium(material2, nlgeom="UL")
solid_assembly2 = fd.Assembly.create(wf2, mesh2)
assembly2 = fd.Assembly.sum(solid_assembly2, ipc_contact)

pb_ipc = fd.problem.NonLinear(assembly2)

res_ipc = pb_ipc.add_output(
    "results/self_contact_ipc", solid_assembly2, ["Disp", "Stress"]
)

pb_ipc.bc.add("Dirichlet", nodes_bottom2, "Disp", 0)
pb_ipc.bc.add("Dirichlet", nodes_top2, "Disp", [0, -70])
pb_ipc.set_nr_criterion("Displacement", tol=5e-3, max_subiter=5)

t0 = time()
pb_ipc.nlsolve(dt=0.01, tmax=1, update_dt=True, print_info=1, interval_output=0.1)
print(f"IPC self-contact solve time: {time() - t0:.2f} s")


# =========================================================================
# Post-processing (requires pyvista)
# =========================================================================
# Uncomment the lines below to visualise and compare the results.

# res_penalty.plot("Stress", "vm", "Node", show=True, scale=1, show_nodes=True)
# res_ipc.plot("Stress", "vm", "Node", show=True, scale=1, show_nodes=True)
