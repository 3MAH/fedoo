"""
Gyroid compression with periodic BCs (3D, IPC method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compresses a 3D gyroid unit cell (50% strain) with IPC self-contact
and periodic boundary conditions.  This is the most advanced contact
example, combining large strain (UL), periodic BCs, and IPC.

A diagnostic callback prints the number of active collisions, the
barrier stiffness, and the contact force norm at each converged step.

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

# --- IPC self-contact (auto-extracts surface) ---
contact = fd.constraint.IPCSelfContact(
    mesh, dhat=5e-3, dhat_is_relative=True, use_ccd=True
)

# --- Assembly ---
wf = fd.weakform.StressEquilibrium(material, nlgeom="UL")
solid_assembly = fd.Assembly.create(wf, mesh)
assembly = fd.Assembly.sum(solid_assembly, contact)

# --- Problem ---
pb = fd.problem.NonLinear(assembly)
if not os.path.isdir("results"):
    os.mkdir("results")
res = pb.add_output("results/gyroid_ipc_periodic", solid_assembly, ["Disp", "Stress"])

# --- BCs: periodic, compress 50% ---
bc_periodic = fd.constraint.PeriodicBC("finite_strain", tol=1e-3)
pb.bc.add(bc_periodic)

# block a node near the center to avoid rigid body motion
pb.bc.add("Dirichlet", mesh.nearest_node(mesh.bounding_box.center), "Disp", 0)

# Uniaxial compression: prescribe DU_xx, fix off-diagonal terms to prevent rotation
pb.bc.add("Dirichlet", "DU_xx", -0.5)
pb.bc.add("Dirichlet", "DU_xy", 0)
pb.bc.add("Dirichlet", "DU_xz", 0)
pb.bc.add("Dirichlet", "DU_yx", 0)
pb.bc.add("Dirichlet", "DU_yz", 0)
pb.bc.add("Dirichlet", "DU_zx", 0)
pb.bc.add("Dirichlet", "DU_zy", 0)

pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=10)


# --- Diagnostic callback ---
def diag(pb):
    n_coll = len(contact._collisions) if contact._collisions is not None else 0
    kappa = getattr(contact, '_kappa', None)
    gv_norm = np.linalg.norm(contact.global_vector) if contact.global_vector is not None else 0
    print(f"  [IPC] t={pb.time:.4f}  collisions={n_coll}  kappa={kappa}  |Fcontact|={gv_norm:.4e}")


# --- Solve ---
pb.nlsolve(dt=0.05, tmax=1, update_dt=True, print_info=1, interval_output=0.1, callback=diag)

# --- Static plot ---
# res.plot("Stress", "vm", "Node", show=True, scale=1)

# --- Video output (uncomment to export MP4) ---
# res.write_movie("results/gyroid_ipc_periodic", "Stress", "vm", "Node",
#                 framerate=10, quality=8)
# print("Movie saved to results/gyroid_ipc_periodic.mp4")
