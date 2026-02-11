"""
Disk indentation (2D plane strain) — OGC contact with Hertz comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same benchmark as ``ipc/indentation_2d.py`` but using the OGC
(Offset Geometric Contact) trust-region method instead of CCD.
OGC filters the displacement per vertex rather than uniformly
scaling the step, which can improve convergence.

.. note::
   Requires ``ipctk`` and ``gmsh``.
"""

import fedoo as fd
import numpy as np
import os
import tempfile

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

# =========================================================================
# Parameters
# =========================================================================

fd.ModelingSpace("2D")

E_plate = 1e3
E_disk = 1e5
nu = 0.3
R = 5.0
plate_half = 30.0
plate_h = 40.0
gap = 0.1
imposed_disp = -2.0

# =========================================================================
# Plate mesh (gmsh -- graded refinement near contact)
# =========================================================================

import gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 1)

plate_tag = gmsh.model.occ.addRectangle(
    -plate_half, 0, 0, 2 * plate_half, plate_h,
)
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(2, [plate_tag], tag=1, name="plate")

gmsh.model.mesh.field.add("Box", 1)
gmsh.model.mesh.field.setNumber(1, "VIn", 0.3)
gmsh.model.mesh.field.setNumber(1, "VOut", 3.0)
gmsh.model.mesh.field.setNumber(1, "XMin", -6)
gmsh.model.mesh.field.setNumber(1, "XMax", 6)
gmsh.model.mesh.field.setNumber(1, "YMin", plate_h - 4)
gmsh.model.mesh.field.setNumber(1, "YMax", plate_h)
gmsh.model.mesh.field.setNumber(1, "Thickness", 4)
gmsh.model.mesh.field.setAsBackgroundMesh(1)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

gmsh.model.mesh.generate(2)
plate_msh = os.path.join(tempfile.gettempdir(), "plate_indent2d.msh")
gmsh.write(plate_msh)
gmsh.finalize()

mesh_plate = fd.mesh.import_msh(plate_msh, mesh_type="surface")
if mesh_plate.nodes.shape[1] == 3:
    mesh_plate.nodes = mesh_plate.nodes[:, :2]
mesh_plate.element_sets["plate"] = np.arange(mesh_plate.n_elements)
print(f"Plate mesh: {mesh_plate.n_nodes} nodes, {mesh_plate.n_elements} elems")

# =========================================================================
# Disk mesh
# =========================================================================

mesh_disk = fd.mesh.disk_mesh(radius=R, nr=12, nt=24, elm_type="tri3")
mesh_disk.nodes += np.array([0, plate_h + R + gap])
mesh_disk.element_sets["disk"] = np.arange(mesh_disk.n_elements)
print(f"Disk mesh:  {mesh_disk.n_nodes} nodes, {mesh_disk.n_elements} elems")

mesh = fd.Mesh.stack(mesh_plate, mesh_disk)
print(f"Total mesh: {mesh.n_nodes} nodes, {mesh.n_elements} elems, type={mesh.elm_type}")

# =========================================================================
# IPC contact with OGC trust-region
# =========================================================================

surf = fd.mesh.extract_surface(mesh)
ipc_contact = fd.constraint.IPCContact(
    mesh,
    surface_mesh=surf,
    dhat=0.05,
    dhat_is_relative=False,
    use_ogc=True,
)

# =========================================================================
# Material, assembly, BCs
# =========================================================================

mat_plate = fd.constitutivelaw.ElasticIsotrop(E_plate, nu)
mat_disk = fd.constitutivelaw.ElasticIsotrop(E_disk, nu)
material = fd.constitutivelaw.Heterogeneous(
    (mat_plate, mat_disk), ("plate", "disk"),
)

wf = fd.weakform.StressEquilibrium(material, nlgeom=False)
solid = fd.Assembly.create(wf, mesh)
assembly = fd.Assembly.sum(solid, ipc_contact)

pb = fd.problem.NonLinear(assembly)

nodes_bottom = mesh.find_nodes("Y", 0)
nodes_disk_top = mesh.find_nodes("Y", mesh.bounding_box.ymax)

pb.bc.add("Dirichlet", nodes_bottom, "Disp", 0)
pb.bc.add("Dirichlet", nodes_disk_top, "Disp", [0, imposed_disp])
pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=8)

# =========================================================================
# Output and tracking callback
# =========================================================================

if not os.path.isdir("results"):
    os.mkdir("results")
res = pb.add_output("results/indentation_2d", solid, ["Disp", "Stress"])

history = {"time": [], "reaction_y": []}


def track(pb):
    history["time"].append(pb.time)
    F = pb.get_ext_forces("Disp")
    history["reaction_y"].append(np.sum(F[1, nodes_disk_top]))


# =========================================================================
# Solve
# =========================================================================

print("=" * 60)
print("2D DISK INDENTATION -- OGC TRUST-REGION")
print("=" * 60)
pb.nlsolve(
    dt=0.05, tmax=1, update_dt=True, print_info=1, callback=track,
)

# =========================================================================
# Hertz comparison
# =========================================================================

try:
    import matplotlib.pyplot as plt

    t = np.array(history["time"])
    Fy = -np.array(history["reaction_y"])

    delta = np.maximum(t * abs(imposed_disp) - gap, 0)

    E_star = 1.0 / ((1 - nu**2) / E_plate + (1 - nu**2) / E_disk)

    def hertz_2d(a):
        d = a**2 / (4 * R) * (2 * np.log(4 * R / a) - 1)
        P = np.pi * E_star / (4 * R) * a**2
        return d, P

    a_vals = np.linspace(0.01, R * 0.95, 500)
    delta_hertz, F_hertz = np.array([hertz_2d(a) for a in a_vals]).T

    mask = delta_hertz <= delta.max() * 1.05
    delta_hertz = delta_hertz[mask]
    F_hertz = F_hertz[mask]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(delta, Fy, "o-", ms=4, label="FEM (OGC)")
    ax.plot(delta_hertz, F_hertz, "--", lw=2,
            label="Hertz (2D half-space)")
    ax.set_xlabel("Indentation depth (mm)")
    ax.set_ylabel("Force per unit thickness (N/mm)")
    ax.set_title("2D Hertz Indentation -- OGC Trust-Region")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/indentation_2d_hertz.png", dpi=150)
    print("Hertz comparison saved to results/indentation_2d_hertz.png")
    plt.show()
except ImportError:
    print("matplotlib not available -- skipping Hertz comparison plot")

# =========================================================================
# Stress plot
# =========================================================================

res.plot("Stress", "vm", "Node", show=False, scale=1)
