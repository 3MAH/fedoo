"""
Sphere indentation (3D) — OGC contact with Hertz comparison
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same benchmark as ``ipc/indentation_3d.py`` but using the OGC
(Offset Geometric Contact) trust-region method instead of CCD.

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

fd.ModelingSpace("3D")

E_plate = 1e3
E_sphere = 1e5
nu = 0.3
R = 5.0
plate_half = 25.0
plate_h = 25.0
gap = 0.1
imposed_disp = -2.0

sphere_cz = plate_h + R + gap

# =========================================================================
# Sphere mesh (gmsh)
# =========================================================================

import gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 1)

sphere_tag = gmsh.model.occ.addSphere(0, 0, sphere_cz, R)
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(3, [sphere_tag], tag=2, name="sphere")

gmsh.model.mesh.field.add("Box", 1)
gmsh.model.mesh.field.setNumber(1, "VIn", 0.4)
gmsh.model.mesh.field.setNumber(1, "VOut", 1.2)
gmsh.model.mesh.field.setNumber(1, "XMin", -R)
gmsh.model.mesh.field.setNumber(1, "XMax", R)
gmsh.model.mesh.field.setNumber(1, "YMin", -R)
gmsh.model.mesh.field.setNumber(1, "YMax", R)
gmsh.model.mesh.field.setNumber(1, "ZMin", sphere_cz - R)
gmsh.model.mesh.field.setNumber(1, "ZMax", sphere_cz - R + 3)
gmsh.model.mesh.field.setNumber(1, "Thickness", 2)
gmsh.model.mesh.field.setAsBackgroundMesh(1)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

gmsh.model.mesh.generate(3)
sphere_msh = os.path.join(tempfile.gettempdir(), "sphere_indent3d.msh")
gmsh.write(sphere_msh)
gmsh.finalize()

mesh_sphere = fd.mesh.import_msh(sphere_msh)
mesh_sphere.element_sets["sphere"] = mesh_sphere.element_sets.get(
    "sphere", np.arange(mesh_sphere.n_elements)
)
print(f"Sphere mesh: {mesh_sphere.n_nodes} nodes, {mesh_sphere.n_elements} tet4")

# =========================================================================
# Plate mesh (gmsh -- graded refinement near contact)
# =========================================================================

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 1)

plate_tag = gmsh.model.occ.addBox(
    -plate_half, -plate_half, 0,
    2 * plate_half, 2 * plate_half, plate_h,
)
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(3, [plate_tag], tag=1, name="plate")

gmsh.model.mesh.field.add("Box", 1)
gmsh.model.mesh.field.setNumber(1, "VIn", 0.6)
gmsh.model.mesh.field.setNumber(1, "VOut", 5.0)
gmsh.model.mesh.field.setNumber(1, "XMin", -6)
gmsh.model.mesh.field.setNumber(1, "XMax", 6)
gmsh.model.mesh.field.setNumber(1, "YMin", -6)
gmsh.model.mesh.field.setNumber(1, "YMax", 6)
gmsh.model.mesh.field.setNumber(1, "ZMin", plate_h - 5)
gmsh.model.mesh.field.setNumber(1, "ZMax", plate_h)
gmsh.model.mesh.field.setNumber(1, "Thickness", 3)
gmsh.model.mesh.field.setAsBackgroundMesh(1)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

gmsh.model.mesh.generate(3)
plate_msh = os.path.join(tempfile.gettempdir(), "plate_indent3d.msh")
gmsh.write(plate_msh)
gmsh.finalize()

mesh_plate = fd.mesh.import_msh(plate_msh)
mesh_plate.element_sets["plate"] = mesh_plate.element_sets.get(
    "plate", np.arange(mesh_plate.n_elements)
)
print(f"Plate mesh: {mesh_plate.n_nodes} nodes, {mesh_plate.n_elements} tet4")

# =========================================================================
# Stack into single mesh
# =========================================================================

mesh = fd.Mesh.stack(mesh_plate, mesh_sphere)
print(f"Total mesh:  {mesh.n_nodes} nodes, {mesh.n_elements} tet4")

# =========================================================================
# IPC contact with OGC trust-region
# =========================================================================

surf = fd.mesh.extract_surface(mesh, quad2tri=True)
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
mat_sphere = fd.constitutivelaw.ElasticIsotrop(E_sphere, nu)
material = fd.constitutivelaw.Heterogeneous(
    (mat_plate, mat_sphere), ("plate", "sphere"),
)

wf = fd.weakform.StressEquilibrium(material, nlgeom=False)
solid = fd.Assembly.create(wf, mesh)
assembly = fd.Assembly.sum(solid, ipc_contact)

pb = fd.problem.NonLinear(assembly)

nodes_bottom = mesh.find_nodes("Z", 0)
nodes_sphere_top = mesh.find_nodes("Z", mesh.bounding_box.zmax)

pb.bc.add("Dirichlet", nodes_bottom, "Disp", 0)
pb.bc.add("Dirichlet", nodes_sphere_top, "Disp", [0, 0, imposed_disp])
pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=8)

# =========================================================================
# Output and tracking callback
# =========================================================================

if not os.path.isdir("results"):
    os.mkdir("results")
res = pb.add_output("results/indentation_3d", solid, ["Disp", "Stress"])

history = {"time": [], "reaction_z": []}


def track(pb):
    history["time"].append(pb.time)
    F = pb.get_ext_forces("Disp")
    history["reaction_z"].append(np.sum(F[2, nodes_sphere_top]))


# =========================================================================
# Solve
# =========================================================================

print("=" * 60)
print("3D SPHERE INDENTATION -- OGC TRUST-REGION")
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
    Fz = -np.array(history["reaction_z"])

    delta = np.maximum(t * abs(imposed_disp) - gap, 0)

    E_star = 1.0 / ((1 - nu**2) / E_plate + (1 - nu**2) / E_sphere)

    delta_hertz = np.linspace(0, delta.max(), 200)
    F_hertz = (4.0 / 3.0) * E_star * np.sqrt(R) * delta_hertz**1.5

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(delta, Fz, "o-", ms=4, label="FEM (OGC)")
    ax.plot(delta_hertz, F_hertz, "--", lw=2, label="Hertz (3D half-space)")
    ax.set_xlabel("Indentation depth (mm)")
    ax.set_ylabel("Force (N)")
    ax.set_title("3D Hertz Indentation -- OGC Trust-Region")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/indentation_3d_hertz.png", dpi=150)
    print("Hertz comparison saved to results/indentation_3d_hertz.png")
    plt.show()
except ImportError:
    print("matplotlib not available -- skipping Hertz comparison plot")

# =========================================================================
# Stress plot
# =========================================================================

res.plot("Stress", "vm", "Node", show=False, scale=1,
         elevation=75, azimuth=20)

# --- Video output (uncomment to export MP4) ---
# res.write_movie("results/indentation_3d", "Stress", "vm", "Node",
#                 framerate=10, quality=8, elevation=75, azimuth=20)
# print("Movie saved to results/indentation_3d.mp4")
