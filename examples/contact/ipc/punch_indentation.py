"""
Hemispherical punch indentation (3D, IPC method)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A cylindrical punch with a hemispherical tip is pressed into a thick
elastic plate.  Both bodies deform visibly at the contact interface --
a classic contact mechanics benchmark.

Both bodies are meshed with gmsh for proper curved geometry.

This example uses the IPC contact method which handles the curved
punch geometry robustly.

.. note::
   Requires ``ipctk`` and ``gmsh``.
"""
import fedoo as fd
import numpy as np
import os
import tempfile

os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")

fd.ModelingSpace("3D")

E_plate, E_punch, nu = 1e4, 3e4, 0.3
R_punch = 3.0       # hemisphere radius
H_cyl = 5.0         # cylinder height above hemisphere
plate_half = 10.0    # plate half-width
plate_h = 10.0       # plate thickness
gap = 0.05           # initial gap

# Punch bottom (hemisphere apex) sits at z = plate_h + gap
punch_base_z = plate_h + gap

# =========================================================================
# Punch mesh (gmsh: hemisphere + cylinder)
# =========================================================================

import gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 1)

# Hemisphere: centred at (0, 0, punch_base_z + R), only bottom half
sphere_tag = gmsh.model.occ.addSphere(0, 0, punch_base_z + R_punch, R_punch)
# Cut box removes the top half of the sphere
cut_box = gmsh.model.occ.addBox(
    -R_punch - 1, -R_punch - 1, punch_base_z + R_punch,
    2 * (R_punch + 1), 2 * (R_punch + 1), R_punch + 1,
)
hemi = gmsh.model.occ.cut([(3, sphere_tag)], [(3, cut_box)])[0]
hemi_tag = hemi[0][1]

# Cylinder on top of hemisphere
cyl_tag = gmsh.model.occ.addCylinder(
    0, 0, punch_base_z + R_punch,
    0, 0, H_cyl,
    R_punch,
)

# Fuse hemisphere + cylinder into one volume
punch_parts = gmsh.model.occ.fuse([(3, hemi_tag)], [(3, cyl_tag)])[0]
punch_vol_tag = punch_parts[0][1]
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(3, [punch_vol_tag], tag=2, name="punch")

# Mesh size: fine at the tip, coarser on the cylinder
gmsh.model.mesh.field.add("Box", 1)
gmsh.model.mesh.field.setNumber(1, "VIn", 0.4)
gmsh.model.mesh.field.setNumber(1, "VOut", 1.5)
gmsh.model.mesh.field.setNumber(1, "XMin", -R_punch)
gmsh.model.mesh.field.setNumber(1, "XMax", R_punch)
gmsh.model.mesh.field.setNumber(1, "YMin", -R_punch)
gmsh.model.mesh.field.setNumber(1, "YMax", R_punch)
gmsh.model.mesh.field.setNumber(1, "ZMin", punch_base_z)
gmsh.model.mesh.field.setNumber(1, "ZMax", punch_base_z + R_punch)
gmsh.model.mesh.field.setNumber(1, "Thickness", 1)
gmsh.model.mesh.field.setAsBackgroundMesh(1)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

gmsh.model.mesh.generate(3)
punch_msh = os.path.join(tempfile.gettempdir(), "punch_hemi.msh")
gmsh.write(punch_msh)
gmsh.finalize()

mesh_punch = fd.mesh.import_msh(punch_msh)
mesh_punch.element_sets["punch"] = mesh_punch.element_sets.get(
    "punch", np.arange(mesh_punch.n_elements)
)
print(f"Punch mesh: {mesh_punch.n_nodes} nodes, {mesh_punch.n_elements} tet4")

# =========================================================================
# Plate mesh (gmsh)
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
gmsh.model.mesh.field.setNumber(1, "VIn", 0.5)
gmsh.model.mesh.field.setNumber(1, "VOut", 4.0)
gmsh.model.mesh.field.setNumber(1, "XMin", -4)
gmsh.model.mesh.field.setNumber(1, "XMax", 4)
gmsh.model.mesh.field.setNumber(1, "YMin", -4)
gmsh.model.mesh.field.setNumber(1, "YMax", 4)
gmsh.model.mesh.field.setNumber(1, "ZMin", plate_h - 4)
gmsh.model.mesh.field.setNumber(1, "ZMax", plate_h)
gmsh.model.mesh.field.setNumber(1, "Thickness", 2)
gmsh.model.mesh.field.setAsBackgroundMesh(1)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)

gmsh.model.mesh.generate(3)
plate_msh = os.path.join(tempfile.gettempdir(), "plate_punch.msh")
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

mesh = fd.Mesh.stack(mesh_plate, mesh_punch)
print(f"Total mesh:  {mesh.n_nodes} nodes, {mesh.n_elements} tet4")

# =========================================================================
# IPC contact, material, BCs
# =========================================================================

nodes_bottom = mesh.find_nodes("Z", 0)
nodes_punch_top = mesh.find_nodes("Z", mesh.bounding_box.zmax)

surf_ipc = fd.mesh.extract_surface(mesh, quad2tri=True)
ipc_contact = fd.constraint.IPCContact(
    mesh, surface_mesh=surf_ipc,
    dhat=1e-2, dhat_is_relative=True,
    use_ccd=True,
)

mat = fd.constitutivelaw.Heterogeneous(
    (fd.constitutivelaw.ElasticIsotrop(E_plate, nu),
     fd.constitutivelaw.ElasticIsotrop(E_punch, nu)),
    ("plate", "punch"),
)
wf = fd.weakform.StressEquilibrium(mat, nlgeom=True)
solid = fd.Assembly.create(wf, mesh)
assembly = fd.Assembly.sum(solid, ipc_contact)

pb = fd.problem.NonLinear(assembly)

if not os.path.isdir("results"):
    os.mkdir("results")
res = pb.add_output("results/punch_ipc", solid, ["Disp", "Stress"])

pb.bc.add("Dirichlet", nodes_bottom, "Disp", 0)
pb.bc.add("Dirichlet", nodes_punch_top, "Disp", [0, 0, -2.0])
pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=8)

print("=" * 60)
print("HEMISPHERICAL PUNCH -- IPC CONTACT")
print("=" * 60)
pb.nlsolve(dt=0.05, tmax=1, update_dt=True, print_info=1,
           interval_output=0.05)

# --- Static plot ---
res.plot("Stress", "vm", "Node", show=False, scale=1,
         elevation=75, azimuth=20)

# --- Video output (uncomment to export MP4) ---
# res.write_movie("results/punch_ipc", "Stress", "vm", "Node",
#                 framerate=10, quality=8, elevation=75, azimuth=20)
# print("Movie saved to results/punch_ipc.mp4")
