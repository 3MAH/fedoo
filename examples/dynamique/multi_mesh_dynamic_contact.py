"""
Dynamic MultiMesh example: shell ping-pong ball dropped on a solid plate.

The example combines:
    - a shell sphere meshed with ``tri3`` elements,
    - a solid plate meshed with ``hex8`` elements,
    - a ``MultiMesh`` obtained with ``mesh_sphere + mesh_plate``,
    - gravity as a distributed load on the shell sphere,
    - IPC contact on the sphere and the top surface of the plate,
    - FDH5 time output from the structural ``AssemblySum``.

The discretisation is intentionally modest so the file stays usable as an
example. Increase the sphere resolution, plate mesh density, and time horizon
for production-quality results.
"""

import os

import fedoo as fd
import numpy as np
import pyvista as pv


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


###############################################################################
# Parameters
#

# Units: N, mm, tonne, s, MPa = N/mm2
E = 2.0e3
NU = 0.37
DENSITY = 10.0e-9  # tonne / mm3
GRAVITY = 9.81e3  # mm / s2

RADIUS = 20.0
THICKNESS = 0.45
GAP = 30  # 0.5

PLATE_X = 100.0
PLATE_Y = 100.0
PLATE_H = 2.0

SPHERE_CENTER = np.array([0.5 * PLATE_X, 0.5 * PLATE_Y, PLATE_H + RADIUS + GAP])


###############################################################################
# Mesh
#

sphere = pv.Sphere(
    radius=RADIUS,
    center=SPHERE_CENTER,
    theta_resolution=24,
    phi_resolution=24,
)
sphere_mesh = fd.Mesh.from_pyvista(sphere)
sphere_mesh.element_sets["sphere"] = np.arange(sphere_mesh.n_elements)

plate_mesh = fd.mesh.box_mesh(
    20,
    20,
    3,
    x_max=PLATE_X,
    y_max=PLATE_Y,
    z_max=PLATE_H,
)
plate_mesh.element_sets["plate"] = np.arange(plate_mesh.n_elements)

mesh = sphere_mesh + plate_mesh
print(f"MultiMesh: {mesh.n_nodes} nodes, {mesh.n_elements} elements, {mesh.elm_types}")


###############################################################################
# Models and structural assemblies
#

fd.ModelingSpace("3D")

material = fd.constitutivelaw.ElasticIsotrop(E, NU)
material.set_density(DENSITY)

shell_section = fd.constitutivelaw.ShellHomogeneous(material, THICKNESS)
shell_wf = fd.weakform.PlateEquilibrium(shell_section, nlgeom=True)
# # Add numerical damping
# shell_wf += fd.weakform.ArtificialDamping(
#     c_stab=0.001,          # target about 1% dissipated energy
#     energy_fraction=True,
#     variables=["Disp", "Rot"],
# )
sphere_assembly = fd.Assembly.create(shell_wf, mesh[0], name="sphere_shell")

solid_wf = fd.weakform.StressEquilibrium(material, nlgeom=True)
plate_assembly = fd.Assembly.create(solid_wf, mesh[1], name="plate_solid")

structural_assembly = sphere_assembly + plate_assembly


###############################################################################
# Gravity and IPC contact
#

gravity_force = [0.0, 0.0, -DENSITY * THICKNESS * GRAVITY]
gravity = fd.constraint.DistributedForce(
    mesh[0],
    gravity_force,
    nlgeom=True,
    time_func=lambda t: 1.0,
    name="sphere_gravity",
)

# IPC contact in 3D expects a tri3 surface mesh with node ids matching the full
# mechanical mesh. The shell sphere already is a surface mesh; the plate uses
# only its top face, converted from quad4 to tri3.
plate_top_nodes = mesh[1].find_nodes("Z", PLATE_H)
plate_surface = fd.mesh.extract_surface(
    mesh[1],
    node_set=plate_top_nodes,
    quad2tri=True,
)
ipc_surface = fd.Mesh(
    mesh.nodes,
    np.vstack((mesh[0].elements, plate_surface.elements)),
    "tri3",
    ndim=3,
)

ipc_contact = fd.constraint.IPCContact(
    mesh,
    surface_mesh=ipc_surface,
    dhat=0.5,
    dhat_is_relative=False,
    friction_coefficient=0.0,
    use_ccd=True,
)

###############################################################################
# Problem, boundary conditions, and output
#

pb = fd.problem.NonLinearNewmark(
    structural_assembly + ipc_contact,
    nlgeom=True,
)
# # or with GeneralizedAlpha with high frequency damping:
# pb = fd.problem.NonLinear(structural_assembly + ipc_contact, nlgeom=True)
# pb.set_time_integrator(
#     fd.time.SECOND_ORDER,
#     fd.time.GeneralizedAlpha(alpha_m=0.0, alpha_f=1.0 / 3.0),
# )

plate_nodes = np.unique(mesh[1].elements)
plate_xyz = mesh.nodes[plate_nodes]
plate_support = plate_nodes[
    np.isclose(plate_xyz[:, 2], 0.0)
    | np.isclose(plate_xyz[:, 0], 0.0)
    | np.isclose(plate_xyz[:, 0], PLATE_X)
    | np.isclose(plate_xyz[:, 1], 0.0)
    | np.isclose(plate_xyz[:, 1], PLATE_Y)
]

pb.bc.add(gravity)
pb.bc.add("Dirichlet", plate_support, "Disp", 0)
pb.bc.add("Dirichlet", plate_nodes, "Rot", 0)

results = pb.add_output(
    os.path.join(RESULTS_DIR, "ping_pong_multimesh"),
    structural_assembly,
    ["Disp", "Stress", "Strain", "Acceleration"],
    file_format="fdh5",
)


###############################################################################
# Solve and post-process
#

sphere_nodes = np.unique(mesh[0].elements)
history = {"time": [], "sphere_z": [], "contact_force_norm": []}


def track(pb):
    disp = pb.get_disp()
    history["time"].append(pb.time)
    history["sphere_z"].append(
        np.mean(mesh.nodes[sphere_nodes, 2] + disp[2, sphere_nodes])
    )
    history["contact_force_norm"].append(np.linalg.norm(ipc_contact.global_vector))


pb.set_nr_criterion("Force", tol=5e-2, max_subiter=20)
pb.nlsolve(
    dt=1.0e-3,
    tmax=2e-1,
    dt_min=2.0e-5,
    update_dt=True,
    print_info=1,
    # interval_output=2.0e-3,  # same as dt by default
    callback=track,
    exec_callback_at_each_iter=False,
)

# results.load(-1)
# results.plot("Stress", "vm", "Node", scale=1.0)
# fd.viewer(results)
