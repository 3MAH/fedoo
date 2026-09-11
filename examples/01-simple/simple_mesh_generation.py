"""
Creating simple meshes and swept geometries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fedoo provides structured mesh generators for common shapes as well as
operations for building new geometries from existing meshes. This example
starts with a few built-in meshes, then creates solids and curved structures
using extrusion, local frames, and surface thickening.
"""

import numpy as np
import pyvista as pv

import fedoo as fd


###############################################################################
# Built-in two-dimensional meshes
# --------------------------------
# Rectangles, disks, plates with holes, and I-shaped profiles can be generated
# directly. Here, a plate with a circular hole and a hollow disk illustrate
# two of the available structured generators.

plate = fd.mesh.hole_plate_mesh(
    nr=7,
    nt=7,
    length=4.0,
    height=3.0,
    radius=0.6,
    elm_type="quad4",
)

annulus = fd.mesh.hollow_disk_mesh(
    radius=1.5,
    thickness=0.45,
    nr=5,
    nt=33,
    elm_type="quad4",
)


###############################################################################
# Straight extrusion
# ------------------
# Extruding a surface mesh along a scalar distance creates a volume mesh. The
# number of nodes controls the discretization through the extrusion direction.

disk = fd.mesh.disk_mesh(radius=1.0, nr=7, nt=9, elm_type="quad4")
cylinder = fd.mesh.extrude(disk, 1.8, n_nodes=6)


###############################################################################
# Sweep along a curved path
# -------------------------
# When the extrusion path is a line mesh, one local frame can be supplied at
# every path node. The frame axes orient the profile as it travels along the
# path. Cylindrical frames conveniently orient a straight profile around a
# circular path, producing a smooth cylindrical surface.

profile = fd.mesh.line_mesh(7, -0.45, 0.45, elm_type="lin2")
circle_path = fd.mesh.circle_mesh(n=40, r=1.5, ndim=3)
circle_frames = fd.mesh.generate_cylindrical_local_frame(circle_path)

curved_surface = fd.mesh.extrude(
    profile,
    circle_path,
    local_frame=circle_frames,
)


###############################################################################
# Surface thickening
# ------------------
# A shell-like surface can be converted into a solid using area-weighted nodal
# normals computed automatically by :func:`fedoo.mesh.thicken`.

thick_ring = fd.mesh.thicken(curved_surface, thickness=0.18, n_nodes=3)


###############################################################################
# Helicoidal extrusion
# --------------------
# For a spatial curve, a custom frame gives direct control over the profile
# orientation. The first axis below follows the helix tangent, the second is
# radial, and the third completes a right-handed orthonormal frame.

i_profile = fd.mesh.I_shape_mesh(
    height=0.65,
    width=0.5,
    web_thickness=0.12,
    flange_thickness=0.12,
    size_elm=0.1,
)

theta_max = 5 * np.pi
helix_height = 2.8
helix_radius = 1.15
helix_path = fd.mesh.line_mesh_cylindric(
    56,
    helix_radius,
    0,
    theta_max,
).as_3d()
helix_path.nodes[:, 2] = np.linspace(0, helix_height, helix_path.n_nodes)

cylindrical_frame = fd.mesh.generate_cylindrical_local_frame(helix_path)
radial = cylindrical_frame[:, 0]
azimuthal = cylindrical_frame[:, 1]
axial = cylindrical_frame[:, 2]

pitch = helix_height / theta_max
tangent = helix_radius * azimuthal + pitch * axial
tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
profile_normal = np.cross(tangent, radial)
helix_frame = np.stack((tangent, radial, profile_normal), axis=1)

helicoidal_beam = fd.mesh.extrude(
    i_profile,
    helix_path,
    local_frame=helix_frame,
)


###############################################################################
# Visualize the generated meshes
# ------------------------------
# Fedoo meshes convert directly to PyVista datasets. Displaying the geometries
# together highlights how the same small set of generators and transformations
# can produce surfaces, solids, and smoothly swept structures.

plotter = pv.Plotter(shape=(2, 3), window_size=(1200, 760))
plotter.set_background("white")

geometries = (
    (plate, "Plate with a hole", "#4C78A8", "xy"),
    (annulus, "Hollow disk", "#72B7B2", "xy"),
    (cylinder, "Extruded disk", "#F2CF5B", "iso"),
    (curved_surface, "Profile swept on a circle", "#B279A2", "iso"),
    (thick_ring, "Thickened curved surface", "#F28E2B", "iso"),
    (helicoidal_beam, "Helicoidal I-beam", "#E15759", "iso"),
)

for index, (mesh, title, color, view) in enumerate(geometries):
    plotter.subplot(*divmod(index, 3))
    plotter.add_mesh(
        mesh.to_pyvista(),
        color=color,
        show_edges=True,
        edge_color="#303030",
        line_width=0.6,
    )
    plotter.add_text(title, font_size=11, color="#202020")
    if view == "xy":
        plotter.view_xy()
    else:
        plotter.view_isometric()
    plotter.camera.zoom(1.2)

plotter.show()
