r"""
Compression of a ping-pong ball
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Example that shows how to use plate (shell) elements with a pressure load in a
geometrically nonlinear analysis.

A ping-pong ball (diameter 40 mm, wall thickness 0.45 mm, celluloid-like
material with :math:`E = 2` GPa) is squeezed by a pressure applied on its two
poles. Each pole carries a resultant of about 175 N. The deflection is of the
order of the wall thickness, which is where the geometric nonlinearity of a
thin shell becomes significant, while the strains remain small (below 1 %).
"""

import fedoo as fd
import pyvista as pv
import numpy as np

###############################################################################
# The problems parameters (units: mm, N, MPa)

E = 2e3  # MPa
nu = 0.37
radius = 20  # mm
thickness = 0.45  # mm
pressure = 0.5  # MPa

###############################################################################
# Create a simple sphere mesh using pyvista.

mesh = fd.Mesh.from_pyvista(pv.Sphere(radius=radius))

###############################################################################
# Define the material and an homogeneous shell section.
#
# The shell kinematics are corotational: large rotations are handled by the
# weak form, and the material only sees the small strains of the corotated
# frame. At such strains any hyperelastic law (Neo-Hookean, ...) reduces to
# its linearization, i.e. the isotropic linear elastic law with the same
# :math:`E` and :math:`\nu` used here.
#
# .. note::
#   Finite-strain hyperelastic laws (simcoon ``NEOHC``, ...) need the full
#   deformation gradient and are not available with the plane-stress
#   assumption of shell elements. Use a solid model for large-strain
#   (e.g. rubber) shells.

material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="Material")
shell_section = fd.constitutivelaw.ShellHomogeneous("Material", thickness)

###############################################################################
# Define the weakform and associated assembly for plate model.
# For plate elements, we first need to create a 3D modeling space.
# ``nlgeom=True`` activates the corotational geometric nonlinearities.

fd.ModelingSpace("3D")
wf = fd.weakform.PlateEquilibrium(shell_section, nlgeom=True)
solid_assembly = fd.Assembly.create(wf, mesh)

###############################################################################
# Select mesh elements where we will apply the pressure.
# The mesh.find_elements method is used with an arbitrary expression.
# Here we select all elements whose z coordinates are less than 3mm from
# minimal or maximal z value (sphere extremities along the z axis).
#
# The loaded caps have a height :math:`h = 3` mm, hence a projected area
# :math:`\pi a^2 = \pi (2Rh - h^2) \approx 349\ \mathrm{mm}^2`, and a resultant force
# :math:`F = p \pi a^2 \approx 175` N on each pole.

boundaries = mesh.find_elements(
    f"Z>{mesh.bounding_box.zmax - 3} or Z<{mesh.bounding_box.zmin + 3}"
)

###############################################################################
# Now we build the pressure load by extracting the loaded surface mesh.
# The pressure is then added to the problem as an external (Neumann) boundary
# condition with ``pb.bc.add``. With geometric nonlinearities, the pressure is
# a follower load (it stays normal to the deformed surface).

pressure_assembly = fd.constraint.Pressure(
    mesh.extract_elements(boundaries),
    pressure,
)

###############################################################################
# Define a nonlinear analysis and solve the problem.
#
# Order of magnitude: in the loaded caps the shell is in a membrane state
# close to that of a sphere under external pressure,
# :math:`\sigma \approx pR/(2t) \approx 11` MPa, i.e. an equibiaxial membrane
# strain :math:`\sigma(1-\nu)/E \approx 0.35` %,
# with additional bending at the edge of the loaded zones. Both stay well
# below the yield stress of the material (about 35 MPa, see the elastoplastic
# version of this example in :ref:`sphx_glr_examples_shells_spherical_shell_compression_plastic.py`).
#
# .. note::
#   To improve numerical stability, a few displacement boundary conditions are
#   added to remove rigid-body motions. These constraints do not affect the
#   strain/stress solution because they only suppress the nullspace modes.
#   Without them, the unconstrained problem is singular; some solvers may still
#   return a usable strain/stress field, but the displacement field can contain
#   arbitrary rigid-body motion.
#
assembly = solid_assembly

pb = fd.problem.NonLinear(assembly, nlgeom=True)
pb.bc.add(pressure_assembly)

nodes = mesh.nodes
node_a = int(np.argmin(nodes[:, 0]))
node_b = int(np.argmax(nodes[:, 0]))
node_c = int(np.argmax(nodes[:, 1]))

pb.bc.add("Dirichlet", node_a, "Disp", 0)
pb.bc.add("Dirichlet", node_b, ["DispY", "DispZ"], 0)
pb.bc.add("Dirichlet", node_c, "DispZ", 0)

pb.set_nr_criterion("Displacement", tol=5e-3, max_subiter=15)
pb.nlsolve(dt=0.25, tmax=1, update_dt=True, print_info=1)

###############################################################################
# Extract the results:
# position = 1 is set for the surface along the positive direction of the
# normal vector (0 is the mean plane). The strains and stresses components are
# defined in the element local coordinate system
# (mesh.get_element_local_frame()).

res = pb.get_results(solid_assembly, ["Disp", "Rot", "Stress", "Strain"], position=1)
pl = pv.Plotter()
res.plot("Stress", component="XX", data_type="Node", plotter=pl)
pl.view_isometric()
pl.show()
