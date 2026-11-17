r"""
2D plate with hole in tension
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A square plate (100 x 100 mm) with a central hole of radius 20 mm is
stretched along :math:`x` by prescribed displacements on its left and right
edges, under the 2D plane stress assumption. The von Mises stress and the
:math:`\sigma_{xx}`, :math:`\sigma_{yy}`, :math:`\sigma_{xy}` components are
plotted in a 2 x 2 :class:`pyvista.Plotter`. The stress concentrates at the top
and bottom of the hole: the concentration factor is about 3 for a small hole in
an infinite plate, and larger relative to the far-field stress here because the
hole is wide (diameter 0.4 times the plate width) and the mesh is coarse.
"""

import fedoo as fd
import pyvista as pv

# Define 2d modeling space using plane stress assumption
fd.ModelingSpace("2Dstress")

# Generate a simple structured mesh "Domain" (plate with a hole).
mesh = fd.mesh.hole_plate_mesh(
    nr=11,
    nt=11,
    length=100,
    height=100,
    radius=20,
    elm_type="quad4",
    sym=False,
    name="Domain",
)

# A plate with a hole is also available as a gmsh file in this folder. A mesh read
# from a file is in 3D (3 coordinates): use
# ``fd.mesh.import_file("plate_with_hole.msh").as_2d()`` to drop the third one.

# Define an elastic isotropic material with E = 2e5 MPa and nu = 0.3 (steel)
material = fd.constitutivelaw.ElasticIsotrop(2e5, 0.3, name="ElasticLaw")

# Create the weak formulation of the mechanical equilibrium equation
fd.weakform.StressEquilibrium("ElasticLaw", name="WeakForm")

# Create a global assembly
fd.Assembly.create("WeakForm", mesh, name="Assembly")

# Define a new static problem
pb = fd.problem.Linear("Assembly")

# Definition of the set of nodes for boundary conditions
left = mesh.find_nodes("X", mesh.bounding_box.xmin)
right = mesh.find_nodes("X", mesh.bounding_box.xmax)

# displacement on left (ux=-0.5mm)
pb.bc.add("Dirichlet", "left", "DispX", -5e-1)
# displacement on right (ux=0.5mm)
pb.bc.add("Dirichlet", "right", "DispX", 5e-1)
# y displacement set in one node to avoid rigid body motion
pb.bc.add("Dirichlet", [0], "DispY", 0)

# Solve problem
pb.set_solver("CG")
pb.solve()

# extract the results from the Assembly object
results = pb.get_results(assemb="Assembly", output_list=["Stress", "Disp", "Strain"])

###############################################################################
# Plot results
# ------------
# The pyvista subplot capability is used to plot in the same figure:
#
# - the von Mises stress,
# - the XX, YY and XY components of the stress tensor.

pl = pv.Plotter(shape=(2, 2))

results.plot("Stress", "vm", "Node", plotter=pl)
pl.subplot(1, 0)
results.plot("Stress", "XX", "Node", plotter=pl)
pl.subplot(0, 1)
results.plot("Stress", "YY", "Node", plotter=pl)
pl.subplot(1, 1)
results.plot("Stress", "XY", "Node", plotter=pl)
pl.show()
