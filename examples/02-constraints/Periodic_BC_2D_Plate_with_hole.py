"""
2D periodic boundary condition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Periodic boundary conditions are enforced on a 2D geometry
with plane stress assumption (plate with hole).
A mean strain tensor is enforced, and the resulting mean stress is
estimated.
"""

import fedoo as fd
import numpy as np

###############################################################################
# Dimension of the problem
# ------------------------------------------------------------------------------
fd.ModelingSpace("2Dstress")

###############################################################################
# Definition of the Geometry
# ------------------------------------------------------------------------------
mesh = fd.mesh.hole_plate_mesh(name="Domain")

# alternative mesh below (uncomment the line)
# Mesh.rectangle_mesh(Nx=51, Ny=51, x_min=-50, x_max=50, y_min=-50, y_max=50, ElementShape = 'quad4', name ="Domain")

###############################################################################
# Adding virtual nodes related the macroscopic strain
# ------------------------------------------------------------------------------
strain_nodes = mesh.add_virtual_nodes(2)
# The position of the virtual node has no importance.
# For a problem in 2D with a 2D periodicity, we need 3 independant strain component
# 2 nodes (with 2 dof per node in 2D) are required

###############################################################################
# Now define the problem to solve

# ------------------------------------------------------------------------------
# Material definition
# ------------------------------------------------------------------------------
fd.constitutivelaw.ElasticIsotrop(1e5, 0.3, name="ElasticLaw")

# ------------------------------------------------------------------------------
# Mechanical weak formulation
# ------------------------------------------------------------------------------
wf = fd.weakform.StressEquilibrium("ElasticLaw")

# ------------------------------------------------------------------------------
# Global Matrix assembly
# ------------------------------------------------------------------------------
fd.Assembly.create(wf, mesh, name="Assembly")

# ------------------------------------------------------------------------------
# Static problem based on the just defined assembly
# ------------------------------------------------------------------------------
pb = fd.problem.Linear("Assembly")

###############################################################################
# Add periodic constraint
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add a periodic conditions (ie a multipoint constraint) linked to the strain dof based on virtual nodes:
#  - the dof 'DispX' of the node strain_nodes[0] will be arbitrary associated to the EXX strain component
#  - the dof 'DispY' of the node strain_nodes[1] will be arbitrary associated to the EYY strain component
#  - the dof 'DispY' of the node strain_nodes[0] will be arbitrary associated to the EXY strain component
#  - the dof 'DispX' of the node strain_nodes[1] is not used and will be blocked to avoid singularity
pb.bc.add(
    fd.constraint.PeriodicBC(
        [strain_nodes[0], strain_nodes[1], strain_nodes[0]], ["DispX", "DispY", "DispY"]
    )
)

###############################################################################
# Add standard boundary conditions

# ------------------------------------------------------------------------------
# Macroscopic strain components to enforce
Exx = 0
Eyy = 0
Exy = 0.1

# Mean strain: Dirichlet (strain) or Neumann (associated mean stress) can be enforced
pb.bc.add("Dirichlet", [strain_nodes[0]], "DispX", Exx)  # EpsXX
pb.bc.add("Dirichlet", [strain_nodes[0]], "DispY", Exy)  # EpsXY

pb.bc.add(
    "Dirichlet", [strain_nodes[1]], "DispX", 0
)  # nothing (blocked to avoir singularity)
pb.bc.add("Dirichlet", [strain_nodes[1]], "DispY", Eyy)  # EpsYY

# Block one node to avoid singularity
center = mesh.nearest_node(mesh.bounding_box.center)
pb.bc.add("Dirichlet", center, "Disp", 0)

###############################################################################
# Solve and plot stress field
pb.solve()

# ------------------------------------------------------------------------------
# Post-treatment
# ------------------------------------------------------------------------------
res = pb.get_results("Assembly", ["Disp", "Stress"])

# plot the deformed mesh with the shear stress (component=3).
res.plot("Stress", "XY", "Node")
# simple matplotlib alternative if pyvista is not installed:
# fd.util.field_plot_2d("Assembly", disp = pb.get_dof_solution(), dataname = 'Stress', component=3, scale_factor = 1, plot_edge = True, nb_level = 6, type_plot = "smooth")

###############################################################################
# print the macroscopic strain tensor and stress tensor
print(
    "Strain tensor ([Exx, Eyy, Exy]): ",
    [pb.get_disp("DispX")[-2], pb.get_disp("DispY")[-1], pb.get_disp("DispY")[-2]],
)

# Compute the mean stress tensor
surf = mesh.bounding_box.volume  # total surface of the domain = volume in 2d
mean_stress = [1 / surf * mesh.integrate_field(res["Stress"][i]) for i in [0, 1, 3]]

print("Stress tensor ([Sxx, Syy, Sxy]): ", mean_stress)
