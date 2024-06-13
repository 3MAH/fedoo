import time

import numpy as np

import fedoo as fd
from fedoo.util.abaqus_inp import ReadINP

# ------------------------------------------------------------------------------
# Défine inplane 2D periodic boundary conditions for a composite ply using a 3D
# unit cell. Don't work very well because the mesh is not periodic.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Dimension of the problem
# ------------------------------------------------------------------------------
fd.ModelingSpace("3D")

# ------------------------------------------------------------------------------
# Definition of the Geometry
# ------------------------------------------------------------------------------
# INP = Util.ReadINP('Job-1.inp')

# warning: this mesh is not periodic and should be replaced by a better one !
INP = ReadINP("cell_taffetas.inp")
mesh = INP.toMesh()

E_fiber = 250e3
E_matrix = 4e3
nu_fiber = 0.3
nu_matrix = 0.33

# E_fiber = E_matrix
# nu_fiber = nu_matrix

list_elm_matrix = mesh.element_sets["alla_matrix"]

# ------------------------------------------------------------------------------
# Set of nodes for boundary conditions
# ------------------------------------------------------------------------------
center = mesh.nearest_node(mesh.bounding_box.center)

# ------------------------------------------------------------------------------
# Adding virtual nodes related the macroscopic strain
# ------------------------------------------------------------------------------
strain_nodes = mesh.add_virtual_nodes()
# The position of the virtual node has no importance (the position is arbitrary set to [0,0,0])
# For a problem in 3D with a 2D periodicity, we have 3 strain component that can be represented with only one node
# In case of a 2D problem, 2 nodes will be required

# ------------------------------------------------------------------------------
# Material definition
# ------------------------------------------------------------------------------
E = E_fiber * np.ones(mesh.n_elements)
nu = nu_fiber * np.ones(mesh.n_elements)
E[list_elm_matrix] = E_matrix
nu[list_elm_matrix] = nu_matrix


fd.constitutivelaw.ElasticIsotrop(E, nu, name="ElasticLaw")

# ------------------------------------------------------------------------------
# Mechanical weak formulation
# ------------------------------------------------------------------------------
wf = fd.weakform.StressEquilibrium("ElasticLaw")

# ------------------------------------------------------------------------------
# Global Matrix assembly
# ------------------------------------------------------------------------------
assemb = fd.Assembly.create(wf, mesh)

# ------------------------------------------------------------------------------
# Static problem based on the just defined assembly
# ------------------------------------------------------------------------------
pb = fd.problem.Linear(assemb)

# ------------------------------------------------------------------------------
# Boundary conditions
# ------------------------------------------------------------------------------
E = [1, 0, 0]  # macroscopic strain tensor [EXX, EYY, EXY]
# For the node StrainNode[0], 'DispX' is a virtual dof for EXX
# For the node StrainNode[0], 'DispY' is a virtual dof for EYY
# For the node StrainNode[0], 'DispZ' is a virtual dof for EXY


# Apply the periodic boundary conditions
pb.bc.add(
    fd.constraint.PeriodicBC(
        [strain_nodes[0], strain_nodes[0], strain_nodes[0]],
        ["DispX", "DispY", "DispY"],
        dim=2,
    )
)

# Homogen.DefinePeriodicBoundaryCondition("Domain",
#         [StrainNodes[0], StrainNodes[0], StrainNodes[0]],
#         ['DispX', 'DispY', 'DispZ'], dim='2d')

# Block a node on the center to avoid rigid body motion
pb.bc.add("Dirichlet", center, "Disp", 0)
pb.bc.add("Dirichlet", strain_nodes, "Disp", E)  # apply specified macro strain


pb.apply_boundary_conditions()

# ------------------------------------------------------------------------------
# Solve
# ------------------------------------------------------------------------------
pb.solve()

# ------------------------------------------------------------------------------
# Post-treatment
# ------------------------------------------------------------------------------
res = pb.get_results(assemb, ["Stress", "Strain", "Disp"])


# Compute the mean stress and strain
# Get the stress tensor (PG values)


volume = mesh.bounding_box.volume  # total volume of the bounding_box
mean_stress = pb.get_ext_forces("Disp")[:, strain_nodes[0]] / volume
# or from the definition:
# mean_stress = [1/volume*mesh.integrate_field(res['Stress'][i]) for i in range(6)]

mean_strain = pb.get_disp()[:, strain_nodes[0]]
# or from the definition (only work if volume with no void because cant comput strain of voids):
# mean_strain = [1/volume*mesh.integrate_field(res['Strain'][i]) for i in range(6)]
# print(fd.ConstitutiveLaw['ElasticLaw'].get_elastic_matrix()@np.array(mean_strain)) #should be the same as MeanStress if homogeneous material and no void

print("Strain tensor ([Exx, Eyy, Ezz, Exy, Exz, Eyz]): ", mean_strain)
print("Stress tensor ([Sxx, Syy, Szz, Sxy, Sxz, Syz]): ", mean_strain)

print("Elastic Energy: " + str(pb.GetElasticEnergy()))

# ------------------------------------------------------------------------------
# Optional: Write data in a vtk file (for visualization with paraview for instance)
# ------------------------------------------------------------------------------
# pb.get_results(assemb, ['Stress','Strain','Disp'], 'Node').save('composite_ply_periodic.vtk', True)


# ------------------------------------------------------------------------------
# Optional: Get data for fibers only and plot results
# ------------------------------------------------------------------------------
res_fibers = pb.get_results(
    assemb, ["Stress", "Strain", "Disp"], element_set="all_fibers"
)
res_fibers.plot("Stress", "XX")
