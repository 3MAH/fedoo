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
E = [0.1, 0, 0]  # macroscopic strain tensor [EXX, EYY, EXY]

# Apply the periodic boundary conditions
pb.bc.add(
    fd.constraint.PeriodicBC(
        periodicity_type="small_strain",
        dim=2,
    )
)

# Block a node on the center to avoid rigid body motion
pb.bc.add("Dirichlet", center, "Disp", 0)
pb.bc.add("Dirichlet", "MeanStrain", E)  # apply specified macro strain

# ------------------------------------------------------------------------------
# Solve
# ------------------------------------------------------------------------------
pb.solve()

# ------------------------------------------------------------------------------
# Post-treatment
# ------------------------------------------------------------------------------
res = pb.get_results(assemb, ["Stress", "Strain", "Disp", "MeanStrain"])


# Compute the mean stress and strain
# Get the stress tensor (PG values)


volume = mesh.bounding_box.volume  # total volume of the bounding_box
mean_stress = pb.get_ext_forces("MeanStrain") / volume
# or from the definition:
# mean_stress = [1/volume*mesh.integrate_field(res['Stress'][i]) for i in range(6)]

mean_strain = pb.get_disp("MeanStrain")
# or from the definition (only work if volume with no void because cant compute strain of voids):
# mean_strain = [1/volume*mesh.integrate_field(res['Strain'][i]) for i in range(6)]
# print(fd.ConstitutiveLaw['ElasticLaw'].get_elastic_matrix()@np.array(mean_strain)) #should be the same as MeanStress if homogeneous material and no void

print("Strain tensor ([Exx, Eyy, Ezz, Exy, Exz, Eyz]): ", mean_strain)
print("Stress tensor ([Sxx, Syy, Szz, Sxy, Sxz, Syz]): ", mean_stress)

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
