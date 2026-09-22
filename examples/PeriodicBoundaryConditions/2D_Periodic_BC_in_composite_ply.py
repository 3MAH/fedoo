import numpy as np

import fedoo as fd

# ------------------------------------------------------------------------------
# Define in-plane 2D periodic boundary conditions for a composite ply using a
# 3D unit cell.
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Dimension of the problem
# ------------------------------------------------------------------------------
fd.ModelingSpace("3D")

# ------------------------------------------------------------------------------
# Definition of the Geometry
# ------------------------------------------------------------------------------
# A structured mesh guarantees matching nodes on opposite faces. Two elliptical
# masks represent orthogonal fiber bundles at different positions through the
# thickness, similar to the warp and weft bundles of a woven ply.
mesh = fd.mesh.box_mesh(
    nx=17 * 2,
    ny=17 * 2,
    nz=7 * 2,
    x_min=0,
    x_max=1,
    y_min=0,
    y_max=1,
    z_min=0,
    z_max=0.3,
    elm_type="hex8",
)

element_centers = mesh.element_centers
fiber_semi_width = 0.16
fiber_semi_thickness = 0.045
fiber_x = ((element_centers[:, 1] - 0.5) / fiber_semi_width) ** 2 + (
    (element_centers[:, 2] - 0.1) / fiber_semi_thickness
) ** 2 <= 1
fiber_y = ((element_centers[:, 0] - 0.5) / fiber_semi_width) ** 2 + (
    (element_centers[:, 2] - 0.2) / fiber_semi_thickness
) ** 2 <= 1
fiber_elements = np.flatnonzero(fiber_x | fiber_y)
matrix_elements = np.flatnonzero(~(fiber_x | fiber_y))
mesh.add_element_set(fiber_elements, "All_fibers")
mesh.add_element_set(matrix_elements, "All_matrix")

E_fiber = 250e3
E_matrix = 4e3
nu_fiber = 0.3
nu_matrix = 0.33

# E_fiber = E_matrix
# nu_fiber = nu_matrix

# ------------------------------------------------------------------------------
# Set of nodes for boundary conditions
# ------------------------------------------------------------------------------
center = mesh.nearest_node(mesh.bounding_box.center)

# ------------------------------------------------------------------------------
# Material definition
# ------------------------------------------------------------------------------
young_modulus = E_fiber * np.ones(mesh.n_elements)
poisson_ratio = nu_fiber * np.ones(mesh.n_elements)
young_modulus[matrix_elements] = E_matrix
poisson_ratio[matrix_elements] = nu_matrix

fd.constitutivelaw.ElasticIsotrop(
    young_modulus,
    poisson_ratio,
    name="ElasticLaw",
)

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
macro_strain = [0.1, 0, 0]  # [EXX, EYY, EXY]

# Apply the periodic boundary conditions
pb.bc.add(
    fd.constraint.PeriodicBC(
        periodicity_type="small_strain",
        dim=2,
    )
)

# Block a node on the center to avoid rigid body motion
pb.bc.add("Dirichlet", center, "Disp", 0)
pb.bc.add("Dirichlet", "MeanStrain", macro_strain)

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

print("Elastic Energy: " + str(pb.get_elastic_energy()))

# ------------------------------------------------------------------------------
# Optional: Write data in a vtk file (for visualization with paraview for instance)
# ------------------------------------------------------------------------------
# pb.get_results(assemb, ['Stress','Strain','Disp'], 'Node').save('composite_ply_periodic.vtk', True)


# ------------------------------------------------------------------------------
# Optional: Get data for fibers only and plot results
# ------------------------------------------------------------------------------
res_fibers = pb.get_results(
    assemb, ["Stress", "Strain", "Disp"], element_set="All_fibers"
)
res_fibers.plot("Stress", "XX")
