import fedoo as fd
import numpy as np

# --------------- Pre-Treatment --------------------------------------------------------

fd.ModelingSpace("3D")
# mesh = fd.mesh.import_file('../../util/meshes/gyroid_per.vtk')
mesh = fd.Mesh.read("../../util/meshes/gyroid_per.vtk")
# assert (mesh.is_periodic())

# Definition of the set of nodes for boundary conditions
crd = mesh.nodes
xmax = np.max(crd[:, 0])
xmin = np.min(crd[:, 0])
ymax = np.max(crd[:, 1])
ymin = np.min(crd[:, 1])
zmax = np.max(crd[:, 2])
zmin = np.min(crd[:, 2])
center = [np.linalg.norm(crd, axis=1).argmin()]

StrainNodes = mesh.add_virtual_nodes(2)  # add virtual nodes for macro strain

# Material definition
material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
wf = fd.weakform.StressEquilibrium(material)

# Assembly
assembly = fd.Assembly.create(wf, mesh)

# Type of problem
pb = fd.problem.Linear(assembly)


# Boundary conditions
E = [0, 0, 0, 0, 0, 1]  # [EXX, EYY, EZZ, EXY, EXZ, EYZ]
# StrainNode[0] - 'DispX' is a virtual dof for EXX
# StrainNode[0] - 'DispY' is a virtual dof for EYY
# StrainNode[0] - 'DispZ' is a virtual dof for EZZ
# StrainNode[1] - 'DispX' is a virtual dof for EXY
# StrainNode[1] - 'DispY' is a virtual dof for EXZ
# StrainNode[1] - 'DispZ' is a virtual dof for EYZ

# define periodic boundary condition
list_strain_nodes = [
    StrainNodes[0],
    StrainNodes[0],
    StrainNodes[0],
    StrainNodes[1],
    StrainNodes[1],
    StrainNodes[1],
]
list_strain_var = ["DispX", "DispY", "DispZ", "DispX", "DispY", "DispZ"]

bc_periodic = fd.constraint.PeriodicBC(list_strain_nodes, list_strain_var)
pb.bc.add(bc_periodic)

# boundary conditions
pb.bc.add("Dirichlet", center, "DispX", 0)
pb.bc.add("Dirichlet", center, "DispY", 0)
pb.bc.add("Dirichlet", center, "DispZ", 0)

pb.bc.add("Dirichlet", [StrainNodes[0]], "DispX", E[0])  # EpsXX
pb.bc.add("Dirichlet", [StrainNodes[0]], "DispY", E[1])  # EpsYY
pb.bc.add("Dirichlet", [StrainNodes[0]], "DispZ", E[2])  # EpsZZ
pb.bc.add("Dirichlet", [StrainNodes[1]], "DispX", E[3])  # EpsXY
pb.bc.add("Dirichlet", [StrainNodes[1]], "DispY", E[4])  # EpsXZ
pb.bc.add("Dirichlet", [StrainNodes[1]], "DispZ", E[5])  # EpsYZ

# lv--------------- Soe --------------------------------------------------------
pb.solve()

tensor_strain = assembly.sv["Strain"]
tensor_stress = assembly.sv["Stress"]

###############################################################################
# print the macroscopic strain tensor and stress tensor
mean_strain = [
    pb.get_disp("DispX")[-2],
    pb.get_disp("DispY")[-2],
    pb.get_disp("DispZ")[-2],
    pb.get_disp("DispX")[-1],
    pb.get_disp("DispY")[-1],
    pb.get_disp("DispZ")[-1],
]
print(
    "Strain tensor: ",
    np.array(
        [
            [mean_strain[0], mean_strain[3], mean_strain[4]],
            [mean_strain[3], mean_strain[1], mean_strain[5]],
            [mean_strain[4], mean_strain[5], mean_strain[2]],
        ]
    ),
)

# Compute the mean stress tensor
surf = mesh.bounding_box.volume  # total surface of the domain = volume in 2d
mean_stress = [1 / surf * mesh.integrate_field(tensor_stress[i]) for i in range(6)]

print(
    "Stress tensor: ",
    np.array(
        [
            [mean_stress[0], mean_stress[3], mean_stress[4]],
            [mean_stress[3], mean_stress[1], mean_stress[5]],
            [mean_stress[4], mean_stress[5], mean_stress[2]],
        ]
    ),
)

pb.get_results(assembly, "Stress").plot("Stress", "XX")
