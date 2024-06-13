import sys
import fedoo as fd
import numpy as np
import os as os
# import matplotlib.pyplot as plt

coeff = 0.2
Nx, Ny = 10000, 10000
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Input parameters
COEFF = 1
MeshSize = 1e-3  # based on the work of C. Fond
# Nx, Ny   = 40, 200 # based on the work of C. Fond

Epsilon = 0.01  # Initial static strain
E = 1.5e9  # Young Modulus
nu = 0.33  # Poisson ratio
rho = 1000  # Density

Gamma = 0.5  # parameters for Newmark scheme (here Newmark-Wilson)
Beta = 0.5  # parameters for Newmark scheme (here Newmark-Wilson)

# -------------------------------------------------------------------
# Intern parameters

# Length, Height = Nx*MeshSize, Ny*MeshSize
TransverseWaveSpeed = (E / (2 * rho * (1 + nu))) ** (1 / 2)
RayleighWaveSpeed = (
    0.874032 + 0.200396 * nu - 0.0756704 * (nu**2)
) * TransverseWaveSpeed
CrackVelocity = RayleighWaveSpeed * coeff

dt = MeshSize / CrackVelocity / COEFF  # StepTime


# -------------------------------------------------------------------
# let's go
fd.ModelingSpace("2Dstress")

meshID = "Domain"
mesh = fd.mesh.import_file("Broberg2D_10000_10000.msh", name=meshID)
mesh.nodes = mesh.nodes[:, :2]
# fd.mesh.rectangle_mesh(Nx, Ny, x_min=0, x_max=Length, y_min=0, y_max=Height, elm_type = 'quad4', name = meshID)


# OUT = Util.ExportData(meshID)
# OUT.toVTK("Broberg2D_Dynamic_PreStress.vtk")

fd.constitutivelaw.ElasticIsotrop(E, nu, name="ElasticLaw")

# ecriture des formes faibles
fd.weakform.StressEquilibrium("ElasticLaw", name="StressEquilibrium")
fd.weakform.Inertia(rho, "Inertia")

fd.Assembly.create("StressEquilibrium", mesh, "tri3", name="StiffAssembling")
fd.Assembly.create("Inertia", mesh, "tri3", name="MassAssembling")

length = mesh.bounding_box.size_x
height = mesh.bounding_box.size_y

ImposedDisplacement = Epsilon * height


Xmin = mesh.find_nodes("X", mesh.bounding_box.xmin)
Xmax = mesh.find_nodes("X", mesh.bounding_box.xmax)
Ymin = mesh.find_nodes("Y", mesh.bounding_box.ymin)
Ymax = mesh.find_nodes("Y", mesh.bounding_box.ymax)
Ymin = Ymin[
    np.argsort(mesh.nodes[Ymin][:, 0])
]  # to get a list of nodes sorted from xmin to xmax

# ------------------------------------------------
# precontrainte statique
# ------------------------------------------------

# mise en place du solveur
pb = fd.problem.Linear("StiffAssembling")


# pb.bc.remove_all()

pb.bc.add("Dirichlet", Xmax, "DispX", 0)
pb.bc.add("Dirichlet", Xmin, "DispX", 0)
pb.bc.add("Dirichlet", Ymax, "DispY", ImposedDisplacement)

deb = pb.bc.add("Dirichlet", Ymin, "DispY", 0, name="deb")


# application des conditions limites et resolution du probleme
pb.apply_boundary_conditions()
pb.solve()

# post traitement
U = pb.get_dof_solution()
Gqs = pb.GetElasticEnergy() / (length)

# OUT = Util.ExportData(meshID)
# OUT.addNodeData(np.reshape(U,(2,-1)).T,'Displacement')
# OUT.toVTK("Broberg2D_Dynamic_PreStress.vtk")

# ------------------------------------------------
# deboutonnage dynamique
# ------------------------------------------------

Ecin, Eela = [], []
# V = U*0.
# A = U*0.
# previousU = U*0.

pb = fd.problem.Newmark("StiffAssembling", "MassAssembling", Beta, Gamma, dt)
pb.SetInitialDisplacement("all", U)

pb.SetInitialVelocity("all", 0)
pb.SetInitialAcceleration("all", 0)
pb.initialize()
# pb.set_solver('cg', precond = True)

# pb.add_output("Broberg2D_Dynamic", "StiffAssembling", [], output_type=None, file_format ='npz', position = 1)
# res = pb.add_output("Broberg2D_Dynamic", "StiffAssembling", ["Disp"], file_format ='vtk')
res = pb.add_output("Broberg2D_Dynamic", "StiffAssembling", ["Disp", "Stress"])

pb.bc.add("Dirichlet", Xmax, "DispX", 0)
pb.bc.add("Dirichlet", Xmin, "DispX", 0)
pb.bc.add("Dirichlet", Ymax, "DispY", ImposedDisplacement)

deb = pb.bc.add("Dirichlet", Ymin, "DispY", 0, name="deb")

popo = len(Ymin)  # 500
for i in range(popo):
    print("\r v = {}cr - {}%".format(str(coeff), i / (500 * COEFF) * 100), end="")

    if i % COEFF == 0:
        deb.node_set = deb.node_set[1:]

    pb.apply_boundary_conditions()
    pb.solve()
    pb.save_results(i)
    pb.update()

    U = pb.get_disp()

    Ecin.append(pb.GetKineticEnergy())
    Eela.append(pb.GetElasticEnergy())

Ecin = np.array(Ecin)
Eela = np.array(Eela)
GId = (-(Eela[1:] - Eela[:-1]) - (Ecin[1:] - Ecin[:-1])) / (MeshSize)

# filename = '{}_{}_{}'.format(coeff,Nx,Ny)
# np.savez(filename+'.npz',CorrCoeff=GId/Gqs)

# if __name__ == "__main__":
#     MainFunction(0.2,10000,10000)
