import fedoo as fd
import numpy as np
from time import time
import os
import pylab as plt
from numpy import linalg

start = time()
# --------------- Pre-Treatment --------------------------------------------------------

fd.ModelingSpace("3D")

meshname = "Domain"
nb_iter = 100

# Mesh.box_mesh(Nx=3, Ny=3, Nz=3, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1, ElementShape = 'hex8', name = meshname)
# Mesh.import_file('octet_surf.msh', meshname = "Domain")
# Mesh.import_file('data/octet_1.msh', meshname = "Domain")
fd.mesh.import_file("../../util/meshes/gyroid.msh", name="Domain")

mesh = fd.Mesh[meshname]

crd = mesh.nodes

K = 500  # K = 18 #W/K/m
c = 0.500  # J/kg/K
rho = 7800  # kg/m2
material = fd.constitutivelaw.ThermalProperties(K, c, rho, name="ThermalLaw")
wf = fd.weakform.HeatEquation("ThermalLaw")
assemb = fd.Assembly.create("ThermalLaw", meshname, name="Assembling")

# note set for boundary conditions
Xmin, Xmax = mesh.bounding_box
bottom = mesh.find_nodes("Z", Xmin[2])
top = mesh.find_nodes("Z", Xmax[2])
left = mesh.find_nodes("X", Xmin[2])
right = mesh.find_nodes("X", Xmax[2])

pb = fd.problem.NonLinear("Assembling")

# Problem.set_solver('cg', precond = True)

pb.set_nr_criterion("Displacement", tol=5e-2, max_subiter=5, err0=100)

# create a 'result' folder and set the desired ouputs
if not (os.path.isdir("results")):
    os.mkdir("results")
results = pb.add_output("results/thermal3D", "Assembling", ["Temp"])
# pb.add_output('results/bendingPlastic', 'Assembling', ['cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')

tmax = 10


# Problem.bc.add('Dirichlet','Temp',0,bottom)
def timeEvolution(timeFactor):
    if timeFactor == 0:
        return 0
    else:
        return 1


# Problem.bc.add('Dirichlet',left,'Temp',100, time_func=timeEvolution)
pb.bc.add("Dirichlet", right, "Temp", 100, time_func=timeEvolution)
# Problem.bc.add('Dirichlet',top,'Temp',100, time_func=timeEvolution)


# Problem.bc.add('Dirichlet','DispY', 0,nodes_bottomLeft)
# Problem.bc.add('Dirichlet','DispY',0,nodes_bottomRight)
# bc = Problem.bc.add('Dirichlet','DispY', uimp, nodes_topCenter)

pb.nlsolve(dt=tmax / nb_iter, tmax=tmax, update_dt=True)

cpos = [
    (-2.090457552750125, 1.7582929402632352, 1.707926514944027),
    (0.20739316009534275, -0.2296587829717462, -0.38339561081860574),
    (0.42357673667356105, -0.37693638734293083, 0.8237121512068624),
]

results.load(46)
pl = results.plot("Temp", show=False)
pl.camera_position = cpos
pl.show()

# results.write_movie('toto','Temp')
