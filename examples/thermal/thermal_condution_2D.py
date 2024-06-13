import fedoo as fd
import numpy as np
from time import time
import os
import pylab as plt
from numpy import linalg

start = time()
# --------------- Pre-Treatment --------------------------------------------------------

fd.ModelingSpace("2Dplane")

# Units: N, mm, MPa
h = 2
w = 10
L = 1
# E = 200e3
# nu=0.3
meshname = "Domain"
uimp = -5

fd.mesh.rectangle_mesh(
    nx=21, ny=21, x_min=0, x_max=L, y_min=0, y_max=L, elm_type="quad4", name=meshname
)
mesh = fd.Mesh[meshname]

K = 18  # W/K/m
c = 0.500  # J/kg/K
rho = 7800  # kg/m2
Material = fd.constitutivelaw.ThermalProperties(K, c, rho, name="ThermalLaw")
wf = fd.weakform.HeatEquation("ThermalLaw")
assemb = fd.Assembly.create("ThermalLaw", meshname, name="Assembling")

left = mesh.find_nodes("X", 0)
right = mesh.find_nodes("X", L)

pb = fd.problem.NonLinear("Assembling")

# Problem.set_solver('cg', precond = True)
pb.set_nr_criterion("Displacement", tol=1e-2, max_subiter=5, err0=100)

# create a 'result' folder and set the desired ouputs
if not (os.path.isdir("results")):
    os.mkdir("results")
res = pb.add_output(
    "results/thermal2D", "Assembling", ["Temp"], output_type="Node", file_format="vtk"
)
# Problem.add_output('results/bendingPlastic', 'Assembling', ['cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')

tmax = 200
pb.bc.add("Dirichlet", left, "Temp", 100, start_value=0)
pb.bc.add("Dirichlet", right, "Temp", 50, start_value=0)
# Problem.bc.add('Dirichlet','DispY', 0,nodes_bottomLeft)
# Problem.bc.add('Dirichlet','DispY',0,nodes_bottomRight)
# bc = Problem.bc.add('Dirichlet','DispY', uimp, nodes_topCenter)

pb.nlsolve(dt=tmax / 10, tmax=tmax, update_dt=True)
