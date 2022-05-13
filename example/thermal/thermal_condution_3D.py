from fedoo import *
import numpy as np
from time import time
import os
import pylab as plt
from numpy import linalg

start = time()
#--------------- Pre-Treatment --------------------------------------------------------

Util.ProblemDimension("3D")

# NLGEOM =False
#Units: N, mm, MPa
h = 2
w = 10
L = 1
# E = 200e3
# nu=0.3
meshID = "Domain2"
uimp = -5

# Mesh.RectangleMesh(Nx=21, Ny=21, x_min=0, x_max=L, y_min=0, y_max=L, ElementShape = 'quad4', ID = meshID) 
Mesh.ImportFromFile('octet_surf.msh', meshID = "Domain")

mesh = Mesh.GetAll()[meshID]

crd = mesh.GetNodeCoordinates() 

K = 18 #W/K/m
c = 0.500 #J/kg/K
rho = 7800 #kg/m2
Material = ConstitutiveLaw.ThermalProperties(K, c, rho, ID='ThermalLaw')
wf = WeakForm.HeatEquation("ThermalLaw")
assemb = Assembly.Create("ThermalLaw", meshID, ID="Assembling")    

#note set for boundary conditions
bottom = mesh.FindNodes('Z', 0)
top = mesh.FindNodes('Z', L)

Problem.NonLinearStatic("Assembling")

# Problem.SetSolver('cg', precond = True)

Problem.SetNewtonRaphsonErrorCriterion("Displacement", tol = 1e-2, max_subiter=5, err0 = 100)

#create a 'result' folder and set the desired ouputs
if not(os.path.isdir('results')): os.mkdir('results')
Problem.AddOutput('results/thermal3D', 'Assembling', ['temp'], output_type='Node', file_format ='vtk')    
# Problem.AddOutput('results/bendingPlastic', 'Assembling', ['cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')    

tmax = 200
Problem.BoundaryCondition('Dirichlet','Temp',0,bottom)
Problem.BoundaryCondition('Dirichlet','Temp',100,top)
# Problem.BoundaryCondition('Dirichlet','DispY', 0,nodes_bottomLeft)
# Problem.BoundaryCondition('Dirichlet','DispY',0,nodes_bottomRight)
# bc = Problem.BoundaryCondition('Dirichlet','DispY', uimp, nodes_topCenter)

Problem.NLSolve(dt = tmax/10, tmax = tmax, update_dt = True)




