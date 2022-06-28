from fedoo import *
import numpy as np
from time import time
import os
import pylab as plt
from numpy import linalg

from simcoon import simmit as sim

start = time()
#--------------- Pre-Treatment --------------------------------------------------------

Util.ProblemDimension("2Dplane")

NLGEOM = True
#Units: N, mm, MPa
h = 2
w = 10
L = 16
E = 200e3
nu=0.3
alpha = 1e-5 #???
rho = 1600e-6
meshID = "Domain"
uimp = -0.5

Mesh.rectangle_mesh(Nx=101, Ny=11,x_min=0, x_max=L, y_min=0, y_max=h, ElementShape = 'quad4', ID = meshID)
mesh = Mesh.get_all()[meshID]

crd = mesh.nodes 

mat =1
if mat == 0:
    props = np.array([[E, nu, alpha]])
    Material = ConstitutiveLaw.Simcoon("ELISO", props, 1, ID='ConstitutiveLaw')
    Material.corate = 0
elif mat == 1:
    Re = 300
    k=1000
    m=0.25
    props = np.array([[E, nu, alpha, Re,k,m]])
    Material = ConstitutiveLaw.Simcoon("EPICP", props, 8, ID='ConstitutiveLaw')
    Material.corate = 0
else:
    Material = ConstitutiveLaw.ElasticIsotrop(E, nu, ID='ConstitutiveLaw')

WeakForm.InternalForce("ConstitutiveLaw", nlgeom = NLGEOM)

#note set for boundary conditions
nodes_bottomLeft = np.where((crd[:,0]==0) * (crd[:,1]==0))[0]
nodes_bottomRight = np.where((crd[:,0]==L) * (crd[:,1]==0))[0]
# nodes_topCenter = np.where((crd[:,0]==L/2) * (crd[:,1]==h))[0]
nodes_top1 = np.where((crd[:,0]==L/4) * (crd[:,1]==h))[0]
nodes_top2 = np.where((crd[:,0]==3*L/4) * (crd[:,1]==h))[0]

Assembly.Create("ConstitutiveLaw", meshID, 'quad4', ID="Assembling", MeshChange = False)     #uses MeshChange=True when the mesh change during the time

#Mass matrix
WeakForm.Inertia(rho,"Inertia")
Assembly.Create("Inertia", meshID, "quad4", ID="MassAssembling")

Problem.NonLinearNewmark("Assembling", "MassAssembling", 0.25, 0.5)

# Problem.SetSolver('cg', precond = True)

Problem.SetNewtonRaphsonErrorCriterion("Displacement")
# Problem.SetNewtonRaphsonErrorCriterion("Work")
# Problem.SetNewtonRaphsonErrorCriterion("Force")

#create a 'result' folder and set the desired ouputs
# if not(os.path.isdir('results')): os.mkdir('results')
# Problem.AddOutput('results/bendingPlasticDyna', 'Assembling', ['disp', 'kirchhoff', 'cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Node', file_format ='vtk')    
# Problem.AddOutput('results/bendingPlasticDyna', 'Assembling', ['kirchhoff', 'cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')    


################### step 1 ################################
tmax = 1
Problem.BoundaryCondition('Dirichlet','DispX',0,nodes_bottomLeft)
Problem.BoundaryCondition('Dirichlet','DispY', 0,nodes_bottomLeft)
Problem.BoundaryCondition('Dirichlet','DispY',0,nodes_bottomRight)
bc1 = Problem.BoundaryCondition('Dirichlet','DispY', uimp, nodes_top1)
bc2 = Problem.BoundaryCondition('Dirichlet','DispY', uimp, nodes_top2)


Problem.NLSolve(dt = 0.2, tmax = 1, update_dt = True, ToleranceNR = 0.005)


################### step 2 ################################
# bc.Remove()

# #We set initial condition to the applied force to relax the load
# F_app = Problem.GetExternalForce('DispY')[nodes_topCenter]
# bc = Problem.BoundaryCondition('Neumann','DispY', 0, nodes_topCenter, initialValue=F_app)#face_center)

# Problem.NLSolve(t0 = 1, tmax = 2, dt = 1., update_dt = True, ToleranceNR = 0.01)


# print(time()-start)

res = Problem.GetResults('Assembling', ['Strain','Stress'], 'Node') 
assert np.abs(res['Strain'][0][941]+0.019591764988376507) < 1e-8
assert np.abs(res['Stress'][3][234]+3.937900318926645) < 1e-4



