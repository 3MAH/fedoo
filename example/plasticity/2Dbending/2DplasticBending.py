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
typeBending = '4nodes' #or '3nodes'
#Units: N, mm, MPa
h = 2
w = 10
L = 16
E = 200e3
nu=0.3
alpha = 1e-5 #???
meshID = "Domain"
uimp = -2

Mesh.RectangleMesh(Nx=41, Ny=21, x_min=0, x_max=L, y_min=0, y_max=h, ElementShape = 'quad4', ID = meshID)
mesh = Mesh.GetAll()[meshID]

crd = mesh.GetNodeCoordinates() 

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
    Material.corate = 3
else:
    Material = ConstitutiveLaw.ElasticIsotrop(E, nu, ID='ConstitutiveLaw')

WeakForm.InternalForce("ConstitutiveLaw", nlgeom = NLGEOM)



#note set for boundary conditions
nodes_bottomLeft = np.where((crd[:,0]==0) * (crd[:,1]==0))[0]
nodes_bottomRight = np.where((crd[:,0]==L) * (crd[:,1]==0))[0]

if typeBending == '3nodes':
    nodes_topCenter = np.where((crd[:,0]==L/2) * (crd[:,1]==h))[0]
else: 
    nodes_top1 = np.where((crd[:,0]==L/4) * (crd[:,1]==h))[0]
    nodes_top2 = np.where((crd[:,0]==3*L/4) * (crd[:,1]==h))[0]
    nodes_topCenter = np.hstack((nodes_top1, nodes_top2))

Assembly.Create("ConstitutiveLaw", meshID, 'quad4', ID="Assembling", MeshChange = False)     #uses MeshChange=True when the mesh change during the time

Problem.NonLinearStatic("Assembling")

# Problem.SetSolver('cg', precond = True)

Problem.SetNewtonRaphsonErrorCriterion("Displacement")
# Problem.SetNewtonRaphsonErrorCriterion("Work")
# Problem.SetNewtonRaphsonErrorCriterion("Force")

#create a 'result' folder and set the desired ouputs
if not(os.path.isdir('results')): os.mkdir('results')
Problem.AddOutput('results/bendingPlastic', 'Assembling', ['disp', 'kirchhoff', 'cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Node', file_format ='vtk')    
Problem.AddOutput('results/bendingPlastic', 'Assembling', ['kirchhoff', 'cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')    


################### step 1 ################################
tmax = 1
Problem.BoundaryCondition('Dirichlet','DispX',0,nodes_bottomLeft)
Problem.BoundaryCondition('Dirichlet','DispY', 0,nodes_bottomLeft)
Problem.BoundaryCondition('Dirichlet','DispY',0,nodes_bottomRight)
bc = Problem.BoundaryCondition('Dirichlet','DispY', uimp, nodes_topCenter)

Problem.NLSolve(dt = 0.05, tmax = 1, update_dt = True, ToleranceNR = 0.005)

################### step 2 ################################
bc.Remove()
#We set initial condition to the applied force to relax the load
F_app = Problem.GetExternalForce('DispY')[nodes_topCenter]
bc = Problem.BoundaryCondition('Neumann','DispY', 0, nodes_topCenter, initialValue=F_app)#face_center)

Problem.NLSolve(dt = 1., update_dt = True, ToleranceNR = 0.01)

print(time()-start)













