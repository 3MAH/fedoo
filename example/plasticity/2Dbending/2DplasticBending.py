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
Mesh.RectangleMesh(Nx=51, Ny=11,x_min=0, x_max=L, y_min=0, y_max=h, ElementShape = 'quad4', ID = 'Domain')

meshID = "Domain"
mesh = Mesh.GetAll()[meshID]

crd = mesh.GetNodeCoordinates() 

mat =1
if mat == 0:
    props = np.array([[E, nu, alpha]])
    Material = ConstitutiveLaw.Simcoon("ELISO", props, 1, ID='ElasticLaw')
    Material.corate = 0
elif mat == 1:
    Re = 300
    k=1000
    m=0.25
    props = np.array([[E, nu, alpha, Re,k,m]])
    Material = ConstitutiveLaw.Simcoon("EPICP", props, 8, ID='ElasticLaw')
    Material.corate = 2
else:
    Material = ConstitutiveLaw.ElasticIsotrop(E, nu, ID='ElasticLaw')

WeakForm.InternalForce("ElasticLaw", nlgeom = NLGEOM)

#note set for boundary conditions
nodes_bottomLeft = np.where((crd[:,0]==0) * (crd[:,1]==0))[0]
nodes_bottomRight = np.where((crd[:,0]==L) * (crd[:,1]==0))[0]
nodes_topCenter = np.where((crd[:,0]==L/2) * (crd[:,1]==h))[0]

Ftot = 0 ; Ftot_sav = []
uimp = -3
u = 0 ; u_sav = []
TotalPKStress = 0
TotalCauchyStress = 0
TotalStrain = 0

Assembly.Create("ElasticLaw", meshID, 'quad4', ID="Assembling", MeshChange = False)     #uses MeshChange=True when the mesh change during the time

Problem.NonLinearStatic("Assembling")

# Problem.SetSolver('cg', precond = True)

Problem.SetNewtonRaphsonErrorCriterion("Displacement")
# Problem.SetNewtonRaphsonErrorCriterion("Work")
# Problem.SetNewtonRaphsonErrorCriterion("Force")

#create a 'result' folder and set the desired ouputs
if not(os.path.isdir('results')): os.mkdir('results')
Problem.AddOutput('results/bendingPlastic', 'Assembling', ['disp', 'kirchhoff', 'cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Node', file_format ='vtk')    

################### step 1 ################################
tmax = 1
Problem.BoundaryCondition('Dirichlet','DispX',0,nodes_bottomLeft)
Problem.BoundaryCondition('Dirichlet','DispY', 0,nodes_bottomLeft)
Problem.BoundaryCondition('Dirichlet','DispY',0,nodes_bottomRight)
bc = Problem.BoundaryCondition('Dirichlet','DispY', uimp, nodes_topCenter)

Problem.NLSolve(dt = 0.1, tmax = 1, update_dt = True, ToleranceNR = 0.01)

################### step 2 ################################
bc.Remove()

#We set initial condition to the applied force to relax the load
F_app = Problem.GetExternalForce('DispY')[nodes_topCenter]
bc = Problem.BoundaryCondition('Neumann','DispY', 0, nodes_topCenter, initialValue=F_app)#face_center)

Problem.NLSolve(t0 = 1, tmax = 2, dt = 1., update_dt = True, ToleranceNR = 0.01)


print(time()-start)













