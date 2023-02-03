from fedoo import *
import numpy as np
from time import time
import os
import pylab as plt
from numpy import linalg

start = time()
#--------------- Pre-Treatment --------------------------------------------------------

ModelingSpace("2Dplane")

NLGEOM =1
typeBending = '3nodes' #'3nodes' or '4nodes'
#Units: N, mm, MPa
h = 2
w = 10
L = 16
E = 200e3
nu=0.3
alpha = 1e-5 #???
meshname = "Domain"
uimp = -5

Mesh.rectangle_mesh(Nx=41, Ny=21, x_min=0, x_max=L, y_min=0, y_max=h, ElementShape = 'quad4', name = meshname)
mesh = Mesh.get_all()[meshname]

crd = mesh.nodes 

mat =1
if mat == 0:
    props = np.array([[E, nu, alpha]])
    Material = ConstitutiveLaw.Simcoon("ELISO", props, 1, name='ConstitutiveLaw')
    Material.corate = 101
elif mat == 1:
    Re = 300
    k=1000
    m=0.25
    props = np.array([[E, nu, alpha, Re,k,m]])
    Material = ConstitutiveLaw.Simcoon("EPICP", props, 8, name='ConstitutiveLaw')
    Material.corate = 0
    # mask = [[3,4,5] for i in range(3)]
    # mask+= [[0,1,2,4,5], [0,1,2,3,5], [0,1,2,3,4]]
    # Material.SetMaskH(mask)
else:
    Material = ConstitutiveLaw.ElasticIsotrop(E, nu, name='ConstitutiveLaw')

WeakForm.StressEquilibrium("ConstitutiveLaw", nlgeom = NLGEOM)



#note set for boundary conditions
nodes_bottomLeft = np.where((crd[:,0]==0) * (crd[:,1]==0))[0]
nodes_bottomRight = np.where((crd[:,0]==L) * (crd[:,1]==0))[0]

if typeBending == '3nodes':
    nodes_topCenter = np.where((crd[:,0]==L/2) * (crd[:,1]==h))[0]
else: 
    nodes_top1 = np.where((crd[:,0]==L/4) * (crd[:,1]==h))[0]
    nodes_top2 = np.where((crd[:,0]==3*L/4) * (crd[:,1]==h))[0]
    nodes_topCenter = np.hstack((nodes_top1, nodes_top2))

Assembly.create("ConstitutiveLaw", meshname, 'quad4', name="Assembling", MeshChange = False)     #uses MeshChange=True when the mesh change during the time

Problem.NonLinear("Assembling")

# Problem.set_solver('cg', precond = True)

Problem.set_nr_criterion("Displacement")
# Problem.set_nr_criterion("Work")
# Problem.set_nr_criterion("Force")

#create a 'result' folder and set the desired ouputs
if not(os.path.isdir('results')): os.mkdir('results')
Problem.add_output('results/bendingPlastic', 'Assembling', ['Disp', 'Cauchy', 'PKII', 'Strain', 'Cauchy_vm', 'Statev', 'Wm'], output_type='Node', file_format ='vtk')    
Problem.add_output('results/bendingPlastic', 'Assembling', ['Cauchy', 'PKII', 'Strain', 'Cauchy_vm', 'Statev'], output_type='Element', file_format ='vtk')    


################### step 1 ################################
tmax = 1
Problem.bc.add('Dirichlet','DispX',0,nodes_bottomLeft)
Problem.bc.add('Dirichlet','DispY', 0,nodes_bottomLeft)
Problem.bc.add('Dirichlet','DispY',0,nodes_bottomRight)
Problem.bc.add('Dirichlet','DispY', uimp, nodes_topCenter, name = "disp")

Problem.nlsolve(dt = 0.2, tmax = 1, update_dt = False, ToleranceNR = 0.05)

################### step 2 ################################
Problem.RemoveBC("disp")
#We set initial condition to the applied force to relax the load
F_app = Problem.get_ext_forces('DispY')[nodes_topCenter]
Problem.bc.add('Neumann','DispY', 0, nodes_topCenter, initialValue=F_app)#face_center)

Problem.nlsolve(dt = 1., update_dt = True, ToleranceNR = 0.05)

print(time()-start)













