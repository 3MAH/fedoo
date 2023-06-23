import fedoo as fd
import numpy as np
from time import time
import os
import pylab as plt
from numpy import linalg

from simcoon import simmit as sim

start = time()
#--------------- Pre-Treatment --------------------------------------------------------

fd.ModelingSpace("2Dplane")

NLGEOM =  'TL'
#Units: N, mm, MPa
h = 2
w = 10
L = 16
E = 200e3
nu=0.3
alpha = 1e-5 #???
rho = 1600e-6
uimp = -0.5

fd.mesh.rectangle_mesh(nx=101, ny=11,x_min=0, x_max=L, y_min=0, y_max=h, elm_type = 'quad4', name = "Domain")
mesh = fd.Mesh["Domain"]

crd = mesh.nodes 

mat =1
if mat == 0:
    props = np.array([E, nu, alpha])
    material = fd.constitutivelaw.Simcoon("ELISO", props, name='ConstitutiveLaw')
    material.corate = 0
elif mat == 1:
    Re = 300
    k=1000
    m=0.25
    props = np.array([E, nu, alpha, Re,k,m])
    material = fd.constitutivelaw.Simcoon("EPICP", props, name='ConstitutiveLaw')
    material.corate = 0
else:
    material = fd.constitutivelaw.ElasticIsotrop(E, nu, name='ConstitutiveLaw')

fd.weakform.StressEquilibrium("ConstitutiveLaw", nlgeom = NLGEOM)

#note set for boundary conditions
nodes_bottomLeft = np.where((crd[:,0]==0) * (crd[:,1]==0))[0]
nodes_bottomRight = np.where((crd[:,0]==L) * (crd[:,1]==0))[0]
# nodes_topCenter = np.where((crd[:,0]==L/2) * (crd[:,1]==h))[0]
nodes_top1 = np.where((crd[:,0]==L/4) * (crd[:,1]==h))[0]
nodes_top2 = np.where((crd[:,0]==3*L/4) * (crd[:,1]==h))[0]

fd.Assembly.create("ConstitutiveLaw", "Domain", 'quad4', name="Assembling", MeshChange = False)     #uses MeshChange=True when the mesh change during the time

#Mass matrix
fd.weakform.Inertia(rho,"Inertia")
fd.Assembly.create("Inertia", "Domain", "quad4", name="MassAssembling")

pb = fd.problem.NonLinearNewmark("Assembling", "MassAssembling", 0.25, 0.5)

# Problem.set_solver('cg', precond = True)

pb.set_nr_criterion("Displacement")
# pb.set_nr_criterion("Work")
# pb.set_nr_criterion("Force")

#create a 'result' folder and set the desired ouputs
# if not(os.path.isdir('results')): os.mkdir('results')
# pb.add_output('results/bendingPlasticDyna', 'Assembling', ['disp', 'kirchhoff', 'cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Node', file_format ='vtk')    
# pb.add_output('results/bendingPlasticDyna', 'Assembling', ['kirchhoff', 'cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')    


################### step 1 ################################
tmax = 1
pb.bc.add('Dirichlet',nodes_bottomLeft,'Disp',0)
pb.bc.add('Dirichlet',nodes_bottomRight,'DispY',0)
bc1 = pb.bc.add('Dirichlet',nodes_top1,'DispY', uimp)
bc2 = pb.bc.add('Dirichlet',nodes_top2,'DispY', uimp)


pb.nlsolve(dt = 0.2, tmax = 1, update_dt = False, ToleranceNR = 0.005)


################### step 2 ################################
# bc.Remove()

# #We set initial condition to the applied force to relax the load
# F_app = Problem.GetExternalForce('DispY')[nodes_topCenter]
# bc = Problem.bc.add('Neumann','DispY', 0, nodes_topCenter, initialValue=F_app)#face_center)

# pb.nlsolve(t0 = 1, tmax = 2, dt = 1., update_dt = True, ToleranceNR = 0.01)


# print(time()-start)

res = pb.get_results('Assembling', ['Strain','Stress'], 'Node') 
assert np.abs(res.node_data['Strain'][0][941]+0.01924864476799095) < 1e-8
assert np.abs(res.node_data['Stress'][3][234]+56.41916240526384) < 1e-4

# assert np.abs(res.node_data['Strain'][0][941]+0.019651029182686866) < 1e-8
# assert np.abs(res.node_data['Stress'][3][234]+2.6359202403886792) < 1e-4
# assert np.abs(res['Stress'][3][234]+3.937900318926645) < 1e-4# assert np.abs(res['Stress'][3][234]+3.937900318926645) < 1e-4



