from fedoo import *
import numpy as np
import time

#--------------- Pre-Treatment --------------------------------------------------------

Util.ProblemDimension("3D")

#Units: N, mm, MPa
#Mesh.box_mesh(Nx=101, Ny=21, Nz=21, x_min=0, x_max=1000, y_min=0, y_max=100, z_min=0, z_max=100, ElementShape = 'hex8', ID = 'Domain')
Mesh.box_mesh(Nx=11, Ny=5, Nz=5, x_min=0, x_max=1000, y_min=0, y_max=100, z_min=0, z_max=100, ElementShape = 'hex8', name = 'Domain')

meshID = "Domain"

#Material definition
ConstitutiveLaw.ElasticIsotrop(200e3, 0.3, ID = 'ElasticLaw')
WeakForm.InternalForce("ElasticLaw")

#Assembly (print the time required for assembling)
Assembly.Create("ElasticLaw", meshID, 'hex8', ID="Assembling") 

#Type of problem 
Problem.Static("Assembling")

#Boundary conditions
nodes_left = Mesh.get_all()[meshID].node_sets["left"]
nodes_right = Mesh.get_all()[meshID].node_sets["right"]
nodes_top = Mesh.get_all()[meshID].node_sets["top"]
nodes_bottom = Mesh.get_all()[meshID].node_sets["bottom"]

Problem.BoundaryCondition('Dirichlet','DispX',0,nodes_left)
Problem.BoundaryCondition('Dirichlet','DispY', 0,nodes_left)
Problem.BoundaryCondition('Dirichlet','DispZ', 0,nodes_left)

Problem.BoundaryCondition('Dirichlet','DispY', -10, nodes_right)

Problem.ApplyBoundaryCondition()

#--------------- Solve --------------------------------------------------------
t0 = time.time() 
# Problem.SetSolver('cg') #uncomment for conjugate gradient solver
print('Solving...')
Problem.Solve() 
print('Done in ' +str(time.time()-t0) + ' seconds')

#--------------- Post-Treatment -----------------------------------------------
#Get the displacement vector on nodes for export to vtk
U = np.reshape(Problem.GetDoFSolution('all'),(3,-1)).T

#Get the stress tensor (nodal values)
TensorStrain = Assembly.get_all()['Assembling'].GetStrainTensor(Problem.GetDoFSolution(), "Nodal", nlgeom=False)       
TensorStress = ConstitutiveLaw.get_all()['ElasticLaw'].GetStressFromStrain(TensorStrain)

assert np.abs(TensorStress[5][-1] + 0.900798346778864) < 1e-15
