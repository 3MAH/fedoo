from fedoo import *
import numpy as np
import time

#--------------- Pre-Treatment --------------------------------------------------------

Util.ProblemDimension("3D")

#Units: N, mm, MPa
#Mesh.BoxMesh(Nx=101, Ny=21, Nz=21, x_min=0, x_max=1000, y_min=0, y_max=100, z_min=0, z_max=100, ElementShape = 'hex8', ID = 'Domain')
Mesh.BoxMesh(Nx=101, Ny=21, Nz=21, x_min=0, x_max=1000, y_min=0, y_max=100, z_min=0, z_max=100, ElementShape = 'hex8', ID = 'Domain')

meshID = "Domain"
#mesh = Mesh.GetAll()[meshID]
#crd = mesh.nodes 
#xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
#mesh.add_node_set(list(np.where(mesh.nodes[:,0] == xmin)[0]), "left")
#mesh.add_node_set(list(np.where(mesh.nodes[:,0] == xmax)[0]), "right")

#Material definition
ConstitutiveLaw.ElasticIsotrop(200e3, 0.3, ID = 'ElasticLaw')
WeakForm.InternalForce("ElasticLaw")

#Assembly (print the time required for assembling)
Assembly.Create("ElasticLaw", meshID, 'hex8', ID="Assembling") 

#Type of problem 
Problem.Static("Assembling")

#Boundary conditions
nodes_left = Mesh.GetAll()[meshID].node_sets["left"]
nodes_right = Mesh.GetAll()[meshID].node_sets["right"]
nodes_top = Mesh.GetAll()[meshID].node_sets["top"]
nodes_bottom = Mesh.GetAll()[meshID].node_sets["bottom"]

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
TensorStrain = Assembly.GetAll()['Assembling'].GetStrainTensor(Problem.GetDoFSolution(), "Nodal")       
TensorStress = Problem.GetResults('Assembling', ['stress'], 'Node')['Stress']
# ConstitutiveLaw.GetAll()['ElasticLaw'].GetStress()

#PrincipalStress, PrincipalDirection = TensorStress.GetPrincipalStress()

# Get the principal directions (vectors on nodes)
#temp = np.array([np.linalg.eig([[TensorStress[0][nd], TensorStress[5][nd]],[TensorStress[5][nd], TensorStress[1][nd]]])[1] for nd in range(len(TensorStress[0]))])
#PrincipalDirection1 = np.c_[temp[:,:,0], np.zeros(len(TensorStress[0]))]
#PrincipalDirection2 = np.c_[temp[:,:,1], np.zeros(len(TensorStress[0]))]

#modification of TensorStress and TensorStressEl for compatibility with the export to vtk 
#TensorStress = np.vstack([TensorStress[i] for i in [0,1,2,5,3,4]]).T                         
#TensorStressEl = np.vstack([TensorStressEl[i] for i in [0,1,2,5,3,4]]).T 
#TensorStrain = np.vstack([TensorStrain[i] for i in [0,1,2,5,3,4]]).T                         
#TensorStrainEl = np.vstack([TensorStrainEl[i] for i in [0,1,2,5,3,4]]).T 
                                                    

#Write the vtk file                            
OUT = Util.ExportData(meshID)

OUT.addNodeData(U.astype(float),'Displacement')
OUT.addNodeData(TensorStress.vtkFormat(),'Stress')
OUT.addNodeData(TensorStress.vonMises(),'VMStress')
OUT.addNodeData(TensorStrain.vtkFormat(),'Strain')
#OUT.addNodeData(PrincipalStress, 'PrincipalStress')
#OUT.addNodeData(PrincipalDirection[0], '1stPrincipalDirection')

OUT.toVTK("simple_cantilever_3D_model.vtk")
print('Elastic Energy: ' + str(Problem.GetElasticEnergy()))

print('Result file "simple_cantilever_3D_model.vtk" written in the active directory')

