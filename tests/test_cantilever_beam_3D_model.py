import fedoo as fd
import numpy as np
import time

#--------------- Pre-Treatment --------------------------------------------------------

fd.ModelingSpace("3D")

#Units: N, mm, MPa
#Mesh.box_mesh(Nx=101, Ny=21, Nz=21, x_min=0, x_max=1000, y_min=0, y_max=100, z_min=0, z_max=100, ElementShape = 'hex8', name = 'Domain')
mesh = fd.mesh.box_mesh(Nx=11, Ny=5, Nz=5, x_min=0, x_max=1000, y_min=0, y_max=100, z_min=0, z_max=100, ElementShape = 'hex8', name = 'Domain')

#Material definition
fd.constitutivelaw.ElasticIsotrop(200e3, 0.3, name = 'ElasticLaw')
fd.weakform.InternalForce("ElasticLaw")

#Assembly (print the time required for assembling)
fd.Assembly.create("ElasticLaw", 'Domain', 'hex8', name="Assembling") 

#Type of problem 
pb = fd.problem.Static("Assembling")

#Boundary conditions
nodes_left = mesh.node_sets["left"]
nodes_right = mesh.node_sets["right"]
nodes_top = mesh.node_sets["top"]
nodes_bottom = mesh.node_sets["bottom"]

pb.BoundaryCondition('Dirichlet','DispX',0,nodes_left)
pb.BoundaryCondition('Dirichlet','DispY', 0,nodes_left)
pb.BoundaryCondition('Dirichlet','DispZ', 0,nodes_left)

pb.BoundaryCondition('Dirichlet','DispY', -10, nodes_right)

pb.ApplyBoundaryCondition()

#--------------- Solve --------------------------------------------------------
t0 = time.time() 
# Problem.SetSolver('cg') #uncomment for conjugate gradient solver
print('Solving...')
pb.Solve() 
print('Done in ' +str(time.time()-t0) + ' seconds')

#--------------- Post-Treatment -----------------------------------------------
#Get the displacement vector on nodes for export to vtk
U = np.reshape(pb.GetDoFSolution('all'),(3,-1)).T

#Get the stress tensor (nodal values)
TensorStrain = fd.Assembly['Assembling'].get_strain(pb.GetDoFSolution(), "Nodal", nlgeom=False)       
TensorStress = fd.ConstitutiveLaw['ElasticLaw'].GetStressFromStrain(TensorStrain)

assert np.abs(TensorStress[5][-1] + 0.900798346778864) < 1e-15
