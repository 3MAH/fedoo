"""
Canteleaver Beam using 3D hexahedral elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import fedoo as fd
import numpy as np
import time

#--------------- Pre-Treatment --------------------------------------------------------

fd.ModelingSpace("3D")

#Units: N, mm, MPa
mesh = fd.mesh.box_mesh(nx=101, ny=21, nz=21, x_min=0, x_max=1000, y_min=0, y_max=100, z_min=0, z_max=100, elm_type = 'hex8', name = 'Domain')

#Material definition
fd.constitutivelaw.ElasticIsotrop(200e3, 0.3, name = 'ElasticLaw')
fd.weakform.StressEquilibrium("ElasticLaw")

#Assembly (print the time required for assembling)
fd.Assembly.create("ElasticLaw", mesh, 'hex8', name="Assembling") 

#Type of problem 
pb = fd.problem.Linear("Assembling")

#Boundary conditions
nodes_left = mesh.node_sets["left"]
nodes_right = mesh.node_sets["right"]
nodes_top = mesh.node_sets["top"]
nodes_bottom = mesh.node_sets["bottom"]

pb.bc.add('Dirichlet',nodes_left, 'Disp', 0)
pb.bc.add('Dirichlet',nodes_right, 'DispY', -10)

pb.apply_boundary_conditions()

#--------------- Solve --------------------------------------------------------
t0 = time.time() 
# pb.set_solver('cg') #uncomment for conjugate gradient solver
print('Solving...')
pb.solve() 
print('Done in ' +str(time.time()-t0) + ' seconds')

#--------------- Post-Treatment -----------------------------------------------
res = pb.get_results('Assembling', ['Stress']).plot('Stress')

#Get the displacement vector on nodes for export to vtk
U = np.reshape(pb.get_dof_solution('all'),(3,-1)).T

#Get the stress tensor (nodal values)
TensorStrain = fd.Assembly['Assembling'].get_strain(pb.get_dof_solution(), "Node")       
TensorStress = pb.get_results('Assembling', ['Stress'], 'Node')['Stress']

