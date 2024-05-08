"""
Canteleaver Beam using 3D wedge elements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

import fedoo as fd
import numpy as np

#--------------- Pre-Treatment --------------------------------------------------------
fd.ModelingSpace("3D")

#Units: N, mm, MPa

#build a mesh with wedge 'wed6' elements by  extrusion from a tri3 mesh
mesh = fd.mesh.extrude(fd.mesh.rectangle_mesh(nx=5, ny=5, x_max = 100, y_max = 100, elm_type = 'tri3'),
                       1000, #extrusion length
                       51) #n_nodes
mesh.nodes = mesh.nodes[:,[2,0,1]] #switch axis to put the extrusion direction along the X axis

#change the type of element
mesh = fd.mesh.functions.change_elm_type(mesh, 'wed15') #or 'wed18'

# fd.DataSet(mesh.extract_elements([0])).plot(node_labels=True, opacity=0.85)

#Material definition
material = fd.constitutivelaw.ElasticIsotrop(200e3, 0.3)
wf = fd.weakform.StressEquilibrium(material)

#Assembly
assemb = fd.Assembly.create(wf, mesh, n_elm_gp=21, name="Assembling") 

#Type of problem 
pb = fd.problem.Linear("Assembling")

#Boundary conditions
nodes_left = mesh.find_nodes('X', mesh.bounding_box.xmin)
nodes_right = mesh.find_nodes('X', mesh.bounding_box.xmax)

pb.bc.add('Dirichlet',nodes_left, 'Disp', 0)
pb.bc.add('Dirichlet',nodes_right, 'DispY', -10)

pb.apply_boundary_conditions()

#--------------- Solve --------------------------------------------------------
# pb.set_solver('cg') #uncomment for conjugate gradient solver
pb.solve() 

res = pb.get_results('Assembling', ['Stress', 'Disp'])
res.plot('Stress', component='XX', scale=10, show_edges=False)



