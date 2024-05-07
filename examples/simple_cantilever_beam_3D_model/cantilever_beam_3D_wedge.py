import fedoo as fd
import numpy as np
import time

#--------------- Pre-Treatment --------------------------------------------------------
fd.ModelingSpace("3D")

#Units: N, mm, MPa

#build a mesh with wedge 'wed6' elements by  extrusion from a tri3 mesh
mesh = fd.mesh.extrude(fd.mesh.rectangle_mesh(nx=11, ny=11, x_max = 100, y_max = 100, elm_type = 'tri3'),
                       1000, #extrusion length
                       101) #n_nodes
mesh.nodes = mesh.nodes[:,[2,0,1]] #switch axis to put the extrusion direction along the X axis

#change the type of element
mesh = fd.mesh.functions.change_elm_type(mesh, 'wed15') #or 'wed18'

# fd.DataSet(mesh.extract_elements([0])).plot(node_labels=True, opacity=0.85)

#Material definition
fd.constitutivelaw.ElasticIsotrop(200e3, 0.3, name = 'ElasticLaw')
fd.weakform.StressEquilibrium("ElasticLaw")

#Assembly (print the time required for assembling)
assemb = fd.Assembly.create("ElasticLaw", mesh, n_elm_gp=21, name="Assembling") 

#Type of problem 
pb = fd.problem.Linear("Assembling")

#Boundary conditions
nodes_left = mesh.find_nodes('X', mesh.bounding_box.xmin)
nodes_right = mesh.find_nodes('X', mesh.bounding_box.xmax)

pb.bc.add('Dirichlet',nodes_left, 'Disp', 0)
pb.bc.add('Dirichlet',nodes_right, 'DispY', -10)

pb.apply_boundary_conditions()

#--------------- Solve --------------------------------------------------------
t0 = time.time() 
# pb.set_solver('cg') #uncomment for conjugate gradient solver
print('Solving...')
pb.solve() 
print('Done in ' +str(time.time()-t0) + ' seconds')

res = pb.get_results('Assembling', ['Stress', 'Disp'])
res.plot('Stress', component='XX', scale=10, show_edges=False)



