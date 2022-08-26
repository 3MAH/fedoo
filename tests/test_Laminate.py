#
# Plate element to model the canteleaver beam using different kind of plate elements
#

import fedoo as fd
import numpy as np
# from matplotlib import pylab as plt
# import os 

fd.Assembly.delete_memory()

fd.ModelingSpace("3D")
E = 1e5 
nu = 0.3

L = 51
h = 11
thickness = 1
F = -100

geomElementType = 'quad4' #choose among 'tri3', 'tri6', 'quad4', 'quad9'
plateElementType = 'p'+geomElementType #plate interpolation. Same as geom interpolation in local element coordinate (change of basis)
reduced_integration = True #if true, use reduce integration for shear 

mat1 = fd.constitutivelaw.ElasticIsotrop(E, nu, name = 'Mat1')
mat2 = fd.constitutivelaw.ElasticIsotrop(E/10, nu, name = 'Mat2')

# ConstitutiveLaw.ShellHomogeneous('Material', thickness, name = 'PlateSection')
fd.constitutivelaw.ShellLaminate(['Mat1', 'Mat2', 'Mat1'], [0.2,1,0.2], name = 'PlateSection')

mesh = fd.mesh.rectangle_mesh(21,5,0,L,-h/2,h/2, geomElementType, ndim = 3, name='plate')

nodes_left = mesh.node_sets['left']
nodes_right = mesh.node_sets['right']

node_right_center = nodes_right[(mesh.nodes[nodes_right,1]**2).argmin()]


if reduced_integration == False:
    fd.weakform.Plate("PlateSection", name = "WFplate") #by default k=0 i.e. no shear effect
    fd.Assembly.create("WFplate", "plate", plateElementType, name="plate")    
    post_tt_assembly = 'plate'
else:    
    fd.weakform.Plate_RI("PlateSection", name = "WFplate_RI") #by default k=0 i.e. no shear effect
    fd.Assembly.create("WFplate_RI", "plate", plateElementType, name="plate_RI", n_elm_gp = 1)    
    
    fd.weakform.Plate_FI("PlateSection", name = "WFplate_FI") #by default k=0 i.e. no shear effect
    fd.Assembly.create("WFplate_FI", "plate", plateElementType, name="plate_FI") 
    
    fd.Assembly.sum("plate_RI", "plate_FI", name = "plate")
    post_tt_assembly = 'plate_FI'


pb = fd.problem.Static("plate")

#create a 'result' folder and set the desired ouputs
# if not(os.path.isdir('results')): os.mkdir('results')
# Problem.add_output('results/simplePlate', post_tt_assembly, ['disp','rot', 'stress', 'strain'], output_type='Node', file_format ='vtk', position = -1)    


pb.bc.add('Dirichlet','DispX',0,nodes_left)
pb.bc.add('Dirichlet','DispY',0,nodes_left)
pb.bc.add('Dirichlet','DispZ',0,nodes_left)
pb.bc.add('Dirichlet','RotX',0,nodes_left)
pb.bc.add('Dirichlet','RotY',0,nodes_left)
pb.bc.add('Dirichlet','RotZ',0,nodes_left)

pb.bc.add('Neumann','DispZ',F,node_right_center)

pb.apply_boundary_conditions()
pb.solve()

assert np.abs(pb.get_disp('DispZ')[node_right_center]+25.768895223177235) < 1e-15


# z, StressDistribution = ConstitutiveLaw.get_all()['PlateSection'].GetStressDistribution(200)
# plt.plot(StressDistribution[0], z)