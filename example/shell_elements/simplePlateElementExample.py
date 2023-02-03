#
# Plate element to model the canteleaver beam using different kind of plate elements
#

from fedoo import *
import numpy as np
from matplotlib import pylab as plt
import os 

ModelingSpace("3D")
E = 1e5 
nu = 0.3

L = 100
h = 20
thickness = 1
F = -10

geomElementType = 'quad4' #choose among 'tri3', 'tri6', 'quad4', 'quad9'
plateElementType = 'p'+geomElementType #plate interpolation. Same as geom interpolation in local element coordinate (change of basis)
reduced_integration = True #if true, use reduce integration for shear 
saveResults = True

material = ConstitutiveLaw.ElasticIsotrop(E, nu, name = 'Material')
ConstitutiveLaw.ShellHomogeneous('Material', thickness, name = 'PlateSection')

mesh = Mesh.rectangle_mesh(201,21,0,L,-h/2,h/2, geomElementType, name='plate', ndim = 3)

nodes_left = mesh.node_sets['left']
nodes_right = mesh.node_sets['right']

node_right_center = nodes_right[(mesh.nodes[nodes_right,1]**2).argmin()]


if reduced_integration == False:
    WeakForm.Plate("PlateSection", name = "WFplate") #by default k=0 i.e. no shear effect
    Assembly.create("WFplate", "plate", plateElementType, name="plate")    
    post_tt_assembly = 'plate'
else:    
    WeakForm.Plate_RI("PlateSection", name = "WFplate_RI") #by default k=0 i.e. no shear effect
    Assembly.create("WFplate_RI", "plate", plateElementType, name="plate_RI", n_elm_gp = 1)    
    
    WeakForm.Plate_FI("PlateSection", name = "WFplate_FI") #by default k=0 i.e. no shear effect
    Assembly.create("WFplate_FI", "plate", plateElementType, name="plate_FI") 
    
    Assembly.Sum("plate_RI", "plate_FI", name = "plate")
    post_tt_assembly = 'plate_FI'


Problem.Linear("plate")

#create a 'result' folder and set the desired ouputs
if not(os.path.isdir('results')): os.mkdir('results')
Problem.add_output('results/simplePlate', post_tt_assembly, ['disp','rot', 'stress', 'strain'], output_type='Node', file_format ='vtk', position = -1)    


Problem.bc.add('Dirichlet','DispX',0,nodes_left)
Problem.bc.add('Dirichlet','DispY',0,nodes_left)
Problem.bc.add('Dirichlet','DispZ',0,nodes_left)
Problem.bc.add('Dirichlet','RotX',0,nodes_left)
Problem.bc.add('Dirichlet','RotY',0,nodes_left)
Problem.bc.add('Dirichlet','RotZ',0,nodes_left)

Problem.bc.add('Neumann','DispZ',F,node_right_center)

Problem.apply_boundary_conditions()
Problem.solve()

I = h*thickness**3/12
print('Beam analitical deflection: ', F*L**3/(3*E*I))
print('Numerical deflection: ', Problem.get_disp('DispZ')[node_right_center])

if saveResults == True: 
    Problem.save_results() #save in vtk

z, StressDistribution = ConstitutiveLaw.get_all()['PlateSection'].GetStressDistribution(20)
plt.plot(StressDistribution[0], z)