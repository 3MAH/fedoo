#
# Plate element to model the canteleaver beam using different kind of plate elements
#

from fedoo import *
import numpy as np
# from matplotlib import pylab as plt
# import os 

Assembly.delete_memory()

Util.ProblemDimension("3D")
E = 1e5 
nu = 0.3

L = 51
h = 11
thickness = 1
F = -100

geomElementType = 'quad4' #choose among 'tri3', 'tri6', 'quad4', 'quad9'
plateElementType = 'p'+geomElementType #plate interpolation. Same as geom interpolation in local element coordinate (change of basis)
reduced_integration = True #if true, use reduce integration for shear 

mat1 = ConstitutiveLaw.ElasticIsotrop(E, nu, name = 'Mat1')
mat2 = ConstitutiveLaw.ElasticIsotrop(E/10, nu, name = 'Mat2')

# ConstitutiveLaw.ShellHomogeneous('Material', thickness, name = 'PlateSection')
ConstitutiveLaw.ShellLaminate(['Mat1', 'Mat2', 'Mat1'], [0.2,1,0.2], name = 'PlateSection')

mesh = Mesh.rectangle_mesh(21,5,0,L,-h/2,h/2, geomElementType, ndim = 3, name='plate')

nodes_left = mesh.node_sets['left']
nodes_right = mesh.node_sets['right']

node_right_center = nodes_right[(mesh.nodes[nodes_right,1]**2).argmin()]


if reduced_integration == False:
    WeakForm.Plate("PlateSection", name = "WFplate") #by default k=0 i.e. no shear effect
    Assembly.Create("WFplate", "plate", plateElementType, name="plate")    
    post_tt_assembly = 'plate'
else:    
    WeakForm.Plate_RI("PlateSection", name = "WFplate_RI") #by default k=0 i.e. no shear effect
    Assembly.Create("WFplate_RI", "plate", plateElementType, name="plate_RI", nb_pg = 1)    
    
    WeakForm.Plate_FI("PlateSection", name = "WFplate_FI") #by default k=0 i.e. no shear effect
    Assembly.Create("WFplate_FI", "plate", plateElementType, name="plate_FI") 
    
    Assembly.Sum("plate_RI", "plate_FI", name = "plate")
    post_tt_assembly = 'plate_FI'


Problem.Static("plate")

#create a 'result' folder and set the desired ouputs
# if not(os.path.isdir('results')): os.mkdir('results')
# Problem.AddOutput('results/simplePlate', post_tt_assembly, ['disp','rot', 'stress', 'strain'], output_type='Node', file_format ='vtk', position = -1)    


Problem.BoundaryCondition('Dirichlet','DispX',0,nodes_left)
Problem.BoundaryCondition('Dirichlet','DispY',0,nodes_left)
Problem.BoundaryCondition('Dirichlet','DispZ',0,nodes_left)
Problem.BoundaryCondition('Dirichlet','RotX',0,nodes_left)
Problem.BoundaryCondition('Dirichlet','RotY',0,nodes_left)
Problem.BoundaryCondition('Dirichlet','RotZ',0,nodes_left)

Problem.BoundaryCondition('Neumann','DispZ',F,node_right_center)

Problem.ApplyBoundaryCondition()
Problem.Solve()

assert np.abs(Problem.GetDisp('DispZ')[node_right_center]+25.768895223177235) < 1e-15


# z, StressDistribution = ConstitutiveLaw.get_all()['PlateSection'].GetStressDistribution(200)
# plt.plot(StressDistribution[0], z)