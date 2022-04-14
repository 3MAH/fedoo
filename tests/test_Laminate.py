#
# Plate element to model the canteleaver beam using different kind of plate elements
#

from fedoo import *
import numpy as np
# from matplotlib import pylab as plt
# import os 

Assembly.DeleteMemory()

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

mat1 = ConstitutiveLaw.ElasticIsotrop(E, nu, ID = 'Mat1')
mat2 = ConstitutiveLaw.ElasticIsotrop(E/10, nu, ID = 'Mat2')

# ConstitutiveLaw.ShellHomogeneous('Material', thickness, ID = 'PlateSection')
ConstitutiveLaw.ShellLaminate(['Mat1', 'Mat2', 'Mat1'], [0.2,1,0.2], ID = 'PlateSection')

mesh = Mesh.RectangleMesh(21,5,0,L,-h/2,h/2, geomElementType, ndim = 3, ID='plate')

nodes_left = mesh.GetSetOfNodes('left')
nodes_right = mesh.GetSetOfNodes('right')

node_right_center = nodes_right[(mesh.GetNodeCoordinates()[nodes_right,1]**2).argmin()]


if reduced_integration == False:
    WeakForm.Plate("PlateSection", ID = "WFplate") #by default k=0 i.e. no shear effect
    Assembly.Create("WFplate", "plate", plateElementType, ID="plate")    
    post_tt_assembly = 'plate'
else:    
    WeakForm.Plate_RI("PlateSection", ID = "WFplate_RI") #by default k=0 i.e. no shear effect
    Assembly.Create("WFplate_RI", "plate", plateElementType, ID="plate_RI", nb_pg = 1)    
    
    WeakForm.Plate_FI("PlateSection", ID = "WFplate_FI") #by default k=0 i.e. no shear effect
    Assembly.Create("WFplate_FI", "plate", plateElementType, ID="plate_FI") 
    
    Assembly.Sum("plate_RI", "plate_FI", ID = "plate")
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


# z, StressDistribution = ConstitutiveLaw.GetAll()['PlateSection'].GetStressDistribution(200)
# plt.plot(StressDistribution[0], z)