#
# Plate element to model the canteleaver beam using different kind of plate elements
#

from fedoo import *
import numpy as np
from matplotlib import pylab as plt
import os 

Util.ProblemDimension("3D")
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

material = ConstitutiveLaw.ElasticIsotrop(E, nu, ID = 'Material')
ConstitutiveLaw.ShellHomogeneous('Material', thickness, ID = 'PlateSection')

mesh = Mesh.RectangleMesh(201,21,0,L,-h/2,h/2, geomElementType, ID='plate', ndim = 3)

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
if not(os.path.isdir('results')): os.mkdir('results')
Problem.AddOutput('results/simplePlate', post_tt_assembly, ['disp','rot', 'stress', 'strain'], output_type='Node', file_format ='vtk', position = -1)    


Problem.BoundaryCondition('Dirichlet','DispX',0,nodes_left)
Problem.BoundaryCondition('Dirichlet','DispY',0,nodes_left)
Problem.BoundaryCondition('Dirichlet','DispZ',0,nodes_left)
Problem.BoundaryCondition('Dirichlet','RotX',0,nodes_left)
Problem.BoundaryCondition('Dirichlet','RotY',0,nodes_left)
Problem.BoundaryCondition('Dirichlet','RotZ',0,nodes_left)

Problem.BoundaryCondition('Neumann','DispZ',F,node_right_center)

Problem.ApplyBoundaryCondition()
Problem.Solve()

I = h*thickness**3/12
print('Beam analitical deflection: ', F*L**3/(3*E*I))
print('Numerical deflection: ', Problem.GetDisp('DispZ')[node_right_center])

if saveResults == True: 
    Problem.SaveResults() #save in vtk

z, StressDistribution = ConstitutiveLaw.GetAll()['PlateSection'].GetStressDistribution(20)
plt.plot(StressDistribution[0], z)