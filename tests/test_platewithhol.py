from fedoo import *
import numpy as np

#Define the Modeling Space - Here 2D problem with plane stress assumption.
Util.ProblemDimension("2Dstress") 

#Generate a simple structured mesh "Domain" (plate with a hole).
meshObject = Mesh.HolePlateMesh(Nx=11, Ny=11, Lx=100, Ly=100, R=20, \
	ElementShape = 'quad4', ID ="Domain") 

#Define an elastic isotropic material with E = 2e5MPa et nu = 0.3 (steel)
ConstitutiveLaw.ElasticIsotrop(2e5, 0.3, ID = 'ElasticLaw') 

#Create the weak formulation of the mechanical equilibrium equation
WeakForm.InternalForce("ElasticLaw", ID = "WeakForm") 

#Create a global assembly
Assembly.Create("WeakForm", "Domain", ID="Assembly", MeshChange = True) 

#Define a new static problem
Problem.Static("Assembly")

#Definition of the set of nodes for boundary conditions
crd = meshObject.GetNodeCoordinates()
left  = np.where(crd[:,0] == np.min(crd[:,0]))[0]
right = np.where(crd[:,0] == np.max(crd[:,0]))[0]  
bottom = np.where(crd[:,1] == np.min(crd[:,1]))[0] 

#Boundary conditions
#symetry condition on left (ux = 0)
Problem.BoundaryCondition('Dirichlet','DispX',    0  , left) 
#symetry condition on bottom edge (ux = 0)
Problem.BoundaryCondition('Dirichlet','DispY',    0  , bottom) 
#displacement on right (ux=0.1mm)
Problem.BoundaryCondition('Dirichlet','DispX', 0.1, right) 

Problem.ApplyBoundaryCondition()

#Solve problem
Problem.Solve()

#---------- Post-Treatment ----------
#Get the stress tensor, strain tensor, and displacement (nodal values)
res_nd = Problem.GetResults("Assembly", ['Stress_VM','Strain'], 'Node')
U = Problem.GetDisp()

assert U[0,22] == 0.1
assert np.abs(U[1,22] +0.010440829731661383) < 1e-15
assert np.abs(res_nd['Stress_Mises'][53]-350.22455046711923) < 1e-15

