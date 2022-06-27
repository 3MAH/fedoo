from fedoo import *
import numpy as np
import os

#Define the Modeling Space - Here 2D problem with plane stress assumption.
Util.ProblemDimension("2Dplane", 'macro_space') 
Util.ProblemDimension("3D", 'micro_space') 

#Generate a simple structured mesh "Domain" (plate with a hole).
mesh_macro = Mesh.HolePlateMesh(Nx=3, Ny=3, Lx=100, Ly=100, R=20, \
	ElementShape = 'quad4', ID ="macro") 
    
Mesh.ImportFromFile('octet_surf.msh', meshID = "micro")
mesh_micro = Mesh.GetAll()['micro2']

# Util.meshPlot2d("macro")

#Define an elastic isotropic material with E = 2e5MPa et nu = 0.3 (steel)

E = 200e3
nu=0.3
Re = 300
k=1000
alpha = 1e-5 #???
m=0.25
uimp = -5

props = np.array([[E, nu, alpha, Re,k,m]])
Material = ConstitutiveLaw.Simcoon("EPICP", props, 8, ID='ConstitutiveLaw')

WeakForm.InternalForce("ConstitutiveLaw", ID = 'micro_wf', space = 'micro_space') 

micro_assembly = Assembly.Create('micro_wf', mesh_micro)

micro_cells = ConstitutiveLaw.FE2(micro_assembly, ID='FEM')

#Create the weak formulation of the mechanical equilibrium equation
WeakForm.InternalForce("FEM", ID = "WeakForm") 

#Create a global assembly
Assembly.Create("WeakForm", "macro", ID="Assembly", MeshChange = True) 

#Define a new static problem
Problem.NonLinearStatic("Assembly")
# Problem.SetNewtonRaphsonErrorCriterion("Displacement")

#create a 'result' folder and set the desired ouputs
if not(os.path.isdir('results')): os.mkdir('results')
Problem.AddOutput('results/FE2', 'Assembly', ['disp', 'stress', 'strain', 'stress_vm', 'wm'], output_type='Node', file_format ='vtk')    
Problem.AddOutput('results/FE2', 'Assembly', ['stress', 'stress', 'stress_vm'], output_type='Element', file_format ='vtk')    

#output result for a random micro cell (here for the 5th integration point)
#Warning : the results of micro_cells are automatically saved at each NR iteration (several iteration per time iteration)
# Problem.Initialize(0) #to build prolems
# micro_cells.list_problem[5].AddOutput('results/micro_cell', micro_cells.list_assembly[5], ['disp', 'stress', 'strain', 'stress_vm'], output_type='Node', file_format ='vtk')    


#Definition of the set of nodes for boundary conditions
crd = mesh_macro.nodes
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
Problem.NLSolve()

#---------- Post-Treatment ----------
#Get the stress tensor, strain tensor, and displacement (nodal values)
res_nd = Problem.GetResults("Assembly", ['Stress_VM','Strain'], 'Node')
U = Problem.GetDisp()
