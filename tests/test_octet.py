from fedoo import *
import numpy as np
import time
 
Assembly.DeleteMemory()

#Define the Modeling Space - Here 3D problem 
Util.ProblemDimension("3D")

#Import the mesh generated with Microgen
Mesh.ImportFromFile('data/MeshPeriodic.msh', meshID = "Domain")

#Get the imported mesh 
mesh = Mesh.GetAll()["Domain2"]

#Get the bounding box (corners coordinates and center)
Xmin, Xmax, crd_center = mesh.GetBoundingBox(return_center = True)
# total volume of the bounding box
Volume = (Xmax-Xmin).prod()

#Nearest node to the center of the bounding box for boundary conditions
center = mesh.GetNearestNode(crd_center)

# Add 2 virtual nodes for macro strain 
StrainNodes = Mesh.GetAll()["Domain2"].AddNodes(crd_center, 2)  

# Material definition and simcoon elasto-plastic constitutive law
Re = 300
k = 1000
m = 0.25
alpha = 1e-5
props = np.array([[1e5, 0.3, alpha, Re, k, m]])
Material = ConstitutiveLaw.Simcoon("EPICP", props, 8, ID='ConstitutiveLaw')

#Create the weak formulation of the mechanical equilibrium equation
wf = WeakForm.InternalForce("ConstitutiveLaw", ID = "WeakForm", nlgeom=False)

# Assembly
assemb = Assembly.Create("WeakForm", "Domain2", 'tet4', ID="Assembly")

# Type of problem
Problem.NonLinearStatic("Assembly")

# Set the desired ouputs at each time step
# Problem.AddOutput('results', 'Assembly', ['disp', 'cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Node', file_format ='vtk')

# Boundary conditions for the linearized strain tensor
E = [0, 0, 0, 0.1, 0, 0]  # [EXX, EYY, EZZ, EXY, EXZ, EYZ]

Util.DefinePeriodicBoundaryCondition('Domain2',
	[StrainNodes[0], StrainNodes[0], StrainNodes[0],
         StrainNodes[1], StrainNodes[1], StrainNodes[1]],
          ['DispX', 'DispY', 'DispZ', 'DispX', 'DispY', 'DispZ'], dim='3D')

#fixed point on the center to avoid rigid body motion
Problem.BoundaryCondition('Dirichlet', 'Disp', 0, center)

#Enforced mean strain
Problem.BoundaryCondition('Dirichlet', 'Disp', [E[0], E[1], E[2]], [
                           StrainNodes[0]])  # EpsXX, EpsYY, EpsZZ
Problem.BoundaryCondition('Dirichlet', 'Disp', [E[3], E[4], E[5]], [
                           StrainNodes[1]])  # EpsXY, EpsXZ, EpsYZ

Problem.ApplyBoundaryCondition()

# ---------------  Non linear solver--------------------------------------------
Problem.SetSolver('CG') #conjugate gradient solver
Problem.NLSolve(dt=0.2, tmax=1, update_dt=False, ToleranceNR=0.1)

# --------------- Post-Treatment -----------------------------------------------
# Get the stress and strain tensor (PG values)
res = Problem.GetResults('Assembly', ['Strain','Stress'], 'GaussPoint') 
TensorStrain = res['Strain']
TensorStress = res['Stress']

assert np.abs(TensorStress[4][222]-72.37808598199845) <1e-15
assert np.abs(TensorStrain[2][876]-0.030469198588675583) <1e-15

