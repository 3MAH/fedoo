from fedoo import *
import numpy as np
import time

#------------------------------------------------------------------------------
# Dimension of the problem
#------------------------------------------------------------------------------
Util.ProblemDimension("2Dstress")

#------------------------------------------------------------------------------
# Definition of the Geometry 
#------------------------------------------------------------------------------
Mesh.ImportFromFile('plate_with_hole.msh', meshID = "Domain")

#alternative mesh below (uncomment the line)
# Mesh.RectangleMesh(Nx=51, Ny=51, x_min=-50, x_max=50, y_min=-50, y_max=50, ElementShape = 'quad4', ID ="Domain")
meshID = "Domain"
type_el = Mesh.GetAll()[meshID].GetElementShape()
Util.meshPlot2d(meshID) #plot the mesh (using matplotlib)

#------------------------------------------------------------------------------
# Set of nodes for boundary conditions
#------------------------------------------------------------------------------
mesh = Mesh.GetAll()[meshID]
crd = mesh.GetNodeCoordinates() 
xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])

center = [np.linalg.norm(crd,axis=1).argmin()]

#------------------------------------------------------------------------------
# Adding virtual nodes related the macroscopic strain
#------------------------------------------------------------------------------
StrainNodes = Mesh.GetAll()[meshID].AddNodes(np.zeros(crd.shape[1]),2) 
#The position of the virtual node has no importance (the position is arbitrary set to [0,0,0])
#For a problem in 2D with a 2D periodicity, we need 3 independant strain component 
#2 nodes (with 2 dof per node in 2D) are required

#The dof 'DispX' of the node StrainNodes[0] will be arbitrary associated to the EXX strain component
#The dof 'DispY' of the node StrainNodes[1] will be arbitrary associated to the EYY strain component
#The dof 'DispY' of the node StrainNodes[0] will be arbitrary associated to the EXY strain component
#The dof 'DispX' of the node StrainNodes[1] is not used and will be blocked to avoid singularity

#------------------------------------------------------------------------------
#Material definition
#------------------------------------------------------------------------------
ConstitutiveLaw.ElasticIsotrop(1e5, 0.3, ID = 'ElasticLaw')

#------------------------------------------------------------------------------
#Mechanical weak formulation
#------------------------------------------------------------------------------
WeakForm.InternalForce("ElasticLaw")

#------------------------------------------------------------------------------
#Global Matrix assembly
#------------------------------------------------------------------------------
Assembly.Create("ElasticLaw", meshID, type_el, ID="Assembling") 

#------------------------------------------------------------------------------
#Static problem based on the just defined assembly
#------------------------------------------------------------------------------
Problem.Static("Assembling")

#------------------------------------------------------------------------------
#Boundary conditions
#------------------------------------------------------------------------------
#Macroscopic strain component to enforce
Exx = 0
Eyy = 0
Exy = 0.1

#Add some multipoint constraint for periodic conditions associated to the defined strain dof
Util.DefinePeriodicBoundaryCondition("Domain", [StrainNodes[0], StrainNodes[1], StrainNodes[0]], ['DispX', 'DispY', 'DispY'], dim='2D')

#Mean strain: Dirichlet (strain) or Neumann (associated mean stress) can be enforced
Problem.BoundaryCondition('Dirichlet','DispX', Exx, [StrainNodes[0]]) #EpsXX
Problem.BoundaryCondition('Dirichlet','DispY', Exy, [StrainNodes[0]]) #EpsXY
# Problem.BoundaryCondition('Neumann','DispX', 1e4, [StrainNodes[0]]) #EpsXY

Problem.BoundaryCondition('Dirichlet','DispX', 0, [StrainNodes[1]]) #nothing (blocked to avoir singularity)
Problem.BoundaryCondition('Dirichlet','DispY', Eyy, [StrainNodes[1]]) #EpsYY

#Block one node to avoid singularity
Problem.BoundaryCondition('Dirichlet','DispX', 0, center)
Problem.BoundaryCondition('Dirichlet','DispY', 0, center)

Problem.ApplyBoundaryCondition()

#------------------------------------------------------------------------------
#Solve
#------------------------------------------------------------------------------
Problem.SetSolver('CG') #Preconditioned Conjugate Gradient
Problem.Solve()

#------------------------------------------------------------------------------
# Post-treatment
#------------------------------------------------------------------------------
#plot the deformed mesh with the shear stress (component=5)
Util.fieldPlot2d("Domain", "ElasticLaw", disp = Problem.GetDisp(), dataID = 'stress', component=3, scale_factor = 1, plot_edge = True, nb_level = 6, type_plot = "smooth")

# print the macroscopic strain tensor and stress tensor
print('Strain tensor ([Exx, Eyy, Exy]): ', [Problem.GetDisp('DispX')[-2], Problem.GetDisp('DispY')[-1], Problem.GetDisp('DispY')[-2]])
#Compute the mean stress 
#Get the stress tensor (PG values)
TensorStrain = Assembly.GetAll()['Assembling'].GetStrainTensor(Problem.GetDisp(), "GaussPoint")       
TensorStress = ConstitutiveLaw.GetAll()['ElasticLaw'].GetStress(TensorStrain)

# Surf = Assembly.GetAll()['Assembling'].IntegrateField(np.ones_like(TensorStress[0])) #surface of domain without the void (hole)
Surf = (xmax-xmin)*(ymax-ymin) #total surface of the domain
MeanStress = [1/Surf*Assembly.GetAll()['Assembling'].IntegrateField(TensorStress[i]) for i in [0,1,5]]

print('Stress tensor ([Sxx, Syy, Sxy]): ', MeanStress)

# print(ConstitutiveLaw.GetAll()['ElasticLaw'].GetH())

#------------------------------------------------------------------------------
#Optional: Compute and write data in a vtk file (for visualization with paraview for instance)
#------------------------------------------------------------------------------

# #Get the stress tensor (nodal values)
# TensorStrain = Assembly.GetAll()['Assembling'].GetStrainTensor(Problem.GetDisp(), "Nodal")       
# TensorStress = ConstitutiveLaw.GetAll()['ElasticLaw'].GetStress(TensorStrain)

# #Get the stress tensor (element values)
# TensorStrainEl = Assembly.GetAll()['Assembling'].GetStrainTensor(Problem.GetDisp(), "Element")       
# TensorStressEl = ConstitutiveLaw.GetAll()['ElasticLaw'].GetStress(TensorStrainEl)

# # Get the principal directions (vectors on nodes)
# PrincipalStress, PrincipalDirection = TensorStress.GetPrincipalStress()

# #Get the displacement vector on nodes for export to vtk
# U = np.reshape(Problem.GetDoFSolution('all'),(2,-1)).T
# N = Mesh.GetAll()[meshID].GetNumberOfNodes()
# U = np.c_[U,np.zeros(N)]

# #write the vtk file                     
# OUT = Util.ExportData(meshID)
# OUT.addNodeData(U,'Displacement')
# OUT.addNodeData(TensorStress.vtkFormat(),'Stress')
# OUT.addElmData(TensorStressEl.vtkFormat(),'Stress')
# OUT.addNodeData(TensorStrain.vtkFormat(),'Strain')
# OUT.addElmData(TensorStrainEl.vtkFormat(),'Strain')
# OUT.addNodeData(TensorStress.vonMises(),'VMStress')
# OUT.addElmData(TensorStressEl.vonMises(),'VMStress')
# OUT.addNodeData(PrincipalStress,'PrincipalStress')
# OUT.addNodeData(PrincipalDirection[0],'DirPrincipal1')
# OUT.addNodeData(PrincipalDirection[1],'DirPrincipal2')
# OUT.toVTK("plate_with_hole_in_tension_BC.vtk")
# print('Result file "Periodic_BC_2D_Plate_with_hole.vtk" written in the active directory')

