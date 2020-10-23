from fedoo import *
import numpy as np
import time

#------------------------------------------------------------------------------
# Dimension of the problem
#------------------------------------------------------------------------------
Util.ProblemDimension("3D") 

#------------------------------------------------------------------------------
# Definition of the Geometry 
#------------------------------------------------------------------------------
# INP = Util.ReadINP('Job-1.inp')

#warning: this mesh is not periodic and should be replaced by a better one !
INP = Util.ReadINP('cell_taffetas.inp') 
INP.toMesh(meshID = "Domain")

E_fiber = 250e3
E_matrix = 4e3
nu_fiber = 0.3
nu_matrix = 0.33

# data = Util.ExportData(Mesh.GetAll()['cell_taffetas'])
# data.toVTK()

#alternative mesh below (uncomment the line)
# Mesh.BoxMesh(Nx=11, Ny=11, Nz=11, x_min=-1, x_max=1, y_min=-1, y_max=1, z_min = -1, z_max = 1, ElementShape = 'hex8', ID ="Domain" )
# E=E_fiber ; nu = nu_fiber

meshID = "Domain" #ID of the mesh    
type_el = Mesh.GetAll()[meshID].GetElementShape() #Type of element for geometrical interpolation
try:
    list_Elm_Matrix = Mesh.GetAll()[meshID].GetSetOfElements('alla_matrix')
except: 
    print('No matrix element')
    list_Elm_Matrix = []

#------------------------------------------------------------------------------
# Set of nodes for boundary conditions
#------------------------------------------------------------------------------

crd = Mesh.GetAll()[meshID].GetNodeCoordinates() 
xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
zmax = np.max(crd[:,2]) ; zmin = np.min(crd[:,2])
center = [np.linalg.norm(crd,axis=1).argmin()]


#------------------------------------------------------------------------------
# Adding virtual nodes related the macroscopic strain
#------------------------------------------------------------------------------
StrainNodes = Mesh.GetAll()[meshID].AddNodes(np.zeros(crd.shape[1]),1) 
#The position of the virtual node has no importance (the position is arbitrary set to [0,0,0])
#For a problem in 3D with a 2D periodicity, we have 3 strain component that can be represented with only one node
#In case of a 2D problem, 2 nodes will be required

#------------------------------------------------------------------------------
#Material definition
#------------------------------------------------------------------------------
E = E_fiber*np.ones(Mesh.GetAll()[meshID].GetNumberOfElements())
nu = nu_fiber*np.ones(Mesh.GetAll()[meshID].GetNumberOfElements())
E[list_Elm_Matrix] = E_matrix
nu[list_Elm_Matrix] = nu_matrix
#convert element value to pg value
E = Assembly.ConvertData(E, meshID)
nu = Assembly.ConvertData(nu, meshID)

ConstitutiveLaw.ElasticIsotrop(E, nu, ID = 'ElasticLaw')

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
E = [1,0,0] #macroscopic strain tensor [EXX, EYY, EXY]
#For the node StrainNode[0], 'DispX' is a virtual dof for EXX
#For the node StrainNode[0], 'DispY' is a virtual dof for EYY
#For the node StrainNode[0], 'DispZ' is a virtual dof for EXY

#Apply the periodic boundary conditions 
Util.DefinePeriodicBoundaryCondition("Domain", 
        [StrainNodes[0], StrainNodes[0], StrainNodes[0]], 
        ['DispX', 'DispY', 'DispZ'], dim='2d')

#Block a node on the center to avoid rigid body motion
Problem.BoundaryCondition('Dirichlet','DispX', 0, center)
Problem.BoundaryCondition('Dirichlet','DispY', 0, center)
Problem.BoundaryCondition('Dirichlet','DispZ', 0, center)

Problem.BoundaryCondition('Dirichlet','DispX', E[0], [StrainNodes[0]]) #EpsXX
Problem.BoundaryCondition('Dirichlet','DispY', E[1], [StrainNodes[0]]) #EpsYY
Problem.BoundaryCondition('Dirichlet','DispZ', E[2], [StrainNodes[0]]) #EpsXY

Problem.ApplyBoundaryCondition()

#------------------------------------------------------------------------------
#Solve
#------------------------------------------------------------------------------
Problem.SetSolver('CG') #Preconditioned Conjugate Gradient
Problem.Solve()

#------------------------------------------------------------------------------
# Post-treatment
#------------------------------------------------------------------------------

#Compute the mean stress and strain
#Get the stress tensor (PG values)
TensorStrain = Assembly.GetAll()['Assembling'].GetStrainTensor(Problem.GetDisp(), "GaussPoint")       
TensorStress = ConstitutiveLaw.GetAll()['ElasticLaw'].GetStress(TensorStrain)

Volume = (xmax-xmin)*(ymax-ymin)*(zmax-zmin) #total volume of the domain
Volume_mesh = Assembly.GetAll()['Assembling'].IntegrateField(np.ones_like(TensorStress[0])) #volume of domain without the void (hole)

MeanStress = [1/Volume*Assembly.GetAll()['Assembling'].IntegrateField(TensorStress[i]) for i in range(6)] 

# MeanStrain only work if volume with no void
# Void = Volume-Volume_mesh 
MeanStrain = [1/Volume*Assembly.GetAll()['Assembling'].IntegrateField(TensorStrain[i]) for i in range(6)] 
# print(ConstitutiveLaw.GetAll()['ElasticLaw'].GetH()@np.array(MeanStrain)) #should be the same as MeanStress if homogeneous material and no void

print('Strain tensor ([Exx, Eyy, Ezz, Exy, Exz, Eyz]): ', MeanStrain)
print('Stress tensor ([Sxx, Syy, Szz, Sxy, Sxz, Syz]): ', MeanStress)

print('Elastic Energy: ' + str(Problem.GetElasticEnergy()))

#------------------------------------------------------------------------------
#Optional: Compute and write data in a vtk file (for visualization with paraview for instance)
#------------------------------------------------------------------------------

#Get the stress tensor (nodal values converted from PG values)
TensorStrainNd = Assembly.ConvertData(TensorStrain, meshID, convertTo = "Node")
TensorStressNd = Assembly.ConvertData(TensorStress, meshID, convertTo = "Node")

#Get the stress tensor (element values)
TensorStrainEl = Assembly.ConvertData(TensorStrain, meshID, convertTo = "Element")       
TensorStressEl = Assembly.ConvertData(TensorStress, meshID, convertTo = "Element")

# Get the principal directions (vectors on nodes)
PrincipalStress, PrincipalDirection = TensorStressNd.GetPrincipalStress()

#Get the displacement vector on nodes for export to vtk
U = np.reshape(Problem.GetDoFSolution('all'),(3,-1)).T
N = Mesh.GetAll()[meshID].GetNumberOfNodes()
# U = np.c_[U,np.zeros(N)]

SetId = 'all_fibers' #or None
if SetId is None:
    OUT = Util.ExportData(meshID)
    ElSet = slice(None) #we keep all elements
else:
    Mesh.GetAll()[meshID].ExtractSetOfElements(SetId, ID="output")
    ElSet = Mesh.GetAll()[meshID].GetSetOfElements(SetId) #keep only some elements    
    OUT = Util.ExportData("output")

#Write the vtk file                            
OUT.addNodeData(U,'Displacement')
OUT.addNodeData(TensorStressNd.vtkFormat(),'Stress')
OUT.addElmData(TensorStressEl.vtkFormat()[ElSet],'Stress')
OUT.addNodeData(TensorStrainNd.vtkFormat(),'Strain')
OUT.addElmData(TensorStrainEl.vtkFormat()[ElSet],'Strain')
OUT.addNodeData(TensorStressNd.vonMises(),'VMStress')
OUT.addElmData(TensorStressEl.vonMises()[ElSet],'VMStress')
OUT.addNodeData(PrincipalStress,'PrincipalStress')
OUT.addNodeData(PrincipalDirection[0],'DirPrincipal1')
OUT.addNodeData(PrincipalDirection[1],'DirPrincipal2')

OUT.toVTK("2D_Periodic_BC_in_composite_ply.vtk")
print('Result file "plate_with_hole_in_tension.vtk" written in the active directory')


