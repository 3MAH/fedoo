from fedoo import *
import numpy as np
import time

#------------------------------------------------------------------------------
# Défine inplane 2D periodic boundary conditions for a composite ply using a 3D
# unit cell. 
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Dimension of the problem
#------------------------------------------------------------------------------
ModelingSpace("3D") 

#------------------------------------------------------------------------------
# Definition of the Geometry 
#------------------------------------------------------------------------------
# INP = Util.ReadINP('Job-1.inp')

#warning: this mesh is not periodic and should be replaced by a better one !
INP = Util.ReadINP('cell_taffetas.inp') 
INP.toMesh(meshname = "Domain")

E_fiber = 250e3
E_matrix = 4e3
nu_fiber = 0.3
nu_matrix = 0.33

# data = Util.ExportData(Mesh.get_all()['cell_taffetas'])
# data.toVTK()

#alternative mesh below (uncomment the line)
# Mesh.box_mesh(Nx=11, Ny=11, Nz=11, x_min=-1, x_max=1, y_min=-1, y_max=1, z_min = -1, z_max = 1, ElementShape = 'hex8', name ="Domain" )
# E=E_fiber ; nu = nu_fiber

meshname = "Domain" #name of the mesh    
type_el = Mesh.get_all()[meshname].elm_type #Type of element for geometrical interpolation
try:
    list_Elm_Matrix = Mesh.get_all()[meshname].element_sets['alla_matrix']
except: 
    print('No matrix element')
    list_Elm_Matrix = []

#------------------------------------------------------------------------------
# Set of nodes for boundary conditions
#------------------------------------------------------------------------------

crd = Mesh.get_all()[meshname].nodes 
xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
zmax = np.max(crd[:,2]) ; zmin = np.min(crd[:,2])
center = [np.linalg.norm(crd,axis=1).argmin()]


#------------------------------------------------------------------------------
# Adding virtual nodes related the macroscopic strain
#------------------------------------------------------------------------------
StrainNodes = Mesh.get_all()[meshname].add_nodes(np.zeros(crd.shape[1]),1) 
#The position of the virtual node has no importance (the position is arbitrary set to [0,0,0])
#For a problem in 3D with a 2D periodicity, we have 3 strain component that can be represented with only one node
#In case of a 2D problem, 2 nodes will be required

#------------------------------------------------------------------------------
#Material definition
#------------------------------------------------------------------------------
E = E_fiber*np.ones(Mesh.get_all()[meshname].n_elements)
nu = nu_fiber*np.ones(Mesh.get_all()[meshname].n_elements)
E[list_Elm_Matrix] = E_matrix
nu[list_Elm_Matrix] = nu_matrix


ConstitutiveLaw.ElasticIsotrop(E, nu, name = 'ElasticLaw')

#------------------------------------------------------------------------------
#Mechanical weak formulation
#------------------------------------------------------------------------------
WeakForm.StressEquilibrium("ElasticLaw")

#------------------------------------------------------------------------------
#Global Matrix assembly
#------------------------------------------------------------------------------
Assembly.create("ElasticLaw", meshname, type_el, name="Assembling") 

#------------------------------------------------------------------------------
#Static problem based on the just defined assembly
#------------------------------------------------------------------------------
Problem.Linear("Assembling")

#------------------------------------------------------------------------------
#Boundary conditions
#------------------------------------------------------------------------------
E = [1,0,0] #macroscopic strain tensor [EXX, EYY, EXY]
#For the node StrainNode[0], 'DispX' is a virtual dof for EXX
#For the node StrainNode[0], 'DispY' is a virtual dof for EYY
#For the node StrainNode[0], 'DispZ' is a virtual dof for EXY

#Apply the periodic boundary conditions 
Homogen.DefinePeriodicBoundaryCondition("Domain", 
        [StrainNodes[0], StrainNodes[0], StrainNodes[0]], 
        ['DispX', 'DispY', 'DispZ'], dim='2d')

#Block a node on the center to avoid rigid body motion
Problem.bc.add('Dirichlet','DispX', 0, center)
Problem.bc.add('Dirichlet','DispY', 0, center)
Problem.bc.add('Dirichlet','DispZ', 0, center)

Problem.bc.add('Dirichlet','DispX', E[0], [StrainNodes[0]]) #EpsXX
Problem.bc.add('Dirichlet','DispY', E[1], [StrainNodes[0]]) #EpsYY
Problem.bc.add('Dirichlet','DispZ', E[2], [StrainNodes[0]]) #EpsXY

Problem.apply_boundary_conditions()

#------------------------------------------------------------------------------
#Solve
#------------------------------------------------------------------------------
Problem.set_solver('CG') #Preconditioned Conjugate Gradient
Problem.solve()

#------------------------------------------------------------------------------
# Post-treatment
#------------------------------------------------------------------------------

#Compute the mean stress and strain
#Get the stress tensor (PG values)
TensorStrain = ConstitutiveLaw.get_all()['ElasticLaw'].GetStrain()
TensorStress = ConstitutiveLaw.get_all()['ElasticLaw'].GetStress()

Volume = (xmax-xmin)*(ymax-ymin)*(zmax-zmin) #total volume of the domain
Volume_mesh = Assembly.get_all()['Assembling'].integrate_field(np.ones_like(TensorStress[0])) #volume of domain without the void (hole)

MeanStress = [1/Volume*Assembly.get_all()['Assembling'].integrate_field(TensorStress[i]) for i in range(6)] 

# MeanStrain only work if volume with no void
# Void = Volume-Volume_mesh 
MeanStrain = [1/Volume*Assembly.get_all()['Assembling'].integrate_field(TensorStrain[i]) for i in range(6)] 
# print(ConstitutiveLaw.get_all()['ElasticLaw'].GetH()@np.array(MeanStrain)) #should be the same as MeanStress if homogeneous material and no void

print('Strain tensor ([Exx, Eyy, Ezz, Exy, Exz, Eyz]): ', MeanStrain)
print('Stress tensor ([Sxx, Syy, Szz, Sxy, Sxz, Syz]): ', MeanStress)

print('Elastic Energy: ' + str(Problem.GetElasticEnergy()))

#------------------------------------------------------------------------------
#Optional: Compute and write data in a vtk file (for visualization with paraview for instance)
#------------------------------------------------------------------------------

#Get the stress tensor (nodal values converted from PG values)
TensorStrainNd = Assembly.get_all()['Assembling'].convert_data(TensorStrain, convertTo = "Node")
TensorStressNd = Assembly.get_all()['Assembling'].convert_data(TensorStress, convertTo = "Node")
# Or using the get_results function
# res = Problem.get_results("Assembling", ["stress", "strain"], 'node')
# TensorStrainNd = res['Strain']
# TensorStressNd = res['Stress']

#Get the stress tensor (element values)
TensorStrainEl = Assembly.get_all()['Assembling'].convert_data(TensorStrain, convertTo = "Element")       
TensorStressEl = Assembly.get_all()['Assembling'].convert_data(TensorStress, convertTo = "Element")

# Get the principal directions (vectors on nodes)
PrincipalStress, PrincipalDirection = TensorStressNd.GetPrincipalStress()

#Get the displacement vector on nodes for export to vtk
U = np.reshape(Problem.get_dof_solution('all'),(3,-1)).T
N = Mesh.get_all()[meshname].n_nodes
# U = np.c_[U,np.zeros(N)]

SetId = 'all_fibers' #or None
if SetId is None:
    OUT = Util.ExportData(meshname)
    ElSet = slice(None) #we keep all elements
else:
    Mesh.get_all()[meshname].extract_elements(SetId, name="output")
    ElSet = Mesh.get_all()[meshname].element_sets[SetId] #keep only some elements    
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


