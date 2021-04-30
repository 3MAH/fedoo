from fedoo import *
import numpy as np
import time

#--------------- Pre-Treatment --------------------------------------------------------

Util.ProblemDimension("3D")

DataINP = Util.ReadINP('Job-1.inp')

INP = Util.ReadINP('cell_taffetas.inp') #Warning: non periodic mesh -> dont work quite well
INP.toMesh(meshID = "Domain")

# data = Util.ExportData(Mesh.GetAll()['Domain'])
# data.toVTK()

#alternative mesh below (uncomment the line)
# Mesh.BoxMesh(Nx=11, Ny=11, Nz=11, x_min=-1, x_max=1, y_min=-1, y_max=1, z_min = -1, z_max = 1, ElementShape = 'hex8', ID ="Domain" )
    
type_el = Mesh.GetAll()['Domain'].GetElementShape()

meshID = "Domain"

#Definition of the set of nodes for boundary conditions
mesh = Mesh.GetAll()[meshID]
crd = mesh.GetNodeCoordinates() 
xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
zmax = np.max(crd[:,2]) ; zmin = np.min(crd[:,2])
center = [np.linalg.norm(crd,axis=1).argmin()]

StrainNodes = Mesh.GetAll()[meshID].AddNodes(np.zeros(crd.shape[1]),2) #add virtual nodes for macro strain

#Material definition
material = ConstitutiveLaw.ElasticIsotrop(1e5, 0.3, ID = 'ElasticLaw')
WeakForm.InternalForce("ElasticLaw")

#Assembly
Assembly.Create("ElasticLaw", meshID, type_el, ID="Assembling") 

#Type of problem 
Problem.Static("Assembling")


#Boundary conditions
E = [0,0,0,0,0,1] #[EXX, EYY, EZZ, EXY, EXZ, EYZ]
#StrainNode[0] - 'DispX' is a virtual dof for EXX
#StrainNode[0] - 'DispY' is a virtual dof for EYY
#StrainNode[0] - 'DispZ' is a virtual dof for EZZ        
#StrainNode[1] - 'DispX' is a virtual dof for EXY        
#StrainNode[1] - 'DispY' is a virtual dof for EXZ
#StrainNode[1] - 'DispZ' is a virtual dof for EYZ

Util.DefinePeriodicBoundaryCondition("Domain", 
        [StrainNodes[0], StrainNodes[0], StrainNodes[0], StrainNodes[1], StrainNodes[1], StrainNodes[1]], 
        ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D')

Problem.BoundaryCondition('Dirichlet','DispX', 0, center)
Problem.BoundaryCondition('Dirichlet','DispY', 0, center)
Problem.BoundaryCondition('Dirichlet','DispZ', 0, center)

Problem.BoundaryCondition('Dirichlet','DispX', E[0], [StrainNodes[0]]) #EpsXX
Problem.BoundaryCondition('Dirichlet','DispY', E[1], [StrainNodes[0]]) #EpsYY
Problem.BoundaryCondition('Dirichlet','DispZ', E[2], [StrainNodes[0]]) #EpsZZ
Problem.BoundaryCondition('Dirichlet','DispX', E[3], [StrainNodes[1]]) #EpsXY
Problem.BoundaryCondition('Dirichlet','DispY', E[4], [StrainNodes[1]]) #EpsXZ
Problem.BoundaryCondition('Dirichlet','DispZ', E[5], [StrainNodes[1]]) #EpsYZ

Problem.ApplyBoundaryCondition()

#--------------- Solve --------------------------------------------------------
Problem.SetSolver('CG')
t0 = time.time() 
print('Solving...')
Problem.Solve()
print('Done in ' +str(time.time()-t0) + ' seconds')

#--------------- Post-Treatment -----------------------------------------------

#Compute the mean stress and strain
#Get the stress tensor (PG values)
TensorStrain = Assembly.GetAll()['Assembling'].GetStrainTensor(Problem.GetDoFSolution(), "GaussPoint")       
TensorStress = ConstitutiveLaw.GetAll()['ElasticLaw'].GetStress(TensorStrain)

Volume = (xmax-xmin)*(ymax-ymin)*(zmax-zmin) #total volume of the domain
Volume_mesh = Assembly.GetAll()['Assembling'].IntegrateField(np.ones_like(TensorStress[0])) #volume of domain without the void (hole)

MeanStress = [1/Volume*Assembly.GetAll()['Assembling'].IntegrateField(TensorStress[i]) for i in range(6)] 

MeanStrain = [Problem.GetDisp('DispX')[-2], Problem.GetDisp('DispY')[-2], Problem.GetDisp('DispZ')[-2], 
              Problem.GetDisp('DispX')[-1], Problem.GetDisp('DispY')[-1], Problem.GetDisp('DispZ')[-1]]
# Other method: only work if volume with no void (Void=0)
# Void = Volume-Volume_mesh 
# MeanStrain = [1/Volume*Assembly.GetAll()['Assembling'].IntegrateField(TensorStrain[i]) for i in range(6)] 

print('Strain tensor ([Exx, Eyy, Ezz, Exy, Exz, Eyz]): ' )
print(MeanStrain)
print('Stress tensor ([Sxx, Syy, Szz, Sxy, Sxz, Syz]): ' )
print(MeanStress)

# print(ConstitutiveLaw.GetAll()['ElasticLaw'].GetH()@np.array(MeanStrain)) #should be the same as MeanStress if homogeneous material and no void


#------------------------------------------------------------------------------
#Optional: Compute and write data in a vtk file (for visualization with paraview for instance)
#------------------------------------------------------------------------------
output_VTK = 1
if output_VTK == 1:
    #Get the stress tensor (nodal values converted from PG values)
    TensorStrainNd = Assembly.ConvertData(TensorStrain, meshID, convertTo = "Node")
    TensorStressNd = Assembly.ConvertData(TensorStress, meshID, convertTo = "Node")
    
    #Get the stress tensor (element values)
    TensorStrainEl = Assembly.ConvertData(TensorStrain, meshID, convertTo = "Element")       
    TensorStressEl = Assembly.ConvertData(TensorStress, meshID, convertTo = "Element")
    
    # #Get the stress tensor (nodal values)
    # TensorStrain = Assembly.GetAll()['Assembling'].GetStrainTensor(Problem.GetDisp(), "Nodal")       
    # TensorStress = ConstitutiveLaw.GetAll()['ElasticLaw'].GetStress(TensorStrain)
    
    # #Get the stress tensor (element values)
    # TensorStrainEl = Assembly.GetAll()['Assembling'].GetStrainTensor(Problem.GetDisp(), "Element")       
    # TensorStressEl = ConstitutiveLaw.GetAll()['ElasticLaw'].GetStress(TensorStrainEl)
    
    # Get the principal directions (vectors on nodes)
    PrincipalStress, PrincipalDirection = TensorStressNd.GetPrincipalStress()
    
    #Get the displacement vector on nodes for export to vtk
    U = np.reshape(Problem.GetDoFSolution('all'),(3,-1)).T
    N = Mesh.GetAll()[meshID].GetNumberOfNodes()
    # U = np.c_[U,np.zeros(N)]
    
    
    # SetId = None
    SetId = 'all_fibers' #to extract only mesh and data related to fibers    
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
    
    OUT.toVTK("3D_Periodic_BC.vtk")
    print('Result file "3D_Periodic_BC.vtk" written in the active directory')
    
