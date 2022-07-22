from fedoo import *
import numpy as np
import time

#--------------- Pre-Treatment --------------------------------------------------------

t0 = time.time()
ModelingSpace("3D")
Mesh.import_file('gyroid.msh', meshname = "Domain")
meshname = "Domain"

type_el = Mesh.get_all()[meshname].elm_type

#Definition of the set of nodes for boundary conditions
mesh = Mesh.get_all()[meshname]
crd = mesh.nodes 
xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
zmax = np.max(crd[:,2]) ; zmin = np.min(crd[:,2])
center = [np.linalg.norm(crd,axis=1).argmin()]

crd_center = (np.array([xmin, ymin, zmin]) + np.array([xmax, ymax, zmax]))/2
StrainNodes = mesh.add_nodes(crd_center,2) #add virtual nodes for macro strain

#Material definition
material = ConstitutiveLaw.ElasticIsotrop(1e5, 0.3, name = 'ElasticLaw')
WeakForm.InternalForce("ElasticLaw")

#Assembly
Assembly.create("ElasticLaw", meshname, type_el, name="Assembling") 

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

#Util.DefinePeriodicBoundaryCondition(meshname,
#        [StrainNodes[0], StrainNodes[0], StrainNodes[0], StrainNodes[1], StrainNodes[1], StrainNodes[1]],
#        ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D')
Homogen.DefinePeriodicBoundaryConditionNonPerioMesh(meshname,
        [StrainNodes[0], StrainNodes[0], StrainNodes[0], StrainNodes[1], StrainNodes[1], StrainNodes[1]],
        ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D', tol=1e-4, nNeighbours = 3, powInter = 1.0)

Problem.BoundaryCondition('Dirichlet','DispX', 0, center)
Problem.BoundaryCondition('Dirichlet','DispY', 0, center)
Problem.BoundaryCondition('Dirichlet','DispZ', 0, center)

Problem.BoundaryCondition('Dirichlet','DispX', E[0], [StrainNodes[0]]) #EpsXX
Problem.BoundaryCondition('Dirichlet','DispY', E[1], [StrainNodes[0]]) #EpsYY
Problem.BoundaryCondition('Dirichlet','DispZ', E[2], [StrainNodes[0]]) #EpsZZ
Problem.BoundaryCondition('Dirichlet','DispX', E[3], [StrainNodes[1]]) #EpsXY
Problem.BoundaryCondition('Dirichlet','DispY', E[4], [StrainNodes[1]]) #EpsXZ
Problem.BoundaryCondition('Dirichlet','DispZ', E[5], [StrainNodes[1]]) #EpsYZ

Problem.apply_boundary_conditions()

#lv--------------- Soe --------------------------------------------------------
Problem.set_solver('CG')
print('Solving...')
print(time.time()-t0)
Problem.solve()
print('Done in ' +str(time.time()-t0) + ' seconds')

TensorStrain = ConstitutiveLaw.get_all()['ElasticLaw'].GetStrain()
TensorStress = ConstitutiveLaw.get_all()['ElasticLaw'].GetStress()

#------------------------------------------------------------------------------
#Optional: Compute and write data in a vtk file (for visualization with paraview for instance)
#------------------------------------------------------------------------------
output_VTK = 1
if output_VTK == 1:
    #Get the stress tensor (nodal values converted from PG values)
    TensorStrainNd = Assembly.convert_data(TensorStrain, meshname, convertTo = "Node")
    TensorStressNd = Assembly.convert_data(TensorStress, meshname, convertTo = "Node")
    
    #Get the stress tensor (element values)
    TensorStrainEl = Assembly.convert_data(TensorStrain, meshname, convertTo = "Element")       
    TensorStressEl = Assembly.convert_data(TensorStress, meshname, convertTo = "Element")
    
    # #Get the stress tensor (nodal values)
    # TensorStrain = Assembly.get_all()['Assembling'].get_strain(Problem.get_disp(), "Nodal")       
    # TensorStress = ConstitutiveLaw.get_all()['ElasticLaw'].GetStress(TensorStrain)
    
    # #Get the stress tensor (element values)
    # TensorStrainEl = Assembly.get_all()['Assembling'].get_strain(Problem.get_disp(), "Element")       
    # TensorStressEl = ConstitutiveLaw.get_all()['ElasticLaw'].GetStress(TensorStrainEl)
    
    # Get the principal directions (vectors on nodes)
    PrincipalStress, PrincipalDirection = TensorStressNd.GetPrincipalStress()
    
    #Get the displacement vector on nodes for export to vtk
    U = np.reshape(Problem.GetDoFSolution('all'),(3,-1)).T
    N = Mesh.get_all()[meshname].n_nodes
    # U = np.c_[U,np.zeros(N)]
    
    SetId = None
    OUT = Util.ExportData(meshname)
    ElSet = slice(None) #we keep all elements
    
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
    
