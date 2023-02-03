import fedoo as fd
import numpy as np
import time

#--------------- Pre-Treatment --------------------------------------------------------

t0 = time.time()
fd.ModelingSpace("3D")
fd.mesh.import_file('gyroid.msh', name = "Domain")
name = "Domain"

mesh = fd.Mesh[name]
type_el = fd.Mesh[name].elm_type

#Definition of the set of nodes for boundary conditions

crd = mesh.nodes 
xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
zmax = np.max(crd[:,2]) ; zmin = np.min(crd[:,2])
center = [np.linalg.norm(crd,axis=1).argmin()]

StrainNodes = mesh.add_nodes(2) #add virtual nodes for macro strain

#Material definition
material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3, name = 'ElasticLaw')
fd.weakform.StressEquilibrium("ElasticLaw")

#Assembly
fd.Assembly.create("ElasticLaw", mesh, type_el, name="Assembling") 

#Type of problem 
pb = fd.problem.Linear("Assembling")


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

# list_strain_nodes = [StrainNodes[0], StrainNodes[0], StrainNodes[0],
#                      StrainNodes[1], StrainNodes[1], StrainNodes[1]]
# list_strain_var = ['DispX', 'DispY', 'DispZ','DispX', 'DispY', 'DispZ']

# bc_periodic = fd.homogen.PeriodicBC(list_strain_nodes, list_strain_var, dim=3) 
# pb.bc.add(bc_periodic)
fd.homogen.DefinePeriodicBoundaryConditionNonPerioMesh(mesh,
        [StrainNodes[0], StrainNodes[0], StrainNodes[0], StrainNodes[1], StrainNodes[1], StrainNodes[1]],
        ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D', tol=1e-4, nNeighbours = 3, powInter = 1.0)

pb.bc.add('Dirichlet','DispX', 0, center)
pb.bc.add('Dirichlet','DispY', 0, center)
pb.bc.add('Dirichlet','DispZ', 0, center)

pb.bc.add('Dirichlet','DispX', E[0], [StrainNodes[0]]) #EpsXX
pb.bc.add('Dirichlet','DispY', E[1], [StrainNodes[0]]) #EpsYY
pb.bc.add('Dirichlet','DispZ', E[2], [StrainNodes[0]]) #EpsZZ
pb.bc.add('Dirichlet','DispX', E[3], [StrainNodes[1]]) #EpsXY
pb.bc.add('Dirichlet','DispY', E[4], [StrainNodes[1]]) #EpsXZ
pb.bc.add('Dirichlet','DispZ', E[5], [StrainNodes[1]]) #EpsYZ

pb.apply_boundary_conditions()

#lv--------------- Soe --------------------------------------------------------
pb.set_solver('CG')
print('Solving...')
print(time.time()-t0)
pb.solve()
print('Done in ' +str(time.time()-t0) + ' seconds')

TensorStrain = fd.ConstitutiveLaw['ElasticLaw'].GetStrain()
TensorStress = fd.ConstitutiveLaw['ElasticLaw'].GetStress()

#------------------------------------------------------------------------------
#Optional: Compute and write data in a vtk file (for visualization with paraview for instance)
#------------------------------------------------------------------------------
output_VTK = 1
if output_VTK == 1:
    #Get the stress tensor (nodal values converted from PG values)
    TensorStrainNd = fd.Assembly.convert_data(TensorStrain, meshname, convertTo = "Node")
    TensorStressNd = fd.Assembly.convert_data(TensorStress, meshname, convertTo = "Node")
    
    #Get the stress tensor (element values)
    TensorStrainEl = fd.Assembly.convert_data(TensorStrain, meshname, convertTo = "Element")       
    TensorStressEl = fd.Assembly.convert_data(TensorStress, meshname, convertTo = "Element")
    
    # #Get the stress tensor (nodal values)
    # TensorStrain = Assembly.get_all()['Assembling'].get_strain(pb.get_disp(), "Nodal")       
    # TensorStress = ConstitutiveLaw.get_all()['ElasticLaw'].GetStress(TensorStrain)
    
    # #Get the stress tensor (element values)
    # TensorStrainEl = Assembly.get_all()['Assembling'].get_strain(pb.get_disp(), "Element")       
    # TensorStressEl = ConstitutiveLaw.get_all()['ElasticLaw'].GetStress(TensorStrainEl)
    
    # Get the principal directions (vectors on nodes)
    PrincipalStress, PrincipalDirection = TensorStressNd.GetPrincipalStress()
    
    #Get the displacement vector on nodes for export to vtk
    U = np.reshape(pb.get_dof_solution('all'),(3,-1)).T
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
    
