from fedoo import *
import numpy as np
import time

#--------------- Pre-Treatment --------------------------------------------------------

t0 = time.time()
ModelingSpace("3D")

DataINP = Util.ReadINP('Job-1.inp')

INP = Util.ReadINP('cell_taffetas.inp') #Warning: non periodic mesh -> dont work quite well
INP.toMesh(meshname = "Domain")

# data = Util.ExportData(Mesh.get_all()['Domain'])
# data.toVTK()

#alternative mesh below (uncomment the line)
# Mesh.box_mesh(Nx=51, Ny=51, Nz=51, x_min=-1, x_max=1, y_min=-1, y_max=1, z_min = -1, z_max = 1, ElementShape = 'hex8', name ="Domain" )
    
type_el = Mesh.get_all()['Domain'].elm_type

meshname = "Domain"

#Definition of the set of nodes for boundary conditions
mesh = Mesh.get_all()[meshname]
crd = mesh.nodes 
xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
zmax = np.max(crd[:,2]) ; zmin = np.min(crd[:,2])
center = [np.linalg.norm(crd,axis=1).argmin()]

StrainNodes = Mesh.get_all()[meshname].add_nodes(np.zeros(crd.shape[1]),2) #add virtual nodes for macro strain

#Material definition
material = ConstitutiveLaw.ElasticIsotrop(1e5, 0.3, name = 'ElasticLaw')
WeakForm.StressEquilibrium("ElasticLaw")

#Assembly
Assembly.create("ElasticLaw", meshname, type_el, name="Assembling") 

#Type of problem 
Problem.Linear("Assembling")


#Boundary conditions
E = [0,0,0,0,0,1] #[EXX, EYY, EZZ, EXY, EXZ, EYZ]
#StrainNode[0] - 'DispX' is a virtual dof for EXX
#StrainNode[0] - 'DispY' is a virtual dof for EYY
#StrainNode[0] - 'DispZ' is a virtual dof for EZZ        
#StrainNode[1] - 'DispX' is a virtual dof for EXY        
#StrainNode[1] - 'DispY' is a virtual dof for EXZ
#StrainNode[1] - 'DispZ' is a virtual dof for EYZ

Homogen.DefinePeriodicBoundaryCondition("Domain", 
        [StrainNodes[0], StrainNodes[0], StrainNodes[0], StrainNodes[1], StrainNodes[1], StrainNodes[1]], 
        ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D')

Problem.bc.add('Dirichlet','DispX', 0, center)
Problem.bc.add('Dirichlet','DispY', 0, center)
Problem.bc.add('Dirichlet','DispZ', 0, center)

Problem.bc.add('Dirichlet','DispX', E[0], [StrainNodes[0]]) #EpsXX
Problem.bc.add('Dirichlet','DispY', E[1], [StrainNodes[0]]) #EpsYY
Problem.bc.add('Dirichlet','DispZ', E[2], [StrainNodes[0]]) #EpsZZ
Problem.bc.add('Dirichlet','DispX', E[3], [StrainNodes[1]]) #EpsXY
Problem.bc.add('Dirichlet','DispY', E[4], [StrainNodes[1]]) #EpsXZ
Problem.bc.add('Dirichlet','DispZ', E[5], [StrainNodes[1]]) #EpsYZ

Problem.apply_boundary_conditions()

#--------------- Solve --------------------------------------------------------
Problem.set_solver('CG')
print('Solving...')
print(time.time()-t0)
Problem.solve()
print('Done in ' +str(time.time()-t0) + ' seconds')

#--------------- Post-Treatment -----------------------------------------------

#Compute the mean stress and strain
#Get the stress tensor (PG values)
TensorStrain = ConstitutiveLaw.get_all()['ElasticLaw'].GetStrain() 
TensorStress = ConstitutiveLaw.get_all()['ElasticLaw'].GetStress()

Volume = (xmax-xmin)*(ymax-ymin)*(zmax-zmin) #total volume of the domain
Volume_mesh = Assembly.get_all()['Assembling'].integrate_field(np.ones_like(TensorStress[0])) #volume of domain without the void (hole)

MeanStress = [1/Volume*Assembly.get_all()['Assembling'].integrate_field(TensorStress[i]) for i in range(6)] 

MeanStrain = [Problem.get_disp('DispX')[-2], Problem.get_disp('DispY')[-2], Problem.get_disp('DispZ')[-2], 
              Problem.get_disp('DispX')[-1], Problem.get_disp('DispY')[-1], Problem.get_disp('DispZ')[-1]]
# Other method: only work if volume with no void (Void=0)
# Void = Volume-Volume_mesh 
# MeanStrain = [1/Volume*Assembly.get_all()['Assembling'].integrate_field(TensorStrain[i]) for i in range(6)] 

print('Strain tensor ([Exx, Eyy, Ezz, Exy, Exz, Eyz]): ' )
print(MeanStrain)
print('Stress tensor ([Sxx, Syy, Szz, Sxy, Sxz, Syz]): ' )
print(MeanStress)

# print(ConstitutiveLaw.get_all()['ElasticLaw'].GetH()@np.array(MeanStrain)) #should be the same as MeanStress if homogeneous material and no void


#------------------------------------------------------------------------------
#Optional: Compute and write data in a vtk file (for visualization with paraview for instance)
#------------------------------------------------------------------------------
output_VTK = 1
if output_VTK == 1:
    #Get the stress tensor (nodal values converted from PG values)
    
    res = Problem.get_results("Assembling", ["stress", "strain"], 'node')
    TensorStrainNd = res['Strain']
    TensorStressNd = res['Stress']
    
    #Get the stress tensor (element values)
    res = Problem.get_results("Assembling", ["stress", "strain"], 'element')
    TensorStrainEl = res['Strain']
    TensorStressEl = res['Stress']    
    
    # Get the principal directions (vectors on nodes)
    PrincipalStress, PrincipalDirection = TensorStressNd.GetPrincipalStress()
    
    #Get the displacement vector on nodes for export to vtk
    U = np.reshape(Problem.get_dof_solution('all'),(3,-1)).T
    N = Mesh.get_all()[meshname].n_nodes
    # U = np.c_[U,np.zeros(N)]
    
    
    # SetId = None
    SetId = 'all_fibers' #to extract only mesh and data related to fibers    
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
    
    OUT.toVTK("3D_Periodic_BC.vtk")
    print('Result file "3D_Periodic_BC.vtk" written in the active directory')
    
