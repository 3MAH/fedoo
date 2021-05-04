from fedoo import *
import numpy as np
import time

#--------------- Pre-Treatment --------------------------------------------------------

Util.ProblemDimension("2Dstress")

Mesh.ImportFromFile('plate_with_hole.msh', meshID = "Domain")

#alternative mesh below (uncomment the line)
#Mesh.RectangleMesh(Nx=101, Ny=101, x_min=-50, x_max=50, y_min=-50, y_max=50, ElementShape = type_el, ID ="Domain")
type_el = Mesh.GetAll()['Domain'].GetElementShape()
meshID = "Domain"

#Material definition
ConstitutiveLaw.ElasticIsotrop(1e5, 0.3, ID = 'ElasticLaw')
WeakForm.InternalForce("ElasticLaw")

#Assembly
Assembly.Create("ElasticLaw", meshID, type_el, ID="Assembling") 

#Type of problem 
Problem.Static("Assembling")

#Boundary conditions

#Definition of the set of nodes for boundary conditions
mesh = Mesh.GetAll()[meshID]
crd = mesh.GetNodeCoordinates() 
xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
mesh.AddSetOfNodes(list(np.where(crd[:,0] == xmin)[0]), "left")
mesh.AddSetOfNodes(list(np.where(crd[:,0] == xmax)[0]), "right")

Problem.BoundaryCondition('Dirichlet','DispX',-5e-1,mesh.GetSetOfNodes("left"))
Problem.BoundaryCondition('Dirichlet','DispX', 5e-1,mesh.GetSetOfNodes("right"))
Problem.BoundaryCondition('Dirichlet','DispY',0,[0])

Problem.ApplyBoundaryCondition()

#--------------- Solve --------------------------------------------------------
Problem.SetSolver('CG')
t0 = time.time() 
print('Solving...')
Problem.Solve()
print('Done in ' +str(time.time()-t0) + ' seconds')

#--------------- Post-Treatment -----------------------------------------------
method_output = 1

if method_output == 1: 
    #Method 1: use the automatic result output
    
    Problem.AddOutput('rplate_with_hole_in_tension', 'Assembling', ['disp', 'strain', 'stress', 'stress_vm'], output_type='Node', file_format ='vtk')    
    Problem.AddOutput('plate_with_hole_in_tension', 'Assembling', ['stress', 'strain'], output_type='Element', file_format ='vtk')    
    Problem.Update() #compute strain and stress
    Problem.SaveResults()

else:

    #Method 2: write the vtk output file by hand
    #Get the stress tensor (nodal values)
    TensorStrain = Assembly.GetAll()['Assembling'].GetStrainTensor(Problem.GetDoFSolution(), "Nodal")       
    TensorStress = ConstitutiveLaw.GetAll()['ElasticLaw'].GetStress(TensorStrain)
    
    #Get the stress tensor (element values)
    TensorStrainEl = Assembly.GetAll()['Assembling'].GetStrainTensor(Problem.GetDoFSolution(), "Element")       
    TensorStressEl = ConstitutiveLaw.GetAll()['ElasticLaw'].GetStress(TensorStrainEl)
    
    # Get the principal directions (vectors on nodes)
    PrincipalStress, PrincipalDirection = TensorStress.GetPrincipalStress()
    
    #Get the displacement vector on nodes for export to vtk
    U = np.reshape(Problem.GetDoFSolution('all'),(2,-1)).T
    N = Mesh.GetAll()[meshID].GetNumberOfNodes()
    U = np.c_[U,np.zeros(N)]
    
    #Write the vtk file                            
    OUT = Util.ExportData(meshID)
    
    OUT.addNodeData(U,'Displacement')
    OUT.addNodeData(TensorStress.vtkFormat(),'Stress')
    OUT.addElmData(TensorStressEl.vtkFormat(),'Stress')
    OUT.addNodeData(TensorStrain.vtkFormat(),'Strain')
    OUT.addElmData(TensorStrainEl.vtkFormat(),'Strain')
    OUT.addNodeData(TensorStress.vonMises(),'VMStress')
    OUT.addElmData(TensorStressEl.vonMises(),'VMStress')
    OUT.addNodeData(PrincipalStress,'PrincipalStress')
    OUT.addNodeData(PrincipalDirection[0],'DirPrincipal1')
    OUT.addNodeData(PrincipalDirection[1],'DirPrincipal2')
    
    OUT.toVTK("plate_with_hole_in_tension.vtk")
    print('Elastic Energy: ' + str(Problem.GetElasticEnergy()))
    
    print('Result file "plate_with_hole_in_tension.vtk" written in the active directory')

