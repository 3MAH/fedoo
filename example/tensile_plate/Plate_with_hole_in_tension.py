from fedoo import *
import numpy as np
import time

#--------------- Pre-Treatment --------------------------------------------------------
method_output = 1
# method_output = 1 to automatically save the results in a vtk file
# method_output = 2 to write the vtk file at the end 
# method_output = 3 to save the results (disp, stress, strain) in a dict

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
if method_output == 1:
    #Method 1: use the automatic result output    
    Problem.AddOutput('plate_with_hole_in_tension', 'Assembling', ['disp', 'strain', 'stress', 'stress_vm'], output_type='Node', file_format ='vtk')    
    Problem.AddOutput('plate_with_hole_in_tension', 'Assembling', ['stress', 'strain'], output_type='Element', file_format ='vtk')    


Problem.SetSolver('CG')
t0 = time.time() 
print('Solving...')
Problem.Solve()
print('Done in ' +str(time.time()-t0) + ' seconds')

#--------------- Post-Treatment -----------------------------------------------


if method_output == 1:
    Problem.SaveResults()
    
elif method_output == 2:
    #Method 2: write the vtk output file by hand
    #Get the nodal values of stress tensor, strain tensor, stress principal component ('Stress_PC') and stress principal directions ('Stress_PDir1', 'Stress_PDir2')
    res = Problem.GetResults("Assembling", ['Stress','Strain', 'Stress_PC', 'Stress_PDir1', 'Stress_PDir2'], 'Node')    
    TensorStrain = res['Strain']
    TensorStress = res['Stress']
    # Get the principal directions (vectors on nodes)
    PrincipalStress = res['Stress_Principal']
    PrincipalDirection1 = res['Stress_PrincipalDir1']
    PrincipalDirection2 = res['Stress_PrincipalDir2']    
    
    #Get the stress tensor (element values)
    res = Problem.GetResults("Assembling", ['Stress','Strain'], 'Element')
    TensorStrainEl = res['Strain']
    TensorStressEl = res['Stress']
        
    #Get the displacement vector on nodes for export to vtk
    U = Problem.GetDisp().T #transpose for comatibility to vtk export
    
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
    OUT.addNodeData(PrincipalDirection1,'DirPrincipal1')
    OUT.addNodeData(PrincipalDirection2,'DirPrincipal2')
    
    OUT.toVTK("plate_with_hole_in_tension.vtk")
    print('Elastic Energy: ' + str(Problem.GetElasticEnergy()))
    
    print('Result file "plate_with_hole_in_tension.vtk" written in the active directory')

elif method_output == 3:
    res = Problem.GetResults("Assembling", ['disp', 'Stress','Strain'], 'Node')