import fedoo as fd 
import numpy as np
import time

#--------------- Pre-Treatment --------------------------------------------------------
method_output = 3
# method_output = 1 to automatically save the results in a vtk file
# method_output = 2 to write the vtk file at the end 
# method_output = 3 to save the results (disp, stress, strain) in a dict

fd.ModelingSpace("2Dstress")

fd.mesh.import_file('plate_with_hole.msh', name = "Domain")
# 
#alternative mesh below (uncomment the line)
#Mesh.rectangle_mesh(Nx=101, Ny=101, x_min=-50, x_max=50, y_min=-50, y_max=50, ElementShape = type_el, name ="Domain")
type_el = fd.Mesh['Domain'].elm_type

#Material definition
fd.constitutivelaw.ElasticIsotrop(1e5, 0.3, name = 'ElasticLaw')
fd.weakform.StressEquilibrium("ElasticLaw")

#Assembly
fd.Assembly.create("ElasticLaw", "Domain", type_el, name="Assembling") 

#Type of problem 
pb = fd.problem.Linear("Assembling")

#Boundary conditions

#Definition of the set of nodes for boundary conditions
mesh = fd.Mesh["Domain"]
crd = mesh.nodes 
xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
mesh.add_node_set(list(np.where(crd[:,0] == xmin)[0]), "left")
mesh.add_node_set(list(np.where(crd[:,0] == xmax)[0]), "right")

pb.bc.add('Dirichlet', "left", 'DispX',-5e-1)
pb.bc.add('Dirichlet', "right", 'DispX', 5e-1)
pb.bc.add('Dirichlet',[0], 'DispY',0)

pb.apply_boundary_conditions()

#--------------- Solve --------------------------------------------------------
if method_output == 1:
    #Method 1: use the automatic result output    
    pb.add_output('plate_with_hole_in_tension', 'Assembling', ['Disp', 'Strain', 'Stress', 'Stress_vm'], output_type='Node', file_format ='vtk')    
    pb.add_output('plate_with_hole_in_tension', 'Assembling', ['Stress', 'Strain'], output_type='Element', file_format ='vtk')    


pb.set_solver('CG')
t0 = time.time() 
print('Solving...')
pb.solve()
print('Done in ' +str(time.time()-t0) + ' seconds')

#--------------- Post-Treatment -----------------------------------------------

if method_output == 1:
    pb.save_results()
    
# elif method_output == 2:
#     #Method 2: write the vtk output file by hand
#     #Get the nodal values of stress tensor, strain tensor, stress principal component ('Stress_PC') and stress principal directions ('Stress_PDir1', 'Stress_PDir2')
#     res = pb.get_results("Assembling", ['Stress','Strain', 'Stress_pc', 'Stress_pdir1', 'Stress_pdir2'], 'Node')    
#     TensorStrain = res['Strain']
#     TensorStress = res['Stress']
#     # Get the principal directions (vectors on nodes)
#     PrincipalStress = res['Stress_Principal']
#     PrincipalDirection1 = res['Stress_PrincipalDir1']
#     PrincipalDirection2 = res['Stress_PrincipalDir2']    
    
#     #Get the stress tensor (element values)
#     res = pb.get_results("Assembling", ['Stress','Strain'], 'Element')
#     TensorStrainEl = res['Strain']
#     TensorStressEl = res['Stress']
        
#     #Get the displacement vector on nodes for export to vtk
#     U = pb.get_disp().T #transpose for comatibility to vtk export
    
#     #Write the vtk file                            
#     OUT = fd.util.ExportData(meshname)
    
#     OUT.addNodeData(U,'Displacement')
#     OUT.addNodeData(TensorStress.vtkFormat(),'Stress')
#     OUT.addElmData(TensorStressEl.vtkFormat(),'Stress')
#     OUT.addNodeData(TensorStrain.vtkFormat(),'Strain')
#     OUT.addElmData(TensorStrainEl.vtkFormat(),'Strain')
#     OUT.addNodeData(TensorStress.vonMises(),'VMStress')
#     OUT.addElmData(TensorStressEl.vonMises(),'VMStress')
#     OUT.addNodeData(PrincipalStress,'PrincipalStress')
#     OUT.addNodeData(PrincipalDirection1,'DirPrincipal1')
#     OUT.addNodeData(PrincipalDirection2,'DirPrincipal2')
    
#     OUT.toVTK("plate_with_hole_in_tension.vtk")
#     print('Elastic Energy: ' + str(pb.GetElasticEnergy()))
    
#     print('Result file "plate_with_hole_in_tension.vtk" written in the active directory')

elif method_output == 3:
    res = pb.get_results("Assembling", ['Disp', 'Stress','Strain'], 'Node')
    import pyvista as pv
    pl = pv.Plotter(shape=(2,2))
    # from pyvistaqt import BackgroundPlotter
    # pl = BackgroundPlotter(shape = (2,2))
    res.plot('Stress','Node','vm', plotter=pl)
    pl.subplot(1,0)
    res.plot('Stress','Node', 'XX', plotter=pl)
    pl.subplot(0,1)
    res.plot('Stress', 'Node', 'YY', plotter=pl)
    pl.subplot(1,1)
    res.plot('Stress', 'Node', 'XY', plotter=pl)
    pl.show()
    