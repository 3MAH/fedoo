import fedoo as fd
import numpy as np
import time

#--------------- Pre-Treatment --------------------------------------------------------

fd.ModelingSpace("3D")

#Units: N, mm, MPa
fd.mesh.box_mesh(nx=31, ny=21, nz=21, x_min=0, x_max=1000, y_min=0, y_max=100, z_min=0, z_max=100, elm_type = 'hex8', name = 'Domain')

meshname = "Domain"

#Material definition
fd.constitutivelaw.ElasticIsotrop(200e3, 0.3, name = 'ElasticLaw')
fd.weakform.StressEquilibrium("ElasticLaw", nlgeom = 2)

#Assembly (print the time required for assembling)
assemb = fd.Assembly.create("ElasticLaw", meshname, 'hex8', name="Assembling") 

#Type of problem 
pb = fd.problem.NonLinear("Assembling")

#Boundary conditions
nodes_left = fd.Mesh[meshname].node_sets["left"]
nodes_right = fd.Mesh[meshname].node_sets["right"]
nodes_top = fd.Mesh[meshname].node_sets["top"]
nodes_bottom = fd.Mesh[meshname].node_sets["bottom"]

pb.bc.add('Dirichlet',nodes_left,'Disp',0)

pb.bc.add('Dirichlet',nodes_right,'DispY', -10)

pb.apply_boundary_conditions()

#--------------- Solve --------------------------------------------------------
t0 = time.time() 
pb.set_solver('cg')
print('Solving...')
pb.nlsolve(dt=1) 
#Problem.solve() 
print('Done in ' +str(time.time()-t0) + ' seconds')

# #--------------- Post-Treatment -----------------------------------------------
# #Get the displacement vector on nodes for export to vtk
# U = np.reshape(pb.get_disp(),(3,-1)).T

# #Get the stress tensor (nodal values)
# TensorStrain = assemb.convert_data(fd.ConstitutiveLaw['ElasticLaw'].GetStrain(), 'GaussPoint', 'Node')
# TensorStress = assemb.convert_data(fd.ConstitutiveLaw['ElasticLaw'].GetStress(), 'GaussPoint', 'Node')

# #PrincipalStress, PrincipalDirection = TensorStress.GetPrincipalStress()
                                                   

# #Write the vtk file                            
# OUT = Util.ExportData(meshname)

# OUT.addNodeData(U.astype(float),'Displacement')
# OUT.addNodeData(TensorStress.vtkFormat(),'Stress')
# OUT.addNodeData(TensorStress.vonMises(),'VMStress')
# OUT.addNodeData(TensorStrain.vtkFormat(),'Strain')
# #OUT.addNodeData(PrincipalStress, 'PrincipalStress')
# #OUT.addNodeData(PrincipalDirection[0], '1stPrincipalDirection')

# OUT.toVTK("simple_cantilever_3D_model.vtk")
# # print('Elastic Energy: ' + str(Problem.GetElasticEnergy()))

# print('Result file "simple_cantilever_3D_model.vtk" written in the active directory')

