from fedoo import *
import numpy as np
import time

# Very simple PGD example using a midplane/out of plane separation of a plate
# with simple bending load

t0=time.time()
ModelingSpace("3D")

mesh.rectangle_mesh(50, 50, -50, 50, -50, 50, 'quad4', name="Midplane")
mesh.line_mesh_1D(20, 0, 5, 'lin2', name = "Thickness")

#spécifique à la PGD
Mesh['Midplane'].crd_name = ('X','Y')
Mesh['Thickness'].crd_name = ('Z')

pgd.Mesh.create("Midplane", "Thickness", name="Domain")

constitutivelaw.ElasticIsotrop(130e6, 0.3, name = 'ElasticLaw')
weakform.StressEquilibrium("ElasticLaw")

pgd.Assembly.create("ElasticLaw", "Domain", name = "Assembling") # attention l'assemblage n'est fait à cette ligne

#pgd.Assembly.SetIntegrationElement("Midplane","quad8") # to be developped
#pgd.Assembly.SetNumber(OfGaussPoint("Midplane",7)
# pgd.Assembly.Launch("Assembling")

#PGD.Assembly.sum à définir

nodes_left   = Mesh.get_all()["Midplane"].node_sets["left"]
nodes_right  = Mesh.get_all()["Midplane"].node_sets["right"]
nodes_top   = Mesh.get_all()["Midplane"].node_sets["top"]

#nodes_all1d = Mesh.get_all()["Thickness"].node_sets["all"]

#
#nodes_face_left = nodes_left * nodes_all1d

Mesh["Domain"].add_node_set([nodes_left ,"all"], name = "faceLeft")
Mesh["Domain"].add_node_set([nodes_right,"all"], name = "faceRight")

pb = pgd.Linear("Assembling")

pb.bc.add('Dirichlet',"faceLeft",'DispX',0)
pb.bc.add('Dirichlet',"faceLeft",'DispY',0)
pb.bc.add('Dirichlet',"faceLeft",'DispZ',0)
pb.bc.add('Dirichlet',"faceRight",'DispZ',-5e-3)

pb.apply_boundary_conditions()

# err0 = Problem.ComputeResidualNorm()
err0 = 1
for i in range(20):
    # old = Problem.get_dof_solution('all')
    pb.AddNewTerm(1)    
    for j in range(5):
        pb.update_pgd([i],0)
        pb.update_pgd([i],1)
        
    pb.update_alpha()
    print(pb.get_X().getTerm(-1).norm())
    # print((pb.get_dof_solution('all') - old).norm())
    
    # print(pb.ComputeResidualNorm(err0))
    
print('Temps de calcul : ', time.time()-t0)

m = Mesh.get_all()["Domain"].ExtractFullMesh(name = 'FullMesh')
U = [pb.get_dof_solution('DispX')[:,:].reshape(-1), \
     pb.get_dof_solution('DispY')[:,:].reshape(-1), \
     pb.get_dof_solution('DispZ')[:,:].reshape(-1) ]
U = np.array(U)

TensorStrain = Assembly['Assembling'].get_strain(pb.get_dof_solution('all'), "Node")
TensorStress = ConstitutiveLaw.get_all()['ElasticLaw'].GetStressFromStrain(TensorStrain)

TensorStrain = util.StrainTensorList([s[:,:].reshape(-1) for s in TensorStrain])
TensorStress = util.StressTensorList([s[:,:].reshape(-1) for s in TensorStress])

data = DataSet(Mesh['FullMesh'])

data.node_data['Stress'] = TensorStress.vtkFormat()
data.node_data['Strain'] = TensorStrain.vtkFormat()

data.node_data['Disp'] = U
data.plot('Stress', scale = 10000)
# data.to_vtk('test.vtk')

