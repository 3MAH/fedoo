from fedoo import *
import numpy as np
import time

# Very simple PGD example using a midplane/out of plane separation of a plate
# with simple bending load

t0=time.time()
ModelingSpace("3D")

Mesh.rectangle_mesh(50, 50, -50, 50, -50, 50, 'quad4', name="Midplane")
Mesh.line_mesh_1D(20, 0, 5, 'lin2', name = "Thickness")

#spécifique à la PGD
Mesh.get_all()['Midplane'].SetCoordinatename(('X','Y'))
Mesh.get_all()['Thickness'].SetCoordinatename(('Z'))

PGD.Mesh.Create("Midplane", "Thickness", name="Domain")

ConstitutiveLaw.ElasticIsotrop(130e6, 0.3, name = 'ElasticLaw')
WeakForm.InternalForce("ElasticLaw")

PGD.Assembly.create("ElasticLaw", "Domain", name = "Assembling") # attention l'assemblage n'est fait à cette ligne

#PGD.Assembly.SetIntegrationElement("Midplane","quad8") # to be developped
#PGD.Assembly.SetNumber(OfGaussPoint("Midplane",7)
PGD.Assembly.Launch("Assembling")

#PGD.Assembly.sum à définir

nodes_left   = Mesh.get_all()["Midplane"].node_sets["left"]
nodes_right  = Mesh.get_all()["Midplane"].node_sets["right"]
nodes_top   = Mesh.get_all()["Midplane"].node_sets["top"]

#nodes_all1d = Mesh.get_all()["Thickness"].node_sets["all"]

#
#nodes_face_left = nodes_left * nodes_all1d

Mesh.get_all()["Domain"].add_node_set([nodes_left ,"all"], name = "faceLeft")
Mesh.get_all()["Domain"].add_node_set([nodes_right,"all"], name = "faceRight")

Problem.Static("Assembling")

Problem.BoundaryCondition('Dirichlet','DispX',0,"faceLeft")
Problem.BoundaryCondition('Dirichlet','DispY',0,"faceLeft")
Problem.BoundaryCondition('Dirichlet','DispZ',0,"faceLeft")
Problem.BoundaryCondition('Dirichlet','DispZ',-5e-3,"faceRight")

Problem.ApplyBoundaryCondition()

# err0 = Problem.ComputeResidualNorm()
err0 = 1
for i in range(20):
    # old = Problem.GetDoFSolution('all')
    Problem.AddNewTerm(1)    
    for j in range(5):
        Problem.UpdatePGD([i],0)
        Problem.UpdatePGD([i],1)
        
    Problem.UpdateAlpha()
    print(Problem.GetX().getTerm(-1).norm())
    # print((Problem.GetDoFSolution('all') - old).norm())
    
    # print(Problem.ComputeResidualNorm(err0))
    
print('Temps de calcul : ', time.time()-t0)

m = Mesh.get_all()["Domain"].ExtractFullMesh(name = 'FullMesh')
U = [Problem.GetDoFSolution('DispX')[:,:].reshape(-1), \
     Problem.GetDoFSolution('DispY')[:,:].reshape(-1), \
     Problem.GetDoFSolution('DispZ')[:,:].reshape(-1) ]
U = np.array(U).T

TensorStrain = Assembly.get_all()['Assembling'].get_strain(Problem.GetDoFSolution('all'), "Nodal")
TensorStress = ConstitutiveLaw.get_all()['ElasticLaw'].GetStressFromStrain(TensorStrain)

TensorStrain = Util.listStrainTensor([s[:,:].reshape(-1) for s in TensorStrain])
TensorStress = Util.listStressTensor([s[:,:].reshape(-1) for s in TensorStress])

OUT = Util.ExportData('FullMesh')

OUT.addNodeData(TensorStress.vtkFormat(),'Stress')
OUT.addNodeData(TensorStrain.vtkFormat(),'Strain')

OUT.addNodeData(U, 'Disp')

OUT.toVTK('test.vtk')


OUT.toVTK()
