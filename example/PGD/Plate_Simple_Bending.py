from fedoo import *
import numpy as np
import time

# Very simple PGD example using a midplane/out of plane separation of a plate
# with simple bending load

t0=time.time()
Util.ProblemDimension("3D")

Mesh.RectangleMesh(50, 50, -50, 50, -50, 50, 'quad4', ID="Midplane")
Mesh.LineMesh1D(20, 0, 5, 'lin2', ID = "Thickness")

#spécifique à la PGD
Mesh.GetAll()['Midplane'].SetCoordinateID(('X','Y'))
Mesh.GetAll()['Thickness'].SetCoordinateID(('Z'))

PGD.Mesh.Create("Midplane", "Thickness", ID="Domain")

ConstitutiveLaw.ElasticIsotrop(130e6, 0.3, ID = 'ElasticLaw')
WeakForm.InternalForce("ElasticLaw")

PGD.Assembly.Create("ElasticLaw", "Domain", ID = "Assembling") # attention l'assemblage n'est fait à cette ligne

#PGD.Assembly.SetIntegrationElement("Midplane","quad8") # to be developped
#PGD.Assembly.SetNumber(OfGaussPoint("Midplane",7)
PGD.Assembly.Launch("Assembling")

#PGD.Assembly.sum à définir

nodes_left   = Mesh.GetAll()["Midplane"].GetSetOfNodes("left")
nodes_right  = Mesh.GetAll()["Midplane"].GetSetOfNodes("right")
nodes_top   = Mesh.GetAll()["Midplane"].GetSetOfNodes("top")

#nodes_all1d = Mesh.GetAll()["Thickness"].GetSetOfNodes("all")

#
#nodes_face_left = nodes_left * nodes_all1d

Mesh.GetAll()["Domain"].AddSetOfNodes([nodes_left ,"all"], ID = "faceLeft")
Mesh.GetAll()["Domain"].AddSetOfNodes([nodes_right,"all"], ID = "faceRight")

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

m = Mesh.GetAll()["Domain"].ExtractFullMesh(ID = 'FullMesh')
U = [Problem.GetDoFSolution('DispX')[:,:].reshape(-1), \
     Problem.GetDoFSolution('DispY')[:,:].reshape(-1), \
     Problem.GetDoFSolution('DispZ')[:,:].reshape(-1) ]
U = np.array(U).T

TensorStrain = Assembly.GetAll()['Assembling'].GetStrainTensor(Problem.GetDoFSolution('all'), "Nodal")
TensorStress = ConstitutiveLaw.GetAll()['ElasticLaw'].GetStressFromStrain(TensorStrain)

TensorStrain = Util.listStrainTensor([s[:,:].reshape(-1) for s in TensorStrain])
TensorStress = Util.listStressTensor([s[:,:].reshape(-1) for s in TensorStress])

OUT = Util.ExportData('FullMesh')

OUT.addNodeData(TensorStress.vtkFormat(),'Stress')
OUT.addNodeData(TensorStrain.vtkFormat(),'Strain')

OUT.addNodeData(U, 'Disp')

OUT.toVTK('test.vtk')


OUT.toVTK()
