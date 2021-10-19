#derive de ConstitutiveLaw
#This law should be used with an InternalForce WeakForm

# from fedoo.libMesh import MeshBase as Mesh
# from fedoo.libConstitutiveLaw import ElasticIsotrop
# from fedoo.libWeakForm import InternalForce
# from fedoo.libAssembly import Assembly
from fedoo.libProblem import Problem, BoundaryCondition
from fedoo.libProblem import ProblemBase
from fedoo.libUtil import DefinePeriodicBoundaryCondition
import numpy as np
import os
import time

def GetHomogenizedStiffness(ProblemID = None):
    #################### PERTURBATION METHODE #############################
    if ProblemID is None: ProblemID = ProblemBase.GetActive().GetID()
    pb = ProblemBase.GetAll()[ProblemID]
    mesh = pb.GetMesh()
    crd = mesh.GetNodeCoordinates()[:-2]
    xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
    ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
    zmax = np.max(crd[:,2]) ; zmin = np.min(crd[:,2])
    crd_center = (np.array([xmin, ymin, zmin]) + np.array([xmax, ymax, zmax]))/2
    center = [np.linalg.norm(crd-crd_center,axis=1).argmin()]    
        
    BC_perturb = np.eye(6)  
    # BC_perturb[3:6,3:6] *= 2 #2xEXY
    
    DStrain = []
    DStress = []
    
    StrainNodes=[len(crd),len(crd)+1] #last 2 nodes
    
    # ElasticIsotrop(1e5,0.3,'none') #not used because we will replace the assembled stiffness matrix
    # InternalForce("none") #not used 
    # Assembly("none", pb.GetMesh(), pb.GetMesh().GetElementShape(), ID="Assembling_post_tt")
    # pb_post_tt = Static("Assembling_post_tt", ID = "_perturbation")
    
    if "_perturbation" in pb.GetAll():
        pb_post_tt = pb.GetAll()["_perturbation"]
        pb_post_tt.MakeActive()
        BoundaryCondition.RemoveID("_Strain")
    else:
        pb_post_tt = Problem(0,0,0, mesh, ID = "_perturbation")
        pb_post_tt.SetA(pb.GetA())
        DefinePeriodicBoundaryCondition(mesh,
                                        [StrainNodes[0], StrainNodes[0], StrainNodes[0],
                                         StrainNodes[1], StrainNodes[1], StrainNodes[1]],
                                        ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D')
    
        BoundaryCondition('Dirichlet', 'DispX', 0, center, ID = 'center')
        BoundaryCondition('Dirichlet', 'DispY', 0, center, ID = 'center')
        BoundaryCondition('Dirichlet', 'DispZ', 0, center, ID = 'center')
    
    pb_post_tt.ApplyBoundaryCondition()
    
    DofFree = pb_post_tt._Problem__DofFree
    MatCB = pb_post_tt._Problem__MatCB
    
    typeBC = 'Dirichlet'
    # typeBC = 'Neumann'
    for i in range(6):
        BoundaryCondition.RemoveID("_Strain")
        BoundaryCondition(typeBC, 'DispX',
              BC_perturb[i][0], [StrainNodes[0]], initialValue=0, ID = '_Strain')  # EpsXX
        BoundaryCondition(typeBC, 'DispY',
              BC_perturb[i][1], [StrainNodes[0]], initialValue=0, ID = '_Strain')  # EpsYY
        BoundaryCondition(typeBC, 'DispZ',
              BC_perturb[i][2], [StrainNodes[0]], initialValue=0, ID = '_Strain')  # EpsZZ
        BoundaryCondition(typeBC, 'DispX',
              BC_perturb[i][3], [StrainNodes[1]], initialValue=0, ID = '_Strain')  # EpsXY
        BoundaryCondition(typeBC, 'DispY',
              BC_perturb[i][4], [StrainNodes[1]], initialValue=0, ID = '_Strain')  # EpsXZ
        BoundaryCondition(typeBC, 'DispZ',
              BC_perturb[i][5], [StrainNodes[1]], initialValue=0, ID = '_Strain')  # EpsYZ
        
        pb_post_tt.ApplyBoundaryCondition()
    
        pb_post_tt.Solve()
        X = pb_post_tt.GetX()  # alias    
        DStrain.append(np.array([pb_post_tt._GetVectorComponent(X, 'DispX')[StrainNodes[0]], pb_post_tt._GetVectorComponent(X, 'DispY')[StrainNodes[0]], pb_post_tt._GetVectorComponent(X, 'DispZ')[StrainNodes[0]],
                                  pb_post_tt._GetVectorComponent(X, 'DispX')[StrainNodes[1]], pb_post_tt._GetVectorComponent(X, 'DispY')[StrainNodes[1]], pb_post_tt._GetVectorComponent(X, 'DispZ')[StrainNodes[1]]]))
    
        F = MatCB.T @ pb_post_tt.GetA() @ MatCB @ pb_post_tt.GetX()[DofFree]
    
        F = F.reshape(3, -1)
        stress = [F[0, -2], F[1, -2], F[2, -2], F[0, -1], F[1, -1], F[2, -1]]
    
        DStress.append(stress)
    
    if typeBC == "Neumann":
        C = np.linalg.inv(np.array(DStrain).T)
    else:
        C = np.array(DStress).T
        
    pb.SetActive(ProblemID) #to reactivate the main problem
    return C



# # ############# CONDENSATION METHOD -> a tester si besoin ##################
# Compute the stress

# Fext_b = Problem.GetExternalForces('Disp')[:,boundary_nodes] #Get the external reaction on boundary nodes
# MeanStress2 = (1/Volume)* D@Fext_b.ravel()

# #Compute tangent matrix
# K = Assembly.GetAll()['Assembling'].GetMatrix()    #Get global FE assembly matrix


# Kbb = K[dof_boundary][:,dof_boundary]
# Kaa = K[dof_internal][:,dof_internal]
# Kab = K[dof_internal][:,dof_boundary]
# Kba = K[dof_boundary][:,dof_internal]

# Condensed_Kbb = (Kbb - Kba @sparse.linalg.inv(Kaa)@ Kab).todense() #matrix should be almost dense

# #macroscopic tangent matrix
# # C = (1/Volume)* D@ Condensed_Kbb @ D.T #linear BC
# C = (1/Volume)*Q@np.linalg.pinv(P @ np.linalg.pinv(Condensed_Kbb) @ P.T)@Q.T
# print(C-Cref)
# ####################################################

    
    