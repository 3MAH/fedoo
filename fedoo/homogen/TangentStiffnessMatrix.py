#derive de ConstitutiveLaw
#This law should be used with an InternalForce WeakForm

from fedoo.core.base import MeshBase as Mesh
from fedoo.constitutivelaw import ElasticAnisotropic
from fedoo.core.base import ConstitutiveLaw
from fedoo.weakform.internal_force import InternalForce
from fedoo.core.assembly import Assembly
from fedoo.core.problem import Problem
from fedoo.problem.static import Static
# from fedoo.core.base import BoundaryCondition
from fedoo.core.base import ProblemBase
from fedoo.homogen.PeriodicBoundaryCondition import DefinePeriodicBoundaryCondition, DefinePeriodicBoundaryConditionNonPerioMesh
import numpy as np
import os
import time

def GetHomogenizedStiffness(assemb,meshperio=True, **kargs):

    #Definition of the set of nodes for boundary conditions
    if isinstance(assemb, str):
        assemb = Assembly.get_all()[assemb]
    mesh = assemb.mesh

    if '_StrainNodes' in mesh.ListSetOfNodes():
        crd = mesh.nodes[:-2]
    else: 
        crd = mesh.nodes

    type_el = mesh.elm_type
    xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
    ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
    zmax = np.max(crd[:,2]) ; zmin = np.min(crd[:,2])
    crd_center = (np.array([xmin, ymin, zmin]) + np.array([xmax, ymax, zmax]))/2
    center = [np.linalg.norm(crd-crd_center,axis=1).argmin()]
        
    BC_perturb = np.eye(6)
    # BC_perturb[3:6,3:6] *= 2 #2xEXY

    DStrain = []
    DStress = []

    if '_StrainNodes' in mesh.ListSetOfNodes():
        StrainNodes = mesh.node_sets['_StrainNodes']
        remove_strain = False
    else:
        StrainNodes = mesh.add_nodes(crd_center,2) #add virtual nodes for macro strain
        mesh.add_node_set(StrainNodes, '_StrainNodes')
        remove_strain = True

    #Type of problem
    pb = Static(assemb)
    
    C = GetTangentStiffness(pb,meshperio, **kargs)
    if remove_strain:
       mesh.remove_nodes(StrainNodes)
       mesh.RemoveSetOfNodes('_StrainNodes')
       
    del pb.get_all()['_perturbation'] #erase the perturbation problem in case of homogenized stiffness is required for another mesh

    return C


def GetHomogenizedStiffness_2(mesh, L, meshperio=True, Problemname =None, **kargs):
    #################### PERTURBATION METHODE #############################

    solver = kargs.get('solver', 'direct')
    
    #Definition of the set of nodes for boundary conditions
    if isinstance(mesh, str):
        mesh = Mesh.get_all()[mesh]

    if '_StrainNodes' in mesh.ListSetOfNodes():
        crd = mesh.nodes[:-2]
    else: 
        crd = mesh.nodes
        
    type_el = mesh.elm_type
    # type_el = 'hex20'
    xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
    ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
    zmax = np.max(crd[:,2]) ; zmin = np.min(crd[:,2])
    crd_center = (np.array([xmin, ymin, zmin]) + np.array([xmax, ymax, zmax]))/2
    center = [np.linalg.norm(crd-crd_center,axis=1).argmin()]
        
    BC_perturb = np.eye(6)
    # BC_perturb[3:6,3:6] *= 2 #2xEXY

    DStrain = []
    DStress = []

    if '_StrainNodes' in mesh.ListSetOfNodes():
        StrainNodes = mesh.node_sets['_StrainNodes']
    else:
        StrainNodes = mesh.add_nodes(crd_center,2) #add virtual nodes for macro strain
        mesh.add_node_set(StrainNodes,'_StrainNodes')

    ElasticAnisotropic(L, name = 'ElasticLaw')
        
    #Assembly
    InternalForce("ElasticLaw")
    Assembly("ElasticLaw", mesh, type_el, name ="Assembling")

    #Type of problem
    pb = Static("Assembling")

    pb_post_tt = Problem(0,0,0, mesh, name = "_perturbation")
    pb_post_tt.SetSolver(solver)
    pb_post_tt.SetA(pb.GetA())    
    
    #Shall add other conditions later on
    if meshperio:
        DefinePeriodicBoundaryCondition(mesh,
        [StrainNodes[0], StrainNodes[0], StrainNodes[0], StrainNodes[1], StrainNodes[1], StrainNodes[1]],
        ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D', Problemname = "_perturbation")
    else:
        DefinePeriodicBoundaryConditionNonPerioMesh(mesh,
        [StrainNodes[0], StrainNodes[0], StrainNodes[0], StrainNodes[1], StrainNodes[1], StrainNodes[1]],
        ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D', Problemname = "_perturbation")

    pb_post_tt.BoundaryCondition('Dirichlet', 'DispX', 0, center, name = 'center')
    pb_post_tt.BoundaryCondition('Dirichlet', 'DispY', 0, center, name = 'center')
    pb_post_tt.BoundaryCondition('Dirichlet', 'DispZ', 0, center, name = 'center')

    pb_post_tt.apply_boundary_conditions()

    DofFree = pb_post_tt._Problem__DofFree
    MatCB = pb_post_tt._Problem__MatCB

    # typeBC = 'Dirichlet' #doesn't work with meshperio = False
    typeBC = 'Neumann'
    for i in range(6):
        pb_post_tt.RemoveBC("_Strain")
        pb_post_tt.BoundaryCondition(typeBC, 'DispX',
              BC_perturb[i][0], [StrainNodes[0]], initialValue=0, name = '_Strain')  # EpsXX
        pb_post_tt.BoundaryCondition(typeBC, 'DispY',
              BC_perturb[i][1], [StrainNodes[0]], initialValue=0, name = '_Strain')  # EpsYY
        pb_post_tt.BoundaryCondition(typeBC, 'DispZ',
              BC_perturb[i][2], [StrainNodes[0]], initialValue=0, name = '_Strain')  # EpsZZ
        pb_post_tt.BoundaryCondition(typeBC, 'DispX',
              BC_perturb[i][3], [StrainNodes[1]], initialValue=0, name = '_Strain')  # EpsXY
        pb_post_tt.BoundaryCondition(typeBC, 'DispY',
              BC_perturb[i][4], [StrainNodes[1]], initialValue=0, name = '_Strain')  # EpsXZ
        pb_post_tt.BoundaryCondition(typeBC, 'DispZ',
              BC_perturb[i][5], [StrainNodes[1]], initialValue=0, name = '_Strain')  # EpsYZ
        
        pb_post_tt.apply_boundary_conditions()

        pb_post_tt.solve()
                
        X = pb_post_tt.GetX()  # alias
        DStrain.append(np.array([pb_post_tt._get_vect_component(X, 'DispX')[StrainNodes[0]], pb_post_tt._get_vect_component(X, 'DispY')[StrainNodes[0]], pb_post_tt._get_vect_component(X, 'DispZ')[StrainNodes[0]],
                                  pb_post_tt._get_vect_component(X, 'DispX')[StrainNodes[1]], pb_post_tt._get_vect_component(X, 'DispY')[StrainNodes[1]], pb_post_tt._get_vect_component(X, 'DispZ')[StrainNodes[1]]]))

    if typeBC == "Neumann":
        C = np.linalg.inv(np.array(DStrain).T)
    else:
        
        F = MatCB.T @ pb_post_tt.GetA() @ MatCB @ pb_post_tt.GetX()[DofFree]

        F = F.reshape(3, -1)
        stress = [F[0, -2], F[1, -2], F[2, -2], F[0, -1], F[1, -1], F[2, -1]]

        DStress.append(stress)
        
        C = np.array(DStress).T
        
    return C


def GetTangentStiffness(pb = None, meshperio = True, **kargs):
    #################### PERTURBATION METHODE #############################    
    solver = kargs.get('solver', 'direct')
    
    if pb is None: 
        pb = ProblemBase.get_active()
    elif isinstance(pb, str):
        pb = ProblemBase.get_all()[pb]
    mesh = pb.mesh
    
    if '_StrainNodes' in mesh.ListSetOfNodes():
        crd = mesh.nodes[:-2]
    else: 
        crd = mesh.nodes
    
    xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
    ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
    zmax = np.max(crd[:,2]) ; zmin = np.min(crd[:,2])
    crd_center = (np.array([xmin, ymin, zmin]) + np.array([xmax, ymax, zmax]))/2
    center = [np.linalg.norm(crd-crd_center,axis=1).argmin()]    
        
    BC_perturb = np.eye(6)  
    # BC_perturb[3:6,3:6] *= 2 #2xEXY
    
    DStrain = []
    DStress = []
    
    if '_StrainNodes' in mesh.ListSetOfNodes():
        StrainNodes = mesh.node_sets['_StrainNodes']
        remove_strain = False
        A = pb.GetA()
    else:
        StrainNodes = mesh.add_nodes(crd_center,2) #add virtual nodes for macro strain
        mesh.add_node_set(StrainNodes,'_StrainNodes')
        remove_strain = True
        A = pb.GetA().copy()
        A.resize(np.array(pb.GetA().shape)+6)
    # StrainNodes=[len(crd),len(crd)+1] #last 2 nodes
    
    if "_perturbation" not in pb.get_all():
        #initialize perturbation problem 
        pb_post_tt = Problem(0,0,0, mesh, name = "_perturbation")
        pb_post_tt.SetSolver('cg')
        
        pb.MakeActive()
        
        #Shall add other conditions later on
        if meshperio:
            DefinePeriodicBoundaryCondition(mesh,
            [StrainNodes[0], StrainNodes[0], StrainNodes[0], StrainNodes[1], StrainNodes[1], StrainNodes[1]],
            ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D', Problemname = "_perturbation")
        else:
            DefinePeriodicBoundaryConditionNonPerioMesh(mesh,
            [StrainNodes[0], StrainNodes[0], StrainNodes[0], StrainNodes[1], StrainNodes[1], StrainNodes[1]],
            ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D', Problemname = "_perturbation")
            
        pb_post_tt.BoundaryCondition('Dirichlet', 'DispX', 0, center, name = 'center')
        pb_post_tt.BoundaryCondition('Dirichlet', 'DispY', 0, center, name = 'center')
        pb_post_tt.BoundaryCondition('Dirichlet', 'DispZ', 0, center, name = 'center')
    else: 
        pb_post_tt = Problem.get_all()["_perturbation"]
    
    pb_post_tt.SetA(pb.GetA())
    
    # typeBC = 'Dirichlet' #doesn't work with meshperio = False
    typeBC = 'Neumann'
    
    pb_post_tt.apply_boundary_conditions()
    
    DofFree = pb_post_tt._Problem__DofFree
    MatCB = pb_post_tt._Problem__MatCB
    
    for i in range(6):
        pb_post_tt.BoundaryCondition(typeBC, 'DispX',
              BC_perturb[i][0], [StrainNodes[0]], initialValue=0, name = '_Strain')  # EpsXX
        pb_post_tt.BoundaryCondition(typeBC, 'DispY',
              BC_perturb[i][1], [StrainNodes[0]], initialValue=0, name = '_Strain')  # EpsYY
        pb_post_tt.BoundaryCondition(typeBC, 'DispZ',
              BC_perturb[i][2], [StrainNodes[0]], initialValue=0, name = '_Strain')  # EpsZZ
        pb_post_tt.BoundaryCondition(typeBC, 'DispX',
              BC_perturb[i][3], [StrainNodes[1]], initialValue=0, name = '_Strain')  # EpsXY
        pb_post_tt.BoundaryCondition(typeBC, 'DispY',
              BC_perturb[i][4], [StrainNodes[1]], initialValue=0, name = '_Strain')  # EpsXZ
        pb_post_tt.BoundaryCondition(typeBC, 'DispZ',
              BC_perturb[i][5], [StrainNodes[1]], initialValue=0, name = '_Strain')  # EpsYZ
        
        pb_post_tt.apply_boundary_conditions()
    
        pb_post_tt.solve()
        X = pb_post_tt.GetX()  # alias    
        DStrain.append(np.array([pb_post_tt._get_vect_component(X, 'DispX')[StrainNodes[0]], pb_post_tt._get_vect_component(X, 'DispY')[StrainNodes[0]], pb_post_tt._get_vect_component(X, 'DispZ')[StrainNodes[0]],
                                  pb_post_tt._get_vect_component(X, 'DispX')[StrainNodes[1]], pb_post_tt._get_vect_component(X, 'DispY')[StrainNodes[1]], pb_post_tt._get_vect_component(X, 'DispZ')[StrainNodes[1]]]))        
        
        pb_post_tt.RemoveBC("_Strain")
    
    if typeBC == "Neumann":
        C = np.linalg.inv(np.array(DStrain).T)
    else:
        F = MatCB.T @ pb_post_tt.GetA() @ MatCB @ pb_post_tt.GetX()[DofFree]
    
        F = F.reshape(3, -1)
        stress = [F[0, -2], F[1, -2], F[2, -2], F[0, -1], F[1, -1], F[2, -1]]
    
        DStress.append(stress)
        
        C = np.array(DStress).T
                
    if remove_strain:
       mesh.remove_nodes(StrainNodes)
       mesh.RemoveSetOfNodes('_StrainNodes')
        
    return C



# # ############# CONDENSATION METHOD -> a tester si besoin ##################
# Compute the stress

# Fext_b = Problem.get_ext_forces('Disp')[:,boundary_nodes] #Get the external reaction on boundary nodes
# MeanStress2 = (1/Volume)* D@Fext_b.ravel()

# #Compute tangent matrix
# K = Assembly.get_all()['Assembling'].get_global_matrix()    #Get global FE assembly matrix


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

    
    