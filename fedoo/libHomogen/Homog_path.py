#derive de ConstitutiveLaw
#This law should be used with an InternalForce WeakForm

from simcoon import simmit as sim
from fedoo.libMesh import MeshBase as Mesh
from fedoo.libConstitutiveLaw import Simcoon
from fedoo.libWeakForm import InternalForce
from fedoo.libAssembly import Assembly
from fedoo.libProblem import NonLinearStatic, BoundaryCondition
from fedoo.libUtil import DefinePeriodicBoundaryCondition, DefinePeriodicBoundaryConditionNonPerioMesh
from fedoo.libHomogen.TangentStiffnessMatrix import GetHomogenizedStiffness
import numpy as np
import os
import re
import time
import pandas as pd

def Read_outputfile(path_data,outputdat_file):
    
    file = path_data + "/" + outputdat_file
    with open(file) as f:
        content = f.read().splitlines()
    for i in range(len(content)):
        content[i] = content[i].replace('\t', ' ')
        content[i] = re.sub(' +', ' ', content[i])
            
    return content

def SolverUnitCell(mesh, umat_name, props, nstatev, solver_type, corate_type, path_data, path_results, path_file, outputfile, outputdat_file, meshperio=True):

    #Definition of the set of nodes for boundary conditions
    if isinstance(mesh, str):
        mesh = Mesh.GetAll()[mesh]
        
    crd = mesh.GetNodeCoordinates()
    type_el = mesh.GetElementShape()

    xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
    ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
    zmax = np.max(crd[:,2]) ; zmin = np.min(crd[:,2])
    crd_center = (np.array([xmin, ymin, zmin]) + np.array([xmax, ymax, zmax]))/2
    center = [np.linalg.norm(crd-crd_center,axis=1).argmin()]

    Volume = (xmax-xmin)*(ymax-ymin)*(zmax-zmin) #total volume of the domain

    StrainNodes = mesh.AddNodes(crd_center,2) #add virtual nodes for macro strain

    if isinstance(umat_name, str):
        material = Simcoon(umat_name, props, nstatev, ID='ConstitutiveLaw')
        material.corate = corate_type
    else:
        material = umat_name

    #Assembly
    InternalForce(material, ID="wf")
    Assembly("wf", mesh, type_el, ID="Assembling")

    #Type of problem
    pb = NonLinearStatic("Assembling")
    
    #Shall add other conditions later on
    if meshperio:
        DefinePeriodicBoundaryCondition(mesh,
        [StrainNodes[0], StrainNodes[0], StrainNodes[0], StrainNodes[1], StrainNodes[1], StrainNodes[1]],
        ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D')
    else:
        DefinePeriodicBoundaryConditionNonPerioMesh(mesh,
        [StrainNodes[0], StrainNodes[0], StrainNodes[0], StrainNodes[1], StrainNodes[1], StrainNodes[1]],
        ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D')
        

    readPath = sim.read_path(path_data,path_file)
    blocks = readPath[2]
    cyclesPerBlocks = readPath[1]

    MeanStress = np.zeros(6)
    MeanStrain = np.zeros(6)
    T = readPath[0] #temperature
    time = 0.

    BoundaryCondition('Dirichlet','DispX', 0, center)
    BoundaryCondition('Dirichlet','DispY', 0, center)
    BoundaryCondition('Dirichlet','DispZ', 0, center)

    #create a 'result' folder and set the desired ouputs
    if not(os.path.isdir(path_results)): os.mkdir(path_results)
    
    content = Read_outputfile(path_data,outputdat_file)
    
    BlocksCyclesSteps = []
    MeanStress_All = []
    MeanStrain_All = []
    MeanWm_All = []
    
    Tangent_bool = None
    if ('Tangent_type 1' in content):
       Tangent_bool = True
    if Tangent_bool:
       TangentMatrix_All = []
    
    time=0    
    
    for blocknumber, block in enumerate(blocks):
        for stepTotNumber, step in enumerate(block):
            step.generate(time, MeanStrain, MeanStress, T)
            
            cyclesPerBlock = cyclesPerBlocks[blocknumber]
            nbStepsPerCycle = len(block)//cyclesPerBlock
            stepNumber = stepTotNumber%nbStepsPerCycle
            cycleNumber = stepTotNumber // nbStepsPerCycle
            
            #Boundary conditions
            BC_meca = step.BC_meca #stress or strain BC
            BC_mecas = step.BC_mecas
            
            BCtype = np.array(['Dirichlet' for i in range(6)])
            BCtype[step.cBC_meca.astype(bool)] = 'Neumann'

            for i, dt in enumerate(step.times):
                
                time+=dt
                BlocksCyclesSteps.append(np.array([int(blocknumber+1),int(cycleNumber+1),int(stepNumber+1),i+1,"{0:.3f}".format(time)]))
                
                initValue = np.array(MeanStrain)
                initValue[step.cBC_meca.astype(bool)] = MeanStress[step.cBC_meca.astype(bool)]
                
                BoundaryCondition.RemoveID("Strain")
                BoundaryCondition(BCtype[0],'DispX', BC_mecas[0,i], [StrainNodes[0]], initialValue = initValue[0], ID = 'Strain') #EpsXX
                BoundaryCondition(BCtype[1],'DispY', BC_mecas[1,i], [StrainNodes[0]], initialValue = initValue[1], ID = 'Strain') #EpsYY
                BoundaryCondition(BCtype[2],'DispZ', BC_mecas[2,i], [StrainNodes[0]], initialValue = initValue[2], ID = 'Strain') #EpsZZ
                BoundaryCondition(BCtype[3],'DispX', BC_mecas[3,i], [StrainNodes[1]], initialValue = initValue[3], ID = 'Strain') #EpsXY
                BoundaryCondition(BCtype[4],'DispY', BC_mecas[4,i], [StrainNodes[1]], initialValue = initValue[4], ID = 'Strain') #EpsXZ
                BoundaryCondition(BCtype[5],'DispZ', BC_mecas[5,i], [StrainNodes[1]], initialValue = initValue[5], ID = 'Strain') #EpsYZ
                
                #pb.ApplyBoundaryCondition()
                pb.NLSolve(dt = dt*step.Dn_init, dt_min = dt*step.Dn_init*step.Dn_mini, tmax = dt, update_dt = True, ToleranceNR = 0.05, intervalOutput = 2.0*dt)
                
                #--------------- Post-Treatment -----------------------------------------------
                #Compute the mean stress and strain
                #Get the stress tensor (PG values)
                # TensorStrain = Assembly.GetAll()['Assembling'].GetStrainTensor(Problem.GetDoFSolution(), "GaussPoint")

                TensorStrain = material.GetStrain()
                TensorStress = material.GetPKII()
                
                MeanStress = np.array([1/Volume*Assembly.GetAll()['Assembling'].IntegrateField(TensorStress[i]) for i in range(6)])

                MeanStrain = np.array([pb.GetDisp('DispX')[-2], pb.GetDisp('DispY')[-2], pb.GetDisp('DispZ')[-2],
                             pb.GetDisp('DispX')[-1], pb.GetDisp('DispY')[-1], pb.GetDisp('DispZ')[-1]])
                
                Wm_mean = (1/Volume) * Assembly.GetAll()['Assembling'].IntegrateField(material.Wm)

                MeanStress_All.append(MeanStress)
                MeanStrain_All.append(MeanStrain)
                MeanWm_All.append(Wm_mean)

                if Tangent_bool:
                
                    TangentMatrix = GetHomogenizedStiffness()
                    TangentMatrix_All.append(TangentMatrix.flatten())
                    
    
    if Tangent_bool:
        return BlocksCyclesSteps,MeanStrain_All,MeanStress_All,MeanWm_All,TangentMatrix_All
    else: 
        return BlocksCyclesSteps,MeanStrain_All,MeanStress_All,MeanWm_All

def GetResultsUnitCell(mesh, umat_name, props, nstatev, solver_type, corate_type, path_data, path_results, path_file, outputfile, outputdat_file):
    
    Res = SolverUnitCell(mesh, umat_name, props, nstatev, solver_type, corate_type, path_data, path_results, path_file, outputfile, outputdat_file)
        
    content = Read_outputfile(path_data,outputdat_file)
    
    Strain_columns = ['Exx', 'Eyy', 'Ezz', 'Exy', 'Exz', 'Eyz']
    Stress_columns = ['Sxx', 'Syy', 'Szz', 'Sxy', 'Sxz', 'Syz']
    BlocksCyclesSteps_columns = ['Block','Cycle','Step','N_TimeStep','Time']
    Wm_columns = ['Wm','Wm_r','Wm_ir','Wm_d']
    
    
    BlocksCyclesSteps_df = pd.DataFrame(np.array(Res[0]) , columns = BlocksCyclesSteps_columns)
    MeanStrain_df = pd.DataFrame(np.array(Res[1]) , columns = Strain_columns)
    MeanStress_df = pd.DataFrame(np.array(Res[2]) , columns = Stress_columns)
    Wm_df = pd.DataFrame(np.array(Res[3]) , columns = Wm_columns)
    
    Tangent_bool = None
    if ('Tangent_type 1' in content):
       Tangent_bool = True
    
    if Tangent_bool:
        TangentMatrix_columns=[]
        for i in range(6):
            for j in range(6):
                TangentMatrix_columns.append('C'+str(i+1)+str(j+1))
        TangentMatrix_df = pd.DataFrame(np.array(Res[4]) , columns = TangentMatrix_columns)
        Results_df = pd.concat([BlocksCyclesSteps_df,MeanStrain_df,MeanStress_df,Wm_df,TangentMatrix_df], axis=1)
    else:
         Results_df = pd.concat([BlocksCyclesSteps_df,MeanStrain_df,MeanStress_df,Wm_df], axis=1)
    
    Results_df.to_csv(path_results + "/" + outputfile, index=None, header=True)