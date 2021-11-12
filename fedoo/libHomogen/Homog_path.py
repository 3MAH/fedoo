#derive de ConstitutiveLaw
#This law should be used with an InternalForce WeakForm

from simcoon import simmit as sim
from fedoo.libMesh import MeshBase as Mesh
from fedoo.libConstitutiveLaw import Simcoon
from fedoo.libWeakForm import InternalForce
from fedoo.libAssembly import Assembly
from fedoo.libProblem import NonLinearStatic, BoundaryCondition
from fedoo.libUtil import DefinePeriodicBoundaryCondition
import numpy as np
import os
import time

def SolverUnitCell(mesh, umat_name, props, nstatev, solver_type, corate_type, path_data, path_results, path_file, outputfile):

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
    DefinePeriodicBoundaryCondition(mesh,
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

    for blocknumber, block in enumerate(blocks):
        for stepTotNumber, step in enumerate(block):
            step.generate(time, MeanStrain, MeanStress, T)
            
            cyclesPerBlock = cyclesPerBlocks[blocknumber]
            nbStepsPerCycle = len(block)//cyclesPerBlock
            stepNumber = stepTotNumber%nbStepsPerCycle
            cycleNumber = stepTotNumber // nbStepsPerCycle
            
            print(blocknumber+1, cycleNumber+1, stepNumber+1)

            #Boundary conditions
            BC_meca = step.BC_meca #stress or strain BC
            BC_mecas = step.BC_mecas
            
            BCtype = np.array(['Dirichlet' for i in range(6)])
            BCtype[step.cBC_meca.astype(bool)] = 'Neumann'

            for i, dt in enumerate(step.times):

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
                
                # print(step.mecas)
                # print(pb.GetA().shape)
                # print(TensorStrain)
                # print(TensorStress)
                # print(mesh.GetNodeCoordinates())
                      
                # Volume_mesh = Assembly.GetAll()['Assembling'].IntegrateField(np.ones_like(TensorStress[0])) #volume of domain without the void (hole)
                
                MeanStress = np.array([1/Volume*Assembly.GetAll()['Assembling'].IntegrateField(TensorStress[i]) for i in range(6)])
                
                MeanStrain = np.array([pb.GetDisp('DispX')[-2], pb.GetDisp('DispY')[-2], pb.GetDisp('DispZ')[-2],
                                       pb.GetDisp('DispX')[-1], pb.GetDisp('DispY')[-1], pb.GetDisp('DispZ')[-1]])
                
                Wm_mean = (1/Volume) * Assembly.GetAll()['Assembling'].IntegrateField(material.Wm)
                
                # Other method: only work if volume with no void (Void=0)
                # Void = Volume-Volume_mesh
                # MeanStrain = [1/Volume*Assembly.GetAll()['Assembling'].IntegrateField(TensorStrain[i]) for i in range(6)]
                
                print('Strain tensor ([Exx, Eyy, Ezz, Exy, Exz, Eyz]): ' )
                print(MeanStrain)
                print('Stress tensor ([Sxx, Syy, Szz, Sxy, Sxz, Syz]): ' )
                print(MeanStress)
    

