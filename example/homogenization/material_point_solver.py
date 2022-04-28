from fedoo import *
import numpy as np
import time
from simcoon import simmit as sim
import os

#--------------- Pre-Treatment --------------------------------------------------------
Util.ProblemDimension("3D")

meshID = "Domain2"

# Mesh.ImportFromFile('octet_surf.msh', meshID = "Domain")
Mesh.BoxMesh(2,2,2, ElementShape = 'hex8', ID = "Domain2")

umat_name = 'ELISO'
props = np.array([[1e5, 0.3, 1]])
nstatev = 1

L = sim.L_iso(1e5, 0.3, 'Enu')
props_test = sim.L_iso_props(L)
print('props', props_test)

solver_type = 0
corate_type = 2

path_data = 'data'
path_results = 'results'
path_file = 'path.txt'
outputfile = 'results_ELISO.txt'
outputdatfile = 'output.dat'

# Homogen.SolverUnitCell(meshID, umat_name, props, nstatev, solver_type, corate_type, path_data, path_results, path_file, outputfile, outputdatfile, meshperio=True)

#Definition of the set of nodes for boundary conditions
mesh = Mesh.GetAll()[meshID]
    
crd = mesh.GetNodeCoordinates()
type_el = mesh.GetElementShape()

xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
zmax = np.max(crd[:,2]) ; zmin = np.min(crd[:,2])

crd_center = (np.array([xmin, ymin, zmin]) + np.array([xmax, ymax, zmax]))/2           
Volume = (xmax-xmin)*(ymax-ymin)*(zmax-zmin) #total volume of the domain

if '_StrainNodes' in mesh.ListSetOfNodes():
    StrainNodes = mesh.GetSetOfNodes('_StrainNodes')            
else:
    StrainNodes = mesh.AddNodes(crd_center,2) #add virtual nodes for macro strain
    mesh.AddSetOfNodes(StrainNodes,'_StrainNodes')

# center = [np.linalg.norm(crd[:-2]-crd_center,axis=1).argmin()] 
# center = [len(crd)-1]
center = [0] 

if isinstance(umat_name, str):
    material = ConstitutiveLaw.Simcoon(umat_name, props, nstatev, ID='ConstitutiveLaw')
    material.corate = corate_type
else:
    material = umat_name

#Assembly
WeakForm.InternalForce(material, ID="wf")
Assembly.Create("wf", mesh, type_el, ID="Assembling", nb_pg = 1)

#Type of problem
pb = Problem.NonLinearStatic("Assembling")

Homogen.DefinePeriodicBoundaryCondition(mesh,
[StrainNodes[0], StrainNodes[0], StrainNodes[0], StrainNodes[1], StrainNodes[1], StrainNodes[1]],
['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D')
    
readPath = sim.read_path(path_data,path_file)
blocks = readPath[2]
cyclesPerBlocks = readPath[1]

MeanStress = np.zeros(6)
MeanStrain = np.zeros(6)
T = readPath[0] #temperature
time = 0.

pb.BoundaryCondition('Dirichlet','DispX', 0, center)
pb.BoundaryCondition('Dirichlet','DispY', 0, center)
pb.BoundaryCondition('Dirichlet','DispZ', 0, center)

pb.SetNewtonRaphsonErrorCriterion('Displacement')

#create a 'result' folder and set the desired ouputs
if not(os.path.isdir(path_results)): os.mkdir(path_results)

content = Homogen.Read_outputfile(path_data,outputdatfile )

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
            
            pb.RemoveBC("Strain")
            pb.BoundaryCondition(BCtype[0],'DispX', BC_mecas[0,i], [StrainNodes[0]], initialValue = initValue[0], ID = 'Strain') #EpsXX
            pb.BoundaryCondition(BCtype[1],'DispY', BC_mecas[1,i], [StrainNodes[0]], initialValue = initValue[1], ID = 'Strain') #EpsYY
            pb.BoundaryCondition(BCtype[2],'DispZ', BC_mecas[2,i], [StrainNodes[0]], initialValue = initValue[2], ID = 'Strain') #EpsZZ
            pb.BoundaryCondition(BCtype[3],'DispX', BC_mecas[3,i], [StrainNodes[1]], initialValue = initValue[3], ID = 'Strain') #EpsXY
            pb.BoundaryCondition(BCtype[4],'DispY', BC_mecas[4,i], [StrainNodes[1]], initialValue = initValue[4], ID = 'Strain') #EpsXZ
            pb.BoundaryCondition(BCtype[5],'DispZ', BC_mecas[5,i], [StrainNodes[1]], initialValue = initValue[5], ID = 'Strain') #EpsYZ
            
            #pb.ApplyBoundaryCondition()
            # pb.NLSolve(dt = dt*step.Dn_init, dt_min = dt*step.Dn_init*step.Dn_mini, tmax = dt, update_dt = False, ToleranceNR = 0.05, intervalOutput = 2.0*dt)
            pb.NLSolve(dt = 0.1, dt_min = 1e-5, tmax = dt, update_dt = True, ToleranceNR = 0.005)
            # print('TIME: ', time)
            
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
            
                TangentMatrix = GetTangentStiffness(ProblemID)
                TangentMatrix_All.append(TangentMatrix.flatten())
                
print(material.GetStress())
print(material.GetStrain())