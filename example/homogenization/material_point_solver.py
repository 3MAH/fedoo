from fedoo import *
import numpy as np
import time
from simcoon import simmit as sim
import os

#--------------- Pre-Treatment --------------------------------------------------------
ModelingSpace("3D")

meshname = "Domain2"

# Mesh.import_file('./meshes/octet_surf.msh', meshname = "Domain")
Mesh.box_mesh(2,2,2, ElementShape = 'hex8', name = "Domain2")

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

# Homogen.SolverUnitCell(meshname, umat_name, props, nstatev, solver_type, corate_type, path_data, path_results, path_file, outputfile, outputdatfile, meshperio=True)

#Definition of the set of nodes for boundary conditions
mesh = Mesh.get_all()[meshname]
    
crd = mesh.nodes
type_el = mesh.elm_type

xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
zmax = np.max(crd[:,2]) ; zmin = np.min(crd[:,2])

crd_center = (np.array([xmin, ymin, zmin]) + np.array([xmax, ymax, zmax]))/2           
Volume = (xmax-xmin)*(ymax-ymin)*(zmax-zmin) #total volume of the domain

if '_StrainNodes' in mesh.node_sets:
    StrainNodes = mesh.node_sets['_StrainNodes']            
else:
    StrainNodes = mesh.add_nodes(crd_center,2) #add virtual nodes for macro strain
    mesh.add_node_set(StrainNodes,'_StrainNodes')

# center = [np.linalg.norm(crd[:-2]-crd_center,axis=1).argmin()] 
# center = [len(crd)-1]
center = [0] 

if isinstance(umat_name, str):
    material = ConstitutiveLaw.Simcoon(umat_name, props, nstatev, name='ConstitutiveLaw')
    material.corate = corate_type
else:
    material = umat_name

#Assembly
WeakForm.StressEquilibrium(material, name="wf")
Assembly.create("wf", mesh, type_el, name="Assembling", n_elm_gp = 1)

#Type of problem
pb = Problem.NonLinear("Assembling")

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

pb.bc.add('Dirichlet','DispX', 0, center)
pb.bc.add('Dirichlet','DispY', 0, center)
pb.bc.add('Dirichlet','DispZ', 0, center)

pb.set_nr_criterion('Displacement')

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
            pb.bc.add(BCtype[0],'DispX', BC_mecas[0,i], [StrainNodes[0]], initialValue = initValue[0], name = 'Strain') #EpsXX
            pb.bc.add(BCtype[1],'DispY', BC_mecas[1,i], [StrainNodes[0]], initialValue = initValue[1], name = 'Strain') #EpsYY
            pb.bc.add(BCtype[2],'DispZ', BC_mecas[2,i], [StrainNodes[0]], initialValue = initValue[2], name = 'Strain') #EpsZZ
            pb.bc.add(BCtype[3],'DispX', BC_mecas[3,i], [StrainNodes[1]], initialValue = initValue[3], name = 'Strain') #EpsXY
            pb.bc.add(BCtype[4],'DispY', BC_mecas[4,i], [StrainNodes[1]], initialValue = initValue[4], name = 'Strain') #EpsXZ
            pb.bc.add(BCtype[5],'DispZ', BC_mecas[5,i], [StrainNodes[1]], initialValue = initValue[5], name = 'Strain') #EpsYZ
            
            #pb.apply_boundary_conditions()
            # pb.nlsolve(dt = dt*step.Dn_init, dt_min = dt*step.Dn_init*step.Dn_mini, tmax = dt, update_dt = False, ToleranceNR = 0.05, intervalOutput = 2.0*dt)
            pb.nlsolve(dt = 0.1, dt_min = 1e-5, tmax = dt, update_dt = True, ToleranceNR = 0.005)
            # print('TIME: ', time)
            
            #--------------- Post-Treatment -----------------------------------------------
            #Compute the mean stress and strain
            #Get the stress tensor (PG values)
            # TensorStrain = Assembly.get_all()['Assembling'].get_strain(Problem.get_dof_solution(), "GaussPoint")

            TensorStrain = material.GetStrain()
            TensorStress = material.GetPKII()
            
            MeanStress = np.array([1/Volume*Assembly.get_all()['Assembling'].integrate_field(TensorStress[i]) for i in range(6)])

            MeanStrain = np.array([pb.get_disp('DispX')[-2], pb.get_disp('DispY')[-2], pb.get_disp('DispZ')[-2],
                         pb.get_disp('DispX')[-1], pb.get_disp('DispY')[-1], pb.get_disp('DispZ')[-1]])
            
            Wm_mean = (1/Volume) * Assembly.get_all()['Assembling'].integrate_field(material.Wm)

            MeanStress_All.append(MeanStress)
            MeanStrain_All.append(MeanStrain)
            MeanWm_All.append(Wm_mean)

            if Tangent_bool:
            
                TangentMatrix = GetTangentStiffness(Problemname)
                TangentMatrix_All.append(TangentMatrix.flatten())
                
print(material.GetStress())
print(material.GetStrain())