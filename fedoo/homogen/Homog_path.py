#derive de ConstitutiveLaw
#This law should be used with an InternalForce WeakForm

USE_SIMCOON = False
if USE_SIMCOON: 
    try:
        from simcoon import simmit as sim
        USE_SIMCOON = True
    except:
        USE_SIMCOON = False
        print('WARNING: Simcoon library not found. The simcoon constitutive law is disabled.')       

if USE_SIMCOON:    
    from fedoo.core.base import MeshBase as Mesh
    from fedoo.constitutivelaw.umat_simcoon import Simcoon
    from fedoo.weakform.stress_equilibrium import StressEquilibrium
    from fedoo.core.assembly import Assembly
    from fedoo.problem.non_linear import NonLinear
    # from fedoo.core.base import BoundaryCondition
    from fedoo.constraint.periodic_bc import PeriodicBC #, DefinePeriodicBoundaryConditionNonPerioMesh
    from fedoo.homogen.tangent_stiffness import get_tangent_stiffness
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
    
    def SolverUnitCell(mesh, umat_name, props, nstatev, solver_type, corate_type, path_data, path_results, path_file, outputfile, outputdat_file, meshperio=True, Problemname ='MainProblem'):
    
        #Definition of the set of nodes for boundary conditions
        if isinstance(mesh, str):
            mesh = Mesh.get_all()[mesh]
            
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
       
        center = [np.linalg.norm(crd[:-2]-crd_center,axis=1).argmin()] 
       
        if isinstance(umat_name, str):
            material = Simcoon(umat_name, props, nstatev, name ='ConstitutiveLaw')
            material.corate = corate_type
        else:
            material = umat_name
    
        #Assembly
        InternalForce(material, name ="wf")
        Assembly("wf", mesh, type_el, name ="Assembling")
    
        #Type of problem
        pb = NonLinearStatic("Assembling", name =Problemname)
        
        #Shall add other conditions later on
        if meshperio:
            DefinePeriodicBoundaryCondition(mesh,
            [StrainNodes[0], StrainNodes[0], StrainNodes[0], StrainNodes[1], StrainNodes[1], StrainNodes[1]],
            ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D', Problemname = Problemname)
        else:
            DefinePeriodicBoundaryConditionNonPerioMesh(mesh,
            [StrainNodes[0], StrainNodes[0], StrainNodes[0], StrainNodes[1], StrainNodes[1], StrainNodes[1]],
            ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D', Problemname = Problemname)
            
    
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
                    
                    pb.RemoveBC("Strain")
                    pb.BoundaryCondition(BCtype[0],'DispX', BC_mecas[0,i], [StrainNodes[0]], initialValue = initValue[0], name = 'Strain') #EpsXX
                    pb.BoundaryCondition(BCtype[1],'DispY', BC_mecas[1,i], [StrainNodes[0]], initialValue = initValue[1], name = 'Strain') #EpsYY
                    pb.BoundaryCondition(BCtype[2],'DispZ', BC_mecas[2,i], [StrainNodes[0]], initialValue = initValue[2], name = 'Strain') #EpsZZ
                    pb.BoundaryCondition(BCtype[3],'DispX', BC_mecas[3,i], [StrainNodes[1]], initialValue = initValue[3], name = 'Strain') #EpsXY
                    pb.BoundaryCondition(BCtype[4],'DispY', BC_mecas[4,i], [StrainNodes[1]], initialValue = initValue[4], name = 'Strain') #EpsXZ
                    pb.BoundaryCondition(BCtype[5],'DispZ', BC_mecas[5,i], [StrainNodes[1]], initialValue = initValue[5], name = 'Strain') #EpsYZ
                    
#                    pb.AddOutput('results/field', 'Assembling', ['Disp', 'Cauchy', 'Strain', 'Cauchy_vm', 'Statev', 'Wm'], output_type='Node', file_format ='vtk')
#                    pb.AddOutput('results/field', 'Assembling', ['Cauchy', 'Strain', 'Cauchy_vm', 'Statev'], output_type='Element', file_format ='vtk')

                    #pb.apply_boundary_conditions()
                    pb.nlsolve(dt = dt*step.Dn_init, dt_min = dt*step.Dn_init*step.Dn_mini, tmax = dt, update_dt = True, ToleranceNR = 0.05, intervalOutput = 2.0*dt)

                    # print('TIME: ', time)
                    
                    #--------------- Post-Treatment -----------------------------------------------
                    #Compute the mean stress and strain
                    #Get the stress tensor (PG values)
                    # TensorStrain = Assembly.get_all()['Assembling'].get_strain(Problem.get_dof_solution(), "GaussPoint")
    
                    TensorStrain = material.get_strain()
                    TensorStress = material.get_pk2()
                    
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
                        
                    ###DEBUG ONLY####
                    # print(TangentMatrix)
                    # from fedoo.utilities import StressTensorArray                     
                    # print(StressTensorArray(TensorStress).vonMises().max())                    
                    # # print(StressTensorArray(MeanStress).vonMises())                  
                    ######
        
        if Tangent_bool:
            return BlocksCyclesSteps,MeanStrain_All,MeanStress_All,MeanWm_All,TangentMatrix_All
        else: 
            return BlocksCyclesSteps,MeanStrain_All,MeanStress_All,MeanWm_All
    
    def get_resultsUnitCell(mesh, umat_name, props, nstatev, solver_type, corate_type, path_data, path_results, path_file, outputfile, outputdat_file,meshperio=True, Problemname = 'MainProblem'):
        
        Res = SolverUnitCell(mesh, umat_name, props, nstatev, solver_type, corate_type, path_data, path_results, path_file, outputfile, outputdat_file, meshperio, Problemname = Problemname)
            
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
        
