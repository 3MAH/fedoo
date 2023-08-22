import numpy as np
from fedoo.core.assembly import Assembly
from fedoo.core.problem import Problem
    
class _NonLinearBase():
    
    def __init__(self, assembly, name = "MainProblem"):   
        
        if isinstance(assembly,str):
            assembly = Assembly.get_all()[assembly]
                               
        #A = assembling.current.get_global_matrix() #tangent stiffness matrix
        A = 0 #tangent stiffness matrix - will be initialized only when required
        B = 0             
        #D = assembling.get_global_vector() #initial stress vector
        D = 0 #initial stress vector #will be initialized later
        self.print_info = 1 #print info of NR convergence during solve
        self._U = 0 #displacement at the end of the previous converged increment
        self._dU = 0 #displacement increment
        
        self._err0 = None #initial error for NR error estimation
    
        self.nr_parameters = {'err0': None, #default error for NR error estimation
                              'criterion': 'Displacement', #
                              'tol': 5e-3,
                              'max_subiter': 5
                              }
        """
        Parameters to set the newton raphson algorithm:
            * 'err0': The reference error. Default is None (automatically computed)
            * 'criterion': Type of convergence test in ['Displacement', 'Force', 'Work']. Default is 'Displacement'.
            * 'tol': Error tolerance for convergence. Default is 5e-3.
            * 'max_subiter': Number of nr iteration before returning a convergence error. Default is 5. 
        """


        self.__assembly = assembly
        super().__init__(A,B,D,assembly.mesh, name, assembly.space)        
        self.t0 = 0 ; self.tmax = 1
        self.__iter = 0
        self.__compteurOutput = 0
        
        self.interval_output = None #save results every self.interval_output iter or time step if self.__saveOutputAtExactTime = True
        self.__saveOutputAtExactTime = True
        self.err_num= 1e-8 #numerical error
    
    #Return the displacement components
    def get_disp(self,name='Disp'):    
        if self._dU is 0: return self._get_vect_component(self._U, name)
        return self._get_vect_component(self._U + self._dU, name)    
    
    #Return the rotation components
    def get_rot(self,name='Rot'):    
        if self._dU is 0: return self._get_vect_component(self._U, name)
        return self._get_vect_component(self._U + self._dU, name)    

    #Return the Temperature
    def get_temp(self):    
        if self._dU is 0: return self._get_vect_component(self._U, 'Temp')
        return self._get_vect_component(self._U + self._dU, 'Temp')    
    
    #Return all the dof for every variable under a vector form
    def get_dof_solution(self,name='all'):
        if self._dU is 0: return self._get_vect_component(self._U, name)
        return self._get_vect_component(self._U + self._dU, name)        
    
    # def get_ext_forces(self, name = 'all'):
    #     return self._get_vect_component(-self.get_D(), name)        
    
    def updateA(self):
        #dt not used for static problem
        self.set_A(self.__assembly.current.get_global_matrix())
    
    def updateD(self,start=False):            
        #dt and start not used for static problem
        self.set_D(self.__assembly.current.get_global_vector()) 
    
    def initialize(self):   
        """
        """
        self.__assembly.initialize(self)
        # self.set_A(self.__assembly.current.get_global_matrix())
        # self.set_D(self.__assembly.current.get_global_vector())
        
    
    def elastic_prediction(self):
        #update the boundary conditions with the time variation
        time_end = self.time + self.dtime
        t_fact    = (time_end-self.t0)/(self.tmax-self.t0) #adimensional time            
        t_fact_old = (self.time-self.t0)/(self.tmax-self.t0)

        self.apply_boundary_conditions(t_fact, t_fact_old)
        
        #build and solve the linearized system with elastic rigidty matrix           
        self.updateA() #should be the elastic rigidity matrix
        self.updateD(start = True) #not modified in principle if dt is not modified, except the very first iteration. May be optimized by testing the change of dt

        self.solve() 


        #set the increment Dirichlet boundray conditions to 0 (i.e. will not change during the NR interations)            
        try:
            self._Xbc *= 0 
        except:
            self._ProblemPGD__Xbc = 0

        #update displacement increment        
        self._dU += self.get_X()
        
        
    def set_start(self,save_results=False):
        #dt not used for static problem
        if self._dU is not 0: 
            self._U += self._dU
            self._dU = 0                
            self._err0 = self.nr_parameters['err0'] #initial error for NR error estimation
            self.__assembly.set_start(self)    
            
            #Save results            
            if save_results: 
                self.save_results(self.__compteurOutput)                              
                self.__compteurOutput += 1     
        else:
            self._err0 = self.nr_parameters['err0'] #initial error for NR error estimation
            self.__assembly.set_start(self)    
        
        
    def to_start(self):   
        self._dU = 0                       
        self._err0 = self.nr_parameters['err0'] #initial error for NR error estimation
        self.__assembly.to_start(self)
        
                          
    def NewtonRaphsonIncrement(self):                                      
        #solve and update total displacement. A and D should up to date
        self.solve()
        self._dU += self.get_X()
        # print(self._dU)
    
    def update(self, compute = 'all', updateWeakForm = True):   
        """
        Assemble the matrix including the following modification:
            - New initial Stress
            - New initial Displacement
            - Modification of the mesh
            - Change in constitutive law (internal variable)
        Don't Update the problem with the new assembled global matrix and global vector -> use UpdateA and UpdateD method for this purpose
        """
        if updateWeakForm == True:
            self.__assembly.update(self, compute)
        else: 
            self.__assembly.current.assemble_global_mat(compute)
            
        if self.bc._update_during_inc:
            self.update_boundary_conditions()

    def reset(self):
        self.__assembly.reset()
        
        self.set_A(0) #tangent stiffness 
        self.set_D(0)                 
        # self.set_A(self.__assembly.current.get_global_matrix()) #tangent stiffness 
        # self.set_D(self.__assembly.current.get_global_vector())            

        B = 0
        self._U = 0
        self._dU = 0
                    
        self._err0 = self.nr_parameters['err0'] #initial error for NR error estimation   
        self.t0 = 0 ; self.tmax = 1
        self.__iter = 0  
        self.apply_boundary_conditions() #perhaps not usefull here as the BC will be applied in the NewTimeIncrement method ?

        
    def change_assembly(self,assembling, update = True):
        """
        Modify the assembly associated to the problem and update the problem (see Assembly.update for more information)
        """
        if isinstance(assembling,str):
            assembling = Assembly[assembling]
            
        self.__assembly = assembling
        if update: self.update()
        
    def NewtonRaphsonError(self):
        """
        Compute the error of the Newton-Raphson algorithm
        For Force and Work error criterion, the problem must be updated
        (Update method).
        """
        dof_free = self._dof_free
        if self._err0 is None: # if self._err0 is None -> initialize the value of err0  
            if self.nr_parameters['criterion'] == 'Displacement':                     
                self._err0 = np.max(np.abs((self._U + self._dU)[dof_free])) #Displacement criterion
                if self._err0 == 0: 
                    self._err0 = 1                
                    return 1
                return np.max(np.abs(self.get_X()[dof_free]))/self._err0
            else:
                self._err0 = 1
                self._err0 = self.NewtonRaphsonError() 
                return 1                
        else: 
            if self.nr_parameters['criterion'] == 'Displacement': 
                return np.max(np.abs(self.get_X()[dof_free]))/self._err0  #Displacement criterion
            elif self.nr_parameters['criterion'] == 'Force': #Force criterion              
                if self.get_D() is 0: return np.max(np.abs(self.get_B()[dof_free]))/self._err0 
                else: return np.max(np.abs(self.get_B()[dof_free]+self.get_D()[dof_free]))/self._err0                     
            else: #self.nr_parameters['criterion'] == 'Work': #work criterion
                if self.get_D() is 0: return np.max(np.abs(self.get_X()[dof_free]) * np.abs(self.get_B()[dof_free]))/self._err0 
                else: return np.max(np.abs(self.get_X()[dof_free]) * np.abs(self.get_B()[dof_free]+self.get_D()[dof_free]))/self._err0 
   
    def set_nr_criterion(self, criterion = 'Displacement', **kargs):
        """
        Define the convergence criterion of the newton raphson algorith. 
        For a problem pb, the newton raphson parameters can also be directly set in the 
        pb.nr_parameters dict.
        
        Parameter: 
            * criterion: Type of convergence test. Str in ['Displacement', 'Force', 'Work'] (default = "Displacement"). 
        
        Optional parameters that can be set as kargs:
            * err0: The reference error. Float or None. (if None, err0 is automatically computed)
            * tol: Error tolerance for convergence. Float.
            * max_subiter: Number of nr iteration before returning a convergence error. Int.
        """
        if criterion not in ["Displacement", "Force", "Work"]:
            raise NameError('criterion must be set to "Displacement", "Force" or "Work"')
        self.nr_parameters['criterion'] = criterion
        
        for key in kargs:
            if key not in ['err0', 'tol', 'max_subiter']:
                raise NameError("Newton Raphson parameters should be in ['err0', 'tol', 'max_subiter']")                

            self.nr_parameters[key] = kargs[key]
           
    
    # def GetElasticEnergy(self): #only work for classical FEM
    #     """
    #     returns : sum (0.5 * U.transposed * K * U)
    #     """

    #     return sum( 0.5*self.get_X().transpose() * self.get_A() * self.get_X() )

    # def GetNodalElasticEnergy(self):
    #     """
    #     returns : 0.5 * K * U . U
    #     """

    #     E = 0.5*self.get_X().transpose() * self.get_A() * self.get_X()

    #     E = np.reshape(E,(3,-1)).T

    #     return E                   
    
    def solve_time_increment(self, max_subiter = None, ToleranceNR = None): 
        if max_subiter is None: max_subiter = self.nr_parameters['max_subiter']
        if ToleranceNR is None: ToleranceNR = self.nr_parameters['tol'] 
       
        self.elastic_prediction()
        for subiter in range(max_subiter): #newton-raphson iterations
            if subiter > 0:
                pass
            #update Stress and initial displacement and Update stiffness matrix
            self.update(compute = 'vector') #update the out of balance force vector
            self.updateD() #required to compute the NR error

            #Check convergence     
            normRes = self.NewtonRaphsonError()    

            if self.print_info > 1:
                print('     Subiter {} - Time: {:.5f} - Err: {:.5f}'.format(subiter, self.time+self.dtime, normRes))

            if normRes < ToleranceNR: #convergence of the NR algorithm                    
                #Initialize the next increment                    
                # self.NewTimeIncrement()                                           
                return 1, subiter, normRes
            
            #--------------- Solve --------------------------------------------------------        
            # self.__Assembly.current.assemble_global_mat(compute = 'matrix')
            # self.set_A(self.__Assembly.current.get_global_matrix())
            self.update(compute = 'matrix', updateWeakForm = False) #assemble the tangeant matrix
            self.updateA()

            self.NewtonRaphsonIncrement()
            
        return 0, subiter, normRes


    def nlsolve(self, **kargs):              
        #parameters
        self.print_info = kargs.get('print_info',self.print_info)
        max_subiter = kargs.get('max_subiter',self.nr_parameters['max_subiter'])
        ToleranceNR = kargs.get('ToleranceNR',self.nr_parameters['tol'])
        self.t0 = kargs.get('t0',self.t0) #time at the start of the time step
        self.tmax = kargs.get('tmax',self.tmax) #time at the end of the time step
        dt = kargs.get('dt',0.1) #initial time step
        dt_min = kargs.get('dt_min',1e-6) #min time step
        
        self.__saveOutputAtExactTime = kargs.get('saveOutputAtExactTime',self.__saveOutputAtExactTime)
        interval_output = kargs.get('interval_output',self.interval_output) # time step for output if saveOutputAtExactTime == 'True' (default) or  number of iter increments between 2 output 
        update_dt = kargs.get('update_dt',True)
        
        if interval_output is None:
            if self.__saveOutputAtExactTime: interval_output = dt
            else: interval_output = 1
        
        if self.__saveOutputAtExactTime: next_time = self.t0 + interval_output
        else: next_time = self.tmax #next_time is the next exact time where the algorithm have to stop for output purpose
        
        self.init_bc_start_value()            
                    
        self.time = self.t0  #time at the begining of the iteration                          
        
        if self._U is 0:#Initialize only if 1st step
            self.initialize()
        
        restart = False #bool to know if the iteration is another attempt
             
        while self.time < self.tmax - self.err_num:                         
    
            save_results = (self.time != self.t0) and \
                ((self.time == next_time) or (self.__saveOutputAtExactTime == False and self.__iter%interval_output == 0))

            #update next_time                
            if self.time == next_time: 
                # self.__saveOutputAtExactTime should be True 
                next_time = next_time + interval_output
                if next_time > self.tmax - self.err_num: next_time = self.tmax
                
            if self.time+dt > next_time - self.err_num: #if dt is too high, it is reduced to reach next_time
                self.dtime = next_time-self.time 
            else: 
                self.dtime = dt
            
            if restart:
                restart = False
            else:                
                self.set_start(save_results)   
            
                
            #self.solve_time_increment = Newton Raphson loop
            convergence, nbNRiter, normRes = self.solve_time_increment(max_subiter, ToleranceNR)
            
            if (convergence) :
                self.time = self.time + self.dtime #update time value 
                self.__iter += 1
                
                if self.print_info > 0:
                    print('Iter {} - Time: {:.5f} - dt {:.5f} - NR iter: {} - Err: {:.5f}'.format(self.__iter, self.time, dt, nbNRiter, normRes))

                #Check if dt can be increased
                if update_dt and nbNRiter < 2 and dt == self.dtime: 
                    dt *= 1.25
                    # print('Increase the time increment to {:.5f}'.format(dt))
                
            else:
                if update_dt:              
                    dt *= 0.25                                     
                    print('NR failed to converge (err: {:.5f}) - reduce the time increment to {:.5f}'.format(normRes, dt ))
                    
                    if dt < dt_min: 
                        raise NameError('Current time step is inferior to the specified minimal time step (dt_min)')   
                    
                    #reset internal variables, update Stress, initial displacement and assemble global matrix at previous time                                      
                    self.to_start()              
                    restart = True
                    continue                    
                else: 
                    raise NameError('Newton Raphson iteration has not converged (err: {:.5f})- Reduce the time step or use update_dt = True'.format(normRes))   
        
        self.set_start(True)
        
    
    @property
    def assembly(self):
        return self.__assembly
                                                                            

class NonLinear(_NonLinearBase, Problem): 
    pass


