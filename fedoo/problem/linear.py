import numpy as np
from fedoo.core.assembly import Assembly
from fedoo.core.problem import Problem
from fedoo.pgd.ProblemPGD import ProblemPGD
# from fedoo.problem.ProblemPGDtest   import ProblemPGDtest as ProblemPGD

#dynamical inheritance. The class is generated inside a function
def Linear(Assembling, name = "MainProblem"):
    if isinstance(Assembling,str):
        Assembling = Assembly.get_all()[Assembling]
                
    if hasattr(Assembling.mesh, 'GetListMesh'): libBase = ProblemPGD
    else: libBase = Problem
    
    class __Linear(libBase):
                
        def __init__(self, assembling, name):   
            assembling.initialize(self,0)                         
            A = assembling.get_global_matrix()
            B = 0             
            D = assembling.get_global_vector()     
            self.__Assembly = Assembling
                        
            libBase.__init__(self,A,B,D,assembling.mesh, name)
    
        def GetElasticEnergy(self): #only work for classical FEM
            """
            returns : sum (0.5 * U.transposed * K * U)
            """
    
            return sum( 0.5*self.get_dof_solution('all').transpose() * self.get_A() * self.get_dof_solution('all') )

        def GetNodalElasticEnergy(self):
            """
            returns : 0.5 * K * U . U
            """
    
            E = 0.5*self.get_dof_solution('all').transpose() * self.get_A() * self.get_dof_solution('all')

            E = np.reshape(E,(3,-1)).T

            return E
        
        def reset(self):
            self.__Assembly.reset()
            
            self.set_A(self.__Assembly.get_global_matrix()) #tangent stiffness 
            self.set_D(self.__Assembly.get_global_vector())            
            B = 0            
            self.apply_boundary_conditions()

        
        def update(self, dtime=1, compute = 'all'):   
            """
            Assemble the matrix including the following modification:
                - New initial Stress
                - New initial Displacement
                - Modification of the mesh
                - Change in constitutive law (internal variable)
            Update the problem with the new assembled global matrix and global vector
            """
#            self.__Assembly.update(self.get_D())
#            self.__Assembly.update(self)
#            self.set_A(Assembling.get_global_matrix())
#            self.set_D(Assembling.get_global_vector())
#            
            outValues = self.__Assembly.update(self, dtime, compute)  
            self.set_A(Assembling.get_global_matrix())
            self.set_D(Assembling.get_global_vector())
            return outValues 
        
        def solve(self, **kargs):
            #Solve and update weakform (compute stress for instance) without updating global matrix
            #to avoid update weakform, use updateWF = True
            updateWF = kargs.pop('updateWF', True)
            libBase.solve(self)
            if updateWF == True:
                self.update(compute = 'none')
            
        def get_Assembly(self):
            return self.__Assembly
        
        def ChangeAssembly(self,Assembling, update = True):
            """
            Modify the assembly associated to the problem and update the problem (see Assembly.update for more information)
            """
            if isinstance(Assembling,str):
                Assembling = Assembly.get_all()[Assembling]
                
            self.__Assembly = Assembling
            if update: self.update()        
        
        def get_disp(self,name='all'):    
            if name == 'all': name = 'Disp'
            return self.get_dof_solution(name)
        
        def get_rot(self,name='all'):    
            if name == 'all': name = 'Rot'
            return self.get_dof_solution(name)
        
    return __Linear(Assembling, name)


