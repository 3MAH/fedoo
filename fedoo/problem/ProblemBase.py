#baseclass
import scipy.sparse.linalg
import scipy.sparse as sparse
import numpy as np
from fedoo.problem.BoundaryCondition import UniqueBoundaryCondition
from fedoo.core.modelingspace  import ModelingSpace

try: 
    from pypardiso import spsolve
    USE_PYPARDISO = True
except: 
    USE_PYPARDISO = False


# =============================================================================
# ============================ Class ProblemBase ==============================
# =============================================================================

#Base class for all Problem objects
class ProblemBase:
    """
    Base class for defining Problems.
    
    All problem object is derived from the ProblemBase class.
    """

    __dic = {}
    __activeProblem = None #name of the current active problem

    def __init__(self, name = "", space = None):
        assert isinstance(name, str) , "An name must be a string" 
        self.__name = name
        self.__solver = ['direct']
        self._BoundaryConditions = [] #list containing boundary contidions associated to the problem        
        
        ProblemBase.__dic[self.__name] = self
        
        if space is None: 
            space = ModelingSpace.get_active()
        self.__space = space
        
        self.MakeActive()

    def name(self):
        return self.__name
    
    @property
    def space(self):
        return self.__space
    
    def MakeActive(self):
        ProblemBase.__activeProblem = self
    
    @staticmethod
    def SetActive(Problemname):
        ProblemBase.__activeProblem = ProblemBase.get_all()[Problemname]
    
    @staticmethod
    def get_active():
        return ProblemBase.__activeProblem
        
    
    def SetSolver(self,solver, tol=1e-5, precond=True):
        """
        Define the solver for the linear system resolution.
        The possible choice are : 
            'direct': direct solver based on the function scipy.sparse.linalg.spsolve
                      No option available
            'cg': conjugate gradient based on the function scipy.sparse.linalg.cg
                      use the tol arg to specify the convergence tolerance (default = 1e-5)
                      use precond = False to desactivate the diagonal matrix preconditionning (default precond=True)                                              
        """
        self.__solver = [solver.lower(), tol, precond]
        
    def __Solve(self, A, B):
        if self.__solver[0] == 'direct':
            if USE_PYPARDISO == True:
                return spsolve(A,B)
            else:
                return sparse.linalg.spsolve(A,B)            
        elif self.__solver[0] == 'cg':
            if self.__solver[2] == True: Mprecond = sparse.diags(1/A.diagonal(), 0)
            else: Mprecond = None
            res, info = sparse.linalg.cg(A,B, tol=self.__solver[1], M=Mprecond) 
            if info > 0: print('Warning: CG convergence to tolerance not achieved') 
            return res
        
    @staticmethod
    def get_all():
        return ProblemBase.__dic
       
    ### Functions related to boundary contidions
    def BoundaryCondition(self,BoundaryType,Var,Value,Index,Constant = None, timeEvolution=None, initialValue = None, name = "No name"):
        """
        Define some boundary conditions        

        Parameters
        ----------
        BoundaryType : str
            Type of boundary conditions : 'Dirichlet', 'Neumann' or 'MPC' for multipoint constraints.
        Var : str, list of str, or list of int
            variable name (str) or list of variable name or for MPC only, list of variable rank 
        Value : scalar or array or list of scalars or list of array
            Variable final value (Dirichlet) or force Value (Neumann) or list of factor (MPC)
            For Neumann and Dirichlet, if Var is a list of str, Value may be :
                (i) scalar if the same Value is applied for all Variable
                (ii) list of scalars, if the scalar values are different for all Variable (in this case the len of Value should be equal to the lenght of Var)
                (iii) list of arrays, if the scalar Value is potentially different for all variables and for all indexes. In this case, Value[num_var][i] should give the value of the num_var variable related to the node i.
        Index : list of int, str, list of list of int, list of str
            For FEM Problem with Neumann/Dirichlet BC: Nodes Index (list of int) 
            For FEM Problem with MPC: list Node Indexes (list of list of int) 
            For PGD Problem with Neumann/Dirichlet BC: SetOfname (type str) defining a set of Nodes of the reference mesh
            For PGD Problem with MPC: list of SetOfname (str)
        Constant : scalar, optional
            For MPC only, constant value on the equation
        timeEvolution : function
            Function that gives the temporal evolution of the BC Value (applyed as a factor to the specified BC). The function y=f(x) where x in [0,1] and y in [0,1]. For x, 0 denote the begining of the step and 1 the end.
        initialValue : float, array or None
            if None, the initialValue is keep to the current state.
            if scalar value: The initialValue is the same for all dof defined in BC
            if array: the len of the array should be = to the number of dof defined in the BC

            Default: None
        name : str, optional
            Define an name for the Boundary Conditions. Default is "No name". The same name may be used for several BC.

        Returns
        -------
        None.

        Remark  
        -------
        To define many MPC in one operation, use array where each line define a single MPC        
        """
        if isinstance(Var, str) and Var not in self.space.list_variables():
            #we assume that Var is a Vector
            try: 
                Var = [self.space.variable_name(var_rank) for var_rank in self.space.get_vector(Var)]
            except:
                raise NameError('Unknown variable name')
                
        if isinstance(Var, list) and BoundaryType != 'MPC':          
            if np.isscalar(Value):
                Value = [Value for var in Var] 
            for i,var in enumerate(Var):
                self._BoundaryConditions.append(UniqueBoundaryCondition(BoundaryType,var,Value[i],Index,Constant, timeEvolution, initialValue, name, self.space))                
        else:
            self._BoundaryConditions.append(UniqueBoundaryCondition(BoundaryType,Var,Value,Index,Constant, timeEvolution, initialValue, name, self.space))


    def GetBC(self, name =None):        
        """
        Return the list of Boundary Conditions
        if an name is specified (str value), return a list of BC whith the specified name
        """
        if name is None: return self._BoundaryConditions
        else: return [bc for bc in self._BoundaryConditions if bc.name == name]          
        

    def RemoveBC(self,name =None):
        """
        Remove all the BC which have the specified name. 
        If name = None (default) remove all boundary conditions
        """
        if name is None: self._BoundaryConditions = []
        else: self._BoundaryConditions = [bc for bc in self._BoundaryConditions if bc.name != name]          
    
    def PrintBC(self):        
        """
        Print all the boundary conditions under the form:
            ind_bc : name - BoundaryType
            ind_bc is the index of the bc in the list of boundary conditions (use GetBC to get the list)
            name is the str name of the BC ("No name") by default
            BoundaryType is the type of BC, ie "Dirichlet", "Neumann" or "MPC"
        """
        listid = [str(i) + ": " + bc.name + " - " + bc.BoundaryType for i,bc in enumerate(self._BoundaryConditions)]
        print("\n".join(listid))
    

    

    
                                

    
    ### Functions that may be defined depending on the type of problem
    def GetDisp(self,name='all'):
         raise NameError("The method 'GetDisp' is not defined for this kind of problem")

    def GetRot(self,name='all'):
         raise NameError("The method 'GetRot' is not defined for this kind of problem")

    def GetTemp(self):
         raise NameError("The method 'GetTemp' is not defined for this kind of problem")
    
    def update(self,):
        raise NameError("The method 'Update' is not defined for this kind of problem")    
        
    def ChangeAssembly(self,Assembling):
        raise NameError("The method 'ChangeAssembly' is not defined for this kind of problem")    
        
    def SetNewtonRaphsonErrorCriterion(self,ErrorCriterion, tol=5e-3, max_subiter = 5, err0 = None):
        """
        Set the error criterion used for the newton raphson algorithm

        Parameters
        ----------
        ErrorCriterion : str in ['Displacement', 'Force', 'Work']             
            Set the type of error criterion.             
        tol : float
            Tolerance of the NewtonRaphson algorithm (default = 5e-3)
        max_subiter: int
            Number of newton raphson iteration before returning an error
        err0 : scalar
            Reference value of error used for normalization
        """
        raise NameError("The method 'SetNewtonRaphsonErrorCriterion' is not defined for this kind of problem")    
        
    def NewtonRaphsonError(self):
        raise NameError("The method 'NewtonRaphsonError' is not defined for this kind of problem")        
        
    def NewTimeIncrement(self,LoadFactor):
        raise NameError("The method 'NewTimeIncrement' is not defined for this kind of problem")    
            
    def NewtonRaphsonIncr(self):                   
        raise NameError("The method 'NewtonRaphsonIncr' is not defined for this kind of problem")            
        
    def to_start(self):
        raise NameError("The method 'to_start' is not defined for this kind of problem")    
    
    def resetLoadFactor(self):             
        raise NameError("The method 'resetLoadFactor' is not defined for this kind of problem")    
    
    def reset(self):
        raise NameError("The method 'reset' is not defined for this kind of problem")    
        
    def GetElasticEnergy(self):
        raise NameError("The method 'GetElasticEnergy' is not defined for this kind of problem")    
    
    def GetNodalElasticEnergy(self):
        raise NameError("The method 'GetNodalElasticEnergy' is not defined for this kind of problem")    
        
    def get_ext_forces(self, name = 'all'):
        raise NameError("The method 'get_ext_forces' is not defined for this kind of problem")    

    def AddOutput(self, filename, assemblyname, output_list, output_type='Node', file_format ='vtk', position = 'top'):
        raise NameError("The method 'AddOutput' is not defined for this kind of problem")    
        
    def SaveResults(self, iterOutput=None):        
        raise NameError("The method 'SaveResults' is not defined for this kind of problem")
    
    def GetResults(self, assemb, output_list, output_type='Node', position = 1, res_format = None):
        raise NameError("The method 'GetResults' is not defined for this kind of problem")        

    #defined in the ProblemPGD classes
    def GetX(self): raise NameError("Method only defined for PGD Problems") 
    def GetXbc(self): raise NameError("Method only defined for PGD Problems") 
    def ComputeResidualNorm(self,err_0=None): raise NameError("Method only defined for PGD Problems") 
    def GetResidual(self): raise NameError("Method only defined for PGD Problems") 
    def updatePGD(self,termToChange, ddcalc='all'): raise NameError("Method only defined for PGD Problems") 
    def updateAlpha(self): raise NameError("Method only defined for PGD Problems") 
    def AddNewTerm(self,numberOfTerm = 1, value = None, variable = 'all'): raise NameError("Method only defined for PGD Problems") 
    
    @property
    def solver(self):
        return self.__solver[0]




# =============================================================================
# Functions that call methods of ProblemBase for the current active problem
# =============================================================================

def get_all():
    return ProblemBase.get_all()
def GetActive():
    return ProblemBase.get_active()
def SetActive(Problemname):
    ProblemBase.SetActive(Problemname)



def SetSolver(solver, tol=1e-5, precond=True):
    ProblemBase.get_active().SetSolver(solver,tol,precond)

## Functions related to boundary contidions
def BoundaryCondition(BoundaryType,Var,Value,Index,Constant = None, timeEvolution=None, initialValue = None, name = "No name", Problemname = None):
    if Problemname is None: problem = ProblemBase.get_active()
    else: problem = ProblemBase.get_all()[Problemname]
    problem.BoundaryCondition(BoundaryType,Var,Value,Index,Constant, timeEvolution, initialValue, name)

def GetBC(): return ProblemBase.get_active()._BoundaryConditions    
def RemoveBC(name =None): ProblemBase.get_active().RemoveBC(name)    
def PrintBC(): ProblemBase.get_active().PrintBC()    
 



### Functions that may be defined depending on the type of problem
def GetDisp(name='Disp'): return ProblemBase.get_active().GetDisp(name)
def GetRot(name='all'): return ProblemBase.get_active().GetRot(name)
def GetTemp(): return ProblemBase.get_active().GetTemp()
def update(**kargs): return ProblemBase.get_active().update(**kargs) 
def ChangeAssembly(Assembling): ProblemBase.get_active().ChangeAssembly(Assembling)
def SetNewtonRaphsonErrorCriterion(ErrorCriterion, tol=5e-3, max_subiter = 5, err0 = None): ProblemBase.get_active().SetNewtonRaphsonErrorCriterion(ErrorCriterion, tol, max_subiter, err0)
def NewtonRaphsonError(): return ProblemBase.get_active().NewtonRaphsonError()
def NewTimeIncrement(LoadFactor): ProblemBase.get_active().NewTimeIncrement(LoadFactor)
def NewtonRaphsonIncr(): ProblemBase.get_active().NewtonRaphsonIncr()
def to_start(): ProblemBase.get_active().to_start()
def reset(): ProblemBase.get_active().reset()
def resetLoadFactor(): ProblemBase.get_active().resetLoadFactor()
def NLSolve(**kargs): return ProblemBase.get_active().NLSolve(**kargs)  
def AddOutput(filename, assemblyname, output_list, output_type='Node', file_format ='vtk', position = 'top'):
    return ProblemBase.get_active().AddOutput(filename, assemblyname, output_list, output_type, file_format, position)
def SaveResults(iterOutput=None):        
    return ProblemBase.get_active().SaveResults(iterOutput)
def GetResults(assemb, output_list, output_type='Node', position = 1, res_format = None):
    return ProblemBase.get_active().GetResults(assemb, output_list, output_type, position, res_format)



#functions that should be define in the Problem and in the ProblemPGD classes
def SetA(A): ProblemBase.get_active().SetA(A)
def GetA(): return ProblemBase.get_active().GetA()
def GetB(): return ProblemBase.get_active().GetB()
def GetD(): return ProblemBase.get_active().GetD()
def GetMesh(): return ProblemBase.get_active().mesh
def SetD(D): ProblemBase.get_active().SetD(D)
def SetB(B): ProblemBase.get_active().SetB(B)
def Solve(**kargs): ProblemBase.get_active().Solve(**kargs)
def GetX(): return ProblemBase.get_active().GetX()
def ApplyBoundaryCondition(): ProblemBase.get_active().ApplyBoundaryCondition()
def GetDoFSolution(name='all'): return ProblemBase.get_active().GetDoFSolution(name)
def SetDoFSolution(name,value): ProblemBase.get_active().SetDoFSolution(name,value)
def SetInitialBCToCurrent(): ProblemBase.get_active().SetInitialBCToCurrent()
def get_global_vectorComponent(vector, name='all'): return ProblemBase.get_active()._get_global_vectorComponent(vector, name)

#functions only defined for Newmark problem 
def GetXdot():
    return ProblemBase.get_active().GetXdot()

def GetXdotdot():
    return ProblemBase.get_active().GetXdotdot()

  
def GetVelocity():
    return ProblemBase.get_active().GetVelocity()

def GetAcceleration():
    return ProblemBase.get_active().GetAcceleration()


def SetInitialDisplacement(name,value):
    """
    name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
    value is an array containing the initial displacement of each nodes
    """
    ProblemBase.get_active().SetInitialDisplacement(name,value)          

def SetInitialVelocity(name,value):
    """
    name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
    value is an array containing the initial velocity of each nodes        
    """
    ProblemBase.get_active().SetInitialVelocity(name,value)          
      

def SetInitialAcceleration(name,value):
    """
    name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
    value is an array containing the initial acceleration of each nodes        
    """
    ProblemBase.get_active().SetInitialAcceleration(name,value)           
    

def SetRayleighDamping(alpha, beta):        
    """
    Compute the damping matrix from the Rayleigh's model:
    [C] = alpha*[M] + beta*[K]         

    where [C] is the damping matrix, [M] is the mass matrix and [K] is the stiffness matrix        
    Note: The rayleigh model with alpha = 0 and beta = Viscosity/YoungModulus is almost equivalent to the multi-axial Kelvin-Voigt model
    
    Warning: the damping matrix is not automatically updated when mass and stiffness matrix are modified.        
    """
    ProblemBase.get_active().SetRayleighDamping(alpha, beta)

def initialize(t0 = 0.):
    ProblemBase.get_active().initialize(t0)           

def GetElasticEnergy():
    """
    returns : sum(0.5 * U.transposed * K * U)
    """
    return ProblemBase.get_active().GetElasticEnergy()
    
def GetNodalElasticEnergy():
    """
    returns : 0.5 * U.transposed * K * U
    """
    return ProblemBase.get_active().GetNodalElasticEnergy()

def get_ext_forces(name='all'):
    return ProblemBase.get_active().get_ext_forces(name)


def GetKineticEnergy():
    """
    returns : 0.5 * Udot.transposed * M * Udot
    """
    return ProblemBase.get_active().GetKineticEnergy()

def GetDampingPower():
    """
    returns : Udot.transposed * C * Udot
    The damping disspated energy can be approximated by:
            Edis = cumtrapz(DampingPower * TimeStep)
    """        
    return ProblemBase.get_active().GetDampingPower()

def updateStiffness(StiffnessAssembling):
    ProblemBase.get_active().updateStiffness(StiffnessAssembling)




#functions only used define in the ProblemPGD subclasses
def GetXbc(): return ProblemBase.get_active().GetXbc() 
def ComputeResidualNorm(err_0=None): return ProblemBase.get_active().ComputeResidualNorm(err_0)
def GetResidual(): return ProblemBase.get_active().GetResidual()
def updatePGD(termToChange, ddcalc='all'): return ProblemBase.get_active().updatePGD(termToChange, ddcalc)
def updateAlpha(): return ProblemBase.get_active().updateAlpha()
def AddNewTerm(numberOfTerm = 1, value = None, variable = 'all'): return ProblemBase.get_active().AddNewTerm(numberOfTerm, value, variable)

