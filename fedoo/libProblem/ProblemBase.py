#baseclass
import scipy.sparse.linalg
import scipy.sparse as sparse
import numpy as np
from fedoo.libProblem.BoundaryCondition import UniqueBoundaryCondition
from fedoo.libUtil.ModelingSpace  import ModelingSpace

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
    __activeProblem = None #ID of the current active problem

    def __init__(self, ID = "", space = None):
        assert isinstance(ID, str) , "An ID must be a string" 
        self.__ID = ID
        self.__solver = ['direct']
        self._BoundaryConditions = [] #list containing boundary contidions associated to the problem        
        
        ProblemBase.__dic[self.__ID] = self
        
        if space is None: 
            space = ModelingSpace.GetActive()
        self.__space = space
        
        self.MakeActive()

    def GetID(self):
        return self.__ID
    
    @property
    def space(self):
        return self.__space
    
    def MakeActive(self):
        ProblemBase.__activeProblem = self
    
    @staticmethod
    def SetActive(ProblemID):
        ProblemBase.__activeProblem = ProblemBase.GetAll()[ProblemID]
    
    @staticmethod
    def GetActive():
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
    def GetAll():
        return ProblemBase.__dic
       
    ### Functions related to boundary contidions
    def BoundaryCondition(self,BoundaryType,Var,Value,Index,Constant = None, timeEvolution=None, initialValue = None, ID = "No ID"):
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
            For PGD Problem with Neumann/Dirichlet BC: SetOfID (type str) defining a set of Nodes of the reference mesh
            For PGD Problem with MPC: list of SetOfID (str)
        Constant : scalar, optional
            For MPC only, constant value on the equation
        timeEvolution : function
            Function that gives the temporal evolution of the BC Value (applyed as a factor to the specified BC). The function y=f(x) where x in [0,1] and y in [0,1]. For x, 0 denote the begining of the step and 1 the end.
        initialValue : float, array or None
            if None, the initialValue is keep to the current state.
            if scalar value: The initialValue is the same for all dof defined in BC
            if array: the len of the array should be = to the number of dof defined in the BC

            Default: None
        ID : str, optional
            Define an ID for the Boundary Conditions. Default is "No ID". The same ID may be used for several BC.

        Returns
        -------
        None.

        Remark  
        -------
        To define many MPC in one operation, use array where each line define a single MPC        
        """
        if isinstance(Var, str) and Var not in self.space.list_variable():
            #we assume that Var is a Vector
            try: 
                Var = [self.space.variable_name(var_rank) for var_rank in self.space.get_vector(Var)]
            except:
                raise NameError('Unknown variable name')
                
        if isinstance(Var, list) and BoundaryType != 'MPC':          
            if np.isscalar(Value):
                Value = [Value for var in Var] 
            for i,var in enumerate(Var):
                self._BoundaryConditions.append(UniqueBoundaryCondition(BoundaryType,var,Value[i],Index,Constant, timeEvolution, initialValue, ID, self.space))                
        else:
            self._BoundaryConditions.append(UniqueBoundaryCondition(BoundaryType,Var,Value,Index,Constant, timeEvolution, initialValue, ID, self.space))


    def GetBC(self):        
        """Return the list of Boundary Conditions"""
        return self._BoundaryConditions

    def RemoveBC(self,ID=None):
        """
        Remove all the BC which have the specified ID. 
        If ID = None (default) remove all boundary conditions
        """
        if ID is None: self._BoundaryConditions = []
        else: self._BoundaryConditions = [bc for bc in self._BoundaryConditions if bc.GetID() != ID]          
    
    def PrintBC(self):        
        """
        Print all the boundary conditions under the form:
            ind_bc : ID - BoundaryType
            ind_bc is the index of the bc in the list of boundary conditions (use GetBC to get the list)
            ID is the str ID of the BC ("No ID") by default
            BoundaryType is the type of BC, ie "Dirichlet", "Neumann" or "MPC"
        """
        listid = [str(i) + ": " + bc.GetID() + " - " + bc.BoundaryType() for i,bc in enumerate(self._BoundaryConditions)]
        print("\n".join(listid))
    

    

    
                                

    
    ### Functions that may be defined depending on the type of problem
    def GetDisp(self,name='all'):
         raise NameError("The method 'GetDisp' is not defined for this kind of problem")

    def GetRot(self,name='all'):
         raise NameError("The method 'GetRot' is not defined for this kind of problem")

    def GetTemp(self):
         raise NameError("The method 'GetTemp' is not defined for this kind of problem")
    
    def Update(self,):
        raise NameError("The method 'Update' is not defined for this kind of problem")    
        
    def ChangeAssembly(self,Assembling):
        raise NameError("The method 'ChangeAssembly' is not defined for this kind of problem")    
        
    def SetNewtonRaphsonErrorCriterion(self,ErrorCriterion):
        raise NameError("The method 'SetNewtonRaphsonErrorCriterion' is not defined for this kind of problem")    
        
    def NewtonRaphsonError(self):
        raise NameError("The method 'NewtonRaphsonError' is not defined for this kind of problem")        
        
    def NewTimeIncrement(self,LoadFactor):
        raise NameError("The method 'NewTimeIncrement' is not defined for this kind of problem")    
            
    def NewtonRaphsonIncr(self):                   
        raise NameError("The method 'NewtonRaphsonIncr' is not defined for this kind of problem")            
        
    def ResetTimeIncrement(self):
        raise NameError("The method 'ResetTimeIncrement' is not defined for this kind of problem")    
    
    def ResetLoadFactor(self):             
        raise NameError("The method 'ResetLoadFactor' is not defined for this kind of problem")    
    
    def Reset(self):
        raise NameError("The method 'Reset' is not defined for this kind of problem")    
        
    def GetElasticEnergy(self):
        raise NameError("The method 'GetElasticEnergy' is not defined for this kind of problem")    
    
    def GetNodalElasticEnergy(self):
        raise NameError("The method 'GetNodalElasticEnergy' is not defined for this kind of problem")    
        
    def GetExternalForces(self, name = 'all'):
        raise NameError("The method 'GetExternalForces' is not defined for this kind of problem")    

    def AddOutput(self, filename, assemblyID, output_list, output_type='Node', file_format ='vtk', position = 'top'):
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
    def UpdatePGD(self,termToChange, ddcalc='all'): raise NameError("Method only defined for PGD Problems") 
    def UpdateAlpha(self): raise NameError("Method only defined for PGD Problems") 
    def AddNewTerm(self,numberOfTerm = 1, value = None, variable = 'all'): raise NameError("Method only defined for PGD Problems") 




# =============================================================================
# Functions that call methods of ProblemBase for the current active problem
# =============================================================================

def GetAll():
    return ProblemBase.GetAll()
def GetActive():
    return ProblemBase.GetActive()
def SetActive(ProblemID):
    ProblemBase.SetActive(ProblemID)



def SetSolver(solver, tol=1e-5, precond=True):
    ProblemBase.GetActive().SetSolver(solver,tol,precond)

## Functions related to boundary contidions
def BoundaryCondition(BoundaryType,Var,Value,Index,Constant = None, timeEvolution=None, initialValue = None, ID = "No ID", ProblemID = None):
    if ProblemID is None: problem = ProblemBase.GetActive()
    else: problem = ProblemBase.GetAll()[ProblemID]
    problem.BoundaryCondition(BoundaryType,Var,Value,Index,Constant, timeEvolution, initialValue, ID)

def GetBC(): return ProblemBase.GetActive()._BoundaryConditions    
def RemoveBC(ID=None): ProblemBase.GetActive().RemoveBC(ID)    
def PrintBC(): ProblemBase.GetActive().PrintBC()    
 



### Functions that may be defined depending on the type of problem
def GetDisp(name='Disp'): return ProblemBase.GetActive().GetDisp(name)
def GetRot(name='all'): return ProblemBase.GetActive().GetRot(name)
def GetTemp(): return ProblemBase.GetActive().GetTemp()
def Update(**kargs): return ProblemBase.GetActive().Update(**kargs) 
def ChangeAssembly(Assembling): ProblemBase.GetActive().ChangeAssembly(Assembling)
def SetNewtonRaphsonErrorCriterion(ErrorCriterion): ProblemBase.GetActive().SetNewtonRaphsonErrorCriterion(ErrorCriterion)
def NewtonRaphsonError(): return ProblemBase.GetActive().NewtonRaphsonError()
def NewTimeIncrement(LoadFactor): ProblemBase.GetActive().NewTimeIncrement(LoadFactor)
def NewtonRaphsonIncr(): ProblemBase.GetActive().NewtonRaphsonIncr()
def ResetTimeIncrement(): ProblemBase.GetActive().ResetTimeIncrement()
def Reset(): ProblemBase.GetActive().Reset()
def ResetLoadFactor(): ProblemBase.GetActive().ResetLoadFactor()
def NLSolve(**kargs): return ProblemBase.GetActive().NLSolve(**kargs)  
def AddOutput(filename, assemblyID, output_list, output_type='Node', file_format ='vtk', position = 'top'):
    return ProblemBase.GetActive().AddOutput(filename, assemblyID, output_list, output_type, file_format, position)
def SaveResults(iterOutput=None):        
    return ProblemBase.GetActive().SaveResults(iterOutput)
def GetResults(assemb, output_list, output_type='Node', position = 1, res_format = None):
    return ProblemBase.GetActive().GetResults(assemb, output_list, output_type, position, res_format)



#functions that should be define in the Problem and in the ProblemPGD classes
def SetA(A): ProblemBase.GetActive().SetA(A)
def GetA(): return ProblemBase.GetActive().GetA()
def GetB(): return ProblemBase.GetActive().GetB()
def GetD(): return ProblemBase.GetActive().GetD()
def GetMesh(): return ProblemBase.GetActive().GetMesh()
def SetD(D): ProblemBase.GetActive().SetD(D)
def SetB(B): ProblemBase.GetActive().SetB(B)
def Solve(**kargs): ProblemBase.GetActive().Solve(**kargs)
def GetX(): return ProblemBase.GetActive().GetX()
def ApplyBoundaryCondition(): ProblemBase.GetActive().ApplyBoundaryCondition()
def GetDoFSolution(name='all'): return ProblemBase.GetActive().GetDoFSolution(name)
def SetDoFSolution(name,value): ProblemBase.GetActive().SetDoFSolution(name,value)
def SetInitialBCToCurrent(): ProblemBase.GetActive().SetInitialBCToCurrent()
def GetVectorComponent(vector, name='all'): return ProblemBase.GetActive()._GetVectorComponent(vector, name)

#functions only defined for Newmark problem 
def GetXdot():
    return ProblemBase.GetActive().GetXdot()

def GetXdotdot():
    return ProblemBase.GetActive().GetXdotdot()

  
def GetVelocity():
    return ProblemBase.GetActive().GetVelocity()

def GetAcceleration():
    return ProblemBase.GetActive().GetAcceleration()


def SetInitialDisplacement(name,value):
    """
    name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
    value is an array containing the initial displacement of each nodes
    """
    ProblemBase.GetActive().SetInitialDisplacement(name,value)          

def SetInitialVelocity(name,value):
    """
    name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
    value is an array containing the initial velocity of each nodes        
    """
    ProblemBase.GetActive().SetInitialVelocity(name,value)          
      

def SetInitialAcceleration(name,value):
    """
    name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
    value is an array containing the initial acceleration of each nodes        
    """
    ProblemBase.GetActive().SetInitialAcceleration(name,value)           
    

def SetRayleighDamping(alpha, beta):        
    """
    Compute the damping matrix from the Rayleigh's model:
    [C] = alpha*[M] + beta*[K]         

    where [C] is the damping matrix, [M] is the mass matrix and [K] is the stiffness matrix        
    Note: The rayleigh model with alpha = 0 and beta = Viscosity/YoungModulus is almost equivalent to the multi-axial Kelvin-Voigt model
    
    Warning: the damping matrix is not automatically updated when mass and stiffness matrix are modified.        
    """
    ProblemBase.GetActive().SetRayleighDamping(alpha, beta)

def Initialize():
    ProblemBase.GetActive().Initialize()           

def GetElasticEnergy():
    """
    returns : sum(0.5 * U.transposed * K * U)
    """
    return ProblemBase.GetActive().GetElasticEnergy()
    
def GetNodalElasticEnergy():
    """
    returns : 0.5 * U.transposed * K * U
    """
    return ProblemBase.GetActive().GetNodalElasticEnergy()

def GetExternalForces(name='all'):
    return ProblemBase.GetActive().GetExternalForces(name)


def GetKineticEnergy():
    """
    returns : 0.5 * Udot.transposed * M * Udot
    """
    return ProblemBase.GetActive().GetKineticEnergy()

def GetDampingPower():
    """
    returns : Udot.transposed * C * Udot
    The damping disspated energy can be approximated by:
            Edis = cumtrapz(DampingPower * TimeStep)
    """        
    return ProblemBase.GetActive().GetDampingPower()

def UpdateStiffness(StiffnessAssembling):
    ProblemBase.GetActive().UpdateStiffness(StiffnessAssembling)




#functions only used define in the ProblemPGD subclasses
def GetXbc(): return ProblemBase.GetActive().GetXbc() 
def ComputeResidualNorm(err_0=None): return ProblemBase.GetActive().ComputeResidualNorm(err_0)
def GetResidual(): return ProblemBase.GetActive().GetResidual()
def UpdatePGD(termToChange, ddcalc='all'): return ProblemBase.GetActive().UpdatePGD(termToChange, ddcalc)
def UpdateAlpha(): return ProblemBase.GetActive().UpdateAlpha()
def AddNewTerm(numberOfTerm = 1, value = None, variable = 'all'): return ProblemBase.GetActive().AddNewTerm(numberOfTerm, value, variable)

