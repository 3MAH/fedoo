#baseclass
import scipy.sparse.linalg
import scipy.sparse as sparse
import numpy as np

class ProblemBase:

    __dic = {}

    def __init__(self, ID = ""):
        assert isinstance(ID, str) , "An ID must be a string" 
        self.__ID = ID
        self.__solver = ['direct']

        ProblemBase.__dic[self.__ID] = self

    def GetID(self):
        return self.__ID
    
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
            return sparse.linalg.spsolve(A,B)
        elif self.__solver[0] == 'cg':
#            print(np.where(A.diagonal()==0))
            if self.__solver[2] == True: Mprecond = sparse.diags(1/A.diagonal(), 0)
            else: Mprecond = None
            res, info = sparse.linalg.cg(A,B, tol=self.__solver[1], M=Mprecond) 
            if info > 0: print('Warning: CG convergence to tolerance not achieved') 
            return res
        
    @staticmethod
    def GetAll():
        return ProblemBase.__dic
       
        
    ### Functions that may be defined depending on the type of problem
    def GetDisp(self,name='all'):
         raise NameError("The method 'GetDisp' is not defined for this kind of problem")
    
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

    #defined in the ProblemPGD classes
    def GetX(self): raise NameError("Method only defined for PGD Problems") 
    def GetXbc(self): raise NameError("Method only defined for PGD Problems") 
    def ComputeResidualNorm(self,err_0=None): raise NameError("Method only defined for PGD Problems") 
    def GetResidual(self): raise NameError("Method only defined for PGD Problems") 
    def UpdatePGD(self,termToChange, ddcalc='all'): raise NameError("Method only defined for PGD Problems") 
    def UpdateAlpha(self): raise NameError("Method only defined for PGD Problems") 
    def AddNewTerm(self,numberOfTerm = 1, value = None, variable = 'all'): raise NameError("Method only defined for PGD Problems") 


def GetAll():
    return ProblemBase.GetAll()

def SetSolver(solver, tol=1e-5, precond=True):
    ProblemBase.GetAll()['MainProblem'].SetSolver(solver,tol,precond)

### Functions that may be defined depending on the type of problem
def GetDisp(name='all'): return ProblemBase.GetAll()['MainProblem'].GetDisp(name)
def Update(): return ProblemBase.GetAll()['MainProblem'].Update() 
def ChangeAssembly(Assembling): ProblemBase.GetAll()['MainProblem'].ChangeAssembly(Assembling)
def SetNewtonRaphsonErrorCriterion(ErrorCriterion): ProblemBase.GetAll()['MainProblem'].SetNewtonRaphsonErrorCriterion(ErrorCriterion)
def NewtonRaphsonError(): return ProblemBase.GetAll()['MainProblem'].NewtonRaphsonError()
def NewTimeIncrement(LoadFactor): ProblemBase.GetAll()['MainProblem'].NewTimeIncrement(LoadFactor)
def NewtonRaphsonIncr(): ProblemBase.GetAll()['MainProblem'].NewtonRaphsonIncr()
def ResetTimeIncrement(): ProblemBase.GetAll()['MainProblem'].ResetTimeIncrement()
def Reset(): ProblemBase.GetAll()['MainProblem'].Reset()
def ResetLoadFactor(): ProblemBase.GetAll()['MainProblem'].ResetLoadFactor()
def NLSolve(**kargs): return ProblemBase.GetAll()['MainProblem'].NLSolve(**kargs)  
def GetElasticEnergy(): return ProblemBase.GetAll()['MainProblem'].GetElasticEnergy()
def GetNodalElasticEnergy(): return ProblemBase.GetAll()['MainProblem'].GetNodalElasticEnergy()

#functions that should be define in the Problem or in the ProblemPGD classes
def SetA(A): ProblemBase.GetAll()["MainProblem"].SetA(A)
def GetA(): return ProblemBase.GetAll()["MainProblem"].GetA()
def GetB(): return ProblemBase.GetAll()["MainProblem"].GetB()
def GetD(): return ProblemBase.GetAll()["MainProblem"].GetD()
def GetMesh(): return ProblemBase.GetAll()["MainProblem"].GetMesh()
def SetD(D): ProblemBase.GetAll()["MainProblem"].SetD(D)
def SetB(B): ProblemBase.GetAll()["MainProblem"].SetB(B)
def Solve(): ProblemBase.GetAll()["MainProblem"].Solve()
def ApplyBoundaryCondition(): ProblemBase.GetAll()["MainProblem"].ApplyBoundaryCondition()
def GetDoFSolution(name): return ProblemBase.GetAll()["MainProblem"].GetDoFSolution(name)
def SetDoFSolution(name,value): ProblemBase.GetAll()["MainProblem"].SetDoFSolution(name,value)

#functions only defined for Newmark problem
def GetX():
    return ProblemBase.GetAll()['MainProblem'].GetX()
    
def GetXdot():
    return ProblemBase.GetAll()['MainProblem'].GetXdot()

def GetXdotdot():
    return ProblemBase.GetAll()['MainProblem'].GetXdotdot()

  
def GetVelocity():
    return ProblemBase.GetAll()['MainProblem'].GetVelocity()

def GetAcceleration():
    return ProblemBase.GetAll()['MainProblem'].GetAcceleration()


def SetInitialDisplacement(name,value):
    """
    name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
    value is an array containing the initial displacement of each nodes
    """
    ProblemBase.GetAll()['MainProblem'].SetInitialDisplacement(name,value)          

def SetInitialVelocity(name,value):
    """
    name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
    value is an array containing the initial velocity of each nodes        
    """
    ProblemBase.GetAll()['MainProblem'].SetInitialVelocity(name,value)          
      

def SetInitialAcceleration(name,value):
    """
    name is the name of the associated variable (generaly 'DispX', 'DispY' or 'DispZ')    
    value is an array containing the initial acceleration of each nodes        
    """
    ProblemBase.GetAll()['MainProblem'].SetInitialAcceleration(name,value)           
    

def SetRayleighDamping(alpha, beta):        
    """
    Compute the damping matrix from the Rayleigh's model:
    [C] = alpha*[M] + beta*[K]         

    where [C] is the damping matrix, [M] is the mass matrix and [K] is the stiffness matrix        
    Note: The rayleigh model with alpha = 0 and beta = Viscosity/YoungModulus is almost equivalent to the multi-axial Kelvin-Voigt model
    
    Warning: the damping matrix is not automatically updated when mass and stiffness matrix are modified.        
    """
    ProblemBase.GetAll()['MainProblem'].SetRayleighDamping(alpha, beta)

def Initialize():
    ProblemBase.GetAll()['MainProblem'].Initialize()           

def GetElasticEnergy():
    """
    returns : sum(0.5 * U.transposed * K * U)
    """
    return ProblemBase.GetAll()['MainProblem'].GetElasticEnergy()
    
def GetNodalElasticEnergy():
    """
    returns : 0.5 * U.transposed * K * U
    """
    return ProblemBase.GetAll()['MainProblem'].GetNodalElasticEnergy()

def GetKineticEnergy():
    """
    returns : 0.5 * Udot.transposed * M * Udot
    """
    return ProblemBase.GetAll()['MainProblem'].GetKineticEnergy()

def GetDampingPower():
    """
    returns : Udot.transposed * C * Udot
    The damping disspated energy can be approximated by:
            Edis = cumtrapz(DampingPower * TimeStep)
    """        
    return ProblemBase.GetAll()['MainProblem'].GetDampingPower()

def UpdateStiffness(StiffnessAssembling):
    ProblemBase.GetAll()['MainProblem'].UpdateStiffness(StiffnessAssembling)




#functions only used define in the ProblemPGD subclasses
def GetX(): return ProblemBase.GetAll()["MainProblem"].GetX() 
def GetXbc(): return ProblemBase.GetAll()["MainProblem"].GetXbc() 
def ComputeResidualNorm(err_0=None): return ProblemBase.GetAll()["MainProblem"].ComputeResidualNorm(err_0)
def GetResidual(): return ProblemBase.GetAll()["MainProblem"].GetResidual()
def UpdatePGD(termToChange, ddcalc='all'): return ProblemBase.GetAll()["MainProblem"].UpdatePGD(termToChange, ddcalc)
def UpdateAlpha(): return ProblemBase.GetAll()["MainProblem"].UpdateAlpha()
def AddNewTerm(numberOfTerm = 1, value = None, variable = 'all'): return ProblemBase.GetAll()["MainProblem"].AddNewTerm(numberOfTerm, value, variable)

