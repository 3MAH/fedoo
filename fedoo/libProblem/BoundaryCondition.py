import numpy as np

from fedoo.libUtil.Dimension import *
from fedoo.libUtil.Variable  import *
from fedoo.libPGD.SeparatedArray import *
from scipy import sparse

class BoundaryCondition :
    """
    Classe de condition limite
    """

    __lbc = {"MainProblem":[]} # variable statique : liste des BC crees

    def __init__(self,BoundaryType,Var,Value,Index,Constant = None, timeEvolution=None, initialValue = None, ProblemID = "MainProblem"):
        ### Var: variable name (str) or (for MPC only) list of variable name or variable rank 
        ### Value: Variable final value (Dirichlet) or force Value (Neumann) or factor list (MPC)
        ### Index : Nodes Index of SetOf Nodes ID           
        ### Constant: for MPC only, constant value on the equation
        ### timeEvolution : function 
        ###
        ### To define many MPC in one operation, use array where each line define a single MPC
        assert BoundaryType in ['Dirichlet', 'Neumann', 'MPC'], "The type of Boundary conditions should be either 'Dirichlet', 'Neumann' or 'MPC'"
        
        if timeEvolution is None: 
            def timeEvolution(timeFactor): return timeFactor            
        self.__timeEvolution = timeEvolution
        self.__initialValue = initialValue # can be a float or an array or None !
        
        self.__BoundaryType = BoundaryType
        if isinstance(Var, str): 
            self.__Var = Variable.GetRank(Var)
            if BoundaryType == 'MPC': self.__VarMaster = self.__Var
        else: #Var should be a list or a numpy array
            assert BoundaryType == 'MPC', "Var should be a string for % Boundary Type".format(BoundaryType)
            if isinstance(Var[0], str): Var = [Variable.GetRank(v) for v in Var]
            self.__Var = Var[0] #Var for slave DOF (eliminated DOF)
            self.__VarMaster = Var[1:] #Var for master DOF (not eliminated DOF in MPC)
        self.__GlobalIndex = None
              
        if BoundaryType in ['Dirichlet', 'Neumann']:
            self.__Value = Value # can be a float or an array !
            if isinstance(Index, str): self.__SetOfID = Index # must be a string defining a set of nodes
            else: self.__Index = np.array(Index).astype(int) # must be a np.array  #Not for PGD
  
        elif BoundaryType == 'MPC':  #only for FEM for now
            
            Factor =  Value #for MPC, Value is a list containing the factor   
            self.__Index = np.array(Index[0], dtype = int) #Node index for slave DOF (eliminated DOF) #use SetOf for PGD
            self.__IndexMaster = np.array(Index[1:], dtype = int) #Node index for master DOF (not eliminated DOF in MPC)
            if Constant is not None: 
                self.__Value = -Constant/Factor[0] # should be a numeric value or a 1D array for multiple MPC       
            else: self.__Value = 0
            
            self.__Fact = -np.array(Factor[1:])/Factor[0] #does not include the master node coef = 1
            
        if not(ProblemID in BoundaryCondition.__lbc):
            BoundaryCondition.__lbc[ProblemID] = []

        BoundaryCondition.__lbc[ProblemID].append(self)     

    def __GetFactor(self,timeFactor=1, timeFactorOld = None): 
        #return the time factor applied to the value of boundary conditions
        if timeFactorOld is None or self.__BoundaryType == 'Neumann': #for Neumann, the force is applied in any cases
            return self.__timeEvolution(timeFactor)
        else: 
            return self.__timeEvolution(timeFactor)-self.__timeEvolution(timeFactorOld)
            
    def GetValue(self,timeFactor=1, timeFactorOld = None): 
        Factor = self.__GetFactor(timeFactor, timeFactorOld) 
        if Factor == 0: return 0
        elif self.__initialValue is None: return Factor * self.__Value         
        else: #in case there is an initial value
            if self.__BoundaryType == 'Neumann': #for Neumann, the true value of force is applied 
                return Factor * (self.__Value - self.__initialValue) + self.__initialValue
            else: #return the incremental value
                return Factor * (self.__Value - self.__initialValue)                               

    def __ApplyTo(self, X, Nnodes, timeFactor=1, timeFactorOld = None): 
        """
        X must be a np.array
        Nnodes must be an int: number of nodes
        timeFactor is the time Factor (default = 1)
            timeFactor = 0 for time=t0 
            timeFactor = 1 for time=tmax 
        timeFactorOld should be defined for incremental approach.
        If timeFactorOld is not None, the incremental value between timeFactorOld and timeFactor is applied
        """

        self.__GlobalIndex = (self.__Var*Nnodes + self.__Index).astype(int)
        X[self.__GlobalIndex] = self.GetValue(timeFactor, timeFactorOld)        
        # X[self.__GlobalIndex] = self.__GetFactor(timeFactor, timeFactorOld) * (self.__Value - self.__InitialValue) + self.__InitialValue
        return X

    @staticmethod
    def Apply(n, timeFactor = 1, timeFactorOld = None, ProblemID = "MainProblem"):
        
        DoF = Variable.GetNumberOfVariable()     
        Uimp = np.zeros(DoF*n)
        F = np.zeros(DoF*n)

        MPC = False
        DofB = np.array([])
        data = []
        row = []
        col = []
        for e in BoundaryCondition.__lbc[ProblemID]:
            if e.__BoundaryType == 'Dirichlet':
                Uimp = e.__ApplyTo(Uimp, n, timeFactor, timeFactorOld)
                DofB = np.hstack((DofB,e.__GlobalIndex))

            elif e.__BoundaryType == 'Neumann':
                F = e.__ApplyTo(F, n, timeFactor)
            
            elif e.__BoundaryType == 'MPC':
                Uimp = e.__ApplyTo(Uimp, n, timeFactor, timeFactorOld) #valid in this case ??? need to be checked
                DofB = np.hstack((DofB,e.__GlobalIndex))        
                MPC = True         
#                if np.isscalar(self.__Index): nbMPC = 1
#                else: nbMPC = len(self.__Index)
                nbFact = len(e.__Fact)
                
                #shape self.__Fact should be nbFact*nbMPC 
                #shape self.__Index should be nbMPC
                #shape self.__IndexMaster should be nbFact*nbMPC
                data.append(np.array(e.__Fact.T).ravel())
                row.append((np.array(e.__GlobalIndex).reshape(-1,1)*np.ones(nbFact)).ravel())
                col.append((e.__IndexMaster + np.c_[e.__VarMaster]*n).T.ravel())
                
        DofB = np.unique(DofB).astype(int)
        DofL = np.setdiff1d(range(DoF*n),DofB).astype(int)
        
        #build matrix MPC
        if MPC:    
            #Treating the case where MPC includes some blocked nodes as master nodes
            #M is a matrix such as Ublocked = M@U + Uimp
            #Compute M + M@M
            M = sparse.coo_matrix( 
                (np.hstack(data), (np.hstack(row),np.hstack(col))), 
                shape=(DoF*n,DoF*n))
                       
            Uimp = Uimp	+ M@Uimp 
            
            M = (M+M@M).tocoo()
            data = M.data
            row = M.row
            col = M.col
            BoundaryCondition.M = M #test : used to compute the reaction - to delete later

            #modification col numbering from DofL to np.arange(len(DofL))
            changeInd = np.full(DoF*n,np.nan) #mettre des nan plutôt que des zeros pour générer une erreur si pb
            changeInd[DofL] = np.arange(len(DofL))
            col = changeInd[np.hstack(col)]
            mask = np.logical_not(np.isnan(col)) #mask to delete nan value 
            
            col = col[mask] ; row = row[mask] ; data = data[mask]

        # #adding identity for free nodes
        col = np.hstack((col,np.arange(len(DofL)))) #np.hstack((col,DofL)) #col.append(DofL)  
        row = np.hstack((row,DofL)) #row.append(DofL)            
        data = np.hstack((data, np.ones(len(DofL)))) #data.append(np.ones(len(DofL)))
        
        MatCB = sparse.coo_matrix( 
                (data,(row,col)), 
                shape=(DoF*n,len(DofL))).tocsr()

        return Uimp, F, DofB, DofL, MatCB


    @staticmethod
    # TODO
    # verifier l'utlisation de var dans boundary conditions PGD
    # reprendre les conditions aux limites en incluant les méthodes de pénalités pour des conditions aux limites plus exotiques
    # verifier qu'il n'y a pas de probleme lié au CL sur les ddl inutiles
    def ApplyToPGD(meshPGD, X, shapeX, timeFactor = 1, timeFactorOld = None, ProblemID = "MainProblem"): 
        Xbc = 0 #SeparatedZeros(shapeX)
        F = 0 

        DofB = [np.array([]) for i in range(meshPGD.GetDimension())] 
        dimBC = None #dimension requiring a modification of Xbc - dimBC = None if Xbc is never modified, dimBC = dd if only the dimension dd is modified, dimBC = 'many' if there is more than one dimension
        
        for e in BoundaryCondition.__lbc[ProblemID]:
            SetOfNodesForBC = meshPGD.GetSetOfNodes(e.__SetOfID)            
            if isinstance(e.__Value, list): e.__Value = np.array(e.__Value)
            
            Value = e.GetValue(timeFactor, timeFactorOld)
            if e.__BoundaryType == 'Neumann':
                if Value == 0: continue #dans ce cas, pas de force à ajouter
#                dd = SetOfNodesForBC[0][0]
#                index = SetOfNodesForBC[1][0]
                var = [meshPGD._GetSpecificVariableRank (d, e.__Var) for d in range(meshPGD.GetDimension())] #specific variable rank related to the submesh dd
                Nnd = [meshPGD.GetListMesh()[d].GetNumberOfNodes() for d in range(meshPGD.GetDimension())]                       
                
                item = [slice(Nnd[d]*var[d], Nnd[d]*(var[d]+1)) for d in range(meshPGD.GetDimension())]
                
                for i, d in enumerate(SetOfNodesForBC[0]):
                    index = np.array(SetOfNodesForBC[1][i], dtype = int)
                    item[d] = var[d]*Nnd[d] + index
                
                if isinstance(e.__Value, np.ndarray): e.__Value = SeparatedArray([e.__Value.reshape(-1,1)])
                if isinstance(e.__Value, SeparatedArray): 
                    if len(e.__Value) != meshPGD.GetDimension():
                        if len(e.__Value) ==  len(SetOfNodesForBC[1]):                            
                            nbt = e.__Value.nbTerm()
                            e.__Value = SeparatedArray( [ e.__Value.data[SetOfNodesForBC[0].index(d)] if d in SetOfNodesForBC[0] \
                                                         else np.ones((1,nbt)) for d in range(len(shapeX))] )
                        else: assert 0, "Dimension doesn't match"
                                    
                if F is 0: 
                    if isinstance(Value, (float, int, np.floating)):
                        Fadd = SeparatedZeros(shapeX)
                        # for d in range(meshPGD.GetDimension()): Fadd.data[d][item[d]] = Value
                        Fadd.data[0][item[0]] = Value
                        for d in range(1,meshPGD.GetDimension()): Fadd.data[d][item[d]] = 1.
                    else: 
                        Fadd = SeparatedZeros(shapeX, nbTerm = Value.nbTerm())
                        Fadd.data[0][item[0]] = Value.data[d]
                        for d in range(1,meshPGD.GetDimension()): Fadd.data[d][item[d]] = Value.data[d]
                    F = F+Fadd                    
                else: F.__setitem__(tuple(item), Value)                        

            elif e.__BoundaryType == 'Dirichlet':  
                if len(SetOfNodesForBC[1]) == 1 and isinstance(Value, (int,float,np.floating,np.ndarray)): #The BC can be applied on only 1 subspace 
                    dd = SetOfNodesForBC[0][0]
                    index = SetOfNodesForBC[1][0]
                    var = meshPGD._GetSpecificVariableRank (dd, e.__Var) #specific variable rank related to the submesh dd
                    Nnd = meshPGD.GetListMesh()[dd].GetNumberOfNodes()
                    GlobalIndex = (var*Nnd + np.array(index)).astype(int)
                    
                    if isinstance(Value, np.ndarray): Value = Value.reshape(-1,1)                                
                    
                    DofB[dd] = np.hstack((DofB[dd],GlobalIndex))                     
                    if dimBC is None: #initialization of Xbc           
                        if Value is not 0: # modification of the second term Xbc
                            dimBC = dd
                            Xbc = SeparatedArray([np.ones((shapeX[d],1)) if d!=dd else np.zeros((shapeX[d],1)) for d in range(len(shapeX))])
                            Xbc.data[dd][GlobalIndex] = Value
                    elif dd == dimBC: #in this case, the definition of Xbc is trivial
                        Xbc.data[dd][GlobalIndex] = Value
                    else: #many dimension required the modification of BC                                          
                        dimBC = 'many'                                                    
                        Xbc_old = Xbc.copy()
                        Xbc_old.data[dd] = 0*Xbc_add.data[dd]
                        Xbc_old.data[dd][GlobalIndex] = Xbc.data[dd][GlobalIndex]
                        Xbc_add = SeparatedArray([np.ones((shapeX[d],1)) if d!=dd else np.zeros((shapeX[d],1)) for d in range(len(shapeX))])
                        Xbc_add.data[dd][GlobalIndex] = Value
                        Xbc = Xbc+Xbc_add - Xbc_old                                                   
                
                else: #a penatly method is required
                    return NotImplemented    
    
            else: assert 0, "Boundary type non recognized"
        
            
#        if F == 0: F = SeparatedZeros(shapeX)              
            
        DofB = [np.unique(dofb).astype(int) for dofb in DofB] #bloqued DoF for all the submeshes
        DofL = [np.setdiff1d(range(shapeX[d]),DofB[d]).astype(int) for d in range(meshPGD.GetDimension())] #free dof for all the submeshes

        if X!=0: 
            for d in range(meshPGD.GetDimension()): 
                X.data[dd][DofB[d]] = 0

        return X, Xbc, F, DofB, DofL
        
    def ChangeIndices(self,newIndices):
        self.__Index = np.array(newIndices).astype(int) # must be a np.array
    
    def ChangeValues(self,newValues, initialValue = None, timeEvolution=None):
        #if initialValue == 'Current', keep current value as initial values (change of step)
        #if initialValue is None, don't change the initial value        
        if initialValue == 'Current': self.__initialValue = self.__Value
        elif initialValue is not None: self.__initialValue = initialValue               
        if timeEvolution is not None: self.__timeEvolution = timeEvolution        
        self.__Value = newValues # can be a float or an array !  
        
    def Remove(self, ProblemID = "MainProblem"):
        BoundaryCondition.__lbc[ProblemID].remove(self)
        del self

    @staticmethod
    def RemoveAll(ProblemID = "MainProblem"):
        del BoundaryCondition.__lbc[ProblemID]

    @staticmethod
    def GetAll(ProblemID = "MainProblem"):        
        return BoundaryCondition.__lbc[ProblemID]

        
if __name__ == "__main__":
    pass
