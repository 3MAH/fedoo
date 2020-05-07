import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from fedoo.libUtil.Variable import *
from fedoo.libPGD.SeparatedArray import *
from fedoo.libPGD.SeparatedOperator import *
#from fedoo.libPGD.libProblemPGD.BoundaryConditionPGD import *
from fedoo.libProblem import ProblemBase, BoundaryCondition

#===============================================================================
# Classes permettant de définir un problème sous forme discrète (forme séparée)
#===============================================================================

class ProblemPGD(ProblemBase): 



    
    def __init__(self, A, B, D, Mesh, ID = "MainProblem"):  
        
        # the problem is AX = B + D
        self.__A = A 
        self.__A.tocsr() #just in case A is in another format as csr
        
        if B is 0: self.__B = self._InitializeVector(A)
        else: self.__B = B        
            
        self.__D = D

        self.__Mesh = Mesh
        self.__NumberOfSubspace = Mesh.GetDimension()
        
        self.__ProblemDimension = self.__A.GetShape() #return a list of number of DoF for each subspace
        
        self.__DofBlocked = None
        self.__DofFree = None
        #    self.__A_BoundaryCondition = None
        self.__X = 0
        self.__Xbc = 0 #for boundary conditions
        
        ProblemBase.__init__(self,ID)


    #===============================================================================
    # Internal Functions
    #===============================================================================
    def _InitializeVector(self,A): #initialize a vector (force vector for instance) being giving the stiffness matrix
        return SeparatedZeros(A.GetShape())  
    
    def _SetVectorComponent(self, vector, name, value): #initialize a vector (force vector for instance) being giving the stiffness matrix

        assert isinstance(name,str), 'argument error'
        
        if name.lower() == 'all': 
            vector = value
        else:
            return NotImplemented
#
#            i = Variable.GetRank(name)
#
#            n = Problem.__Mesh.GetNumberOfNodes()
#
#            NewmarkPGD.__Xdotdot[i*n : (i+1)*n] = value            
    
    def _GetVectorComponent(self, vector, name):
        assert isinstance(name,str), 'argument error'
        
        if name.lower() == 'all': 
            return vector

        else:
            return NotImplemented



    def SecTerm_RS(self,R,d1): #détermine le second terme du problème restreint à la dimension d1 et connaissant R
        prod_aux_3 = sp.ones((R.nbTerm(), self.__C.nbTerm()))
        for d2 in list(range(d1))+list(range(d1+1,len(self.__C))):  
            prod_aux_3 = prod_aux_3 * (sp.dot(R.data[d2].T,self.__C.data[d2])) 
        return sp.reshape(sp.dot(prod_aux_3, self.__C.data[d1].T) , (-1,1))

    def calcMat_RS(self,R, d1): #détermine la matrice équivalente sur la dimension d1 connaissant R
        N_d1 = self.__A.GetShape()[d1]        
        
        #===============================================================================
        #         PGD standard(galerkin)
        #===============================================================================           
        # Possibilité d'optimisation à tester 
        # 1 - utiliser un tableau de sparse matrix pour vectorisé les opérations
        # 2 - faire un assemblage de toutes les matrices creuses dans une grande matrice creuse par bloc pour vectoriser les opérations
        
        if R.nbTerm() == 1: #sinon erreur
            Mat_K  = sparse.csr_matrix((N_d1 , N_d1))
            for kk in range(self.__A.NumberOfOperators()):
                prod_aux = 1
                for d2 in list(range(d1))+list(range(d1+1,self.__A.GetDimension())):
                    prod_aux = prod_aux * sp.dot(R.data[d2].T,self.__A.data[kk][d2]*R.data[d2])
                Mat_K = Mat_K + self.__A.data[kk][d1] * float(prod_aux)
            return Mat_K      
        else: 
            Mat_K  = sparse.csr_matrix((R.nbTerm()*N_d1 , R.nbTerm()*N_d1))
            for kk in range(self.__A.NumberOfOperators()):
                prod_aux = 1
                for d2 in list(range(d1))+list(range(d1+1,self.__A.GetDimension())):
                    prod_aux = prod_aux * sp.dot(R.data[d2].T,self.__A.data[kk][d2]*R.data[d2])                   
                Mat_K = Mat_K + sparse.bmat([[self.__A.data[kk][d1] * prod_aux[i,j] for j in range(R.nbTerm())] for i in range(R.nbTerm())])
            return Mat_K  
    
#    def copy(self):
#        return self.__class__(self)
 
    def SecTerm_Alpha(self,FF, BB): #détermine le second terme du calcul des alpha connaissant FF
        V=1 
        for dd in range(BB.dim):
            if sparse.issparse(FF.data[dd]): #bug fonction dot pour sparse matrice (utile pour calcul FE)
                V = V*(FF.data[dd].T * BB.data[dd])
            else:                
                V = V*sp.dot(FF.data[dd].T , BB.data[dd])
        return sp.sum(V,1)
     
    def calcMat_Alpha(self,FF): #détermine la matrice équivalente sur la dimension pour le calcul des alpha
        # traitement différent si FF contient des matrices creuses
        nbFF = FF.nbTerm()
        if sparse.issparse(FF.data[0]): #si FF contient des matrices creuses (cas particulier)
            Mat_K = sparse.csr_matrix((nbFF, nbFF))
            for Op in self.__A.data: 
                M=FF.data[0].T * Op[0] * FF.data[0]
                for dd in range(1,self.__A.dim):
                    M = sparse.csr_matrix.multiply(M, FF.data[dd].T * Op[dd] * FF.data[dd])
                Mat_K = Mat_K + M     
            return Mat_K #au format csr        
        else: #si FF contient des matrices pleines (cas général)
            Mat_K = 0
            for Op in self.__A.data: 
                M=1
                for dd in range(self.__A.GetDimension()):
                    M = M*sp.dot(FF.data[dd].T , Op[dd]*FF.data[dd])
                Mat_K +=M    
            return Mat_K       

    def GetB(self):
        return self.__B

    def SetD(self,D):
        self.__D = D 

    def GetX(self):
        return self.__X 
   
    def GetXbc(self):
        return self.__Xbc 
    
    def ApplyBoundaryCondition(self, timeFactor=1, timeFactorOld=None):                
        self.__X, self.__Xbc, F, self.__DofBlocked, self.__DofFree = BoundaryCondition.ApplyToPGD(self.__Mesh, self.__X, self.__ProblemDimension, timeFactor, timeFactorOld, self.GetID())
        self.__B = F 

    def GetDoFSolution(self,name):
        assert isinstance(name,str), 'argument error'
        
        if name.lower() == 'all': 
            return self.__X + self.__Xbc
        
        if self.__Xbc == 0:
            return self.__X.GetVariable(Variable.GetRank(name), self.__Mesh)
        else:
            return self.__X.GetVariable(Variable.GetRank(name), self.__Mesh) + self.__Xbc.GetVariable(Variable.GetRank(name), self.__Mesh)

    def SetDoFSolution(self,name,value):
        assert isinstance(name,str), 'argument error'
        
        if name.lower() == 'all': 
            self.__X = value
        else:
            return NotImplemented
#            i = Variable.GetRank(name)
#
#            self.__X[i*n : (i+1)*n] = value        



                           
#===============================================================================
# Fonction to build the PGD solution
#===============================================================================


    def ComputeResidualNorm(self,err_0=None):
        if err_0 == None: err_0 = 1
#            if self.err_0 == None: err_0 = 1
#            else: err_0 = self.err_0
        if self.__Xbc == 0 and self.__X == 0: res = self.__B + self.__D
        else: res = self.__B + self.__D - self.__A*(self.__Xbc+self.__X)  
        
        # CL à intégrer???
        for dd in range(self.__Mesh.GetDimension()): 
            res.data[dd][self.__DofBlocked[dd]] =  0  
        return res.norm(nbvar = Variable.GetNumberOfVariable())/err_0


    def GetResidual(self):
        if self.__Xbc == 0 and self.__X == 0: res = self.__B + self.__D
        else: res = self.__B + self.__D - self.__A*(self.__Xbc+self.__X)  
        
        # CL à intégrer???
        for dd in range(self.__Mesh.GetDimension()): 
            res.data[dd][self.__DofBlocked[dd]] =  0  
        return res


    def UpdatePGD(self,termToChange, ddcalc='all'): #extended PGD
        
        if ddcalc == 'all': 
            for nMesh in range(self.__Mesh.GetDimension()):
                self.UpdatePGD(termToChange, nMesh)
            return
        
        termToKeep = [t for t in range(self.__X.nbTerm()) if not t in termToChange]
        
        # --- Initialisation du second membre (BB/CC/SecTerm) --- #
        if termToKeep == []: self.__C = self.__B + self.__D - self.__A * self.__Xbc
        else: self.__C = self.__B + self.__D - self.__A * (self.__X.getTerm(termToKeep) + self.__Xbc)
        
#        for dd in range(self.__NumberOfSubspace): self.__C.data[dd][self.__CL[dd],:] = 0                      
                
        RS = self.__X.getTerm(termToChange) #value of the solution to update
        M = self.calcMat_RS(RS, ddcalc)                
        V = self.SecTerm_RS(RS,ddcalc)

        NbDoF = self.__C.shape[ddcalc]
        DofFree = np.hstack([self.__DofFree[ddcalc] + i*NbDoF for i in range(len(termToChange))])
        
#        self.__X.data[ddcalc][:,termToChange] = sp.reshape(sparse.linalg.spsolve(M, V) , (len(termToChange),-1)).T
#        self.__X.data[ddcalc][DofFree,termToChange]  = sparse.linalg.spsolve(M[np.c_[DofFree],DofFree], V[DofFree] )

        self.__X.data[ddcalc][self.__DofFree[ddcalc].reshape(-1,1),termToChange]  = np.reshape(self._ProblemBase__Solve(M[DofFree.reshape(-1,1),DofFree], V[DofFree] ) , (len(termToChange), -1)).T
#        self.__X.data[ddcalc][self.__DofFree[ddcalc].reshape(-1,1),termToChange]  = np.reshape(sparse.linalg.spsolve(M[DofFree.reshape(-1,1),DofFree], V[DofFree] ) , (len(termToChange), -1)).T
#        self.__X.data[ddcalc][self.__DofFree[ddcalc].reshape(-1,1),termToChange]  = np.reshape(sparse.linalg.cg(M[DofFree.reshape(-1,1),DofFree], V[DofFree] )[0] , (len(termToChange), -1)).T

#        from numpy import linalg
#        print(np.shape(DofFree))
#        print(M[DofFree.reshape(-1,1),DofFree].todense())
#        self.__X.data[ddcalc][self.__DofFree[ddcalc].reshape(-1,1),termToChange]  = np.reshape(linalg.solve(M[DofFree.reshape(-1,1),DofFree].todense(), V[DofFree] ) , (len(termToChange), -1)).T


    def UpdateAlpha(self):
        BB = SeparatedArray(self.__B + self.__D - self.__A*self.__Xbc)
        alpha = sp.c_[linalg.solve(self.calcMat_Alpha(self.__X), self.SecTerm_Alpha(self.__X, BB))] 
#        alpha = self.solve_Alpha(self.__X,SeparatedArray(self.__B - self.__A*self.__Xbc))   
        self.__X.data[0] = sp.tile(alpha.T, (self.__ProblemDimension[0],1) )*self.__X.data[0]  
                       

    def AddNewTerm(self,numberOfTerm = 1, value = None, variable = 'all'): 
        if variable != 'all': return NotImplemented    
        if value == None: self.__X += SeparatedArray([np.random.random((nn,numberOfTerm)) for nn in self.__ProblemDimension])    
        elif isinstance(value, (int,float)): self.__X += value * SeparatedOnes(self.__ProblemDimension, numberOfTerm)
        elif isinstance(value, SeparatedArray): 
            for t in range(numberOfTerm): self.__X += value.getTerm(t%value.nbTerm())
        #for boundary conditions        
        for dd in range(self.__Mesh.GetDimension()): self.__X.data[dd][self.__DofBlocked[dd]] = 0




##    def CL_ml(self, dd, QQ): #conditions aux limites via des multiplicateurs de lagrange        
##        #QQ matrice définissant les CL        
##        #à développer
##        pass
#        
#    def linkNodes(self, dd, nds, var): #lient des noeuds (CL périodiques)
#        """     
#        Apply Periodic Boundary Conditions
#
#        Parameters
#        ----------
#        dd : int
#            id of the considered space             
#        nds : list of list or arrays of integers 
#            Each element on the list is a list of two nodes to link 
#        var : int (default = 0)
#            id of variable to applied the boundary conditions 
#
#        See also
#        ----------                      
#        addCL : Method used to apply Dirichelet boundary conditions    
#        """     
#        
#        #vérifier le fonctionnement avec EqD_min
#        #Non optimisée
#        N = self.N[dd] #nombre de noeuds
#        self.CL[dd].extend(var*N + sp.array(nds)[:,1])        
#        
#        for nd in nds:        
#            for Op in self.AA.data:
#                Op[dd] = Op[dd].tolil()
#                Op[dd][var*N +nd[0],:] += Op[dd][var*N + nd[1],:]
#                Op[dd].rows[var*N + nd[1]]=[]
#                Op[dd].data[var*N + nd[1]]=[]                
#            self.AA[0][dd].rows[var*N + nd[1]] = [var*N + nd[0], var*N + nd[1]]
#            self.AA[0][dd].data[var*N + nd[1]] = [-1, 1]
#            
#            
#            
#            
#                        
#
#class EqD_min(EqD): #Equation discrete avec minimisation du résidu
#    def SecTerm_RS(self,R,d1): #détermine le vecteur équivalent sur la dimension d1 connaissant R
#        Vec_F = sp.zeros((self.CC.NN[d1], 1))
#        for kk in range(self.AA.NumberOfOperators()):
#            prod_aux_3 = sp.ones((1, self.CC.nbTerm()))
#            for d2 in range(d1)+range(d1+1,len(self.CC)):  
#                prod_aux_3 = prod_aux_3 * (sp.dot(R.data[d2].T,self.AA.data[kk][d2].T*self.CC.data[d2])) 
#            Vec_F += sp.dot(self.AA.data[kk][d1].T*self.CC.data[d1], prod_aux_3.T) 
#        return Vec_F
#        
#    def SecTerm_Alpha(self,FF,BB): #détermine le vecteur équivalent pour calcul des alpha   
#        # ne marche pas !!!         
#        SecM = 0
#        for Op in self.AA.data:         
#            V=1 
#            for dd in range(BB.dim):
#                V = V*sp.dot(FF.data[dd].T , Op[dd].T*BB.data[dd])
#            SecM += sp.sum(V,1)   
#        return SecM        
#                
#    def calcMat_RS(self, R, d1): #détermine la matrice équivalente sur la dimension d1 connaissant R
#        N_d1 = self.AA.NN[d1]
#        Mat_K  = sparse.csr_matrix((N_d1 , N_d1))
#                
#        #===============================================================================
#        #         Symétrisation de l'opérateur (minimisation du résidu)
#        #===============================================================================
#        prod_aux = sp.ones((self.AA.NumberOfOperators(), self.AA.NumberOfOperators()))
#        AA_R = self.AA*R #on n'est pas obligé de calculer AA_R pour la dimension d1 (à optimiser)
#        for d2 in range(d1)+range(d1+1,self.AA.dim):                
#            prod_aux = prod_aux * sp.dot(AA_R.data[d2].T , AA_R.data[d2])
#        for k1 in range(self.AA.NumberOfOperators()):
#            Mat_K = Mat_K + (self.AA.data[k1][d1].T * self.AA.data[k1][d1]) * float(prod_aux[k1][k1])
#            for k2 in range(k1+1,self.AA.NumberOfOperators()):
#                M = self.AA.data[k2][d1].T*self.AA.data[k1][d1]
#                Mat_K = Mat_K + (M+M.T) * float(prod_aux[k1][k2])
#        return Mat_K            
#        
#    def MatEquiv_Alpha(self, FF): #détermine la matrice équivalente pour le calcul des alpha 
#        # ne marche pas !!!
#        #Pas optimisé        
#        Mat_K = 0
#        for Op in self.AA.data: 
#            for Op2 in self.AA.data:
#                M=1
#                for dd in range(FF.dim):
#                    M = M*sp.dot(FF.data[dd].T * Op2[dd].T , Op[dd]*FF.data[dd])
#                Mat_K +=M      
#        return Mat_K
#
#

def GetX(): return ProblemPGD.GetAll()["MainProblem"].GetX() 
def GetXbc(): return ProblemPGD.GetAll()["MainProblem"].GetXbc() 
def ComputeResidualNorm(err_0=None): return ProblemPGD.GetAll()["MainProblem"].ComputeResidualNorm(err_0)
def GetResidual(): return ProblemPGD.GetAll()["MainProblem"].GetResidual()
def UpdatePGD(termToChange, ddcalc='all'): return ProblemPGD.GetAll()["MainProblem"].UpdatePGD(termToChange, ddcalc)
def UpdateAlpha(): return ProblemPGD.GetAll()["MainProblem"].UpdateAlpha()
def AddNewTerm(numberOfTerm = 1, value = None, variable = 'all'): return ProblemPGD.GetAll()["MainProblem"].AddNewTerm(numberOfTerm, value, variable)

