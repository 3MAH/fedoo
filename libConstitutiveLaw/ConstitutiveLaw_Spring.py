#derive de ConstitutiveLaw

from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.DispOperator   import GetDispOperator
from fedoo.libUtil.Variable       import *
from fedoo.libUtil.Dimension      import *
from fedoo.libAssembly import AssemblyBase
import numpy as np
from numpy import linalg


class Spring(ConstitutiveLaw): 
    #Similar to CohesiveLaw but with different rigidity axis and without damage variable
    #Use with WeakForm.InterfaceForce
    def __init__(self, Kx=0, Ky = 0, Kz = 0, ID=""):        
        ConstitutiveLaw.__init__(self, ID) # heritage        
        self.__parameters = {'Kx':Kx, 'Ky':Ky, 'Kz':Kz}   
        
        Variable("DispX")
        Variable("DispY")        
        
        if ProblemDimension.Get() == "3D": # or ProblemDimension.Get() == "2Dstress" :
            Variable("DispZ")                       

    def GetK(self):
        return [[self.__parameters['Kx'], 0, 0], [0, self.__parameters['Ky'], 0], [0,0,self.__parameters['Kz']]]        

    def __ChangeBasisK(self, K):
        #Change of basis capability for spring type laws on the form : ForceVector = K * DispVector
        if self._ConstitutiveLaw__localFrame is not None:
            #building the matrix to change the basis of the stress and the strain
            B = self._ConstitutiveLaw__localFrame     

            if len(B.shape) == 3:    
                Binv = np.transpose(B, [2,1,0])
                B = np.transpose(B, [1,2,0])
                
            elif len(B.shape) == 2:
                Binv = B.T
            
            dim = len(B)
                
            KB = [[sum([K[i][j]*B[j][k] for j in range(dim)]) for k in range(dim)] for i in range(dim)]     
            K= [[sum([Binv[i][j]*KB[j][k] for j in range(dim)]) for k in range(dim)] for i in range(dim)]

            if dim == 2:
                K[0].append(0) ; K[1].append(0)
                K.append([0, 0, 0])
            
        return K

    def GetInterfaceStressOperator(self, **kargs): 
        #TODO test if it work in 2D and add the 2D case if needed
        K = self.__ChangeBasisK(self.GetK())
        
        U, U_vir = GetDispOperator() #relative displacement if used with cohesive element
        return [sum([U[j]*K[i][j] for j in range(3)]) for i in range(3)]

    def GetOperartorDelta(self): #operator to get the relative displacement
        U, U_vir = GetDispOperator()  #relative displacement if used with cohesive element
        return U 
        
    def GetInterfaceStress(self, Delta, time = None): 
        #Delta is the relative displacement vector
        K = self.__ChangeBasisK(self.GetK())
        return [sum([Delta[j]*K[i][j] for j in range(3)]) for i in range(3)] #list of 3 objects        
    


#    def GetStressOperator(self, localFrame=None): # methode virtuel
#    
#        U, U_vir = GetDispOperator()
#        
#        if self._ConstitutiveLaw__localFrame is None:
#            if ProblemDimension.Get() == "3D":        # tester si contrainte plane ou def plane              
#                return [U[0] * self.__parameters['Kx'], U[1] * self.__parameters['Ky'], U[2] * self.__parameters['Kz']]
#            else:
#                return [U[0] * self.__parameters['Kx'], U[1] * self.__parameters['Ky'], 0]
#        else: 
#            #TODO test if it work in 2D and add the 2D case if needed
#            K = [[self.__parameters['Kx'], 0, 0], [0, self.__parameters['Ky'], 0], [0,0,self.__parameters['Kz']]]
#            K= self._ConstitutiveLaw__ChangeBasisK(K)
#            return [sum([U[j]*K[i][j] for j in range(3)]) for i in range(3)]
