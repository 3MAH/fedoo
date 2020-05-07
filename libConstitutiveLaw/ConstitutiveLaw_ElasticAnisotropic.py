#derive de ConstitutiveLaw

from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.StrainOperator import *
from fedoo.libUtil.Variable       import *
from fedoo.libUtil.Dimension      import *
from fedoo.libUtil.PostTreatement import listStressTensor

import numpy as np

class ElasticAnisotropic(ConstitutiveLaw):
    def __init__(self, H, ID=""):
        ConstitutiveLaw.__init__(self, ID) # heritage
        
        Variable("DispX")
        Variable("DispY")        
        
        if ProblemDimension.Get() == "3D": 
            Variable("DispZ")

        self.__H = H
        
    def GetH(self):
        return self.__H
       
    def __ChangeBasisH(self, H):
        #Change of basis capability for laws on the form : StressTensor = H * StrainTensor
        #StressTensor and StrainTensor are column vectors based on the voigt notation 
        if self._ConstitutiveLaw__localFrame is not None:
            localFrame = self._ConstitutiveLaw__localFrame
            #building the matrix to change the basis of the stress and the strain
#            theta = np.pi/8
#            np.array([[np.cos(theta),np.sin(theta),0], [-np.sin(theta),np.cos(theta),0], [0,0,1]]) 
            R_epsilon = np.empty((len(localFrame), 6,6))
            R_epsilon[:,  :3,  :3] = localFrame**2 
            R_epsilon[:,  :3, 3:6] = localFrame[:,:,[1,2,0]]*localFrame[:,:,[2,0,1]]
            R_epsilon[:, 3:6,  :3] = 2*localFrame[:,[1,2,0]]*localFrame[:,[2,0,1]] 
            R_epsilon[:, 3:6, 3:6] = localFrame[:,[[1],[2],[0]], [1,2,0]]*localFrame[:,[[2],[0],[1]],[2,0,1]] + localFrame[:,[[2],[0],[1]],[1,2,0]]*localFrame[:,[[1],[2],[0]],[2,0,1]] 
            R_sigma_inv = np.transpose(R_epsilon,[0,2,1])        
            
            if len(H.shape) == 3: H = np.rollaxis(H,2,0)
            H = np.matmul(R_sigma_inv, np.matmul(H,R_epsilon))
            if len(H.shape) == 3: H = np.rollaxis(H,0,3)  
            
        return H
    
    def GetStressOperator(self, **kargs): 
        H = self.__ChangeBasisH(self.GetH())
                      
        eps, eps_vir = GetStrainOperator()            
        sigma = [sum([eps[j]*H[i][j] for j in range(6)]) for i in range(6)]

        return sigma # list de 6 objets de type OpDiff
       
    def GetStress(self, StrainTensor, time = None): 
        H = self.__ChangeBasisH(self.GetH())
        sigma = listStressTensor([sum([StrainTensor[j]*H[i][j] for j in range(6)]) for i in range(6)])

        return sigma # list of 6 objets 