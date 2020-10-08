#derive de ConstitutiveLaw

from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.StrainOperator import *
from fedoo.libUtil.Variable       import *
from fedoo.libUtil.Dimension      import *
from fedoo.libUtil.PostTreatement import listStressTensor, listStrainTensor

import numpy as np

class ElasticAnisotropic(ConstitutiveLaw):
    def __init__(self, H, ID=""):
        ConstitutiveLaw.__init__(self, ID) # heritage
        
        Variable("DispX")
        Variable("DispY")        
        
        if ProblemDimension.Get() == "3D": 
            Variable("DispZ")

        self.__H = H
        self.__currentSigma = None
        self.__currentGradDisp = None
    
    
    def GetH(self):
        return self.__H
        
    def GetCurrentStress(self):
        return self.__currentSigma
    
    def GetCurrentGradDisp(self):
        return self.__currentGradDisp    
    
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
       
    def Update(self,assembly, pb, time, nlgeom):
        displacement = pb.GetDisp()
        
        if displacement is 0: 
            self.__currentGradDisp = 0
            self.__currentSigma = 0                        
        else:
            self.__currentGradDisp = assembly.GetGradTensor(displacement, "GaussPoint")
            GradValues = self.__currentGradDisp
            if nlgeom == False:
                Strain  = [GradValues[i][i] for i in range(3)] 
                Strain += [GradValues[1][2] + GradValues[2][1], GradValues[0][2] + GradValues[2][0], GradValues[0][1] + GradValues[1][0]]
            else:            
                Strain  = [GradValues[i][i] + 0.5*sum([GradValues[k][i]**2 for k in range(3)]) for i in range(3)] 
                Strain += [GradValues[1][2] + GradValues[2][1] + sum([GradValues[k][1]*GradValues[k][2] for k in range(3)])]
                Strain += [GradValues[0][2] + GradValues[2][0] + sum([GradValues[k][0]*GradValues[k][2] for k in range(3)])]
                Strain += [GradValues[0][1] + GradValues[1][0] + sum([GradValues[k][0]*GradValues[k][1] for k in range(3)])] 
        
            TotalStrain = listStrainTensor(Strain)
            self.__currentSigma = self.GetStress(TotalStrain, time) #compute the total stress in self.__currentSigma
        
        if nlgeom:
            if displacement is 0: self.__InitialGradDispTensor = 0
            else: self.__InitialGradDispTensor = assembly.GetGradTensor(displacement, "GaussPoint") 
       
    def GetStress(self, StrainTensor, time = None):         
        H = self.__ChangeBasisH(self.GetH())
        sigma = listStressTensor([sum([StrainTensor[j]*H[i][j] for j in range(6)]) for i in range(6)])

        return sigma # list of 6 objets 