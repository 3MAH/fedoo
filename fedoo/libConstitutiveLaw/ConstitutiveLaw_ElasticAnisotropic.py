#derive de ConstitutiveLaw
#compatible with the simcoon strain and stress notation

from fedoo.libConstitutiveLaw.ConstitutiveLaw import Mechanical3D
from fedoo.libUtil.PostTreatement import listStressTensor, listStrainTensor

import numpy as np

class ElasticAnisotropic(Mechanical3D):
    """
    Linear full Anistropic constitutive law defined from the rigidity matrix H.

    The constitutive Law should be associated with :mod:`fedoo.libWeakForm.InternalForce`    
    
    Parameters
    ----------
    H : list of list or an array (shape=(6,6)) of scalars or arrays of gauss point values.
        The rigidity matrix. 
        If H is a list of gauss point values, the shape shoud be H.shape = (6,6,NumberOfGaussPoints)
    ID : str, optional
        The ID of the constitutive law      
    """
    def __init__(self, H, ID=""):
        Mechanical3D.__init__(self, ID) # heritage

        self.__H = H
        self.__currentSigma = None
        self.__currentGradDisp = None            
    
    def GetTangentMatrix(self):
        return self.__H

    def GetPKII(self):
        return self.__currentSigma
    
    def GetCurrentStress(self):
        #alias of GetPKII mainly use for small strain displacement problems
        print('Warning : GetCurrentStress will be removed in future versions. Use GetStress instead')
        return (self.__currentSigma)

    def GetStress(self, **kargs):
        #alias of GetPKII mainly use for small strain displacement problems
        return (self.__currentSigma)
    
    def GetStrain(self, **kargs):
        return self.__currentStrain
    
    # def ComputeStrain(self, assembly, pb, nlgeom, type_output='GaussPoint'):
    #     displacement = pb.GetDoFSolution()                
    #     if displacement is 0: 
    #         return 0 #if displacement = 0, Strain = 0
    #     else:
    #         return assembly.GetStrainTensor(displacement, type_output)  
    
    
    def GetCurrentGradDisp(self):
        return self.__currentGradDisp    
    
    # def GetStressOperator(self, **kargs): 
    #     H = self.GetH(**kargs )
                      
    #     eps, eps_vir = GetStrainOperator(self.__currentGradDisp)            
    #     sigma = [sum([eps[j]*H[i][j] for j in range(6)]) for i in range(6)]

    #     return sigma # list de 6 objets de type OpDiff
       
    
    def Initialize(self, assembly, pb, initialTime = 0., nlgeom=False):
        if self._dimension is None:   
            self._dimension = assembly.space.GetDimension()
        self.nlgeom = nlgeom
    
    def Update(self,assembly, pb, dtime):
        displacement = pb.GetDoFSolution()
        
        if displacement is 0: 
            self.__currentGradDisp = 0
            self.__currentSigma = 0                        
        else:
            self.__currentGradDisp = assembly.GetGradTensor(displacement, "GaussPoint")

            GradValues = self.__currentGradDisp
            if self.nlgeom == False:
                Strain  = [GradValues[i][i] for i in range(3)] 
                Strain += [GradValues[0][1] + GradValues[1][0], GradValues[0][2] + GradValues[2][0], GradValues[1][2] + GradValues[2][1]]
            else:            
                Strain  = [GradValues[i][i] + 0.5*sum([GradValues[k][i]**2 for k in range(3)]) for i in range(3)] 
                Strain += [GradValues[0][1] + GradValues[1][0] + sum([GradValues[k][0]*GradValues[k][1] for k in range(3)])] 
                Strain += [GradValues[0][2] + GradValues[2][0] + sum([GradValues[k][0]*GradValues[k][2] for k in range(3)])]
                Strain += [GradValues[1][2] + GradValues[2][1] + sum([GradValues[k][1]*GradValues[k][2] for k in range(3)])]
            TotalStrain = listStrainTensor(Strain)
                
            self.__currentStrain = TotalStrain                            
       
            H = self.GetH()
        
            self.__currentSigma = listStressTensor([sum([TotalStrain[j]*assembly.ConvertData(H[i][j]) for j in range(6)]) for i in range(6)]) #H[i][j] are converted to gauss point excepted if scalar
            # self.__currentSigma = self.GetStress(TotalStrain) #compute the total stress in self.__currentSigma            
       
    def GetStressFromStrain(self, StrainTensor):     
        H = self.GetH()
        
        sigma = listStressTensor([sum([StrainTensor[j]*H[i][j] for j in range(6)]) for i in range(6)])

        return sigma # list of 6 objets 
    
     
                        
        