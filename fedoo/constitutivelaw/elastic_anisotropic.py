#derive de ConstitutiveLaw
#compatible with the simcoon strain and stress notation

from fedoo.core.mechanical3d import Mechanical3D
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList

import numpy as np

class ElasticAnisotropic(Mechanical3D):
    """
    Linear full Anistropic constitutive law defined from the rigidity matrix H.

    The constitutive Law should be associated with :mod:`fedoo.weakform.InternalForce`    
    
    Parameters
    ----------
    H : list of list or an array (shape=(6,6)) of scalars or arrays of gauss point values.
        The rigidity matrix. 
        If H is a list of gauss point values, the shape shoud be H.shape = (6,6,NumberOfGaussPoints)
    name : str, optional
        The name of the constitutive law      
    """
    def __init__(self, H, name =""):
        Mechanical3D.__init__(self, name) # heritage

        self.__H = H
        self._stress = 0
        self._grad_disp = 0            
    
    def GetTangentMatrix(self):
        return self.__H

    def get_stress(self, **kargs):
        #alias of GetStress mainly use for small strain displacement problems
        return (self._stress)
    
    def get_pk2(self):
        #alias of GetPKII mainly use for small strain displacement problems
        return self._stress
    
    def get_cauchy(self):
        #alias of GetStress mainly use for small strain displacement problems
        return self._stress
    
    def get_strain(self, **kargs):
        return self.__currentStrain
    
    # def ComputeStrain(self, assembly, pb, nlgeom, type_output='GaussPoint'):
    #     displacement = pb.GetDoFSolution()                
    #     if displacement is 0: 
    #         return 0 #if displacement = 0, Strain = 0
    #     else:
    #         return assembly.get_strain(displacement, type_output)  
    
    
    def get_disp_grad(self):
        return self._grad_disp          
    
    def initialize(self, assembly, pb, t0 = 0., nlgeom=False):
        if self._dimension is None:   
            self._dimension = assembly.space.get_dimension()
        self.nlgeom = nlgeom
    
    def update(self,assembly, pb, dtime):
        displacement = pb.GetDoFSolution()
        
        if displacement is 0: 
            self._grad_disp = 0
            self._stress = 0                        
        else:
            self._grad_disp = assembly.get_grad_disp(displacement, "GaussPoint")

            GradValues = self._grad_disp #alias
            if self.nlgeom == False:
                Strain  = [GradValues[i][i] for i in range(3)] 
                Strain += [GradValues[0][1] + GradValues[1][0], GradValues[0][2] + GradValues[2][0], GradValues[1][2] + GradValues[2][1]]
            else:            
                Strain  = [GradValues[i][i] + 0.5*sum([GradValues[k][i]**2 for k in range(3)]) for i in range(3)] 
                Strain += [GradValues[0][1] + GradValues[1][0] + sum([GradValues[k][0]*GradValues[k][1] for k in range(3)])] 
                Strain += [GradValues[0][2] + GradValues[2][0] + sum([GradValues[k][0]*GradValues[k][2] for k in range(3)])]
                Strain += [GradValues[1][2] + GradValues[2][1] + sum([GradValues[k][1]*GradValues[k][2] for k in range(3)])]
            TotalStrain = StrainTensorList(Strain)
                
            self.__currentStrain = TotalStrain                            
       
            H = self.GetH()
        
            self._stress = StressTensorList([sum([TotalStrain[j]*assembly.convert_data(H[i][j]) for j in range(6)]) for i in range(6)]) #H[i][j] are converted to gauss point excepted if scalar
       
    def GetStressFromStrain(self, StrainTensor):     
        H = self.GetH()
        
        sigma = StressTensorList([sum([StrainTensor[j]*H[i][j] for j in range(6)]) for i in range(6)])

        return sigma # list of 6 objets 
    
     
                        
        