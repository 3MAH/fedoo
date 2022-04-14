#derive de ConstitutiveLaw
#simcoon compatible

from fedoo.libConstitutiveLaw.ConstitutiveLaw import Mechanical3D
from fedoo.libConstitutiveLaw.ConstitutiveLaw_ElasticAnisotropic import ElasticAnisotropic

import numpy as np

class ElasticIsotrop(ElasticAnisotropic):
    """
    A simple linear elastic isotropic constitutive law defined from a Yound Modulus and a Poisson Ratio.
     
    The constitutive Law should be associated with :mod:`fedoo.libWeakForm.InternalForce`    
    
    Parameters
    ----------
    YoungModulus : scalars or arrays of gauss point values.
        The Young Modulus of the elastic isotropic material
    PoissonRatio : scalars or arrays of gauss point values.
        The PoissonRatio of the elastic isotropic material
    ID : str, optional
        The ID of the constitutive law       
    """
    
    def __init__(self, YoungModulus, PoissonRatio, ID=""):

        Mechanical3D.__init__(self, ID) # heritage
        self.__YoungModulus = YoungModulus
        self.__PoissonRatio = PoissonRatio    

    def GetYoungModulus(self):
        """
        Return the Young Modulus 
        """
        return self.__YoungModulus

    def GetPoissonRatio(self):
        """
        Return the Poisson Ratio
        """
        return self.__PoissonRatio       
    
    def GetTangentMatrix(self):     
        #the returned stiffness matrix is 6x6 even in 2D
        H  = np.zeros((6,6), dtype='object')
        E  = self.__YoungModulus 
        nu = self.__PoissonRatio       

        H[0,0]=H[1,1]=H[2,2]= E*(1./(1+nu) + nu/((1.+nu)*(1-2*nu))) #H1 = 2*mu+lamb
        H[0,1]=H[0,2]=H[1,2]= E*(nu/((1+nu)*(1-2*nu)))  #H2 = lamb
        H[3,3]=H[4,4]=H[5,5] = 0.5*E/(1+nu) #H3 = mu
        H[1,0]=H[0,1] ; H[2,0]=H[0,2] ; H[2,1] = H[1,2] #symétrie 
            
        return H        

    def GetTangentMatrix_2Dstress(self):
        #for 2D stress problems       
        #the returned stiffness matrix is 6x6 even in 2D
        H  = np.zeros((6,6), dtype='object')
        E  = self.__YoungModulus 
        nu = self.__PoissonRatio       

        H[0,0]=H[1,1]= E/(1-nu**2)
        H[0,1]= nu*E/(1-nu**2)
        H[3,3] = 0.5*E/(1+nu)    
        H[1,0]=H[0,1]  #symétrie                                  
            
        return H        
    
    
if __name__=="__main__":
    law = ElasticIsotrop(5e9,0.3)
    print(law.GetTangentMatrix())
