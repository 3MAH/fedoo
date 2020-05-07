#derive de ConstitutiveLaw

from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libConstitutiveLaw.ConstitutiveLaw_ElasticAnisotropic import ElasticAnisotropic
from fedoo.libUtil.StrainOperator import *
from fedoo.libUtil.Variable       import *
from fedoo.libUtil.Dimension      import *

import scipy as sp

class ElasticIsotrop(ElasticAnisotropic):
    def __init__(self, YoungModulus, PoissonRatio, ID=""):
        ConstitutiveLaw.__init__(self, ID) # heritage
        self.__YoungModulus = YoungModulus
        self.__PoissonRatio = PoissonRatio    

    def GetYoungModulus(self):
        return self.__YoungModulus

    def GetPoissonRatio(self):
        return self.__PoissonRatio       
    
    def GetH (self):
        H  = sp.zeros((6,6), dtype='object')
        E  = self.__YoungModulus 
        nu = self.__PoissonRatio       

        # tester si contrainte plane ou def plane 
        if ProblemDimension.Get() == "2Dstress":
            H[0,0]=H[1,1]= E/(1-nu**2)
            H[0,1]= nu*E/(1-nu**2)
            H[5,5] = 0.5*E/(1+nu)    
            H[1,0]=H[0,1]  #symétrie            
                      
        else:
            H[0,0]=H[1,1]=H[2,2]= E*(1./(1+nu) + nu/((1.+nu)*(1-2*nu))) #H1 = 2*mu+lamb
            H[0,1]=H[0,2]=H[1,2]= E*(nu/((1+nu)*(1-2*nu)))  #H2 = lamb
            H[3,3]=H[4,4]=H[5,5] = 0.5*E/(1+nu) #H3 = mu
            H[1,0]=H[0,1] ; H[2,0]=H[0,2] ; H[2,1] = H[1,2] #symétrie 
            
        return H
    
#    def GetStressOperator(self, localFrame = None): # methode virtuel
#        #localFrame not used for isotropic material
#        # tester si contrainte plane ou def plane
#        
#        H = self.GetH()            
#        sigma = [sum([eps[j]*H[i][j] for j in range(6)]) for i in range(6)]
#
#        return sigma # list de 6 objets de type OpDiff

#    def GetStress(self, localFrame = None): # methode virtuel
#        #localFrame not used for isotropic material
#        
#        
#           
#        eps, eps_vir = GetStrainOperator()            
#        sigma = [sum([eps[j]*H[i][j] for j in range(6)]) for i in range(6)]
#
#        return sigma # list de 6 objets de type OpDiff

if __name__=="__main__":
    law = ElasticIsotrop(5e9,0.3)
    StressTensor = law.GetStress()
