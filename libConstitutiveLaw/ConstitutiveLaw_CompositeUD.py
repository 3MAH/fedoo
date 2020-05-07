#derive de ConstitutiveLaw

from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libConstitutiveLaw.ConstitutiveLaw_ElasticAnisotropic import ElasticAnisotropic
from fedoo.libUtil.StrainOperator import *
from fedoo.libUtil.Variable       import *
from fedoo.libUtil.Dimension      import *

import numpy as np




class CompositeUD(ElasticAnisotropic):
    
    def __init__(self, Vf=0.6, E_f=250000, E_m = 3500, nu_f = 0.33, nu_m = 0.3, angle=0, ID=""):
        ConstitutiveLaw.__init__(self, ID) # heritage
#        self.__YoungModulus = YoungModulus
#        self.__PoissonRatio = PoissonRatio
        
        Variable("DispX")
        Variable("DispY")        
        
        if ProblemDimension.Get() == "3D": # or ProblemDimension.Get() == "2Dstress" :
            Variable("DispZ")

        self.__parameters = {'Vf':Vf, 'E_f':E_f, 'E_m':E_m, 'nu_f':nu_f, 'nu_m':nu_m, 'angle':angle}   
        
    def GetEngineeringConstants(self):
        Vf = self.__parameters['Vf']
        #carac composites (cf Berthelot)
        #Vf taux de fibres      
        Gf = 0.5*self.__parameters['E_f']/(1+self.__parameters['nu_f']) #shear modulus fibers
        Gm = 0.5*self.__parameters['E_m']/(1+self.__parameters['nu_m']) #shear modulus matrix
        kf = self.__parameters['E_f']/(3*(1-2*self.__parameters['nu_f']))
        km = self.__parameters['E_m']/(3*(1-2*self.__parameters['nu_m'])) #modules de compressibilité
        Kf = kf+Gf/3. ; Km = km+Gm/3. #modules de compression latérale        
        
        EL = self.__parameters['E_f']*Vf + self.__parameters['E_m']*(1-Vf) #loi des mélanges
    
        nuLT = Vf*self.__parameters['nu_f'] +(1-Vf)*self.__parameters['nu_m'] #loi des mélanges
        
        #GLT =  1./(Vf/Gm + (1-Vf)/Gf) #approche simplifiée
        GLT = Gm* (Gf*(1.+Vf) + Gm*(1.-Vf))/(Gf*(1.-Vf) + Gm*(1.+Vf)) #approche exacte
        GTT = Gm*(1. + Vf/(Gm/(Gf-Gm) + (km+7.*Gm/3)/(2*km+8.*Gm/3)*(1-Vf)) ) #approche exacte
        KL = Km + Vf/(1/(kf-km+(Gf-Gm)/3) + (1-Vf)/(km+4/3.*Gm) )
        
        #ET = 1./(Vf/E_f + (1-Vf)/E_m) #simplified approach        
        ET = 2./(0.5/KL + 0.5/GTT + 2*nuLT**2/EL) #exact approach
        nuTT = 0.5*ET/GTT-1 
        
        return {'EX':EL, 'EY':ET, 'EZ':ET, 'GYZ':GTT, 'GXZ':GLT, 'GXY':GLT, 'nuYZ':nuTT, 'nuXZ':nuLT, 'nuXY':nuLT}

    def GetH (self):
        if ProblemDimension.Get() == "2Dstress":
            print('CompositeUD law for 2Dstress is not implemented')
            return NotImplemented
        
        Vf = self.__parameters['Vf']
        #carac composites (cf Berthelot)
        #Vf taux de fibres      
        Gf = 0.5*self.__parameters['E_f']/(1+self.__parameters['nu_f']) #shear modulus fibers
        Gm = 0.5*self.__parameters['E_m']/(1+self.__parameters['nu_m']) #shear modulus matrix
        kf = self.__parameters['E_f']/(3*(1-2*self.__parameters['nu_f']))
        km = self.__parameters['E_m']/(3*(1-2*self.__parameters['nu_m'])) #modules de compressibilité
        Kf = kf+Gf/3. ; Km = km+Gm/3. #modules de compression latérale        
        
        EL = self.__parameters['E_f']*Vf + self.__parameters['E_m']*(1-Vf) #loi des mélanges
    
        nuLT = Vf*self.__parameters['nu_f'] +(1-Vf)*self.__parameters['nu_m'] #loi des mélanges
        
        #GLT =  1./(Vf/Gm + (1-Vf)/Gf) #approche simplifiée
        GLT = Gm* (Gf*(1.+Vf) + Gm*(1.-Vf))/(Gf*(1.-Vf) + Gm*(1.+Vf)) #approche exacte
        GTT = Gm*(1. + Vf/(Gm/(Gf-Gm) + (km+7.*Gm/3)/(2*km+8.*Gm/3)*(1-Vf)) ) #approche exacte
        KL = Km + Vf/(1/(kf-km+(Gf-Gm)/3) + (1-Vf)/(km+4/3.*Gm) )
        
        #ET = 1./(Vf/E_f + (1-Vf)/E_m) #simplified approach        
        ET = 2./(0.5/KL + 0.5/GTT + 2*nuLT**2/EL) #exact approach
        nuTT = 0.5*ET/GTT-1

#         stiffness matrix for unidirectional composites
        
#         compliance matrix for unidirectional composites :
#        S = np.array([[1/EL    , -nuLT/EL, -nuLT/EL, 0    , 0    , 0    ], \
#                      [-nuLT/EL, 1/ET    , -nuTT/ET, 0    , 0    , 0    ], \
#                      [-nuLT/EL, -nuTT/ET, 1/ET    , 0    , 0    , 0    ], \
#                      [0       , 0       , 0       , 1/GTT, 0    , 0    ], \
#                      [0       , 0       , 0       , 0    , 1/GLT, 0    ], \
#                      [0       , 0       , 0       , 0    , 0    , 1/GLT]]) 
#        H1 = linalg.inv(S) #H  = np.zeros((6,6), dtype='object')
   
        if isinstance(EL, float): H = np.zeros((6,6))
        elif isinstance(EL,(np.ndarray,list)): H = np.zeros((6,6,len(EL)))
        else: H = np.zeros((6,6), dtype='object')     
            
        nuTL = nuLT*ET/EL 
        k = 1-nuTT**2 - 2*nuLT*nuTL  - 2*nuLT*nuTT*nuTL
        H[0,0] = EL*(1-nuTT**2)/k 
        H[1,1] = H[2,2] = ET*(1-nuLT*nuTL)/k
        H[0,1] = H[1,0] = H[0,2] = H[2,0] = EL*(nuTT*nuTL+nuTL)/k
        H[1,2] = H[2,1] = ET*(nuLT*nuTL+nuTT)/k
        H[3,3] = GTT ; H[4,4] = H[5,5] = GLT

        if self.__parameters['angle'] is not 0:
            # angle in degree
            angle_pli = self.__parameters['angle']/180.*np.pi
            s = np.sin(angle_pli) ; c = np.cos(angle_pli)
            zero = 0*s ; one = zero+1           
            R_epsilon = np.array([[c**2  , s**2 , zero, zero, zero, s*c      ], \
                                  [s**2  , c**2 , zero, zero, zero, -s*c     ], \
                                  [zero  , zero , one , zero, zero, zero     ], \
                                  [zero  , zero , zero, c   , -s  , zero     ], \
                                  [zero  , zero , zero, s   , c   , zero     ], \
                                  [-2*s*c, 2*s*c, zero, zero, zero, c**2-s**2]])

            if len(R_epsilon.shape) == 3: 
                R_epsilon = np.transpose(R_epsilon,[2,0,1])
                R_sigma_inv = np.transpose(R_epsilon,[0,2,1])
            else: R_sigma_inv = R_epsilon.T
            if len(H.shape) == 3: H = np.rollaxis(H,2,0)
            H = np.matmul(R_sigma_inv, np.matmul(H,R_epsilon))
            if len(H.shape) == 3: H = np.rollaxis(H,0,3)
            
        return H
        
       
