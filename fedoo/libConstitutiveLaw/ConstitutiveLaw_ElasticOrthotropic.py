#derive de ConstitutiveLaw
#simcoon compatible

from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libConstitutiveLaw.ConstitutiveLaw_ElasticAnisotropic import ElasticAnisotropic
from fedoo.libUtil.Variable       import *
from fedoo.libUtil.Dimension      import *

import scipy as sp

class ElasticOrthotropic(ElasticAnisotropic):
    def __init__(self, EX, EY, EZ, GYZ, GXZ, GXY, nuYZ, nuXZ, nuXY, ID=""):
        ConstitutiveLaw.__init__(self, ID) # heritage
#        self.__YoungModulus = YoungModulus
#        self.__PoissonRatio = PoissonRatio
        
        Variable("DispX")
        Variable("DispY")        
        
        if ProblemDimension.Get() == "3D": 
            Variable("DispZ")

        self.__parameters = {'EX':EX, 'EY':EY, 'EZ':EZ, 'GYZ':GYZ, 'GXZ':GXZ, 'GXY':GXY, 'nuYZ':nuYZ, 'nuXZ':nuXZ, 'nuXY':nuXY}
        
    def GetEngineeringConstants(self):
        return self.__parameters
    
    def GetH (self):
        #need to be checked
        if ProblemDimension.Get() == "2Dstress":
            print('ElasticOrthotropic law for 2Dstress is not implemented')
            return NotImplemented
        
#        S = sp.array([[1/EX    , -nuXY/EX, -nuXZ/EX, 0    , 0    , 0    ], \
#                      [-nuXY/EX, 1/EY    , -nuYZ/EY, 0    , 0    , 0    ], \
#                      [-nuXZ/EX, -nuYZ/EY, 1/EZ    , 0    , 0    , 0    ], \
#                      [0       , 0       , 0       , 1/GXY, 0    , 0    ], \
#                      [0       , 0       , 0       , 0    , 1/GXZ, 0    ], \
#                      [0       , 0       , 0       , 0    , 0    , 1/GYZ]])                  
#        H = linalg.inv(S) #H  = sp.zeros((6,6), dtype='object')

        for key in self.__parameters: 
            temporary = self.__parameters[key]
            exec(key + '= temporary')
        
        if isinstance(EX, float): H = sp.empty((6,6))
        elif isinstance(EX,(sp.ndarray,list)): H = sp.zeros((6,6,len(EX)))
        else: H = sp.zeros((6,6), dtype='object')
        
        nuYX = nuXY*EY/EX ; nuZX = nuXZ*EZ/EX ; nuZY = nuYZ*EZ/EY
        k = 1-nuYZ*nuZY - nuXY*nuYX - nuXZ*nuZX - nuXY*nuYZ*nuZX - nuYX*nuZY*nuXZ
        H[0,0] = EX*(1-nuYZ*nuZY)/k ; H[1,1] = EY*(1-nuXZ*nuZX)/k ; H[2,2] = EZ*(1-nuXY*nuYX)/k
        H[0,1] = H[1,0] = EX*(nuYZ*nuZX+nuYX)/k
        H[0,2] = H[2,0] = EX*(nuYX*nuZY+nuZX)/k
        H[1,2] = H[2,1] = EY*(nuXY*nuZX+nuZY)/k
        H[3,3] = GXY ; H[4,4] = GXZ ; H[5,5] = GYZ
        
        return H
    
   