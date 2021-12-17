#derive de ConstitutiveLaw
#simcoon compatible

from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libConstitutiveLaw.ConstitutiveLaw_ElasticAnisotropic import ElasticAnisotropic
from fedoo.libUtil.ModelingSpace      import Variable, GetDimension

import scipy as sp

class ElasticOrthotropic(ElasticAnisotropic):
    """
    Linear Orthotropic constitutive law defined from the engineering coefficients in local material coordinates.  
    
    The constitutive Law should be associated with :mod:`fedoo.libWeakForm.InternalForce`    
    
    Parameters
    ----------
    EX: scalars or arrays of gauss point values
        Young modulus along the X direction
    EY: scalars or arrays of gauss point values
        Young modulus along the Y direction
    EZ: scalars or arrays of gauss point values
        Young modulus along the Z direction
    GYZ, GXZ, GXY: scalars or arrays of gauss point values
        Shear modulus 
    nuYZ, nuXZ, nuXY: scalars or arrays of gauss point values
        Poisson's ratio
    ID: str, optional
        The ID of the constitutive law
    """
    def __init__(self, EX, EY, EZ, GYZ, GXZ, GXY, nuYZ, nuXZ, nuXY, ID=""):
        ConstitutiveLaw.__init__(self, ID) # heritage
#        self.__YoungModulus = YoungModulus
#        self.__PoissonRatio = PoissonRatio
        
        Variable("DispX")
        Variable("DispY")        
        
        if GetDimension() == "3D": 
            Variable("DispZ")

        self.__parameters = {'EX':EX, 'EY':EY, 'EZ':EZ, 'GYZ':GYZ, 'GXZ':GXZ, 'GXY':GXY, 'nuYZ':nuYZ, 'nuXZ':nuXZ, 'nuXY':nuXY}
        
    def GetEngineeringConstants(self):
        """
        Return a dict containing the engineering constants
        """
        return self.__parameters
    
    def GetTangentMatrix(self): 
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
    
   