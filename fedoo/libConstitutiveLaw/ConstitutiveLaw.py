#baseclass
import numpy as np
from fedoo.libUtil.ModelingSpace import GetDimension

class ConstitutiveLaw:

    __dic = {}

    def __init__(self, ClID = ""):
        assert isinstance(ClID, str) , "An ID must be a string" 
        self.__ID = ClID
        self.__localFrame = None

        ConstitutiveLaw.__dic[self.__ID] = self

    def GetID(self):
        return self.__ID

    def SetLocalFrame(self, localFrame):
        self.__localFrame = localFrame

    def GetLocalFrame(self):
        return self.__localFrame 
    
    def Reset(self): 
        #function called to restart a problem (reset all internal variables)
        pass
    
    def NewTimeIncrement(self):  
        #function called when the time is increased. Not used for elastic laws
        pass
    
    def ResetTimeIncrement(self):
        #function called if the time step is reinitialized. Not used for elastic laws
        pass

    def Initialize(self, assembly, pb, initialTime = 0., nlgeom=True):
        #function called to initialize the constutive law 
        pass
    
    def Update(self,assembly, pb, dtime, nlgeom):
        #function called to update the state of constitutive law 
        pass
   
    @staticmethod
    def GetAll():
        return ConstitutiveLaw.__dic




class Mechanical3D(ConstitutiveLaw):  
    # model of constitutive law for InternalForce Weakform
       
    def GetPKII(self):
        return NotImplemented
        
    def GetKirchhoff(self):
        return NotImplemented        
    
    def GetCauchy(self):
        return NotImplemented        
    
    def GetStrain(self):
        return NotImplemented
           
    def GetStatev(self):
        return NotImplemented

    def GetStress(self, **kargs): #same as GetPKII (used for small def)
        return NotImplemented
    
    def GetCurrentGradDisp(self): #use if nlgeom == True
        return NotImplemented
    
    def GetTangentMatrix(self): #Tangent Matrix in local coordinate system (no change of basis)
        return NotImplemented

    def GetTangentMatrix_2Dstress(self): #Tangent Matrix in local coordinate system (no change of basis)
        return NotImplemented
    
    def GetH(self, **kargs): #Tangent Matrix in global coordinate system (apply change of basis)        
        if kargs.get('pbdim', GetDimension()) == "2Dstress":
            H = self.GetTangentMatrix_2Dstress()
            if H is NotImplemented:
                H = self.__ApplyChangeOfBasis(self.GetTangentMatrix())
                return [[H[i][j]-H[i][2]*H[j][2]/H[2][2] if j in [0,1,3] else 0 for j in range(6)] \
                        if i in [0,1,3] else [0,0,0,0,0,0]for i in range(6)] 
            else: 
                return self.__ApplyChangeOfBasis(H)
                    
        return self.__ApplyChangeOfBasis(self.GetTangentMatrix())
    
    def __ApplyChangeOfBasis(self, H):        
        #Change of basis capability for laws on the form : StressTensor = H * StrainTensor
        #StressTensor and StrainTensor are column vectors based on the voigt notation 
        if self._ConstitutiveLaw__localFrame is not None:
            localFrame = self._ConstitutiveLaw__localFrame
            #building the matrix to change the basis of the stress and the strain
#            theta = np.pi/8
#            np.array([[np.cos(theta),np.sin(theta),0], [-np.sin(theta),np.cos(theta),0], [0,0,1]]) 
            R_epsilon = np.empty((len(localFrame), 6,6))
            R_epsilon[:,  :3,  :3] = localFrame**2 
            R_epsilon[:,  :3, 3:6] = localFrame[:,:,[0,2,1]]*localFrame[:,:,[1,0,2]]
            R_epsilon[:, 3:6,  :3] = 2*localFrame[:,[0,2,1]]*localFrame[:,[1,0,2]] 
            R_epsilon[:, 3:6, 3:6] = localFrame[:,[[0],[2],[1]], [0,2,1]]*localFrame[:,[[1],[0],[2]],[1,0,2]] + localFrame[:,[[1],[0],[2]],[0,2,1]]*localFrame[:,[[0],[2],[1]],[1,0,2]] 
            R_sigma_inv = R_epsilon.transpose(0,2,1)    # np.transpose(R_epsilon,[0,2,1])        
            
            if len(H.shape) == 3: H = np.rollaxis(H,2,0)
            H = np.matmul(R_sigma_inv, np.matmul(H,R_epsilon))
            if len(H.shape) == 3: H = np.rollaxis(H,0,3)  
            
        return H
    

class ThermalProperties(ConstitutiveLaw):  
    
    def __init__(self, thermal_conductivity, specific_heat, density, ID=""):
        ConstitutiveLaw.__init__(self, ID)
        if np.isscalar(thermal_conductivity): 
            self.thermal_conductivity = [[thermal_conductivity,0,0], [0,thermal_conductivity,0], [0,0,thermal_conductivity]]
        else: 
            self.thermal_conductivity = thermal_conductivity
        
        self.specific_heat = specific_heat
        self.density = density
        
        
def GetAll():
    return ConstitutiveLaw.GetAll()


