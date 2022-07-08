#baseclass
import numpy as np
from copy import deepcopy

class ConstitutiveLaw:

    __dic = {}

    def __init__(self, name = ""):
        assert isinstance(name, str) , "An name must be a string" 
        self.__name = name
        self.__localFrame = None
        self._dimension = None #str or None to specify a space and associated model (for instance "2Dstress" for plane stress)

        ConstitutiveLaw.__dic[self.__name] = self        

    def SetLocalFrame(self, localFrame):
        self.__localFrame = localFrame

    def GetLocalFrame(self):
        return self.__localFrame 
    
    def reset(self): 
        #function called to restart a problem (reset all internal variables)
        pass
    
    def set_start(self):  
        #function called when the time is increased. Not used for elastic laws
        pass
    
    def to_start(self):
        #function called if the time step is reinitialized. Not used for elastic laws
        pass

    def initialize(self, assembly, pb, t0 = 0., nlgeom=False):
        #function called to initialize the constutive law 
        pass
    
    def update(self,assembly, pb, dtime):
        #function called to update the state of constitutive law 
        pass
    
    def copy(self, new_id = ""):
        """
        Return a raw copy of the constitutive law without keeping current internal variables.

        Parameters
        ----------
        new_id : TYPE, optional
            The name of the created constitutive law. The default is "".

        Returns
        -------
        The copy of the constitutive law
        """
        new_cl = deepcopy(self)        
        new_cl._ConstitutiveLaw__name = new_id
        self.__dic[new_id] = new_cl
        new_cl.reset()
        return new_cl
   
    @staticmethod
    def get_all():
        return ConstitutiveLaw.__dic
    
    @property
    def name(self):
        return self.__name




class Mechanical3D(ConstitutiveLaw):  
    # model of constitutive law for InternalForce Weakform

    def __init__(self, name = ""):
        ConstitutiveLaw.__init__(self,name)
        self._stress = 0 #current stress (pk2 if nlgeom) at integration points
        self._grad_disp = 0 #current grad_disp at integration points
        
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

    def GetWm(self):
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
        if kargs.get('dimension') == "2Dstress" or self._dimension == "2Dstress":
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
    

class ListConstitutiveLaw(ConstitutiveLaw):
    def __init__(self, list_constitutivelaw, name =""):    
        ConstitutiveLaw.__init__(self,name)   
        
        self.__list_constitutivelaw = set(list_constitutivelaw) #remove duplicated cl
    
    def initialize(self, assembly, pb, t0=0., nlgeom=False):
        for cl in self.__list_constitutivelaw:
            cl.initialize(assembly, pb, t0)

    def InitTimeIncrement(self, assembly, pb, dtime):
        for cl in self.__list_constitutivelaw:
            cl.InitTimeIncrement(assembly, pb, dtime)
    
    def update(self, assembly, pb, dtime):        
        for cl in self.__list_constitutivelaw:
            cl.update(assembly, pb, dtime)
    
    def set_start(self):  
        for cl in self.__list_constitutivelaw:
            cl.set_start()
    
    def to_start(self):
        for cl in self.__list_constitutivelaw:
            cl.to_start()

    def reset(self):
        for cl in self.__list_constitutivelaw:
            cl.reset()
    
    def copy(self):
        #function to copy a weakform at the initial state
        raise NotImplementedError()

        
    # def initializeConstitutiveLaw(self, assembly, pb, t0=0.):
    #     if hasattr(self,'nlgeom'): nlgeom = self.nlgeom
    #     else: nlgeom=False
    #     constitutivelaw = self.GetConstitutiveLaw()
        
    #     if constitutivelaw is not None:
    #         if isinstance(constitutivelaw, list):
    #             for cl in constitutivelaw:
    #                 cl.initialize(assembly, pb, t0, nlgeom)
    #         else:
    #             constitutivelaw.initialize(assembly, pb, t0, nlgeom)
    
    # def updateConstitutiveLaw(self,assembly, pb, dtime):   
    #     if hasattr(self,'nlgeom'): nlgeom = self.nlgeom
    #     else: nlgeom=False
    #     constitutivelaw = self.GetConstitutiveLaw()
        
    #     if constitutivelaw is not None:
    #         if isinstance(constitutivelaw, list):
    #             for cl in constitutivelaw:
    #                 cl.update(assembly, pb, dtime, nlgeom)
    #         else:
    #             constitutivelaw.update(assembly, pb, dtime, nlgeom)

        

    # def resetConstitutiveLaw(self):
    #     constitutivelaw = self.GetConstitutiveLaw()
        
    #     if constitutivelaw is not None:
    #         if isinstance(constitutivelaw, list):
    #             for cl in constitutivelaw:
    #                 cl.reset()
    #         else:
    #             constitutivelaw.reset()


class ThermalProperties(ConstitutiveLaw):  
    
    def __init__(self, thermal_conductivity, specific_heat, density, name =""):
        ConstitutiveLaw.__init__(self, name)
        if np.isscalar(thermal_conductivity): 
            self.thermal_conductivity = [[thermal_conductivity,0,0], [0,thermal_conductivity,0], [0,0,thermal_conductivity]]
        else: 
            self.thermal_conductivity = thermal_conductivity
        
        self.specific_heat = specific_heat
        self.density = density
        
        
def get_all():
    return ConstitutiveLaw.get_all()


