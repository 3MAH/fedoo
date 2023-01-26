"""Not intended for public use, excepted to derive new mechanical constitutivelaw """


#baseclass
import numpy as np
from fedoo.core.base import ConstitutiveLaw

class Mechanical3D(ConstitutiveLaw):  
    """Base class for mechanical constitutive laws."""

    # model of constitutive law for InternalForce Weakform

    def __init__(self, name = ""):
        ConstitutiveLaw.__init__(self,name)
        self._stress = 0 #current stress (pk2 if nlgeom) at integration points
        self._grad_disp = 0 #current grad_disp at integration points
        
    def get_pk2(self):
        return NotImplemented
        
    def get_kirchhoff(self):
        return NotImplemented        
    
    def get_cauchy(self):
        return NotImplemented        
    
    def get_strain(self):
        return NotImplemented
           
    def get_statev(self):
        return NotImplemented

    def get_wm(self):
        return NotImplemented

    def get_stress(self, **kargs): #same as GetPKII (used for small def)
        return NotImplemented
    
    def get_disp_grad(self): #use if nlgeom == True
        return NotImplemented
    
    def get_tangent_matrix(self): #Tangent Matrix in local coordinate system (no change of basis)
        return NotImplemented

    def get_tangent_matrix_2Dstress(self): #Tangent Matrix in local coordinate system (no change of basis)
        return NotImplemented
    
    def GetH(self, **kargs): #Tangent Matrix in global coordinate system (apply change of basis)        
        if kargs.get('dimension') == "2Dstress" or self._dimension == "2Dstress":
            H = self.get_tangent_matrix_2Dstress()
            if H is NotImplemented:
                H = self.__ApplyChangeOfBasis(self.get_tangent_matrix())
                return [[H[i][j]-H[i][2]*H[j][2]/H[2][2] if j in [0,1,3] else 0 for j in range(6)] \
                        if i in [0,1,3] else [0,0,0,0,0,0]for i in range(6)] 
            else: 
                return self.__ApplyChangeOfBasis(H)
                    
        return self.__ApplyChangeOfBasis(self.get_tangent_matrix())
    
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
    


        

    
