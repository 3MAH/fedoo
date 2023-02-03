#derive de ConstitutiveLaw

from fedoo.core.base import ConstitutiveLaw
from fedoo.core.base import AssemblyBase
import numpy as np
from numpy import linalg


class Spring(ConstitutiveLaw): 
    """
    Simple directional spring connector between nodes or surfaces

    This constitutive Law should be associated with :mod:`fedoo.weakform.InterfaceForce`    

    Parameters
    ----------
    Kx: scalar
        the rigidity along the X direction in material coordinates
    Ky: scalar
        the rigidity along the Y direction in material coordinates
    Kz: scalar
        the rigidity along the Z direction in material coordinates        
    name: str, optional
        The name of the constitutive law
    """
    #Similar to CohesiveLaw but with different rigidity axis and without damage variable
    #Use with WeakForm.InterfaceForce
    def __init__(self, Kx=0, Ky = 0, Kz = 0, name =""):        
        ConstitutiveLaw.__init__(self, name) # heritage        
        self.__parameters = {'Kx':Kx, 'Ky':Ky, 'Kz':Kz}  
        self._InterfaceStress = 0           

    def GetRelativeDisp(self):
        return self.__Delta

    def GetInterfaceStress(self):
        return self._InterfaceStress

    def get_tangent_matrix(self):
        return [[self.__parameters['Kx'], 0, 0], [0, self.__parameters['Ky'], 0], [0,0,self.__parameters['Kz']]]     
    
    def GetK(self):
        return self.__ChangeBasisK(self.get_tangent_matrix())
    
    def __ChangeBasisK(self, K):
        #Change of basis capability for spring type laws on the form : ForceVector = K * DispVector
        if self._ConstitutiveLaw__localFrame is not None:
            #building the matrix to change the basis of the stress and the strain
            B = self._ConstitutiveLaw__localFrame     

            if len(B.shape) == 3:    
                Binv = np.transpose(B, [2,1,0])
                B = np.transpose(B, [1,2,0])
                
            elif len(B.shape) == 2:
                Binv = B.T
            
            dim = len(B)
                
            KB = [[sum([K[i][j]*B[j][k] for j in range(dim)]) for k in range(dim)] for i in range(dim)]     
            K= [[sum([Binv[i][j]*KB[j][k] for j in range(dim)]) for k in range(dim)] for i in range(dim)]

            if dim == 2:
                K[0].append(0) ; K[1].append(0)
                K.append([0, 0, 0])
            
        return K

    def initialize(self, assembly, pb, t0 = 0., nlgeom=False):
       #nlgeom not implemented
       pass

    def update(self,assembly, pb, dtime):            
        #dtime not used for this law
        
        displacement = pb.get_dof_solution()
        if displacement is 0: self._InterfaceStress = self.__Delta = 0
        else:
            op_delta = assembly.space.op_disp() #relative displacement = disp if used with cohesive element
            self.__Delta = [assembly.get_gp_results(op, displacement) for op in op_delta]
        
            self.ComputeInterfaceStress(self.__Delta)        

    # def GetOperartorDelta(self): #operator to get the relative displacement
    #     U, U_vir = get_DispOperator()  
    #     return U 
        
    def ComputeInterfaceStress(self, Delta, dtime = None): 
        #Delta is the relative displacement vector
        K = self.GetK()
        dim = len(Delta)
        self._InterfaceStress = [sum([Delta[j]*K[i][j] for j in range(dim)]) for i in range(dim)] #list of 3 objects        
    


#    def GetStressOperator(self, localFrame=None): # methode virtuel
#    
#        U, U_vir = get_DispOperator()
#        
#        if self._ConstitutiveLaw__localFrame is None:
#            if GetNumberOfDimensions() == "3D":        # tester si contrainte plane ou def plane              
#                return [U[0] * self.__parameters['Kx'], U[1] * self.__parameters['Ky'], U[2] * self.__parameters['Kz']]
#            else:
#                return [U[0] * self.__parameters['Kx'], U[1] * self.__parameters['Ky'], 0]
#        else: 
#            #TODO test if it work in 2D and add the 2D case if needed
#            K = [[self.__parameters['Kx'], 0, 0], [0, self.__parameters['Ky'], 0], [0,0,self.__parameters['Kz']]]
#            K= self._ConstitutiveLaw__ChangeBasisK(K)
#            return [sum([U[j]*K[i][j] for j in range(3)]) for i in range(3)]
