# -*- coding: utf-8 -*-
import numpy as np


#simcoon compatible
class arrayStressTensor(np.ndarray):

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        if len(input_array) != 6: raise NameError('lenght for arrayStressTensor object must be 6')
        obj = np.asarray(input_array).view(cls)
        
        # # add the new attribute to the created instance
        # obj.info = info
        # Finally, we must return the newly created object:
        return obj

    def vtkFormat(self):
        """
        Return a array adapted to export symetric tensor data in a vtk file
        See the Util.ExportData class for more details
        """
        return np.vstack((self[0:4].reshape(4,-1), self[5], self[4])).astype(float).T 
    
    def GetFullTensor(self): 
        return np.array([[self[0],self[3],self[4]], [self[3], self[1], self[5]], [self[4], self[5], self[2]]])  

    def vonMises(self):
        """
        Return the vonMises stress
        """
        return np.sqrt( 0.5 * ((self[0]-self[1])**2 + (self[1]-self[2])**2 + (self[0]-self[2])**2 \
                         + 6 * (self[3]**2 + self[4]**2 + self[5]**2) ) )            


    # def __array_finalize__(self, obj):
    #     # see InfoArray.__array_finalize__ for comments
    #     if obj is None: return
    #     self.info = getattr(obj, 'info', None)

#simcoon compatible
class listStressTensor(list):
    
    def __init__(self, l):
        if len(l) != 6: raise NameError('list lenght for listStressTensor object must be 6')
        list.__init__(self,l)
    
    def vtkFormat(self):
        """
        Return a array adapted to export symetric tensor data in a vtk file
        See the Util.ExportData class for more details
        """
        try: 
            return np.vstack([self[i] for i in [0,1,2,3,5,4] ]).astype(float).T 
        except:
            self.fill_zeros()
            return np.vstack([self[i] for i in [0,1,2,3,5,4] ]).astype(float).T 
        
    def GetFullTensor(self): 
        return np.array([[self[0],self[3],self[4]], [self[3], self[1], self[5]], [self[4], self[5], self[2]]])  

    def asarray(self):
        return np.array(self)
    
    def convertPiolaToCauchy(self, GradDeformedCoordinates): 
        PiolaKStress = self.GetFullTensor().transpose(2,0,1)          
        
    #            GradX = [[Assembly.get_all()['Assembling'].GetNodeResult(GradOp[i][j], Mesh.get_all()[meshID].nodes.T.reshape(-1)+Problem.GetDisp()) for j in range(3)] for i in range(3)] 
        GradX = np.transpose(np.array(GradDeformedCoordinates)[:,:,:],(2,0,1))
        DetGradX = np.linalg.det(GradX)
    
        CauchyStress =  (1/DetGradX).reshape(-1,1,1)*np.matmul(GradX,np.matmul(PiolaKStress,GradX.transpose(0,2,1)))
        return listStressTensor([CauchyStress[:,0,0],CauchyStress[:,1,1],CauchyStress[:,2,2],CauchyStress[:,0,1],CauchyStress[:,0,2],CauchyStress[:,1,2]])

    def vonMises(self):
        """
        Return the vonMises stress
        """
        return np.sqrt( 0.5 * ((self[0]-self[1])**2 + (self[1]-self[2])**2 + (self[0]-self[2])**2 \
                         + 6 * (self[3]**2 + self[4]**2 + self[5]**2) ) )
    def deviatoric(self):
        """
        Return the deviatoric part of the Stress Tensor using void form
        """
        return listStressTensor([ 2/3*self[0]-1/3*self[1]-1/3*self[2], \
                                 -1/3*self[0]+2/3*self[1]-1/3*self[2], \
                                 -1/3*self[0]-1/3*self[1]+2/3*self[2], \
                                  self[3], self[4], self[5]])

    def hydrostatic(self):
        """
        Return the hydrostatic part of the Stress Tensor using void form
        """
        traceStress = (1/3)*(self[0]+self[1]+self[2])
        return listStressTensor([traceStress, traceStress, traceStress, 0, 0, 0])

    def GetPrincipalStress(self): 
        """
        Return the principal stress and principal directions for all given points
        Return PrincipalStress, PrincipalDirection
        The line of principalStress are the values of the principal stresses for all points.
        PrincipalDirection[i] is the principal direction associated to the ith principal stress.
        The line of PrincipalDirection[i] are component of the vector, for all points.
        """
        FullStressTensor = self.GetFullTensor().transpose(2,0,1)  
        PrincipalStress, PrincipalDirection = np.linalg.eig(FullStressTensor)
        return PrincipalStress, PrincipalDirection.transpose(2,0,1)

    def toStrain(self):
        return listStrainTensor(self[:3] + [self[i]*2 for i in [3,4,5]])
        
    def toStress(self):
        return self
    
    def fill_zeros(self):
        for i in range(6):
            if np.isscalar(self[i]) != True:
                N = len(self[i]) #number of stress values
                break
        for i in range(6):
            if self[i] is 0:
                self[i] = np.zeros(N)
            

    def Convert(self,assemb, ConvertFrom=None, ConvertTo='GaussPoint'):
        return listStressTensor([assemb.ConvertData(S, ConvertFrom, ConvertTo) for S in self])

#simcoon compatible    
class listStrainTensor(list):
    
    def __init__(self, l):
        if len(l) != 6: raise NameError('list lenght for listStrainTensor object must be 6')
        list.__init__(self,l)
    
    def vtkFormat(self):
        """
        Return a array adapted to export symetric tensor data in a vtk file
        See the Util.ExportData class for more details
        """

        try: 
            return np.vstack(self[:3] + [self[i]/2 for i in [3,5,4]]).astype(float).T
        except:
            self.fill_zeros()
            return np.vstack(self[:3] + [self[i]/2 for i in [3,5,4]]).astype(float).T
        
    def asarray(self):
        try:
            return np.array(self)
        except:
            self.fill_zeros()
            return np.array(self)

    def GetFullTensor(self):
        return np.array([[self[0],self[3]/2,self[4]/2], [self[3]/2, self[1], self[5]/2], [self[4]/2, self[5]/2, self[2]]])
        
    def GetPrincipalStrain(self): 
        """
        Return the principal strain and principal directions for all given points
        Return PrincipalStrain, PrincipalDirection
        The line of principalStrain are the values of the principal strains for all points.
        PrincipalDirection[i] is the principal direction associated to the ith principal strain.
        The line of PrincipalDirection[i] are component of the vector, for all points.
        """
        FullStrainTensor = self.GetFullTensor().transpose(2,0,1)  
        PrincipalStrain, PrincipalDirection = np.linalg.eig(FullStrainTensor)
        return PrincipalStrain, PrincipalDirection.transpose(2,0,1)
    
    def toStress(self):
        return listStrainTensor(self[:3] + [self[i]/2 for i in [3,4,5]])
        
    def toStrain(self):
        return self

    def fill_zeros(self):
        for i in range(6):
            if np.isscalar(self[i]) != True:
                N = len(self[i]) #number of stress values
                break
        for i in range(6):
            if self[i] is 0:
                self[i] = np.zeros(N)

    def Convert(self,assemb, ConvertFrom=None, ConvertTo='GaussPoint'):
        return listStrainTensor([assemb.ConvertData(S, ConvertFrom, ConvertTo) for S in self])
