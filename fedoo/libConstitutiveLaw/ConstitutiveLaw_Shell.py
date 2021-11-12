#derive de ConstitutiveLaw
#compatible with the simcoon strain and stress notation

from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.StrainOperator import *
from fedoo.libUtil.ModelingSpace      import Variable, GetDimension, Vector
from fedoo.libUtil.PostTreatement import listStressTensor, listStrainTensor

import numpy as np

class ShellHomogeneous(ConstitutiveLaw):
    
    def __init__(self, MatConstitutiveLaw, thickness, k=1, ID=""):        
        assert GetDimension() == '3D', "No 2D model for a shell kinematic. Choose '3D' problem dimension."

        if isinstance(MatConstitutiveLaw, str):
            MatConstitutiveLaw = ConstitutiveLaw.GetAll()[MatConstitutiveLaw]


        ConstitutiveLaw.__init__(self, ID) # heritage

        Variable("DispX") 
        Variable("DispY")            
        Variable("DispZ")   
        Variable("RotX") #torsion rotation 
        Variable("RotY")   
        Variable("RotZ")
        Vector('Disp' , ('DispX', 'DispY', 'DispZ'))
        Vector('Rot' , ('RotX', 'RotY', 'RotZ'))     

        self.__thickness = thickness
        self.__k = k 
        self.__material = MatConstitutiveLaw
        self.__GeneralizedStrain = None
        self.__GeneralizedStress = None
    
    def GetMaterial(self):
        return self.__material
        
    def GetThickness(self):
        return self.__thickness
    
    def Get_k(self):
        return self.__k
          
    def GetShellRigidityMatrix(self):
        Hplane = self.__material.GetH(pbdim="2Dstress") #membrane rigidity matrix with plane stress assumption
        Hplane = np.array([[Hplane[i][j] for j in [0,1,3]] for i in[0,1,3]], dtype='object')
        Hshear = self.__material.GetH()
        Hshear = np.array([[Hshear[i][j] for j in [4,5]] for i in[4,5]], dtype='object')
        
        H = np.zeros((8,8), dtype='object')  
        H[:3,:3] = self.__thickness*Hplane #Membrane
        H[3:6,3:6] = (self.__thickness**3/12) * Hplane #Flexual rigidity matrix
        H[6:8,6:8] = Hshear
        
        return H
        
    def GetShellRigidityMatrix_RI(self):
        #only shear component are given for reduce integration part
        Hshear = self.__material.GetH()
        Hshear = np.array([[Hshear[i][j] for j in [4,5]] for i in[4,5]], dtype='object')                
        
        return Hshear
               
    def GetShellRigidityMatrix_FI(self):
        #membrane and flexural component are given for full integration part
        Hplane = self.__material.GetH(pbdim="2Dstress") #membrane rigidity matrix with plane stress assumption
        Hplane = np.array([[Hplane[i][j] for j in [0,1,3]] for i in[0,1,3]], dtype='object')        
        
        H = np.zeros((6,6), dtype='object')  
        H[:3,:3] = self.__thickness*Hplane #Membrane
        H[3:6,3:6] = (self.__thickness**3/12) * Hplane #Flexual rigidity matrix
        
        return H    
                      
    def Update(self,assembly, pb, dtime, nlgeom):
        # disp = pb.GetDisp()
        # rot = pb.GetRot()
        U = pb.GetDoFSolution()
        if U is 0: 
            self.__GeneralizedStrain = 0
            self.__GeneralizedStress = 0                        
        else:
            GeneralizedStrainOp = assembly.GetWeakForm().GetGeneralizedStrainOperator()
            GeneralizedStrain = [0 if op is 0 else assembly.GetGaussPointResult(op, U) for op in GeneralizedStrainOp]
       
            H = self.GetShellRigidityMatrix()
        
            self.__GeneralizedStress = [sum([GeneralizedStrain[j]*assembly.ConvertData(H[i][j]) for j in range(8)]) for i in range(8)] #H[i][j] are converted to gauss point excepted if scalar
            self.__GeneralizedStrain = GeneralizedStrain
    
    def GetStrain(self, position = 'top'):
        z = position
        if z == 'top': z = self.__thickness/2
        elif z == 'bottom': z = -self.__thickness/2
        
        Strain = listStrainTensor([0 for i in range(6)])
        Strain[0] = self.__GeneralizedStrain[0] + z*self.__GeneralizedStrain[4] #epsXX -> membrane and bending
        Strain[1] = self.__GeneralizedStrain[1] - z*self.__GeneralizedStrain[3] #epsYY -> membrane and bending
        Strain[3] = self.__GeneralizedStrain[2] #2epsXY
        Strain[4:6] = self.__GeneralizedStrain[6:8] #2epsXZ and 2epsYZ -> shear
        
        return Strain
    
        
    def GetStress(self, position = 'top'):      
        Strain = self.GetStrain(position)
        Hplane = self.__material.GetH(pbdim="2Dstress") #membrane rigidity matrix with plane stress assumption
        Stress = [sum([0 if Strain[j] is 0 else Strain[j]*Hplane[i][j] for j in range(4)]) for i in range(4)] #SXX, SYY, SXY (SZZ should be = 0)
        Hshear = self.__material.GetH()                       
        Stress += [sum([0 if Strain[j] is 0 else Strain[j]*Hshear[i][j] for j in [4,5]]) for i in [4,5]] #SXX, SYY, SXY (SZZ should be = 0)
        
        return listStressTensor(Stress)
    
        
    
    
    
#     # def ComputeStrain(self, assembly, pb, nlgeom, type_output='GaussPoint'):
#     #     displacement = pb.GetDoFSolution()                
#     #     if displacement is 0: 
#     #         return 0 #if displacement = 0, Strain = 0
#     #     else:
#     #         return assembly.GetStrainTensor(displacement, type_output)  
    
    
#     def Update(self,assembly, pb, dtime, nlgeom):
#         displacement = pb.GetDoFSolution()
        
#         if displacement is 0: 
#             self.__currentGradDisp = 0
#             self.__currentSigma = 0                        
#         else:
#             self.__currentGradDisp = assembly.GetGradTensor(displacement, "GaussPoint")
#             GradValues = self.__currentGradDisp
#             if nlgeom == False:
#                 Strain  = [GradValues[i][i] for i in range(3)] 
#                 Strain += [GradValues[0][1] + GradValues[1][0], GradValues[0][2] + GradValues[2][0], GradValues[1][2] + GradValues[2][1]]
#             else:            
#                 Strain  = [GradValues[i][i] + 0.5*sum([GradValues[k][i]**2 for k in range(3)]) for i in range(3)] 
#                 Strain += [GradValues[0][1] + GradValues[1][0] + sum([GradValues[k][0]*GradValues[k][1] for k in range(3)])] 
#                 Strain += [GradValues[0][2] + GradValues[2][0] + sum([GradValues[k][0]*GradValues[k][2] for k in range(3)])]
#                 Strain += [GradValues[1][2] + GradValues[2][1] + sum([GradValues[k][1]*GradValues[k][2] for k in range(3)])]
#             TotalStrain = listStrainTensor(Strain)
                
#             self.__currentStrain = TotalStrain                            
       
#             H = self.__ChangeBasisH(self.GetH())
        
#             self.__currentSigma = listStressTensor([sum([TotalStrain[j]*assembly.ConvertData(H[i][j]) for j in range(6)]) for i in range(6)]) #H[i][j] are converted to gauss point excepted if scalar
#             # self.__currentSigma = self.GetStress(TotalStrain) #compute the total stress in self.__currentSigma
       
        
       
#     def GetStressFromStrain(self, StrainTensor):     
#         H = self.__ChangeBasisH(self.GetH())
        
#         sigma = listStressTensor([sum([StrainTensor[j]*H[i][j] for j in range(6)]) for i in range(6)])

#         return sigma # list of 6 objets 
    
     
                        
        