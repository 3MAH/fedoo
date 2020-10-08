from fedoo.libWeakForm.WeakForm   import *
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.BeamStrainOperator import GetBeamStrainOperator
from fedoo.libUtil.Variable import Variable
from fedoo.libUtil.Dimension import ProblemDimension

class Beam(WeakForm):
    def __init__(self, CurrentConstitutiveLaw, Section, Jx, Iyy, Izz, k=0, ID = ""):
        #k: shear shape factor
        
        if isinstance(CurrentConstitutiveLaw, str):
            CurrentConstitutiveLaw = ConstitutiveLaw.GetAll()[CurrentConstitutiveLaw]

        if ID == "":
            ID = CurrentConstitutiveLaw.GetID()
            
        WeakForm.__init__(self,ID)

        Variable("DispX") 
        Variable("DispY")            
        if ProblemDimension.Get() == '3D':
            Variable("DispZ")   
            Variable("RotX") #torsion rotation 
            Variable("RotY")   
            Variable("RotZ")
            Variable.SetVector('Disp' , ('DispX', 'DispY', 'DispZ') , 'global')
            Variable.SetVector('Rot' , ('RotX', 'RotY', 'RotZ') , 'global')            
        elif ProblemDimension.Get() == '2Dplane':
            Variable("RotZ")
            Variable.SetVector('Disp' , ('DispX', 'DispY') )            
        elif ProblemDimension.Get() == '2Dstress':
            assert 0, "No 2Dstress model for a beam kinematic. Choose '2Dplane' instead."
                          
        self.__ConstitutiveLaw = CurrentConstitutiveLaw
        self.__parameters = {'Section': Section, 'Jx': Jx, 'Iyy':Iyy, 'Izz':Izz, 'k':k}        
    
    def GetDifferentialOperator(self, localFrame):
        E  = self.__ConstitutiveLaw.GetYoungModulus()
        nu = self.__ConstitutiveLaw.GetPoissonRatio()       
        G = E/(1+nu)/2
        
        eps, eps_vir = GetBeamStrainOperator()           
        beamShearStifness = self.__parameters['k'] * G * self.__parameters['Section']

        Ke = [E*self.__parameters['Section'], beamShearStifness, beamShearStifness, G*self.__parameters['Jx'], E*self.__parameters['Iyy'], E*self.__parameters['Izz']]
        return sum([eps_vir[i] * eps[i] * Ke[i] for i in range(6)])                

    
    def GetGeneralizedStress(self):
        #only for post treatment

        eps, eps_vir = GetBeamStrainOperator()
        E  = self.__ConstitutiveLaw.GetYoungModulus()
        nu = self.__ConstitutiveLaw.GetPoissonRatio()       
        G = E/(1+nu)/2
        beamShearStifness = self.__parameters['k'] * G * self.__parameters['Section']

        Ke = [E*self.__parameters['Section'], beamShearStifness, beamShearStifness, G*self.__parameters['Jx'], E*self.__parameters['Iyy'], E*self.__parameters['Izz']]
        return [eps[i] * Ke[i] for i in range(6)]


def BernoulliBeam(CurrentConstitutiveLaw, Section, Jx, Iyy, Izz, ID = ""):
    #same as beam with k=0 (no shear effect)
    return Beam(CurrentConstitutiveLaw, Section, Jx, Iyy, Izz, k=0, ID = ID)

# class BernoulliBeam(WeakForm):
#     def __init__(self, CurrentConstitutiveLaw, Section, Jx, Iyy, Izz, ID = ""):
#         if isinstance(CurrentConstitutiveLaw, str):
#             CurrentConstitutiveLaw = ConstitutiveLaw.GetAll()[CurrentConstitutiveLaw]

#         if ID == "":
#             ID = CurrentConstitutiveLaw.GetID()
            
#         WeakForm.__init__(self,ID)

#         Variable("DispX") 
#         Variable("DispY")     
        
#         if ProblemDimension.Get() == '3D':
#             Variable("DispZ")   
#             Variable("RotX") #torsion rotation   
#             Variable("RotY") #flexion   
#             Variable("RotZ") #flexion   
#             Variable.SetVector('Disp' , ('DispX', 'DispY', 'DispZ') , 'global')
#             Variable.SetVector('Rot' , ('RotX', 'RotY', 'RotZ') , 'global')
#         elif ProblemDimension.Get() == '2Dplane':
#             Variable("RotZ")
#             # Variable.SetDerivative('DispY', 'RotZ') #only valid with Bernoulli model       
#             Variable.SetVector('Disp' , ('DispX', 'DispY') )            
#         elif ProblemDimension.Get() == '2Dstress':
#             assert 0, "No 2Dstress model for a beam kinematic. Choose '2Dplane' instead."
                  
#         self.__ConstitutiveLaw = CurrentConstitutiveLaw
#         self.__parameters = {'Section': Section, 'Jx': Jx, 'Iyy':Iyy, 'Izz':Izz}
    
#     def GetDifferentialOperator(self, localFrame):
#         E  = self.__ConstitutiveLaw.GetYoungModulus()
#         nu = self.__ConstitutiveLaw.GetPoissonRatio()       
#         G = E/(1+nu)/2
        
#         eps, eps_vir = GetBernoulliBeamStrainOperator()           

#         Ke = [E*self.__parameters['Section'], 0, 0, G*self.__parameters['Jx'], E*self.__parameters['Iyy'], E*self.__parameters['Izz']]
#         return sum([eps_vir[i] * eps[i] * Ke[i] for i in range(6)])                

    
#     def GetGeneralizedStress(self):
#         #only for post treatment

#         eps, eps_vir = GetBernoulliBeamStrainOperator()
#         E  = self.__ConstitutiveLaw.GetYoungModulus()
#         nu = self.__ConstitutiveLaw.GetPoissonRatio()       
#         G = E/(1+nu)/2
#         Ke = [E*self.__parameters['Section'], 0, 0, G*self.__parameters['Jx'], E*self.__parameters['Iyy'], E*self.__parameters['Izz']]
        
#         return [eps[i] * Ke[i] for i in range(6)]



    


    
