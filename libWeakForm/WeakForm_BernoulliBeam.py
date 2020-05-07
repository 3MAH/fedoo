from fedoo.libWeakForm.WeakForm   import *
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.BernoulliBeamStrainOperator import GetBernoulliBeamStrainOperator
from fedoo.libUtil.Variable import Variable
from fedoo.libUtil.Dimension import ProblemDimension

class BernoulliBeam(WeakForm):
    def __init__(self, CurrentConstitutiveLaw, Section, Jx, Iyy, Izz, ID = ""):
        if isinstance(CurrentConstitutiveLaw, str):
            CurrentConstitutiveLaw = ConstitutiveLaw.GetAll()[CurrentConstitutiveLaw]

        if ID == "":
            ID = CurrentConstitutiveLaw.GetID()
            
        WeakForm.__init__(self,ID)

        Variable("DispX") 
        Variable("DispY")     
        if ProblemDimension.Get() == '3D':
            Variable("DispZ")   
            Variable("ThetaX") #torsion rotation            
            Variable.SetDerivative('DispZ', 'ThetaY', sign = -1) #only valid with Bernoulli model
            Variable.SetDerivative('DispY', 'ThetaZ') #only valid with Bernoulli model       
            Variable.SetVector('Disp' , ('DispX', 'DispY', 'DispZ') , 'global')
            Variable.SetVector('Theta' , ('ThetaX', 'ThetaY', 'ThetaZ') , 'global')
        elif ProblemDimension.Get() == '2Dplane':
            Variable.SetDerivative('DispY', 'ThetaZ') #only valid with Bernoulli model       
            Variable.SetVector('Disp' , ('DispX', 'DispY') )            
        elif ProblemDimension.Get() == '2Dstress':
            assert 0, "No 2Dstress model for a beam kinematic. Choose '2Dplane' instead."
                  
        self.__ConstitutiveLaw = CurrentConstitutiveLaw
        self.__parameters = {'Section': Section, 'Jx': Jx, 'Iyy':Iyy, 'Izz':Izz}
        self.__typeOperator = 'all'       

    def Bending(self):
        self.__typeOperator = 'Bending'

    def TractionTorsion(self):
        self.__typeOperator = 'TractionTorsion'
    
    def GetDifferentialOperator(self, localFrame):
        try:
            E  = self.__ConstitutiveLaw.GetYoungModulus()
            nu = self.__ConstitutiveLaw.GetPoissonRatio()       
            G = E/(1+nu)/2
            
            eps, eps_vir = GetBernoulliBeamStrainOperator()           

            if self.__typeOperator == 'all':            
                Ke = [E*self.__parameters['Section'], 0, 0, G*self.__parameters['Jx'], E*self.__parameters['Iyy'], E*self.__parameters['Izz']]
                return sum([eps_vir[i] * eps[i] * Ke[i] for i in range(6)])                

            elif self.__typeOperator == 'Bending':
                return eps_vir[4] * eps[4] * E*self.__parameters['Iyy'] + eps_vir[5] * eps[5] * E*self.__parameters['Izz']    

            elif self.__typeOperator == 'TractionTorsion':
                return eps_vir[0] * eps[0] * E*self.__parameters['Section'] + eps_vir[3] * eps[3] * G*self.__parameters['Jx']   

        except:
            assert 0, "Warning, you put a non mechanical consitutive law in the InternalForce"
    
    def GetGeneralizedStress(self):
        #only for post treatment

        eps, eps_vir = GetBernoulliBeamStrainOperator()
        E  = self.__ConstitutiveLaw.GetYoungModulus()
        nu = self.__ConstitutiveLaw.GetPoissonRatio()       
        G = E/(1+nu)/2
        Ke = [E*self.__parameters['Section'], 0, 0, G*self.__parameters['Jx'], E*self.__parameters['Iyy'], E*self.__parameters['Izz']]
        
        return [eps[i] * Ke[i] for i in range(6)]



    
