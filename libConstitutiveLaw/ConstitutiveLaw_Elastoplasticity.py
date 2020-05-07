#derive de ConstitutiveLaw
#The elastoplastic law should be used with an InternalForce WeakForm

from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.StrainOperator import *
from fedoo.libUtil.Variable       import *
from fedoo.libUtil.Dimension      import *
from fedoo.libUtil.PostTreatement import listStressTensor, listStrainTensor

import numpy as np

class ElastoPlasticity(ConstitutiveLaw):
    def __init__(self,YoungModulus, PoissonRatio, YieldStress, ID=""):
        #only scalar values of YoungModulus and PoissonRatio are possible
        ConstitutiveLaw.__init__(self, ID) # heritage
        
        Variable("DispX")
        Variable("DispY")        
        
        if ProblemDimension.Get() == "3D": 
            Variable("DispZ")

        self.__YoungModulus = YoungModulus
        self.__PoissonRatio = PoissonRatio
        self.__YieldStress = YieldStress

        self.__P = None #irrevesrible plasticity
        self.__currentP = None #current iteration plasticity (reversible)
        self.__PlasticStrainTensor = None         
        self.__currentPlasticStrainTensor = None 
        self.__currentSigma = None #lissStressTensor object describing the last computed stress (GetStress method)
        
        self.__tol = 1e-6 #tolerance of Newton Raphson used to get the updated plasticity state (constutive law alogorithm)    

    def GetYoungModulus(self):
        return self.__YoungModulus

    def GetPoissonRatio(self):
        return self.__PoissonRatio

    def GetYieldStress(self):
        return self.__YieldStress        
    
    def SetNewtonRaphsonTolerance(self, tol):
        """
        Set the tolerance of the Newton Raphson algorithm used to get the updated plasticity state (constutive law alogorithm)
        """
        self.__tol = tol
    
    def GetHelas (self):        
        H  = np.zeros((6,6), dtype='object')
        E  = self.__YoungModulus 
        nu = self.__PoissonRatio       

        # tester si contrainte plane ou def plane 
        if ProblemDimension.Get() == "2Dstress":
            H[0,0]=H[1,1]= E/(1-nu**2)
            H[0,1]= nu*E/(1-nu**2)
            H[5,5] = 0.5*E/(1+nu)    
            H[1,0]=H[0,1]  #symétrie            
                      
        else:
            H[0,0]=H[1,1]=H[2,2]= E*(1./(1+nu) + nu/((1.+nu)*(1-2*nu))) #H1 = 2*mu+lamb
            H[0,1]=H[0,2]=H[1,2]= E*(nu/((1+nu)*(1-2*nu)))  #H2 = lamb
            H[3,3]=H[4,4]=H[5,5] = 0.5*E/(1+nu) #H3 = mu
            H[1,0]=H[0,1] ; H[2,0]=H[0,2] ; H[2,1] = H[1,2] #symétrie 
            
        return H    
    
    def HardeningFunction(self, p): 
        raise NameError('Hardening function not defined. Use the method SetHardeningFunction')

    def HardeningFunctionDerivative(self, p):
        raise NameError('Hardening function not defined. Use the method SetHardeningFunction')
    
    def SetHardeningFunction(self, FunctionType, **kargs):
        if FunctionType.lower() == 'power':
            H = None ; beta = None
            for item in kargs:
                if item.lower() == 'h': H = kargs[item]  
                if item.lower() == 'beta': beta = kargs[item]  

            if H is None: raise NameError("Keyword arguments 'H' missing")
            if beta is None: raise NameError("Keyword arguments 'beta' missing")       
            
            def HardeningFunction(p): 
                return H*p**beta
            
            def HardeningFunctionDerivative(p):        
                return beta*H*p**(beta-1)
            
        elif FunctionType.lower() == 'user':
            HardeningFunction = None ; HardeningFunctionDerivative = None
            for item in kargs:
                if item.lower() == 'hardeningfunction': HardeningFunction = kargs[item]  
                if item.lower() == 'hardeningfunctionderivative':  HardeningFunctionDerivative = kargs[item]  
            
            if HardeningFunction is None: raise NameError("Keyword arguments 'HardeningFunction' missing")
            if HardeningFunctionDerivative is None: raise NameError("Keyword arguments 'HardeningFunctionDerivative' missing")  
            
        self.HardeningFunction = HardeningFunction
        self.HardeningFunctionDerivative = HardeningFunctionDerivative
            
    def YieldFunction(self, Stress, p):
        return Stress.vonMises() - self.__YieldStress - self.HardeningFunction(p)
    
    def YieldFunctionDerivativeSigma(self, sigma):
        """
        Derivative of the Yield Function with respect to the stress tensor defined in sigma
        sigma should be a listStressTensor object
        """
        return listStressTensor((3/2)*np.array(sigma.deviatoric())/sigma.vonMises()).toStrain()
    
    def GetPlasticity(self):
        return self.__currentP
    
    def GetPlasticStrainTensor(self):
        return self.__currentPlasticStrainTensor
    
    def GetCurrentStress(self):
        return self.__currentSigma
        
    def GetH(self):        
        Helas = self.GetHelas() #Elastic Rigidity matrix: no change of basis because only isotropic behavior are considered      

        if self.__currentSigma is None: return Helas
                        
        dphi_dp = self.HardeningFunctionDerivative(self.__currentP)
        dphi_dsigma = self.YieldFunctionDerivativeSigma(self.__currentSigma)
        Lambda = dphi_dsigma #for isotropic hardening only
        test = self.YieldFunction(self.__currentSigma, self.__P) > self.__tol                      
        
        ##### Compute new tangeant moduli
        B = sum([sum([dphi_dsigma[j]*Helas[i][j] for j in range(6)]) * Lambda[i] for i in range(6)])        
        Ap = (B-dphi_dp)
        CL = [sum([Lambda[j]*Helas[i][j] for j in range(6)]) for i in range(6)] # [C:Lambda]
        Peps = [sum([dphi_dsigma[i]*Helas[i][j] for i in range(6)])/Ap for j in range(6)]  #Peps 
#        TangeantModuli = [[Helas[i][j] - CL[i]*Peps[j] for j in range(6)] for i in range(6)]                
        TangeantModuli = [[Helas[i][j] - (CL[i]*Peps[j] * test) for j in range(6)] for i in range(6)]                
        return TangeantModuli
        ##### end Compute new tangeant moduli                        
    
    def __ChangeBasisH(self, H):
        #Change of basis capability for laws on the form : StressTensor = H * StrainTensor
        #StressTensor and StrainTensor are column vectors based on the voigt notation 
        if self._ConstitutiveLaw__localFrame is not None:
            localFrame = self._ConstitutiveLaw__localFrame
            #building the matrix to change the basis of the stress and the strain
#            theta = np.pi/8
#            np.array([[np.cos(theta),np.sin(theta),0], [-np.sin(theta),np.cos(theta),0], [0,0,1]]) 
            R_epsilon = np.empty((len(localFrame), 6,6))
            R_epsilon[:,  :3,  :3] = localFrame**2 
            R_epsilon[:,  :3, 3:6] = localFrame[:,:,[1,2,0]]*localFrame[:,:,[2,0,1]]
            R_epsilon[:, 3:6,  :3] = 2*localFrame[:,[1,2,0]]*localFrame[:,[2,0,1]] 
            R_epsilon[:, 3:6, 3:6] = localFrame[:,[[1],[2],[0]], [1,2,0]]*localFrame[:,[[2],[0],[1]],[2,0,1]] + localFrame[:,[[2],[0],[1]],[1,2,0]]*localFrame[:,[[1],[2],[0]],[2,0,1]] 
            R_sigma_inv = np.transpose(R_epsilon,[0,2,1])
            
            if len(H.shape) == 3: H = np.rollaxis(H,2,0)
            H = np.matmul(R_sigma_inv, np.matmul(H,R_epsilon))
            if len(H.shape) == 3: H = np.rollaxis(H,0,3)  

        return H
    
    def GetStressOperator(self, localFrame=None): 
        H = self.__ChangeBasisH(self.GetH())
                      
        eps, eps_vir = GetStrainOperator()         
        sigma = [sum([0 if eps[j] is 0 else eps[j]*H[i][j] for j in range(6)]) for i in range(6)]

        return sigma # list de 6 objets de type OpDiff
    
    def NewTimeIncrement(self):
        #Set Irreversible Plasticity
        if self.__P is not None:     
            self.__P = self.__currentP.copy()
            self.__PlasticStrainTensor = self.__currentPlasticStrainTensor.copy()        
        self.__currentSigma = None
        
    def ResetTimeIncrement(self):
        if self.__P is None: 
            self.__currentP = None
            self.__currentPlasticStrainTensor = None
        else:
            self.__currentP = self.__P.copy()
            self.__currentPlasticStrainTensor = self.__PlasticStrainTensor.copy()            
        self.__currentSigma = None       
    
    def Reset(self): 
        """
        Reset the constitutive law (time history)
        """
        self.__P = None #irrevesrible plasticity
        self.__currentP = None #current iteration plasticity (reversible)
        self.__PlasticStrainTensor = None         
        self.__currentPlasticStrainTensor = None 
        self.__currentSigma = None #lissStressTensor object describing the last computed stress (GetStress method)

    
    def GetStress(self, StrainTensor, time = None): 
        # time not used here because this law require no time effect
        # initilialize values plasticity variables if required
        if self.__P is None: 
            self.__P = np.zeros(len(StrainTensor[0]))
            self.__currentP = np.zeros(len(StrainTensor[0]))
        if self.__PlasticStrainTensor is None: 
            self.__PlasticStrainTensor = listStrainTensor(np.zeros((6,len(StrainTensor[0]))))
            self.__currentPlasticStrainTensor = listStrainTensor(np.zeros((6,len(StrainTensor[0]))))
            
        H = self.GetHelas() #no change of basis because only isotropic behavior are considered            
        sigma = listStressTensor([sum([(StrainTensor[j]-self.__PlasticStrainTensor[j])*H[i][j] for j in range(6)]) for i in range(6)])
        test = self.YieldFunction(sigma, self.__P) > self.__tol
#        print(sum(test)/len(test)*100)

        sigmaFull = np.array(sigma).T
        Ep = np.array(self.__PlasticStrainTensor).T
#        Ep = np.array(self.__currentPlasticStrainTensor).T
        
        for pg in range(len(sigmaFull)):
            if test[pg] > 0:                 
                sigma = listStressTensor(sigmaFull[pg])
                p = self.__P[pg]               
                iter = 0
                while abs(self.YieldFunction(sigma, p)) > self.__tol:                       
                    dphi_dp = self.HardeningFunctionDerivative(p)
                    dphi_dsigma = np.array(self.YieldFunctionDerivativeSigma(sigma))
                    
                    Lambda = dphi_dsigma #for associated plasticity
                    B = sum([sum([dphi_dsigma[j]*H[i][j] for j in range(6)]) * Lambda[i] for i in range(6)])
        
                    dp = self.YieldFunction(sigma,p)/(B-dphi_dp)
                    p += dp
                    Ep[pg] += Lambda * dp
                    sigma = listStressTensor([sum([(StrainTensor[j][pg]-Ep[pg][j])*H[i][j] for j in range(6)]) for i in range(6)])                                
                self.__currentP[pg] = p
                sigmaFull[pg] = sigma                                                  
                
        self.__currentPlasticStrainTensor = listStrainTensor(Ep.T)
        self.__currentSigma = listStressTensor(sigmaFull.T) # list of 6 objets 
               
        return self.__currentSigma 
    
    
