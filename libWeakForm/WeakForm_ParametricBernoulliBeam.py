from fedoo.libWeakForm.WeakForm   import *
from fedoo.libConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from fedoo.libUtil.Variable import Variable
from fedoo.libUtil.Dimension import ProblemDimension
from fedoo.libUtil.Coordinate import Coordinate
from fedoo.libUtil.BernoulliBeamStrainOperator import GetBernoulliBeamStrainOperator
from fedoo.libPGD.MeshPGD import MeshPGD
from fedoo.libPGD.SeparatedArray import SeparatedOnes

import numpy as np

class ParametricBernoulliBeam(WeakForm):
    def __init__(self, E=None, nu = None, S=None, Jx=None, Iyy=None, Izz = None, R = None, ID = ""):
        """
        Weak formulation dedicated to treat parametric problems using Bernoulli beams for isotropic materials
        
        Arugments
        ----------
        ID: str
            ID of the weak formulation
        List of other optional parameters
            E: Young Modulus
            nu: Poisson Ratio
            S: Section area
            Jx: Torsion constant
            Iyy, Izz: Second moment of area, if Izz is not specified, Iyy = Izz is assumed
        When the differntial operator is generated (using PGD.Assembly) the parameters are searched among the CoordinateID defining the associated mesh.
        If a parameter is not found , a Numeric value should be specified in argument.
        
        In the particular case of cylindrical beam, the radius R can be specified instead of S, Jx, Iyy and Izz.
            S = pi * R**2
            Jx = pi * R**4/2
            Iyy = Izz = pi * R**4/4         
        """
    
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
        
        if R is not None:
            S = np.pi * R**2
            Jx = np.pi * R**4/2
            Iyy = Izz = np.pi * R**4/4 
        
        self.__parameters = {'E':E, 'nu':nu, 'S':S, 'Jx':Jx, 'Iyy':Iyy, 'Izz':Izz}
        self.__typeOperator = 'all'

    def Bending(self):
        self.__typeOperator = 'Bending'

    def TractionTorsion(self):
        self.__typeOperator = 'TractionTorsion'
   
    def __GetKe(self, mesh):
        NN = mesh.GetNumberOfNodes() #number of nodes for every submeshes
        E_S = SeparatedOnes(NN) 
        G_Jx = SeparatedOnes(NN) #G = E/(1+nu)/2
        E_Iyy = SeparatedOnes(NN)
        E_Izz = SeparatedOnes(NN)                                             
                               
        for param in ['E', 'nu', 'R', 'S', 'Jx', 'Iyy', 'Izz']:     
            if mesh.FindCoordinateID(param) is not None:
                Coordinate(param)
                id_mesh = mesh.FindCoordinateID(param)
                mesh._SetSpecificVariableRank(id_mesh, 'default', 0) #all the variables use the same function for the submeshes related to parametric coordinates                              
                col = mesh.GetListMesh()[id_mesh].GetCoordinateID().index(param)

                if param == 'R':
                    E_S.data[id_mesh][:,0] = E_S.data[id_mesh][:,0] * (mesh.GetListMesh()[id_mesh].GetNodeCoordinates()[:,col] ** 2 *np.pi)
                    G_Jx.data[id_mesh][:,0] = G_Jx.data[id_mesh][:,0] * (mesh.GetListMesh()[id_mesh].GetNodeCoordinates()[:,col] ** 4 * (np.pi/2))
                    E_Iyy.data[id_mesh][:,0] = E_Iyy.data[id_mesh][:,0] * (mesh.GetListMesh()[id_mesh].GetNodeCoordinates()[:,col] ** 4 * (np.pi/4))
                    E_Izz = E_Iyy
                    break
                    
                if param in ['E','S']:
                    E_S.data[id_mesh][:,0] = E_S.data[id_mesh][:,0] * mesh.GetListMesh()[id_mesh].GetNodeCoordinates()[:,col]
                    if param == 'E':
                        G_Jx.data[id_mesh][:,0] = G_Jx.data[id_mesh][:,0] * mesh.GetListMesh()[id_mesh].GetNodeCoordinates()[:,col]            
                if param == 'Jx':
                    G_Jx.data[id_mesh][:,0] = G_Jx.data[id_mesh][:,0] * mesh.GetListMesh()[id_mesh].GetNodeCoordinates()[:,col]
                if param == 'nu':                    
                    G_Jx.data[id_mesh][:,0] = G_Jx.data[id_mesh][:,0] * (0.5 / (1 + mesh.GetListMesh()[id_mesh].GetNodeCoordinates()[:,col]))
                if param in ['E','Iyy']:
                    E_Iyy.data[id_mesh][:,0] = E_Iyy.data[id_mesh][:,0] * mesh.GetListMesh()[id_mesh].GetNodeCoordinates()[:,col]
                if param in ['E','Izz']:
                    E_Izz.data[id_mesh][:,0] = E_Izz.data[id_mesh][:,0] * mesh.GetListMesh()[id_mesh].GetNodeCoordinates()[:,col]                             
            
            elif self.__parameters[param] is not None:                
                if param in ['E','S']: E_S = E_S * self.__parameters[param]     
                if param in ['E', 'Jx']: G_Jx = G_Jx * self.__parameters[param] 
                if param == 'nu': G_Jx = G_Jx * (0.5/(1+self.__parameters['nu']))
                if param in ['E','Iyy']: E_Iyy = E_Iyy * self.__parameters[param]                 
                if param in ['E','Izz']: E_Izz = E_Izz * self.__parameters[param]                      

            elif param != 'R':                
                if param == 'Izz': E_Izz = E_Iyy                
                else: assert 0, "The parameter " + param + " is not defined"
        
        return [E_S, 0, 0, G_Jx, E_Iyy, E_Izz]  


    def GetDifferentialOperator(self, mesh, localFrame=None):
        
        if isinstance(mesh, str):
            mesh = MeshPGD.GetAll()[mesh]

        Ke = self.__GetKe(mesh)            
        eps, eps_vir = GetBernoulliBeamStrainOperator()    
        
        if self.__typeOperator == 'all':            
            return sum([eps_vir[i] * eps[i] * Ke[i] for i in range(6)])
        elif self.__typeOperator == 'Bending':
            return eps_vir[4] * eps[4] * Ke[4] + eps_vir[5] * eps[5] * Ke[5]
        elif self.__typeOperator == 'TractionTorsion':
            return eps_vir[0] * eps[0] * Ke[0] + eps_vir[3] * eps[3] * Ke[3]
            
    def GetGeneralizedStress(self, mesh):
        
        Ke = self.__GetKe(mesh)
        eps, eps_vir = GetBernoulliBeamStrainOperator()
        
        temp = [eps[i] * Ke[i] for i in range(6)]

        return temp

    
    
    
    
    