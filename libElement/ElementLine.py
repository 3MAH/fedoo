import scipy as sp
from fedoo.libElement.Element import *


class lin2(element1DGeom2,element1D):
    def __init__(self, nb_pg=2, **kargs):
        self.xi_nd = sp.c_[[0., 1.]]                     
        self.nb_pg = nb_pg
        element1D.__init__(self, nb_pg)
            
    #Dans les fonctions suivantes, xi doit toujours être une matrice colonne      
    def ShapeFunction(self,xi): 
        return sp.c_[(1-xi), xi]
    def ShapeFunctionDerivative(self,xi):               
        return [sp.array([[-1., 1.]]) for x in xi] 

class lin2Bubble(lin2):
    def ShapeFunction(self,xi):    
        return sp.c_[(1-xi), xi, xi*(1-xi)]
    def ShapeFunctionDerivative(self,xi):               
        return [sp.array([[-1., 1., 1.-2*x]]) for x in xi[:,0]] 


class lin3(element1D):
    def __init__(self, nb_pg=3, **kargs):
        self.xi_nd = sp.c_[[0., 1., 0.5]]  
        self.nb_pg = nb_pg
        element1D.__init__(self, nb_pg)
            
    def ShapeFunction(self,xi):       
        return sp.c_[2*xi**2-3*xi+1, xi*(2*xi-1), 4*xi*(1-xi)]
    def ShapeFunctionDerivative(self,xi):       
        return [sp.array([[4*x-3, 4*x-1, 4-8*x]]) for x in xi[:,0]]
    
class lin3Bubble(lin3):        
    def ShapeFunction(self,xi):    
        return sp.c_[2*xi**2-3*xi+1, xi*(2*xi-1), 4*xi*(1-xi), 64./3*xi**3 - 32*xi**2 + 32./3*xi]
    def ShapeFunctionDerivative(self,xi):               
        return [sp.array([[4*x-3, 4*x-1, 4-8*x, 64*x**2 - 64*x + 32./3]]) for x in xi[:,0]]

#lin4 needs to modify the initial position of nodes for compatibility with other elements
#class lin4(element1D): 
#    def __init__(self, nb_pg=4, avec_bulle = 0, **kargs):
#        self.xi_nd = sp.c_[[-1., 1., -1./3, 1./3]]  
#        self.nb_pg = nb_pg
#        element1D.__init__(self, nb_pg)
#            
#    def nn(self,xi):       
#        return sp.c_[-4.5*xi**3+9*xi**2-5.5*xi+1, 4.5*xi**3-4.5*xi**2+xi, 13.5*xi**3-22.5*xi**2+9*xi, -13.5*xi**3+18*xi**2-4.5*xi]
#    def dnn(self,xi):       
#        return [sp.array([[-13.5*x**2+18*x-5.5, 13.5*x**2-9*x+1, 40.5*x**2-45*x+9, -40.5*x**2+36*x-4.5]]) for x in xi[:,0]]


class lin2C1(element1D): #2 nodes with derivatative dof
    def __init__(self, nb_pg=4, **kargs): # pour la matrice de masse on est sous-integré (il en faut 6), pour la matrice de rigidite -> reste à voir
        self.xi_nd = sp.c_[[0., 1.]]                     
        self.nb_pg = nb_pg
        element1D.__init__(self, nb_pg)
        self.ShapeFunctionSecondDerivativePG = self.ShapeFunctionSecondDerivative(self.xi_pg)
            
    #Dans les fonctions suivantes, xi doit toujours être une matrice colonne
    
    def ShapeFunction(self,xi):
        # [(vi,vj,tetai,tetaj)]
        return sp.c_[(1-3*xi**2+2*xi**3), (3*xi**2-2*xi**3), (xi-2*xi**2+xi**3), (-xi**2+xi**3)]
    def ShapeFunctionDerivative(self,xi):  
        return [sp.array([[-6*x+6*x**2, 6*x-6*x**2, 1-4*x+3*x**2, -2*x+3*x**2]]) for x in xi[:,0]]
    def ShapeFunctionSecondDerivative(self,xi):               
        return [sp.array([[-6+12*x, 6-12*x, -4+6*x, -2+6*x]]) for x in xi[:,0]]