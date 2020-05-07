import scipy as sp
from scipy import linalg


#class lin2C1(element1D): #2 nodes with derivatative dof
#    def __init__(self, nb_pg=4, **kargs): # pour la matrice de masse on est sous-integré (il en faut 6), pour la matrice de rigidite -> reste à voir
#        self.xi_nd = sp.c_[[0., 1.]]                     
#        self.nb_pg = nb_pg
#        element1D.__init__(self, nb_pg)
#        self.ShapeFunctionSecondDerivativePG = self.ShapeFunctionSecondDerivative(self.xi_pg)
#            
#    #Dans les fonctions suivantes, xi doit toujours être une matrice colonne
#    
#    def ShapeFunction(self,xi):
#        # [(vi,vj,tetai,tetaj)]
#        return sp.c_[(1-3*xi**2+2*xi**3), (3*xi**2-2*xi**3), (xi-2*xi**2+xi**3), (-xi**2+xi**3)]
#    def ShapeFunctionDerivative(self,xi):  
#        return [sp.array([[-6*x+6*x**2, 6*x-6*x**2, 1-4*x+3*x**2, -2*x+3*x**2]]) for x in xi[:,0]]
#    def ShapeFunctionSecondDerivative(self,xi):               
#        return [sp.array([[-6+12*x, 6-12*x, -4+6*x, -2+6*x]]) for x in xi[:,0]]
    

beam = {'DispX':'lin2', 'DispY':'lin2C1', 'DispZ':'lin2C1', 'ThetaX':'lin2', 'default':'lin2'}        
        
