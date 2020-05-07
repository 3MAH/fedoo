import scipy as sp
from scipy import linalg
from fedoo.libElement.Element import *
from fedoo.libElement.ElementQuadrangle import *

class cohesive1D(element):
    def __init__(self, nb_pg=1, **kargs):
        self.nb_pg = 1 #pas de point de gauss pour les éléments cohésifs car pas de point d'intégration
        self.xi_pg = sp.c_[[0.]] ; self.xi_nd = sp.c_[[0., 0.]] #The values are arbitrary, only the size is important
        self.w_pg = sp.array([1.]) 
    
    def ComputeJacobianMatrix(self,vec_x, vec_xi): 
        self.detJ = [1. for xi in vec_xi]   

    #In the following functions, xi shall always be a column matrix 
    def ShapeFunction(self,xi): 
        return sp.array([[-1., 1.] for x in xi])
    def ShapeFunctionDerivative(self,xi): #inutile en principe
        return [sp.array([[0., 0.]]) for x in xi]     
#    def GeometricalShapeFunction(self,xi): 
#        return 0.5*sp.array([[1., 1.] for x in xi])
    
class cohesive2D(element1DGeom2, element1D):
    def __init__(self,nb_pg=2, **kargs):
        """
        An element is defined with 4 nodes [0, 1, 2, 3]
        [0, 1] is a lin2 defining the face on the negative side of the cohesive zone
        [2, 3] is a lin2 defining the face on the positive side of the cohesive zone
        Node 0 is in front of node 2 and node 1 is in front of node 3
        """
        element1D.__init__(self, nb_pg)
        self.xi_nd = sp.c_[[0., 1.]]              
        self.nb_pg = nb_pg
        
    def ShapeFunction(self,xi): 
        return sp.c_[-xi, -1+xi, xi, 1-xi]
    def ShapeFunctionDerivative(self,xi): #is it required for cohesive elements ?
        return [sp.array([[-1., 1.,1., -1.]]) for x in xi] 
#    def GeometricalShapeFunction(self,xi): 
#        return 0.5*sp.c_[xi, 1-xi, xi, 1-xi]

class cohesive3D(quad4): # à vérifier

    def __init__(self,nb_pg=4, **kargs):
        """
        An element is defined with 8 nodes [0, 1, 2, 3, 4, 5, 6, 7]
        [0, 1, 2, 3] is a quad4 defining the face on the negative side of the cohesive zone
        [4, 5, 6, 7] is a quad4 defining the face on the positive side of the cohesive zone
        """
        quad4.__init__(self, nb_pg, **kargs)
                        
    #Dans les fonctions suivantes vec_xi contient une liste de points dans le repère de référence (xi, eta)
    #vec_xi[:,0] -> liste des valeurs de xi pour chaque point (points de gauss en général)
    #vec_xi[:,1] -> liste des valeurs de eta pour chaque point (points de gauss en général)    
    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:,0] ; eta = vec_xi[:,1]
        return sp.c_[ -0.25*(1-xi)*(1-eta) , -0.25*(1+xi)*(1-eta) , -0.25*(1+xi)*(1+eta) , -0.25*(1-xi)*(1+eta), 0.25*(1-xi)*(1-eta) , 0.25*(1+xi)*(1-eta) , 0.25*(1+xi)*(1+eta) , 0.25*(1-xi)*(1+eta) ]
    def ShapeFunctionDerivative(self, vec_xi): #quad4 shape functions based on the mean values from two adjacent nodes 
        return [ 0.5*sp.array([ [0.25*(xi[1]-1), 0.25*(1-xi[1]), 0.25*(1+xi[1]), -0.25*(1+xi[1]), 0.25*(xi[1]-1), 0.25*(1-xi[1]), 0.25*(1+xi[1]), -0.25*(1+xi[1])] , [0.25*(xi[0]-1), -0.25*(1+xi[0]), 0.25*(1+xi[0]), 0.25*(1-xi[0]), 0.25*(xi[0]-1), -0.25*(1+xi[0]), 0.25*(1+xi[0]), 0.25*(1-xi[0])] ]) for xi in vec_xi]
#    def GeometricalShapeFunction(self, vec_xi):
#        xi = vec_xi[:,0] ; eta = vec_xi[:,1]
#        return 0.5 * sp.c_[ 0.25*(1-xi)*(1-eta) , 0.25*(1+xi)*(1-eta) , 0.25*(1+xi)*(1+eta) , 0.25*(1-xi)*(1+eta),  0.25*(1-xi)*(1-eta) , 0.25*(1+xi)*(1-eta) , 0.25*(1+xi)*(1+eta) , 0.25*(1-xi)*(1+eta)]
#    def ShapeFunctionDerivative_quad4(self, vec_xi): 
#        return [ sp.array([ [0.25*(xi[1]-1), 0.25*(1-xi[1]), 0.25*(1+xi[1]), -0.25*(1+xi[1])] , [0.25*(xi[0]-1), -0.25*(1+xi[0]), 0.25*(1+xi[0]), 0.25*(1-xi[0])] ]) for xi in vec_xi]      

    
#    def ComputeJacobianMatrix(self,vec_x, vec_xi):
#        """
#        Calcul le Jacobien aux points de gauss dans le cas d'un élément isoparamétrique (c'est à dire que les mêmes fonctions de forme sont utilisées)
#        vec_x est un tabeau dont les lignes donnent les coordonnées de chacun des noeuds de l'éléments
#        vec_xi est un tableau dont les lignes donnent les coordonnées dans le repère de référence où on souhaite avoir le jacobien (en général pg)
#        Calcul le jacobien dans self.JacobianMatrix le jacobien sous la forme [[dx/dxi, dy/dxi, ...], [dx/deta, dy/deta, ...], ...]
#        Renvoie le déterminant du jacobien
#        """      
#        vec_x_quad = 0.5*(vec_x[0:4]+vec_x[4:8])
#        
#        dnn_xi = self.ShapeFunctionDerivative_quad4(vec_xi)        
#        self.JacobianMatrix = [sp.dot(dnn,vec_x_quad) for dnn in dnn_xi]         
#        
#        if sp.shape(self.JacobianMatrix[0])[0] == sp.shape(self.JacobianMatrix[0])[1]:
#            self.detJ = [abs(linalg.det(J)) for J in self.JacobianMatrix]
#        else: #l'espace réel est dans une dimension plus grande que l'espace de l'élément de référence         
#            if sp.shape(self.JacobianMatrix[0])[0] == 1: self.detJ = [linalg.norm(J) for J in self.JacobianMatrix]
#            else: #On doit avoir sp.shape(JacobianMatrix)[0]=2 (l'elm de ref est défini en 2D) et sp.shape(JacobianMatrix)[1]=3  (l'espace réel est 3D)
#                self.detJ = [sp.sqrt(abs(J[0,1]*J[1,2]-J[0,2]*J[1,1])**2 +\
#                             abs(J[0,2]*J[1,0]-J[1,2]*J[0,0])**2 +\
#                             abs(J[0,0]*J[1,1]-J[1,0]*J[0,1])**2 ) for J in self.JacobianMatrix]