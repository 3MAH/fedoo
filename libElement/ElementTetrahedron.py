import scipy as sp
from fedoo.libElement.Element import *

class elementTetrahedron(element):
    def __init__(self,nb_pg):                 
        #initialize the gauss points and the associated weight           
        if nb_pg == 1:
            xi = sp.c_[[1./4]] ; eta = sp.c_[[1./4]] ; zeta = sp.c_[[1./4]]
            w_pg = sp.array([1./6])                                
        elif nb_pg == 4:
            a = 0.1381966011250105 ; b = 0.5854101966249685
            xi =  sp.c_[[a, a, a, b]] ; eta = sp.c_[[a, a, b, a]] ; zeta = sp.c_[[a, b, a, a]]
            w_pg = sp.array([1./24, 1./24, 1./24, 1./24])
        elif nb_pg == 5:
            a = 0.25 ; b = 0.16666666666666666 ; c = 0.5
            xi =  sp.c_[[a, b, b, b, c]] ; eta = sp.c_[[a, b, b, c, b]] ; zeta = sp.c_[[a, b, c, b, b]]
            w_pg = sp.array([-2./15, 3./40, 3./40, 3./40, 3./40])
        elif nb_pg == 15:
            a = 0.25 ; b1 = 0.3197936278296299 ; b2 = 0.09197107805272303 ; c1 = 0.040619116511110234 ; c2 = 0.724086765841831 ; d = 0.05635083268962915 ; e = 0.4436491673103708
            xi =  sp.c_[[a, b1, b1, b1, c1, b2, b2, b2, c2, d, d, e, d, e, e]] ; eta = sp.c_[[a, b1, b1, c1, b1, b2, b2, c2, b2, d, e, d, e, d, e]] ; zeta = sp.c_[[a, b1, c1, b1, b1, b2, c2, b2, b2, e, d, d, e, e, d]]
            f1 = 0.011511367871045397 ; f2 = 0.01198951396316977
            w_pg = sp.array([8./405, f1, f1, f1, f1, f2, f2, f2, f2, 5./567, 5./567, 5./567, 5./567, 5./567, 5./567])
        elif nb_pg == 0:
            self.xi_pg = self.xi_nd #if nb_pg == 0, we take the position of the nodes
        else:
            assert 0, "Number of gauss points "+str(nb_pg)+" unavailable for tetrahedron element"  
    
        if nb_pg != 0:    
            self.xi_pg = sp.c_[xi,eta,zeta]
            self.w_pg = w_pg

        self.ShapeFunctionPG = self.ShapeFunction(self.xi_pg)
        self.ShapeFunctionDerivativePG = self.ShapeFunctionDerivative(self.xi_pg)

class tet4(elementTetrahedron):
    def __init__(self, nb_pg=4, **kargs):
        self.xi_nd = sp.c_[[0. , 0. , 0. , 1.],\
                           [1. , 0. , 0. , 0.],\
                           [0. , 1. , 0. , 0.]]
        self.nb_pg = nb_pg
        elementTetrahedron.__init__(self,nb_pg)
            
    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:,0] ; eta = vec_xi[:,1]; zeta = vec_xi[:,2]
        return sp.c_[eta, zeta, 1-xi-eta-zeta, xi]    
    def ShapeFunctionDerivative(self, vec_xi): 
        return [ sp.array([ [0. , 0. , -1. , 1.] , \
                         [1. , 0. , -1. , 0.] , \
                         [0. , 1. , -1. , 0.] ]) for xi in vec_xi]   
                         
class tet10(elementTetrahedron):
    def __init__(self, nb_pg=15, **kargs):
        self.xi_nd = sp.c_[[0. , 0. , 0. , 1. , 0. , 0. , 0. , 0.5 , 0.5 , 0.5],\
                           [1. , 0. , 0. , 0. , 0.5 , 0. , 0.5 , 0.5 , 0. , 0.],\
                           [0. , 1. , 0. , 0. , 0.5 , 0.5 , 0. , 0. , 0.5 , 0.]]
        self.nb_pg = nb_pg
        elementTetrahedron.__init__(self,nb_pg)
            
    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:,0] ; eta = vec_xi[:,1]; zeta = vec_xi[:,2]
        return sp.c_[eta*(2*eta-1), zeta*(2*zeta-1), (1-xi-eta-zeta)*(1-2*xi-2*eta-2*zeta), xi*(2*xi-1), 4*eta*zeta, 4*zeta*(1-xi-eta-zeta), 4*eta*(1-xi-eta-zeta), 4*xi*eta, 4*xi*zeta, 4*xi*(1-xi-eta-zeta)]    
    def ShapeFunctionDerivative(self, vec_xi):
        m = 1-xi[0]-xi[1]-xi[2]    
        return [ sp.array([ [ 0., 0., 1-4*m, -1+4*xi[0], 0., -4*xi[2], -4*xi[1], 4*xi[1], 4*xi[2], 4*(m-xi[0]) ] , \
                         [ -1+4*xi[1], 0., 1-4*m, 0., 4*xi[2],-4*xi[2], 4*(m-xi[1]), 4*xi[0], 0., -4*xi[0]  ] , \
                         [ 0., -1+4*xi[2], 1-4*m, 0., 4*xi[1], 4*(m-xi[2]), 4*xi[1], 0. , 4*xi[0], -4*xi[0] ] ]) for xi in vec_xi]   
