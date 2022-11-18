import numpy as np
from fedoo.lib_elements.element_base import *

class ElementQuadrangle(Element2D):
    def __init__(self,n_elm_gp): 
        if n_elm_gp == 0:  #if n_elm_gp == 0, we take the position of the nodes
            self.xi_pg = self.xi_nd            
        else: 
            self.xi_pg = self.get_gp_elm_coordinates(n_elm_gp) # = np.c_[xi,eta]             
            self.w_pg = self.get_gp_weight(n_elm_gp)
            
        self.ShapeFunctionPG = self.ShapeFunction(self.xi_pg)
        self.ShapeFunctionDerivativePG = self.ShapeFunctionDerivative(self.xi_pg)
    

    def get_gp_elm_coordinates(self,n_elm_gp=None):
        if n_elm_gp is None: return self.xi_pg
        
        if n_elm_gp == 1: #ordre exact 1
            return np.array([[0, 0]])
        elif n_elm_gp == 4: # ordre exacte 2
            a = 1/np.sqrt(3)
            return np.array([[-a,-a], [a,-a], [a,a], [-a,a]])    #= np.c_[xi, eta]
        elif n_elm_gp == 9: # ordre exacte 3
            a = 0.774596669241483
            return  np.array([[-a,-a], [a,-a], [a,a], [-a,a], [0,-a], [a,0], [0,a], [-a,0], [0,0]]) 
        elif n_elm_gp == 16: # ordre exacte 5
            a = 0.339981043584856 ; b = 0.861136311594053
            return np.array([[-b,-b], [-a,-b],[a,-b],[b,-b],
                             [-b,-a], [-a,-a],[a,-a],[b,-a],
                             [-b,a], [-a,a],[a,a],[b,a],
                             [-b,b], [-a,b],[a,b],[b,b]])     
        else:
            assert 0, "Number of gauss points "+str(n_elm_gp)+" unavailable for triangle element"  


    def get_gp_weight(self,n_elm_gp=None):
        if n_elm_gp is None: return self.w_pg
            
        if n_elm_gp == 1:
            return np.array([4.])
        elif n_elm_gp == 4: # ordre exacte 2
            return np.array([1., 1., 1., 1.])
        elif n_elm_gp == 9: # ordre exacte 3
            return np.array([25/81., 25/81., 25/81., 25/81., 40/81., 40/81., 40/81., 40/81., 64/81.])
        elif n_elm_gp == 16: # ordre exacte 4
            w_a = 0.652145154862546 ; w_b = 0.347854845137454        
            return np.array([w_b**2, w_a*w_b, w_a*w_b, w_b**2, w_a*w_b, w_a**2, w_a**2, w_a*w_b, w_a*w_b, w_a**2, w_a**2, w_a*w_b, w_b**2, w_a*w_b, w_a*w_b, w_b**2])
        else:
            assert 0, "Number of gauss points "+str(n_elm_gp)+" unavailable for triangle element"  


class Quad4(ElementQuadrangle):
    name = 'quad4'
    default_n_gp = 4
    n_nodes = 4
    
    def __init__(self, n_elm_gp=4, **kargs):
        self.xi_nd =  np.c_[[-1. , 1., 1., -1.],\
                            [-1. , -1., 1., 1.]]
        self.n_elm_gp = n_elm_gp
        ElementQuadrangle.__init__(self, n_elm_gp)
            
    #In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta)
    #vec_xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    #vec_xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:,0] ; eta = vec_xi[:,1]
        return np.c_[ 0.25*(1-xi)*(1-eta) , 0.25*(1+xi)*(1-eta) , 0.25*(1+xi)*(1+eta) , 0.25*(1-xi)*(1+eta) ]
    def ShapeFunctionDerivative(self, vec_xi): 
        return [ np.array([ [0.25*(xi[1]-1), 0.25*(1-xi[1]), 0.25*(1+xi[1]), -0.25*(1+xi[1])] , [0.25*(xi[0]-1), -0.25*(1+xi[0]), 0.25*(1+xi[0]), 0.25*(1-xi[0])] ]) for xi in vec_xi]        

class Quad8(ElementQuadrangle):
    name = 'quad8'
    default_n_gp = 9
    n_nodes = 8
    
    def __init__(self, n_elm_gp=9, **kargs):
        self.xi_nd = np.c_[[-1. , 1. , 1., -1., 0., 1., 0.,-1.],\
                           [-1. , -1., 1., 1. ,-1., 0., 1., 0.]]
        self.n_elm_gp = n_elm_gp
        ElementQuadrangle.__init__(self, n_elm_gp)
            
    #In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta)
    #xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    #xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)  
    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:,0] ; eta = vec_xi[:,1]
        return np.c_[0.25*(1-xi)*(1-eta)*(-1-xi-eta) , 0.25*(1+xi)*(1-eta)*(-1+xi-eta), 0.25*(1+xi)*(1+eta)*(-1+xi+eta), 0.25*(1-xi)*(1+eta)*(-1-xi+eta), 0.5*(1-xi**2)*(1-eta) , 0.5*(1+xi)*(1-eta**2) , 0.5*(1-xi**2)*(1+eta), 0.5*(1-xi)*(1-eta**2) ]
    def ShapeFunctionDerivative(self, vec_xi): 
        return [ np.array([ [0.25*(1-xi[1])*(2*xi[0]+xi[1]), 0.25*( 1-xi[1])*(2*xi[0]-xi[1]), 0.25*(1+xi[1])*(2*xi[0]+xi[1]), 0.25*(-1-xi[1])*(-2*xi[0]+xi[1]), -xi[0]*(1-xi[1]) , 0.5*(1-xi[1]**2), -xi[0]*(1+xi[1]), -0.5*(1-xi[1]**2)] , \
                         [0.25*(1-xi[0])*(2*xi[1]+xi[0]), 0.25*(-1-xi[0])*(xi[0]-2*xi[1]), 0.25*(1+xi[0])*(2*xi[1]+xi[0]), 0.25*( 1-xi[0])*(-xi[0]+2*xi[1]), -0.5*(1-xi[0]**2), -xi[1]*(1+xi[0]), 0.5*(1-xi[0]**2), -xi[1]*(1-xi[0]) ] ]) for xi in vec_xi]        


class Quad9(ElementQuadrangle):
    name = 'quad9'
    default_n_gp = 9
    n_nodes = 9
    
    def __init__(self, n_elm_gp=9, **kargs):
        self.xi_nd =  np.c_[[-1. , 1. , 1., -1., 0., 1., 0.,-1., 0.],\
                            [-1. , -1., 1., 1. ,-1., 0., 1., 0., 0.]]
        self.n_elm_gp = n_elm_gp
        ElementQuadrangle.__init__(self, n_elm_gp)
        
    #In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta)
    #xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    #xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:,0] ; eta = vec_xi[:,1]
        return np.c_[0.25*xi*eta*(xi-1)*(eta-1) , 0.25*xi*eta*(xi+1)*(eta-1), 0.25*xi*eta*(xi+1)*(eta+1), 0.25*xi*eta*(xi-1)*(eta+1), 0.5*(1-xi**2)*eta*(eta-1), 0.5*xi*(xi+1)*(1-eta**2), 0.5*(1-xi**2)*eta*(eta+1), 0.5*xi*(xi-1)*(1-eta**2), (1-xi**2)*(1-eta**2) ]
    def ShapeFunctionDerivative(self, vec_xi):
        return [ np.array([ [0.25*xi[1]*(xi[1]-1)*(2*xi[0]-1) , 0.25*xi[1]*(xi[1]-1)*(2*xi[0]+1) , 0.25*xi[1]*(xi[1]+1)*(2*xi[0]+1) , 0.25*xi[1]*(xi[1]+1)*(2*xi[0]-1) , -xi[0]*xi[1]*(xi[1]-1)       , 0.5*(2*xi[0]+1)*(1-xi[1]**2) , -xi[0]*xi[1]*(xi[1]+1)       , 0.5*(2*xi[0]-1)*(1-xi[1]**2) , -2*xi[0]*(1-xi[1]**2) ] , \
                         [0.25*xi[0]*(xi[0]-1)*(2*xi[1]-1) , 0.25*xi[0]*(xi[0]+1)*(2*xi[1]-1) , 0.25*xi[0]*(xi[0]+1)*(2*xi[1]+1) , 0.25*xi[0]*(xi[0]-1)*(2*xi[1]+1) , 0.5*(1-xi[0]**2)*(2*xi[1]-1) , -xi[0]*xi[1]*(xi[0]+1)       , 0.5*(1-xi[0]**2)*(2*xi[1]+1) , -xi[0]*xi[1]*(xi[0]-1)          , -2*xi[1]*(1-xi[0]**2) ] ]) for xi in vec_xi]        
