import numpy as np
from fedoo.lib_elements.element_base import *

class ElementTriangle(Element2D):
    def __init__(self,n_elm_gp): 
        #initialize the gauss points and the associated weight
        if n_elm_gp == 1:
            xi = np.c_[[1./3]]
            eta = np.c_[[1./3]]
            w_pg = np.array([1./2])
        elif n_elm_gp == 3: # ordre exacte 2
            xi = np.c_[[1./6 , 2./3 , 1./6]]
            eta = np.c_[[1./6 , 1./6 , 2./3]]
            w_pg = np.array([1./6, 1./6, 1./6])
        elif n_elm_gp == 4: # ordre exacte 3
            xi = np.c_[[1./3 , 1./5 , 3./5 , 1./5 ]]
            eta = np.c_[[1./3 , 1./5 , 1./5 , 3./5 ]]
            w_pg = np.array([-27./96, 25./96, 25./96, 25./96])
        elif n_elm_gp == 7: # ordre exacte 5
            a_pg = (6.+np.sqrt(15))/21; b_pg = 4./7-a_pg
            AA_pg = (155.+np.sqrt(15))/2400; BB_pg = 31./240 - AA_pg                
            xi = np.c_[[1./3 , a_pg , 1-2*a_pg , a_pg , b_pg , 1-2*b_pg , b_pg ]]
            eta = np.c_[[1./3 , a_pg , a_pg , 1-2*a_pg , b_pg , b_pg , 1-2*b_pg ]]
            w_pg = np.array([9./80 , AA_pg , AA_pg , AA_pg , BB_pg , BB_pg , BB_pg ])
        elif n_elm_gp == 12: #ordre exacte 6
            a = 0.063089014491502; b = 0.249286745170910; c = 0.310352451033785; d = 0.053145049844816
            w1 = 0.025422453185103; w2 = 0.058393137863189; w3 = 0.041425537809187
            xi = np.c_[[ a , 1-2*a , a , b , 1-2*b , b , c , d , 1-(c+d) , 1-(c+d), c , d ]]
            eta = np.c_[[ a , a , 1-2*a , b , b , 1-2*b , d , c , c , d , 1-(c+d) , 1-(c+d) ]]
            w_pg = np.array([ w1 , w1 , w1 , w2 , w2 , w2 , w3 , w3 , w3 , w3 , w3 , w3 ])
        elif n_elm_gp == 0: #if n_elm_gp == 0, we take the position of the nodes
            self.xi_pg = self.xi_nd
        else:
            assert 0, "Number of gauss points "+str(n_elm_gp)+" unavailable for triangle element"           
                                                  
        if n_elm_gp != 0:    
            self.xi_pg = np.c_[xi,eta]
            self.w_pg = w_pg
            
        self.ShapeFunctionPG = self.ShapeFunction(self.xi_pg)
        self.ShapeFunctionDerivativePG = self.ShapeFunctionDerivative(self.xi_pg)
        
    

class Tri3(ElementTriangle):
    def __init__(self, n_elm_gp=3, **kargs):        
        self.xi_nd = np.c_[[0. , 1., 0.],\
                           [0. , 0., 1.]]
        self.n_elm_gp = n_elm_gp
        ElementTriangle.__init__(self,n_elm_gp)     
            
    #In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta)
    #xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    #xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
        
    def ShapeFunction(self, xi):       
        return np.c_[(1-xi[:,0]-xi[:,1]), xi[:,0], xi[:,1]]
    def ShapeFunctionDerivative(self, xi): 
        return [ np.array([[-1.,1.,0.],[-1.,0.,1.]]) for x in xi]        
    
class Tri3Bubble(Tri3):    
    def ShapeFunction(self, xi):    
        return np.c_[1-xi[:,0]-xi[:,1], xi[:,0], xi[:,1], (1-xi[:,0]-xi[:,1])*xi[:,0]*xi[:,1]] 
    def ShapeFunctionDerivative(self, xi):
        return [np.array([ [-1.,1.,0.,x[1]*(1-2*x[0]-x[1])] , [-1.,0.,1.,x[0]*(1-2*x[1]-x[0])] ]) for x in xi]                

class Tri6(ElementTriangle):
    def __init__(self, n_elm_gp=4, **kargs):
        self.xi_nd =  np.c_[[0. , 1., 0., 0.5, 0.5, 0. ],\
                            [0. , 0., 1., 0. , 0.5, 0.5]]  
        self.n_elm_gp = n_elm_gp
        ElementTriangle.__init__(self,n_elm_gp)     
            
    #In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta)
    #xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    #xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
    def ShapeFunction(self, xi):       
        return np.c_[(1-xi[:,0]-xi[:,1])*(1-2*xi[:,0]-2*xi[:,1]), xi[:,0]*(2*xi[:,0]-1), xi[:,1]*(2*xi[:,1]-1), 4*xi[:,0]*(1-xi[:,0]-xi[:,1]), 4*xi[:,0]*xi[:,1], 4*xi[:,1]*(1-xi[:,0]-xi[:,1])]
    def ShapeFunctionDerivative(self, xi): 
        return [ np.array([ [4*(x[0]+x[1])-3, 4*x[0]-1, 0., 4*(1-2*x[0]-x[1]), 4*x[1], -4*x[1]] , [4*(x[0]+x[1])-3, 0., 4*x[1]-1, -4*x[0], 4*x[0], 4*(1-x[0]-2*x[1])] ]) for x in xi]        