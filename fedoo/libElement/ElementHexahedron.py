import numpy as np
from fedoo.libElement.Element import *

class elementHexahedron(element):
    def __init__(self,nb_gp):                 
        #initialize the gauss points and the associated weight           
        if nb_gp == 1:
            xi = np.c_[[0.]] ; eta = np.c_[[0.]] ; zeta = np.c_[[0.]]
            w_pg = np.array([8.])    
        elif nb_gp == 8:
            a = 0.57735026918962573
            xi =  np.c_[[-a, -a, -a, -a, a, a, a, a]] ; eta = np.c_[[-a, -a, a,  a, -a, -a, a, a]] ; zeta = np.c_[[-a, a, -a, a, -a, a , -a, a]]
            w_pg = np.array([1., 1., 1., 1., 1., 1., 1., 1.])
        elif nb_gp == 27:
            a = 0.7745966692414834 ; b = 0.5555555555555556 ; c = 0.8888888888888888 
            xi =  np.c_[[-a, -a, -a, -a, -a, -a, -a, -a, -a, 0., 0., 0., 0., 0., 0., 0., 0., 0., a, a, a, a, a, a, a, a, a]] ; eta = np.c_[[-a, -a, -a, 0., 0., 0., a, a, a, -a, -a, -a, 0., 0., 0., a, a, a, -a, -a, -a, 0., 0., 0., a, a, a]] ; zeta = np.c_[[-a, 0., a, -a, 0., a, -a, 0., a, -a, 0., a, -a, 0., a, -a, 0., a, -a, 0., a, -a, 0., a, -a, 0., a]]
            w_pg = np.array([b**3, (b**2)*c, b**3, (b**2)*c, b*(c**2), (b**2)*c, b**3, (b**2)*c, b**3, (b**2)*c, b*(c**2), (b**2)*c, b*(c**2), c**3, b*(c**2), (b**2)*c, b*(c**2), (b**2)*c, b**3, (b**2)*c, b**3, (b**2)*c, b*(c**2), (b**2)*c, b**3, (b**2)*c, b**3])
        elif nb_gp == 0: 
            self.xi_pg = self.xi_nd #if nb_gp == 0, we take the position of the nodes            
        else:
            assert 0, "Number of gauss points "+str(nb_gp)+" unavailable for hexahedron element"

        if nb_gp != 0:    
            self.xi_pg = np.c_[xi,eta,zeta]
            self.w_pg = w_pg
        
        self.ShapeFunctionPG = self.ShapeFunction(self.xi_pg)
        self.ShapeFunctionDerivativePG = self.ShapeFunctionDerivative(self.xi_pg)

class hex8(elementHexahedron):
    def __init__(self, nb_gp=8, **kargs):
        self.xi_nd = np.c_[[-1. ,  1. , 1. , -1. , -1.,  1. , 1. ,-1.],\
                           [-1. , -1. , 1. ,  1. , -1., -1. , 1. , 1.],\
                           [-1. , -1. , -1., -1. ,  1.,  1. , 1. , 1.]]
        self.nb_gp = nb_gp
        elementHexahedron.__init__(self,nb_gp)
     
    #In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta, zeta)
    #vec_xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    #vec_xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
    #vec_xi[:,2] -> list of values of zeta for all points (gauss points in general but may be used with other points)
    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:,0] ; eta = vec_xi[:,1]; zeta = vec_xi[:,2]
        return np.c_[0.125*(1-xi)*(1-eta)*(1-zeta) , 0.125*(1+xi)*(1-eta)*(1-zeta), 0.125*(1+xi)*(1+eta)*(1-zeta), 0.125*(1-xi)*(1+eta)*(1-zeta), 0.125*(1-xi)*(1-eta)*(1+zeta) , 0.125*(1+xi)*(1-eta)*(1+zeta) , 0.125*(1+xi)*(1+eta)*(1+zeta), 0.125*(1-xi)*(1+eta)*(1+zeta) ]    
    def ShapeFunctionDerivative(self, vec_xi): 
        return [ np.array([ [-0.125*(1-xi[1])*(1-xi[2]) , 0.125*(1-xi[1])*(1-xi[2]), 0.125*(1+xi[1])*(1-xi[2]), -0.125*(1+xi[1])*(1-xi[2]), -0.125*(1-xi[1])*(1+xi[2]) , 0.125*(1-xi[1])*(1+xi[2]) , 0.125*(1+xi[1])*(1+xi[2]), -0.125*(1+xi[1])*(1+xi[2]) ] , \
                         [-0.125*(1-xi[0])*(1-xi[2]) , -0.125*(1+xi[0])*(1-xi[2]), 0.125*(1+xi[0])*(1-xi[2]), 0.125*(1-xi[0])*(1-xi[2]), -0.125*(1-xi[0])*(1+xi[2]) , -0.125*(1+xi[0])*(1+xi[2]) , 0.125*(1+xi[0])*(1+xi[2]), 0.125*(1-xi[0])*(1+xi[2]) ] , \
                         [-0.125*(1-xi[0])*(1-xi[1]) , -0.125*(1+xi[0])*(1-xi[1]), -0.125*(1+xi[0])*(1+xi[1]), -0.125*(1-xi[0])*(1+xi[1]), 0.125*(1-xi[0])*(1-xi[1]) , 0.125*(1+xi[0])*(1-xi[1]) , 0.125*(1+xi[0])*(1+xi[1]), 0.125*(1-xi[0])*(1+xi[1]) ] ]) for xi in vec_xi]      

class hex20(elementHexahedron):
    def __init__(self, nb_gp=27, **kargs):
        self.xi_nd = np.c_[[-1. ,  1. , 1. , -1. , -1.,  1. , 1. ,-1. , 0. ,  1. , 0. , -1. , -1.,  1. , 1. ,-1. , 0.,  1. , 0. ,-1.],\
                           [-1. , -1. , 1. ,  1. , -1., -1. , 1. , 1. , -1. , 0. , 1. ,  0. , -1., -1. , 1. , 1. , -1.,  0. , 1. ,0.],\
                           [-1. , -1. , -1., -1. ,  1.,  1. , 1. , 1. , -1. , -1. , -1., -1. ,  0.,  0. , 0. , 0. , 1.,  1. , 1. ,1.]]
        self.nb_gp = nb_gp 
        elementHexahedron.__init__(self,nb_gp)

    #In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta, zeta)
    #vec_xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    #vec_xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
    #vec_xi[:,2] -> list of values of zeta for all points (gauss points in general but may be used with other points)
    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:,0] ; eta = vec_xi[:,1]; zeta = vec_xi[:,2]
        return np.c_[0.125*(1-xi)*(1-eta)*(1-zeta)*(-2-xi-eta-zeta) , 0.125*(1+xi)*(1-eta)*(1-zeta)*(-2+xi-eta-zeta), 0.125*(1+xi)*(1+eta)*(1-zeta)*(-2+xi+eta-zeta), 0.125*(1-xi)*(1+eta)*(1-zeta)*(-2-xi+eta-zeta), 0.125*(1-xi)*(1-eta)*(1+zeta)*(-2-xi-eta+zeta) , 0.125*(1+xi)*(1-eta)*(1+zeta)*(-2+xi-eta+zeta) , 0.125*(1+xi)*(1+eta)*(1+zeta)*(-2+xi+eta+zeta), 0.125*(1-xi)*(1+eta)*(1+zeta)*(-2-xi+eta+zeta), \
                     0.25*(1-xi**2)*(1-eta)*(1-zeta), 0.25*(1-eta**2)*(1+xi)*(1-zeta), 0.25*(1-xi**2)*(1+eta)*(1-zeta), 0.25*(1-eta**2)*(1-xi)*(1-zeta), 0.25*(1-zeta**2)*(1-xi)*(1-eta), 0.25*(1-zeta**2)*(1+xi)*(1-eta), 0.25*(1-zeta**2)*(1+xi)*(1+eta), 0.25*(1-zeta**2)*(1-xi)*(1+eta), 0.25*(1-xi**2)*(1-eta)*(1+zeta), 0.25*(1-eta**2)*(1+xi)*(1+zeta), 0.25*(1-xi**2)*(1+eta)*(1+zeta), 0.25*(1-eta**2)*(1-xi)*(1+zeta)]    

    def ShapeFunctionDerivative(self, vec_xi): 
        return [ np.array([ [0.125*(1-xi[1])*(1-xi[2])*(1+2*xi[0]+xi[1]+xi[2]) , 0.125*(1-xi[1])*(1-xi[2])*(-1+2*xi[0]-xi[1]-xi[2]),  0.125*(1+xi[1])*(1-xi[2])*(-1+2*xi[0]+xi[1]-xi[2]), -0.125*(1+xi[1])*(1-xi[2])*(-1-2*xi[0]+xi[1]-xi[2]), -0.125*(1-xi[1])*(1+xi[2])*(-1-2*xi[0]-xi[1]+xi[2]),  0.125*(1-xi[1])*(1+xi[2])*(-1+2*xi[0]-xi[1]+xi[2]), 0.125*(1+xi[1])*(1+xi[2])*(-1+2*xi[0]+xi[1]+xi[2]), -0.125*(1+xi[1])*(1+xi[2])*(-1-2*xi[0]+xi[1]+xi[2]), -0.5*xi[0]*(1-xi[1])*(1-xi[2]),  0.25*(1-xi[1]**2)*(1-xi[2])  , -0.5*xi[0]*(1+xi[1])*(1-xi[2]), -0.25*(1-xi[1]**2)*(1-xi[2])  , -0.25*(1-xi[1])*(1-xi[2]**2)  ,  0.25*(1-xi[1])*(1-xi[2]**2)  ,  0.25*(1+xi[1])*(1-xi[2]**2)  , -0.25*(1+xi[1])*(1-xi[2]**2)  , -0.5*xi[0]*(1-xi[1])*(1+xi[2]),  0.25*(1-xi[1]**2)*(1+xi[2])  , -0.5*xi[0]*(1+xi[1])*(1+xi[2]), -0.25*(1-xi[1]**2)*(1+xi[2])  ] , \
                            [0.125*(1-xi[0])*(1-xi[2])*(1+xi[0]+2*xi[1]+xi[2]) , 0.125*(1+xi[0])*(1-xi[2])*(1-xi[0]+2*xi[1]+xi[2]) , 0.125*(1+xi[0])*(1-xi[2])*(-1+xi[0]+2*xi[1]-xi[2]) ,  0.125*(1-xi[0])*(1-xi[2])*(-1-xi[0]+2*xi[1]-xi[2]), -0.125*(1-xi[0])*(1+xi[2])*(-1-xi[0]-2*xi[1]+xi[2]), -0.125*(1+xi[0])*(1+xi[2])*(-1+xi[0]-2*xi[1]+xi[2]), 0.125*(1+xi[0])*(1+xi[2])*(-1+xi[0]+2*xi[1]+xi[2]),  0.125*(1-xi[0])*(1+xi[2])*(-1-xi[0]+2*xi[1]+xi[2]), -0.25*(1-xi[0]**2)*(1-xi[2])  , -0.5*xi[1]*(1+xi[0])*(1-xi[2]), 0.25*(1-xi[0]**2)*(1-xi[2])   , -0.5*xi[1]*(1-xi[0])*(1-xi[2]), -0.25*(1-xi[0])*(1-xi[2]**2)  , -0.25*(1+xi[0])*(1-xi[2]**2)  ,  0.25*(1+xi[0])*(1-xi[2]**2)  ,  0.25*(1-xi[0])*(1-xi[2]**2)  , -0.25*(1-xi[0]**2)*(1+xi[2])  , -0.5*xi[1]*(1+xi[0])*(1+xi[2]),  0.25*(1-xi[0]**2)*(1+xi[2])  , -0.5*xi[1]*(1-xi[0])*(1+xi[2])] , \
                            [0.125*(1-xi[0])*(1-xi[1])*(1+xi[0]+xi[1]+2*xi[2]) , 0.125*(1+xi[0])*(1-xi[1])*(1-xi[0]+xi[1]+2*xi[2]) , -0.125*(1+xi[0])*(1+xi[1])*(-1+xi[0]+xi[1]-2*xi[2]), -0.125*(1-xi[0])*(1+xi[1])*(-1-xi[0]+xi[1]-2*xi[2]), 0.125*(1-xi[0])*(1-xi[1])*(-1-xi[0]-xi[1]+2*xi[2]) ,  0.125*(1+xi[0])*(1-xi[1])*(-1+xi[0]-xi[1]+2*xi[2]), 0.125*(1+xi[0])*(1+xi[1])*(-1+xi[0]+xi[1]+2*xi[2]),  0.125*(1-xi[0])*(1+xi[1])*(-1-xi[0]+xi[1]+2*xi[2]), -0.25*(1-xi[0]**2)*(1-xi[1])  , -0.25*(1+xi[0])*(1-xi[1]**2)  , -0.25*(1-xi[0]**2)*(1+xi[1])  , -0.25*(1-xi[0])*(1-xi[1]**2)  , -0.5*xi[2]*(1-xi[0])*(1-xi[1]), -0.5*xi[2]*(1+xi[0])*(1-xi[1]), -0.5*xi[2]*(1+xi[0])*(1+xi[1]), -0.5*xi[2]*(1-xi[0])*(1+xi[1]),  0.25*(1-xi[0]**2)*(1-xi[1])  ,  0.25*(1+xi[0])*(1-xi[1]**2)  ,  0.25*(1-xi[0]**2)*(1+xi[1])  ,  0.25*(1-xi[0])*(1-xi[1]**2)  ] ]) for xi in vec_xi]