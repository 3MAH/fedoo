import numpy as np

from fedoo.lib_elements.element_base import Element1D, Element1DGeom2


class Lin2(Element1DGeom2, Element1D):
    name = "lin2"
    default_n_gp = 2
    n_nodes = 2

    def __init__(self, n_elm_gp=2, **kargs):
        self.xi_nd = np.c_[[0.0, 1.0]]
        self.n_elm_gp = n_elm_gp
        Element1D.__init__(self, n_elm_gp)

    # Dans les fonctions suivantes, xi doit toujours être une matrice colonne
    def ShapeFunction(self, xi):
        return np.c_[(1 - xi), xi]

    def ShapeFunctionDerivative(self, xi):
        return [np.array([[-1.0, 1.0]]) for x in xi]


class Lin2Bubble(Lin2):
    name = "lin2bubble"
    n_nodes = 3

    def ShapeFunction(self, xi):
        return np.c_[(1 - xi), xi, xi * (1 - xi)]

    def ShapeFunctionDerivative(self, xi):
        return [np.array([[-1.0, 1.0, 1.0 - 2 * x]]) for x in xi[:, 0]]


class Lin3(Element1D):
    name = "lin3"
    default_n_gp = 3
    n_nodes = 3

    def __init__(self, n_elm_gp=3, **kargs):
        self.xi_nd = np.c_[[0.0, 1.0, 0.5]]
        self.n_elm_gp = n_elm_gp
        Element1D.__init__(self, n_elm_gp)

    def ShapeFunction(self, xi):
        return np.c_[2 * xi**2 - 3 * xi + 1, xi * (2 * xi - 1), 4 * xi * (1 - xi)]

    def ShapeFunctionDerivative(self, xi):
        return [np.array([[4 * x - 3, 4 * x - 1, 4 - 8 * x]]) for x in xi[:, 0]]


class Lin3Bubble(Lin3):
    def ShapeFunction(self, xi):
        return np.c_[
            2 * xi**2 - 3 * xi + 1,
            xi * (2 * xi - 1),
            4 * xi * (1 - xi),
            64.0 / 3 * xi**3 - 32 * xi**2 + 32.0 / 3 * xi,
        ]

    def ShapeFunctionDerivative(self, xi):
        return [
            np.array([[4 * x - 3, 4 * x - 1, 4 - 8 * x, 64 * x**2 - 64 * x + 32.0 / 3]])
            for x in xi[:, 0]
        ]


# lin4 needs to modify the initial position of nodes for compatibility with other elements
# class lin4(Element1D):
# name = 'lin4'
# default_n_gp = 4
# n_nodes = 4

#    def __init__(self, n_elm_gp=4, avec_bulle = 0, **kargs):
#        self.xi_nd = np.c_[[-1., 1., -1./3, 1./3]]
#        self.n_elm_gp = n_elm_gp
#        Element1D.__init__(self, n_elm_gp)
#
#    def nn(self,xi):
#        return np.c_[-4.5*xi**3+9*xi**2-5.5*xi+1, 4.5*xi**3-4.5*xi**2+xi, 13.5*xi**3-22.5*xi**2+9*xi, -13.5*xi**3+18*xi**2-4.5*xi]
#    def dnn(self,xi):
#        return [np.array([[-13.5*x**2+18*x-5.5, 13.5*x**2-9*x+1, 40.5*x**2-45*x+9, -40.5*x**2+36*x-4.5]]) for x in xi[:,0]]
