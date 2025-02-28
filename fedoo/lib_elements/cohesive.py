import numpy as np
from numpy import linalg
from fedoo.lib_elements.element_base import Element, Element1DGeom2, Element1D
from fedoo.lib_elements.quadrangle import ElementQuadrangle
from fedoo.lib_elements.element_list import CombinedElement


class Node2Jump(Element1DGeom2):
    name = "node_jump"
    default_n_gp = 1
    n_nodes = 2

    def __init__(self, n_elm_gp=1, **kargs):
        self.n_elm_gp = 1  # pas de point de gauss pour les éléments cohésifs car pas de point d'intégration
        self.xi_pg = np.array([[0.0]])
        self.xi_nd = np.c_[
            [0.0, 0.0]
        ]  # The values are arbitrary, only the size is important
        self.w_pg = np.array([1.0])
        self.ShapeFunctionPG = self.ShapeFunction(self.xi_pg)

    def ComputeJacobianMatrix(self, vec_x, vec_xi, local_frame=None):
        ### local_frame not used here ###
        self.detJ = [1.0 for xi in vec_xi]

    # In the following functions, xi shall always be a column matrix
    def ShapeFunction(self, xi):
        return np.array([[-1.0, 1.0] for x in xi])

    def ShapeFunctionDerivative(self, xi):  # inutile en principe
        return [np.array([[0.0, 0.0]]) for x in xi]


#    def GeometricalShapeFunction(self,xi):
#        return 0.5*np.array([[1., 1.] for x in xi])


class Lin2InterfaceJump(Element1DGeom2, Element1D):
    name = "lin2interface_jump"
    default_n_gp = 2
    n_nodes = 4
    # local_csys = True

    def __init__(self, n_elm_gp=2, **kargs):
        """
        An element is defined with 4 nodes [0, 1, 2, 3]
        [0, 1] is a lin2 defining the face on the negative side of the cohesive zone
        [2, 3] is a lin2 defining the face on the positive side of the cohesive zone
        Node 0 is in front of node 2 and node 1 is in front of node 3
        """
        Element1D.__init__(self, n_elm_gp)
        self.xi_nd = np.c_[[0.0, 1.0]]
        self.n_elm_gp = n_elm_gp

    def ShapeFunction(self, xi):
        return np.c_[-xi, -1 + xi, xi, 1 - xi]

    def ShapeFunctionDerivative(self, xi):  # is it required for cohesive elements ?
        return [np.array([[-1.0, 1.0, 1.0, -1.0]]) for x in xi]


#    def GeometricalShapeFunction(self,xi):
#        return 0.5*np.c_[xi, 1-xi, xi, 1-xi]


class Quad4InterfaceJump(ElementQuadrangle):  # à vérifier
    name = "quad4interface_jump"
    default_n_gp = 4
    n_nodes = 8
    # local_csys = True

    def __init__(self, n_elm_gp=4, **kargs):
        """
        An element is defined with 8 nodes [0, 1, 2, 3, 4, 5, 6, 7]
        [0, 1, 2, 3] is a quad4 defining the face on the negative side of the cohesive zone
        [4, 5, 6, 7] is a quad4 defining the face on the positive side of the cohesive zone
        """
        self.xi_nd = np.c_[
            [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
        ]
        self.n_elm_gp = n_elm_gp
        ElementQuadrangle.__init__(self, n_elm_gp)

    # Dans les fonctions suivantes vec_xi contient une liste de points dans le repère de référence (xi, eta)
    # vec_xi[:,0] -> liste des valeurs de xi pour chaque point (points de gauss en général)
    # vec_xi[:,1] -> liste des valeurs de eta pour chaque point (points de gauss en général)
    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:, 0]
        eta = vec_xi[:, 1]
        return np.c_[
            -0.25 * (1 - xi) * (1 - eta),
            -0.25 * (1 + xi) * (1 - eta),
            -0.25 * (1 + xi) * (1 + eta),
            -0.25 * (1 - xi) * (1 + eta),
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta),
        ]

    def ShapeFunctionDerivative(
        self, vec_xi
    ):  # quad4 shape functions based on the mean values from two adjacent nodes
        return [
            0.5
            * np.array(
                [
                    [
                        0.25 * (xi[1] - 1),
                        0.25 * (1 - xi[1]),
                        0.25 * (1 + xi[1]),
                        -0.25 * (1 + xi[1]),
                        0.25 * (xi[1] - 1),
                        0.25 * (1 - xi[1]),
                        0.25 * (1 + xi[1]),
                        -0.25 * (1 + xi[1]),
                    ],
                    [
                        0.25 * (xi[0] - 1),
                        -0.25 * (1 + xi[0]),
                        0.25 * (1 + xi[0]),
                        0.25 * (1 - xi[0]),
                        0.25 * (xi[0] - 1),
                        -0.25 * (1 + xi[0]),
                        0.25 * (1 + xi[0]),
                        0.25 * (1 - xi[0]),
                    ],
                ]
            )
            for xi in vec_xi
        ]


class Node2Middle(Node2Jump):
    # define the midle between two nodes.
    name = "node2middle"
    default_n_gp = 1
    n_nodes = 2

    # Dans les fonctions suivantes, xi doit toujours être une matrice colonne
    def ShapeFunction(self, xi):
        return 0.5 * np.c_[1 + 0 * xi, 1 + 0 * xi]


class Lin2MeanPlane(Element1D):
    # define the mean plane between two lin2 elements defining an interface (used for cohesive elements).
    name = "lin2meanplane"
    default_n_gp = 2
    n_nodes = 4

    def __init__(self, n_elm_gp=4, **kargs):
        self.xi_nd = np.c_[[0.0, 1.0, 0.0, 1.0]]
        self.n_elm_gp = n_elm_gp
        Element1D.__init__(self, n_elm_gp)

    # Dans les fonctions suivantes, xi doit toujours être une matrice colonne
    def ShapeFunction(self, xi):
        return 0.5 * np.c_[(1 - xi), xi, (1 - xi), xi]

    def ShapeFunctionDerivative(self, xi):
        return [0.5 * np.array([[-1.0, 1.0, -1.0, 1.0]]) for x in xi]


class Quad4MeanPlane(ElementQuadrangle):
    # interpolate the mean plane between two quad4 elements, used for an interface element (cohesive)
    name = "quad4meanplane"
    default_n_gp = 4
    n_nodes = 8

    def __init__(self, n_elm_gp=4, **kargs):
        self.xi_nd = np.c_[
            [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
        ]
        self.n_elm_gp = n_elm_gp
        ElementQuadrangle.__init__(self, n_elm_gp)

    # In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta)
    # vec_xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    # vec_xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:, 0]
        eta = vec_xi[:, 1]
        return (
            0.5
            * np.c_[
                0.25 * (1 - xi) * (1 - eta),
                0.25 * (1 + xi) * (1 - eta),
                0.25 * (1 + xi) * (1 + eta),
                0.25 * (1 - xi) * (1 + eta),
                0.25 * (1 - xi) * (1 - eta),
                0.25 * (1 + xi) * (1 - eta),
                0.25 * (1 + xi) * (1 + eta),
                0.25 * (1 - xi) * (1 + eta),
            ]
        )

    def ShapeFunctionDerivative(self, vec_xi):
        return [
            0.5
            * np.array(
                [
                    [
                        0.25 * (xi[1] - 1),
                        0.25 * (1 - xi[1]),
                        0.25 * (1 + xi[1]),
                        -0.25 * (1 + xi[1]),
                        0.25 * (xi[1] - 1),
                        0.25 * (1 - xi[1]),
                        0.25 * (1 + xi[1]),
                        -0.25 * (1 + xi[1]),
                    ],
                    [
                        0.25 * (xi[0] - 1),
                        -0.25 * (1 + xi[0]),
                        0.25 * (1 + xi[0]),
                        0.25 * (1 - xi[0]),
                        0.25 * (xi[0] - 1),
                        -0.25 * (1 + xi[0]),
                        0.25 * (1 + xi[0]),
                        0.25 * (1 - xi[0]),
                    ],
                ]
            )
            for xi in vec_xi
        ]


Node2 = CombinedElement("spring", Node2Jump, default_n_gp=1, local_csys=True)
Node2.geometry_elm = Node2Middle

Quad4Interface = CombinedElement(
    "quad4interface", Quad4InterfaceJump, default_n_gp=4, local_csys=True
)
Quad4Interface.geometry_elm = Quad4MeanPlane

Lin2Interface = CombinedElement(
    "lin2interface", Lin2InterfaceJump, default_n_gp=2, local_csys=True
)
Lin2Interface.geometry_elm = Lin2MeanPlane
