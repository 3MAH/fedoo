import numpy as np


from fedoo.lib_elements.element_base import Element


class ElementHexahedron(Element):
    def __init__(self, n_elm_gp):
        # initialize the gauss points and the associated weight
        if n_elm_gp == 0:  # if n_elm_gp == 0, we take the position of the nodes
            self.xi_pg = self.xi_nd
        else:
            self.xi_pg = self.get_gp_elm_coordinates(n_elm_gp)  # = np.c_[xi,eta]
            self.w_pg = self.get_gp_weight(n_elm_gp)

        self.ShapeFunctionPG = self.ShapeFunction(self.xi_pg)
        self.ShapeFunctionDerivativePG = self.ShapeFunctionDerivative(self.xi_pg)

    def get_gp_elm_coordinates(self, n_elm_gp):
        if n_elm_gp == 1:
            return np.array([[0.0, 0.0, 0.0]])  # = np.c_[xi,eta,zeta]
        elif n_elm_gp == 8:
            a = 0.5773502691896258  # 1/np.sqrt(3)
            xi = np.c_[[-a, -a, -a, -a, a, a, a, a]]
            eta = np.c_[[-a, -a, a, a, -a, -a, a, a]]
            zeta = np.c_[[-a, a, -a, a, -a, a, -a, a]]
            return np.c_[xi, eta, zeta]
        elif n_elm_gp == 27:
            a = 0.7745966692414834
            xi = np.c_[
                [
                    -a,
                    -a,
                    -a,
                    -a,
                    -a,
                    -a,
                    -a,
                    -a,
                    -a,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    a,
                    a,
                    a,
                    a,
                    a,
                    a,
                    a,
                    a,
                    a,
                ]
            ]
            eta = np.c_[
                [
                    -a,
                    -a,
                    -a,
                    0.0,
                    0.0,
                    0.0,
                    a,
                    a,
                    a,
                    -a,
                    -a,
                    -a,
                    0.0,
                    0.0,
                    0.0,
                    a,
                    a,
                    a,
                    -a,
                    -a,
                    -a,
                    0.0,
                    0.0,
                    0.0,
                    a,
                    a,
                    a,
                ]
            ]
            zeta = np.c_[
                [
                    -a,
                    0.0,
                    a,
                    -a,
                    0.0,
                    a,
                    -a,
                    0.0,
                    a,
                    -a,
                    0.0,
                    a,
                    -a,
                    0.0,
                    a,
                    -a,
                    0.0,
                    a,
                    -a,
                    0.0,
                    a,
                    -a,
                    0.0,
                    a,
                    -a,
                    0.0,
                    a,
                ]
            ]
            return np.c_[xi, eta, zeta]
        else:
            assert 0, (
                "Number of gauss points "
                + str(n_elm_gp)
                + " unavailable for hexahedron element"
            )

    def get_gp_weight(self, n_elm_gp):
        if n_elm_gp == 1:
            return np.array([8.0])
        elif n_elm_gp == 8:
            return np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        elif n_elm_gp == 27:
            b = 0.5555555555555556
            c = 0.8888888888888888
            return np.array(
                [
                    b**3,
                    (b**2) * c,
                    b**3,
                    (b**2) * c,
                    b * (c**2),
                    (b**2) * c,
                    b**3,
                    (b**2) * c,
                    b**3,
                    (b**2) * c,
                    b * (c**2),
                    (b**2) * c,
                    b * (c**2),
                    c**3,
                    b * (c**2),
                    (b**2) * c,
                    b * (c**2),
                    (b**2) * c,
                    b**3,
                    (b**2) * c,
                    b**3,
                    (b**2) * c,
                    b * (c**2),
                    (b**2) * c,
                    b**3,
                    (b**2) * c,
                    b**3,
                ]
            )
        else:
            assert 0, (
                "Number of gauss points "
                + str(n_elm_gp)
                + " unavailable for hexahedron element"
            )


class Hex8(ElementHexahedron):
    name = "hex8"
    default_n_gp = 8
    n_nodes = 8

    def __init__(self, n_elm_gp=8, **kargs):
        self.xi_nd = np.c_[
            [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        ]
        self.n_elm_gp = n_elm_gp
        ElementHexahedron.__init__(self, n_elm_gp)

    # In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta, zeta)
    # vec_xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    # vec_xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
    # vec_xi[:,2] -> list of values of zeta for all points (gauss points in general but may be used with other points)
    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:, 0]
        eta = vec_xi[:, 1]
        zeta = vec_xi[:, 2]
        return np.c_[
            0.125 * (1 - xi) * (1 - eta) * (1 - zeta),
            0.125 * (1 + xi) * (1 - eta) * (1 - zeta),
            0.125 * (1 + xi) * (1 + eta) * (1 - zeta),
            0.125 * (1 - xi) * (1 + eta) * (1 - zeta),
            0.125 * (1 - xi) * (1 - eta) * (1 + zeta),
            0.125 * (1 + xi) * (1 - eta) * (1 + zeta),
            0.125 * (1 + xi) * (1 + eta) * (1 + zeta),
            0.125 * (1 - xi) * (1 + eta) * (1 + zeta),
        ]

    def ShapeFunctionDerivative(self, vec_xi):
        return [
            np.array(
                [
                    [
                        -0.125 * (1 - xi[1]) * (1 - xi[2]),
                        0.125 * (1 - xi[1]) * (1 - xi[2]),
                        0.125 * (1 + xi[1]) * (1 - xi[2]),
                        -0.125 * (1 + xi[1]) * (1 - xi[2]),
                        -0.125 * (1 - xi[1]) * (1 + xi[2]),
                        0.125 * (1 - xi[1]) * (1 + xi[2]),
                        0.125 * (1 + xi[1]) * (1 + xi[2]),
                        -0.125 * (1 + xi[1]) * (1 + xi[2]),
                    ],
                    [
                        -0.125 * (1 - xi[0]) * (1 - xi[2]),
                        -0.125 * (1 + xi[0]) * (1 - xi[2]),
                        0.125 * (1 + xi[0]) * (1 - xi[2]),
                        0.125 * (1 - xi[0]) * (1 - xi[2]),
                        -0.125 * (1 - xi[0]) * (1 + xi[2]),
                        -0.125 * (1 + xi[0]) * (1 + xi[2]),
                        0.125 * (1 + xi[0]) * (1 + xi[2]),
                        0.125 * (1 - xi[0]) * (1 + xi[2]),
                    ],
                    [
                        -0.125 * (1 - xi[0]) * (1 - xi[1]),
                        -0.125 * (1 + xi[0]) * (1 - xi[1]),
                        -0.125 * (1 + xi[0]) * (1 + xi[1]),
                        -0.125 * (1 - xi[0]) * (1 + xi[1]),
                        0.125 * (1 - xi[0]) * (1 - xi[1]),
                        0.125 * (1 + xi[0]) * (1 - xi[1]),
                        0.125 * (1 + xi[0]) * (1 + xi[1]),
                        0.125 * (1 - xi[0]) * (1 + xi[1]),
                    ],
                ]
            )
            for xi in vec_xi
        ]


class Hex8r(Hex8):
    name = "hex8r"
    default_n_gp = 1
    n_nodes = 8

    # In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta, zeta)
    # vec_xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    # vec_xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
    # vec_xi[:,2] -> list of values of zeta for all points (gauss points in general but may be used with other points)
    def ShapeFunction(self, vec_xi):
        # return center value for all gp
        return 0.125 * np.ones((len(vec_xi), 8))

    def ShapeFunctionDerivative(self, vec_xi):
        return [
            0.125
            * np.array(
                [
                    [-1, 1, 1, -1, -1, 1, 1, -1],
                    [-1, -1, 1, 1, -1, -1, 1, 1],
                    [-1, -1, -1, -1, 1, 1, 1, 1],
                ]
            )
            for xi in vec_xi
        ]


class Hex20(ElementHexahedron):
    name = "hex20"
    default_n_gp = 27
    n_nodes = 20

    def __init__(self, n_elm_gp=27, **kargs):
        self.xi_nd = np.array(
            [
                [-1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0],
                [1.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, 1.0],
                [-1.0, 1.0, 1.0],
                [0.0, -1.0, -1.0],
                [1.0, 0.0, -1.0],
                [0.0, 1.0, -1.0],
                [-1.0, 0.0, -1.0],
                [0.0, -1.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [-1.0, 0.0, 1.0],
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ]
        )

        self.n_elm_gp = n_elm_gp
        ElementHexahedron.__init__(self, n_elm_gp)

    # In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta, zeta)
    # vec_xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    # vec_xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
    # vec_xi[:,2] -> list of values of zeta for all points (gauss points in general but may be used with other points)
    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:, 0]
        eta = vec_xi[:, 1]
        zeta = vec_xi[:, 2]
        return np.c_[
            0.125 * (1 - xi) * (1 - eta) * (1 - zeta) * (-2 - xi - eta - zeta),
            0.125 * (1 + xi) * (1 - eta) * (1 - zeta) * (-2 + xi - eta - zeta),
            0.125 * (1 + xi) * (1 + eta) * (1 - zeta) * (-2 + xi + eta - zeta),
            0.125 * (1 - xi) * (1 + eta) * (1 - zeta) * (-2 - xi + eta - zeta),
            0.125 * (1 - xi) * (1 - eta) * (1 + zeta) * (-2 - xi - eta + zeta),
            0.125 * (1 + xi) * (1 - eta) * (1 + zeta) * (-2 + xi - eta + zeta),
            0.125 * (1 + xi) * (1 + eta) * (1 + zeta) * (-2 + xi + eta + zeta),
            0.125 * (1 - xi) * (1 + eta) * (1 + zeta) * (-2 - xi + eta + zeta),
            0.25 * (1 - xi**2) * (1 - eta) * (1 - zeta),
            0.25 * (1 - eta**2) * (1 + xi) * (1 - zeta),
            0.25 * (1 - xi**2) * (1 + eta) * (1 - zeta),
            0.25 * (1 - eta**2) * (1 - xi) * (1 - zeta),
            0.25 * (1 - xi**2) * (1 - eta) * (1 + zeta),
            0.25 * (1 - eta**2) * (1 + xi) * (1 + zeta),
            0.25 * (1 - xi**2) * (1 + eta) * (1 + zeta),
            0.25 * (1 - eta**2) * (1 - xi) * (1 + zeta),
            0.25 * (1 - zeta**2) * (1 - xi) * (1 - eta),
            0.25 * (1 - zeta**2) * (1 + xi) * (1 - eta),
            0.25 * (1 - zeta**2) * (1 + xi) * (1 + eta),
            0.25 * (1 - zeta**2) * (1 - xi) * (1 + eta),
        ]

    def ShapeFunctionDerivative(self, vec_xi):
        return [
            np.array(
                [
                    [
                        0.125
                        * (1 - xi[1])
                        * (1 - xi[2])
                        * (1 + 2 * xi[0] + xi[1] + xi[2]),
                        0.125
                        * (1 - xi[1])
                        * (1 - xi[2])
                        * (-1 + 2 * xi[0] - xi[1] - xi[2]),
                        0.125
                        * (1 + xi[1])
                        * (1 - xi[2])
                        * (-1 + 2 * xi[0] + xi[1] - xi[2]),
                        -0.125
                        * (1 + xi[1])
                        * (1 - xi[2])
                        * (-1 - 2 * xi[0] + xi[1] - xi[2]),
                        -0.125
                        * (1 - xi[1])
                        * (1 + xi[2])
                        * (-1 - 2 * xi[0] - xi[1] + xi[2]),
                        0.125
                        * (1 - xi[1])
                        * (1 + xi[2])
                        * (-1 + 2 * xi[0] - xi[1] + xi[2]),
                        0.125
                        * (1 + xi[1])
                        * (1 + xi[2])
                        * (-1 + 2 * xi[0] + xi[1] + xi[2]),
                        -0.125
                        * (1 + xi[1])
                        * (1 + xi[2])
                        * (-1 - 2 * xi[0] + xi[1] + xi[2]),
                        -0.5 * xi[0] * (1 - xi[1]) * (1 - xi[2]),
                        0.25 * (1 - xi[1] ** 2) * (1 - xi[2]),
                        -0.5 * xi[0] * (1 + xi[1]) * (1 - xi[2]),
                        -0.25 * (1 - xi[1] ** 2) * (1 - xi[2]),
                        -0.5 * xi[0] * (1 - xi[1]) * (1 + xi[2]),
                        0.25 * (1 - xi[1] ** 2) * (1 + xi[2]),
                        -0.5 * xi[0] * (1 + xi[1]) * (1 + xi[2]),
                        -0.25 * (1 - xi[1] ** 2) * (1 + xi[2]),
                        -0.25 * (1 - xi[1]) * (1 - xi[2] ** 2),
                        0.25 * (1 - xi[1]) * (1 - xi[2] ** 2),
                        0.25 * (1 + xi[1]) * (1 - xi[2] ** 2),
                        -0.25 * (1 + xi[1]) * (1 - xi[2] ** 2),
                    ],
                    [
                        0.125
                        * (1 - xi[0])
                        * (1 - xi[2])
                        * (1 + xi[0] + 2 * xi[1] + xi[2]),
                        0.125
                        * (1 + xi[0])
                        * (1 - xi[2])
                        * (1 - xi[0] + 2 * xi[1] + xi[2]),
                        0.125
                        * (1 + xi[0])
                        * (1 - xi[2])
                        * (-1 + xi[0] + 2 * xi[1] - xi[2]),
                        0.125
                        * (1 - xi[0])
                        * (1 - xi[2])
                        * (-1 - xi[0] + 2 * xi[1] - xi[2]),
                        -0.125
                        * (1 - xi[0])
                        * (1 + xi[2])
                        * (-1 - xi[0] - 2 * xi[1] + xi[2]),
                        -0.125
                        * (1 + xi[0])
                        * (1 + xi[2])
                        * (-1 + xi[0] - 2 * xi[1] + xi[2]),
                        0.125
                        * (1 + xi[0])
                        * (1 + xi[2])
                        * (-1 + xi[0] + 2 * xi[1] + xi[2]),
                        0.125
                        * (1 - xi[0])
                        * (1 + xi[2])
                        * (-1 - xi[0] + 2 * xi[1] + xi[2]),
                        -0.25 * (1 - xi[0] ** 2) * (1 - xi[2]),
                        -0.5 * xi[1] * (1 + xi[0]) * (1 - xi[2]),
                        0.25 * (1 - xi[0] ** 2) * (1 - xi[2]),
                        -0.5 * xi[1] * (1 - xi[0]) * (1 - xi[2]),
                        -0.25 * (1 - xi[0] ** 2) * (1 + xi[2]),
                        -0.5 * xi[1] * (1 + xi[0]) * (1 + xi[2]),
                        0.25 * (1 - xi[0] ** 2) * (1 + xi[2]),
                        -0.5 * xi[1] * (1 - xi[0]) * (1 + xi[2]),
                        -0.25 * (1 - xi[0]) * (1 - xi[2] ** 2),
                        -0.25 * (1 + xi[0]) * (1 - xi[2] ** 2),
                        0.25 * (1 + xi[0]) * (1 - xi[2] ** 2),
                        0.25 * (1 - xi[0]) * (1 - xi[2] ** 2),
                    ],
                    [
                        0.125
                        * (1 - xi[0])
                        * (1 - xi[1])
                        * (1 + xi[0] + xi[1] + 2 * xi[2]),
                        0.125
                        * (1 + xi[0])
                        * (1 - xi[1])
                        * (1 - xi[0] + xi[1] + 2 * xi[2]),
                        -0.125
                        * (1 + xi[0])
                        * (1 + xi[1])
                        * (-1 + xi[0] + xi[1] - 2 * xi[2]),
                        -0.125
                        * (1 - xi[0])
                        * (1 + xi[1])
                        * (-1 - xi[0] + xi[1] - 2 * xi[2]),
                        0.125
                        * (1 - xi[0])
                        * (1 - xi[1])
                        * (-1 - xi[0] - xi[1] + 2 * xi[2]),
                        0.125
                        * (1 + xi[0])
                        * (1 - xi[1])
                        * (-1 + xi[0] - xi[1] + 2 * xi[2]),
                        0.125
                        * (1 + xi[0])
                        * (1 + xi[1])
                        * (-1 + xi[0] + xi[1] + 2 * xi[2]),
                        0.125
                        * (1 - xi[0])
                        * (1 + xi[1])
                        * (-1 - xi[0] + xi[1] + 2 * xi[2]),
                        -0.25 * (1 - xi[0] ** 2) * (1 - xi[1]),
                        -0.25 * (1 + xi[0]) * (1 - xi[1] ** 2),
                        -0.25 * (1 - xi[0] ** 2) * (1 + xi[1]),
                        -0.25 * (1 - xi[0]) * (1 - xi[1] ** 2),
                        0.25 * (1 - xi[0] ** 2) * (1 - xi[1]),
                        0.25 * (1 + xi[0]) * (1 - xi[1] ** 2),
                        0.25 * (1 - xi[0] ** 2) * (1 + xi[1]),
                        0.25 * (1 - xi[0]) * (1 - xi[1] ** 2),
                        -0.5 * xi[2] * (1 - xi[0]) * (1 - xi[1]),
                        -0.5 * xi[2] * (1 + xi[0]) * (1 - xi[1]),
                        -0.5 * xi[2] * (1 + xi[0]) * (1 + xi[1]),
                        -0.5 * xi[2] * (1 - xi[0]) * (1 + xi[1]),
                    ],
                ]
            )
            for xi in vec_xi
        ]


#### Hourglass control shape function ####


class Hex8Hourglass(Hex8):
    name = "hex8hourglass"
    default_n_gp = 1
    n_nodes = 8

    def __init__(self, n_elm_gp=1, **kargs):
        assembly = kargs.get("assembly", None)
        if assembly is not None:
            self.x_nd = assembly.mesh.nodes[assembly.mesh.elements]
            self.mesh = assembly.mesh
        else:
            assert 0, "Internal error, contact developper"
        if n_elm_gp != 1:
            raise ValueError(
                "Hourglass stiffness only applicable with " "1 integration point"
            )
        self._b_matrix = self._get_b_matrix()
        assembly._b_matrix = self._b_matrix  # for use in weakform
        super().__init__(n_elm_gp, **kargs)

    def ShapeFunction(self, vec_xi):
        list_h = [
            np.array([[1, 1, -1, -1, -1, -1, 1, 1]]),  # 1st mode
            np.array([[1, -1, -1, 1, -1, 1, 1, -1]]),  # 2nd mode
            np.array([[1, -1, 1, -1, 1, -1, 1, -1]]),  # 3rd mode
            np.array([[-1, 1, -1, 1, 1, -1, 1, -1]]),
        ]  # mode 4
        b = self._b_matrix

        return [
            (1 / np.sqrt(8))
            * (
                h
                - (self.x_nd[:, :, 0] @ h.T) * b[0].T
                - (self.x_nd[:, :, 1] @ h.T) * b[1].T
                - (self.x_nd[:, :, 2] @ h.T) * b[2].T
            )[:, np.newaxis, :]
            for h in list_h
        ]
        # list (len = nb hourglass mode)
        # of array of shape = (Nel, Nb_pg=1, Nddl=4)

    def _get_b_matrix(self):
        xi_center = self.get_gp_elm_coordinates(1)
        self.ComputeJacobianMatrix(self.x_nd, xi_center)
        # should try to avoid computing multiple time J
        return (self.inverseJacobian @ self.ShapeFunctionDerivative(xi_center)[0])[
            :, 0
        ].transpose(1, 2, 0)
