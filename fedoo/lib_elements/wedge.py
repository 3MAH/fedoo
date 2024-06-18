import numpy as np

from fedoo.lib_elements.element_base import Element


class ElementWedge(Element):
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
            return np.array([[0.0, 0.5, 0.5]])  # = np.c_[xi,eta,zeta]
        elif n_elm_gp == 6:  # order 3 in xi , order 2 in eta and zeta
            a = 0.5773502691896258  # 1/np.sqrt(3)
            xi = np.c_[[-a, -a, -a, a, a, a]]
            eta = np.c_[[0.5, 0.0, 0.5, 0.5, 0.0, 0.5]]
            zeta = np.c_[[0.5, 0.5, 0.0, 0.5, 0.5, 0.0]]
            return np.c_[xi, eta, zeta]
        elif n_elm_gp == 8:  # order 3 in xi, eta and zeta
            a = 0.5773502691896258  # 1/np.sqrt(3)
            xi = np.c_[[-a, -a, -a, -a, a, a, a, a]]
            eta = np.c_[[1 / 3, 0.6, 0.2, 0.2, 1 / 3, 0.6, 0.2, 0.2]]
            zeta = np.c_[[1 / 3, 0.2, 0.6, 0.2, 1 / 3, 0.2, 0.6, 0.2]]
            return np.c_[xi, eta, zeta]
        elif n_elm_gp == 21:
            # alpha = np.sqrt(3/5)
            # a = (6+np.sqrt(15))/21 ; b = (6-np.sqrt(15))/21
            a = 0.47014206410511505
            b = 0.10128650732345633
            alpha = 0.7745966692414834
            xi = np.c_[
                [
                    -alpha,
                    -alpha,
                    -alpha,
                    -alpha,
                    -alpha,
                    -alpha,
                    -alpha,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    alpha,
                    alpha,
                    alpha,
                    alpha,
                    alpha,
                    alpha,
                    alpha,
                ]
            ]
            # 16th term in eta could be b. Need to be checked
            eta = np.c_[
                [
                    1 / 3,
                    a,
                    1 - 2 * a,
                    a,
                    b,
                    1 - 2 * b,
                    b,
                    1 / 3,
                    a,
                    1 - 2 * a,
                    a,
                    b,
                    1 - 2 * b,
                    b,
                    1 / 3,
                    a,
                    1 - 2 * a,
                    a,
                    b,
                    1 - 2 * b,
                    b,
                ]
            ]
            zeta = np.c_[
                [
                    1 / 3,
                    a,
                    a,
                    1 - 2 * a,
                    b,
                    b,
                    1 - 2 * b,
                    1 / 3,
                    a,
                    a,
                    1 - 2 * a,
                    b,
                    b,
                    1 - 2 * b,
                    1 / 3,
                    a,
                    a,
                    1 - 2 * a,
                    b,
                    b,
                    1 - 2 * b,
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
            return np.array([1.0])
        elif n_elm_gp == 6:
            return np.array([1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])
        elif n_elm_gp == 8:
            return np.array(
                [
                    -27 / 96,
                    25 / 96,
                    25 / 96,
                    25 / 96,
                    -27 / 96,
                    25 / 96,
                    25 / 96,
                    25 / 96,
                ]
            )
        elif n_elm_gp == 21:
            # c1 = 5/9 ; c2 = 8/9
            c1 = 0.5555555555555556
            c2 = 0.8888888888888888
            d = 9 / 80
            e1 = (155 + np.sqrt(15)) / 2400
            e2 = (155 - np.sqrt(15)) / 2400
            return np.array(
                [
                    c1 * d,
                    c1 * e1,
                    c1 * e1,
                    c1 * e1,
                    c1 * e2,
                    c1 * e2,
                    c1 * e2,
                    c2 * d,
                    c2 * e1,
                    c2 * e1,
                    c2 * e1,
                    c2 * e2,
                    c2 * e2,
                    c2 * e2,
                    c1 * d,
                    c1 * e1,
                    c1 * e1,
                    c1 * e1,
                    c1 * e2,
                    c1 * e2,
                    c1 * e2,
                ]
            )
        else:
            assert 0, (
                "Number of gauss points "
                + str(n_elm_gp)
                + " unavailable for hexahedron element"
            )


class Wed6(ElementWedge):
    name = "wed6"
    default_n_gp = 6
    n_nodes = 6

    def __init__(self, n_elm_gp=6, **kargs):
        self.xi_nd = np.c_[
            [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        ]
        self.n_elm_gp = n_elm_gp
        ElementWedge.__init__(self, n_elm_gp)

    # In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta, zeta)
    # vec_xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    # vec_xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
    # vec_xi[:,2] -> list of values of zeta for all points (gauss points in general but may be used with other points)
    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:, 0]
        eta = vec_xi[:, 1]
        zeta = vec_xi[:, 2]
        return np.c_[
            0.5 * (1 - xi) * eta,
            0.5 * (1 - xi) * zeta,
            0.5 * (1 - xi) * (1 - eta - zeta),
            0.5 * (1 + xi) * eta,
            0.5 * (1 + xi) * zeta,
            0.5 * (1 + xi) * (1 - eta - zeta),
        ]

    def ShapeFunctionDerivative(self, vec_xi):
        return [
            np.array(
                [
                    [
                        -0.5 * xi[1],
                        -0.5 * xi[2],
                        -0.5 * (1 - xi[1] - xi[2]),
                        0.5 * xi[1],
                        0.5 * xi[2],
                        0.5 * (1 - xi[1] - xi[2]),
                    ],
                    [
                        0.5 * (1 - xi[0]),
                        0 * xi[1],
                        -0.5 * (1 - xi[0]),
                        0.5 * (1 + xi[0]),
                        0 * xi[1],
                        -0.5 * (1 + xi[0]),
                    ],
                    [
                        0 * xi[2],
                        0.5 * (1 - xi[0]),
                        -0.5 * (1 - xi[0]),
                        0 * xi[2],
                        0.5 * (1 + xi[0]),
                        -0.5 * (1 + xi[0]),
                    ],
                ]
            )
            for xi in vec_xi
        ]


class Wed15(ElementWedge):
    name = "wed15"
    default_n_gp = 21
    n_nodes = 15

    def __init__(self, n_elm_gp=8, **kargs):
        self.xi_nd = np.c_[
            [
                -1.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ],
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0],
        ]
        self.n_elm_gp = n_elm_gp
        ElementWedge.__init__(self, n_elm_gp)

    # In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta, zeta)
    # vec_xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    # vec_xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
    # vec_xi[:,2] -> list of values of zeta for all points (gauss points in general but may be used with other points)
    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:, 0]
        eta = vec_xi[:, 1]
        zeta = vec_xi[:, 2]
        return np.c_[
            0.5 * eta * (1 - xi) * (2 * eta - 2 - xi),
            0.5 * zeta * (1 - xi) * (2 * zeta - 2 - xi),
            0.5 * (xi - 1) * (1 - zeta - eta) * (xi + 2 * zeta + 2 * eta),
            0.5 * eta * (1 + xi) * (2 * eta - 2 + xi),
            0.5 * zeta * (1 + xi) * (2 * zeta - 2 + xi),
            0.5 * (-xi - 1) * (1 - zeta - eta) * (-xi + 2 * zeta + 2 * eta),
            2 * eta * zeta * (1 - xi),
            2 * zeta * (1 - eta - zeta) * (1 - xi),
            2 * eta * (1 - eta - zeta) * (1 - xi),
            2 * eta * zeta * (1 + xi),
            2 * zeta * (1 - eta - zeta) * (1 + xi),
            2 * eta * (1 - eta - zeta) * (1 + xi),
            eta * (1 - xi**2),
            zeta * (1 - xi**2),
            (1 - xi**2) * (1 - eta - zeta),
        ]

    def ShapeFunctionDerivative(self, vec_xi):
        xi = vec_xi[:, 0]
        eta = vec_xi[:, 1]
        zeta = vec_xi[:, 2]
        dn_dxi = np.array(
            [
                0.5 * eta * (-2 * eta + 1 + 2 * xi),
                0.5 * zeta * (-2 * zeta + 1 + 2 * xi),
                0.5 * (1 - zeta - eta) * (2 * xi + 2 * zeta + 2 * eta - 1),
                0.5 * eta * (2 * eta - 1 + 2 * xi),
                0.5 * zeta * (2 * zeta - 1 + 2 * xi),
                -0.5 * (1 - zeta - eta) * (-2 * xi + 2 * zeta + 2 * eta - 1),
                -2 * eta * zeta,
                -2 * zeta * (1 - eta - zeta),
                -2 * eta * (1 - eta - zeta),
                2 * eta * zeta,
                2 * zeta * (1 - eta - zeta),
                2 * eta * (1 - eta - zeta),
                -eta * 2 * xi,
                -zeta * 2 * xi,
                -2 * xi * (1 - eta - zeta),
            ]
        )
        dn_deta = np.array(
            [
                0.5 * (1 - xi) * (4 * eta - 2 - xi),
                0.0 * eta,
                0.5 * (xi - 1) * (-xi - 4 * zeta - 4 * eta + 2),
                0.5 * (1 + xi) * (4 * eta - 2 + xi),
                0.0 * eta,
                0.5 * (-xi - 1) * (+xi - 4 * zeta - 4 * eta + 2),
                2 * zeta * (1 - xi),
                -2 * zeta * (1 - xi),
                2 * (1 - 2 * eta - zeta) * (1 - xi),
                2 * zeta * (1 + xi),
                -2 * zeta * (1 + xi),
                2 * (1 - 2 * eta - zeta) * (1 + xi),
                (1 - xi**2),
                0.0 * eta,
                -(1 - xi**2),
            ]
        )
        dn_dzeta = np.array(
            [
                0.0 * zeta,
                0.5 * (1 - xi) * (4 * zeta - 2 - xi),
                0.5 * (xi - 1) * (2 - 4 * eta - xi - 4 * zeta),
                0.0 * zeta,
                0.5 * (1 + xi) * (4 * zeta - 2 + xi),
                0.5 * (-xi - 1) * (2 - 4 * eta + xi - 4 * zeta),
                2 * eta * (1 - xi),
                2 * (1 - eta - 2 * zeta) * (1 - xi),
                -2 * eta * (1 - xi),
                2 * eta * (1 + xi),
                2 * (1 - eta - 2 * zeta) * (1 + xi),
                -2 * eta * (1 + xi),
                0.0 * zeta,
                (1 - xi**2),
                -(1 - xi**2),
            ]
        )
        return np.array([dn_dxi, dn_deta, dn_dzeta]).transpose(2, 0, 1)
        # return [np.array([dn_dxi[:,i], dn_deta[:,i], dn_dzeta[:,i]]) for i in range(len(vec_xi))]


class Wed18(ElementWedge):
    name = "wed18"
    default_n_gp = 21
    n_nodes = 18

    def __init__(self, n_elm_gp=8, **kargs):
        self.xi_nd = np.c_[
            [
                -1.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
                -1.0,
                -1.0,
                -1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.5,
                0.0,
                0.5,
                0.5,
                0.0,
                0.5,
                1.0,
                0.0,
                0.0,
                0.5,
                0.0,
                0.5,
            ],
            [
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.5,
                0.5,
                0.0,
                0.5,
                0.5,
                0.0,
                0.0,
                1.0,
                0.0,
                0.5,
                0.5,
                0.0,
            ],
        ]
        self.n_elm_gp = n_elm_gp
        ElementWedge.__init__(self, n_elm_gp)

    # In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta, zeta)
    # vec_xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    # vec_xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
    # vec_xi[:,2] -> list of values of zeta for all points (gauss points in general but may be used with other points)
    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:, 0]
        eta = vec_xi[:, 1]
        zeta = vec_xi[:, 2]
        return np.c_[
            0.5 * xi * eta * (xi - 1) * (2 * eta - 1),
            0.5 * xi * zeta * (xi - 1) * (2 * zeta - 1),
            0.5 * xi * (xi - 1) * (zeta + eta - 1) * (2 * zeta + 2 * eta - 1),
            0.5 * xi * eta * (xi + 1) * (2 * eta - 1),
            0.5 * xi * zeta * (xi + 1) * (2 * zeta - 1),
            0.5 * xi * (xi + 1) * (zeta + eta - 1) * (2 * zeta + 2 * eta - 1),
            2 * xi * eta * zeta * (xi - 1),
            -2 * xi * zeta * (xi - 1) * (eta + zeta - 1),
            -2 * xi * eta * (xi - 1) * (eta + zeta - 1),
            2 * xi * eta * zeta * (xi + 1),
            -2 * xi * zeta * (xi + 1) * (eta + zeta - 1),
            -2 * xi * eta * (xi + 1) * (eta + zeta - 1),
            eta * (1 - xi**2) * (2 * eta - 1),
            zeta * (1 - xi**2) * (2 * zeta - 1),
            (1 - xi**2) * (eta + zeta - 1) * (2 * eta + 2 * zeta - 1),
            4 * eta * zeta * (1 - xi**2),
            4 * zeta * (xi**2 - 1) * (eta + zeta - 1),
            4 * eta * (xi**2 - 1) * (eta + zeta - 1),
        ]

    def ShapeFunctionDerivative(self, vec_xi):
        xi = vec_xi[:, 0]
        eta = vec_xi[:, 1]
        zeta = vec_xi[:, 2]
        dn_dxi = np.array(
            [
                0.5 * eta * (2 * xi - 1) * (2 * eta - 1),
                0.5 * zeta * (2 * xi - 1) * (2 * zeta - 1),
                0.5 * (2 * xi - 1) * (zeta + eta - 1) * (2 * zeta + 2 * eta - 1),
                0.5 * eta * (2 * xi + 1) * (2 * eta - 1),
                0.5 * zeta * (2 * xi + 1) * (2 * zeta - 1),
                0.5 * (2 * xi + 1) * (zeta + eta - 1) * (2 * zeta + 2 * eta - 1),
                2 * eta * zeta * (2 * xi - 1),
                -2 * zeta * (2 * xi - 1) * (eta + zeta - 1),
                -2 * eta * (2 * xi - 1) * (eta + zeta - 1),
                2 * eta * zeta * (2 * xi + 1),
                -2 * zeta * (2 * xi + 1) * (eta + zeta - 1),
                -2 * eta * (2 * xi + 1) * (eta + zeta - 1),
                -2 * eta * xi * (2 * eta - 1),
                -2 * zeta * xi * (2 * zeta - 1),
                -2 * xi * (eta + zeta - 1) * (2 * eta + 2 * zeta - 1),
                -8 * eta * zeta * xi,
                8 * zeta * xi * (eta + zeta - 1),
                8 * eta * xi * (eta + zeta - 1),
            ]
        )
        dn_deta = np.array(
            [
                0.5 * xi * (xi - 1) * (4 * eta - 1),
                0.0 * eta,
                0.5 * xi * (xi - 1) * (4 * zeta + 4 * eta - 3),
                0.5 * xi * (xi + 1) * (4 * eta - 1),
                0.0 * eta,
                0.5 * xi * (xi + 1) * (4 * zeta + 4 * eta - 3),
                2 * xi * zeta * (xi - 1),
                -2 * xi * zeta * (xi - 1),
                -2 * xi * (xi - 1) * (2 * eta + zeta - 1),
                2 * xi * zeta * (xi + 1),
                -2 * xi * zeta * (xi + 1),
                -2 * xi * (xi + 1) * (2 * eta + zeta - 1),
                (1 - xi**2) * (4 * eta - 1),
                0.0 * eta,
                (1 - xi**2) * (4 * eta + 4 * zeta - 3),
                4 * zeta * (1 - xi**2),
                4 * zeta * (xi**2 - 1),
                4 * (xi**2 - 1) * (2 * eta + zeta - 1),
            ]
        )
        dn_dzeta = np.array(
            [
                0.0 * zeta,
                0.5 * xi * (xi - 1) * (4 * zeta - 1),
                0.5 * xi * (xi - 1) * (4 * zeta + 4 * eta - 3),
                0.0 * zeta,
                0.5 * xi * (xi + 1) * (4 * zeta - 1),
                0.5 * xi * (xi + 1) * (4 * zeta + 4 * eta - 3),
                2 * xi * eta * (xi - 1),
                -2 * xi * (xi - 1) * (eta + 2 * zeta - 1),
                -2 * xi * eta * (xi - 1),
                2 * xi * eta * (xi + 1),
                -2 * xi * (xi + 1) * (eta + 2 * zeta - 1),
                -2 * xi * eta * (xi + 1),
                0.0 * zeta,
                (1 - xi**2) * (4 * zeta - 1),
                (1 - xi**2) * (4 * eta + 4 * zeta - 3),
                4 * eta * (1 - xi**2),
                4 * (xi**2 - 1) * (eta + 2 * zeta - 1),
                4 * eta * (xi**2 - 1),
            ]
        )
        return np.array([dn_dxi, dn_deta, dn_dzeta]).transpose(2, 0, 1)
        # return [np.array([dn_dxi[:,i], dn_deta[:,i], dn_dzeta[:,i]]) for i in range(len(vec_xi))]
