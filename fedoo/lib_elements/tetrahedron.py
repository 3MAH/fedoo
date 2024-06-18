import numpy as np

from fedoo.lib_elements.element_base import Element


class ElementTetrahedron(Element):
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
            return np.array([[0.25, 0.25, 0.25]])
        elif n_elm_gp == 4:
            a = 0.1381966011250105
            b = 0.5854101966249685
            xi = np.c_[[a, a, a, b]]
            eta = np.c_[[a, a, b, a]]
            zeta = np.c_[[a, b, a, a]]
            return np.array(
                [[a, a, a], [a, a, b], [a, b, a], [b, a, a]]
            )  # = np.c_[xi, eta, zeta]
        elif n_elm_gp == 5:
            a = 0.25
            b = 0.16666666666666666
            c = 0.5
            return np.array([[a, a, a], [b, b, b], [b, b, c], [b, c, b], [c, b, b]])
        elif n_elm_gp == 15:
            a = 0.25
            b1 = 0.3197936278296299
            b2 = 0.09197107805272303
            c1 = 0.040619116511110234
            c2 = 0.724086765841831
            d = 0.05635083268962915
            e = 0.4436491673103708
            return np.array(
                [
                    [a, a, a],
                    [b1, b1, b1],
                    [b1, b1, c1],
                    [b1, c1, b1],
                    [c1, b1, b1],
                    [b2, b2, b2],
                    [b2, b2, c2],
                    [b2, c2, b2],
                    [c2, b2, b2],
                    [d, d, e],
                    [d, e, d],
                    [e, d, d],
                    [d, e, e],
                    [e, d, e],
                    [e, e, d],
                ]
            )
        else:
            assert 0, (
                "Number of gauss points "
                + str(n_elm_gp)
                + " unavailable for triangle element"
            )

    def get_gp_weight(self, n_elm_gp):
        if n_elm_gp == 1:
            return np.array([1.0 / 6])
        elif n_elm_gp == 4:
            return np.array([1 / 24, 1 / 24, 1 / 24, 1 / 24])
        elif n_elm_gp == 5:
            return np.array([-2 / 15, 3 / 40, 3 / 40, 3 / 40, 3 / 40])
        elif n_elm_gp == 15:
            f1 = 0.011511367871045397
            f2 = 0.01198951396316977
            return np.array(
                [
                    8.0 / 405,
                    f1,
                    f1,
                    f1,
                    f1,
                    f2,
                    f2,
                    f2,
                    f2,
                    5.0 / 567,
                    5.0 / 567,
                    5.0 / 567,
                    5.0 / 567,
                    5.0 / 567,
                    5.0 / 567,
                ]
            )
        else:
            assert 0, (
                "Number of gauss points "
                + str(n_elm_gp)
                + " unavailable for triangle element"
            )


class Tet4(ElementTetrahedron):
    name = "tet4"
    default_n_gp = 4
    n_nodes = 4

    def __init__(self, n_elm_gp=4, **kargs):
        self.xi_nd = np.c_[
            [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]
        ]
        self.n_elm_gp = n_elm_gp
        ElementTetrahedron.__init__(self, n_elm_gp)

    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:, 0]
        eta = vec_xi[:, 1]
        zeta = vec_xi[:, 2]
        return np.c_[eta, zeta, 1 - xi - eta - zeta, xi]

    def ShapeFunctionDerivative(self, vec_xi):
        return [
            np.array(
                [[0.0, 0.0, -1.0, 1.0], [1.0, 0.0, -1.0, 0.0], [0.0, 1.0, -1.0, 0.0]]
            )
            for xi in vec_xi
        ]


class Tet10(ElementTetrahedron):
    name = "tet10"
    default_n_gp = 15
    n_nodes = 10

    def __init__(self, n_elm_gp=15, **kargs):
        self.xi_nd = np.c_[
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0],
        ]
        self.n_elm_gp = n_elm_gp
        ElementTetrahedron.__init__(self, n_elm_gp)

    def ShapeFunction(self, vec_xi):
        xi = vec_xi[:, 0]
        eta = vec_xi[:, 1]
        zeta = vec_xi[:, 2]
        return np.c_[
            eta * (2 * eta - 1),
            zeta * (2 * zeta - 1),
            (1 - xi - eta - zeta) * (1 - 2 * xi - 2 * eta - 2 * zeta),
            xi * (2 * xi - 1),
            4 * eta * zeta,
            4 * zeta * (1 - xi - eta - zeta),
            4 * eta * (1 - xi - eta - zeta),
            4 * xi * eta,
            4 * xi * zeta,
            4 * xi * (1 - xi - eta - zeta),
        ]

    def ShapeFunctionDerivative(self, vec_xi):
        vec_m = [1 - xi[0] - xi[1] - xi[2] for xi in vec_xi]
        return [
            np.array(
                [
                    [
                        0.0,
                        0.0,
                        1 - 4 * m,
                        -1 + 4 * xi[0],
                        0.0,
                        -4 * xi[2],
                        -4 * xi[1],
                        4 * xi[1],
                        4 * xi[2],
                        4 * (m - xi[0]),
                    ],
                    [
                        -1 + 4 * xi[1],
                        0.0,
                        1 - 4 * m,
                        0.0,
                        4 * xi[2],
                        -4 * xi[2],
                        4 * (m - xi[1]),
                        4 * xi[0],
                        0.0,
                        -4 * xi[0],
                    ],
                    [
                        0.0,
                        -1 + 4 * xi[2],
                        1 - 4 * m,
                        0.0,
                        4 * xi[1],
                        4 * (m - xi[2]),
                        -4 * xi[1],
                        0.0,
                        4 * xi[0],
                        -4 * xi[0],
                    ],
                ]
            )
            for m, xi in zip(vec_m, vec_xi)
        ]
