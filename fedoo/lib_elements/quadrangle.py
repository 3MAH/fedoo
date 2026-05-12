import numpy as np


from fedoo.lib_elements.element_base import Element2D


class ElementQuadrangle(Element2D):
    def __init__(self, n_elm_gp):
        if n_elm_gp == 0:  # if n_elm_gp == 0, we take the position of the nodes
            self.xi_pg = self.xi_nd
        else:
            self.xi_pg = self.get_gp_elm_coordinates(n_elm_gp)  # = np.c_[xi,eta]
            self.w_pg = self.get_gp_weight(n_elm_gp)

        self.shape_function_gp = self.shape_function(self.xi_pg)
        self.shape_function_derivative_gp = self.shape_function_derivative(self.xi_pg)

    def get_gp_elm_coordinates(self, n_elm_gp=None):
        if n_elm_gp is None:
            return self.xi_pg

        if n_elm_gp == 1:  # ordre exact 1
            return np.array([[0, 0]])
        elif n_elm_gp == 4:  # ordre exacte 2
            a = 1 / np.sqrt(3)
            return np.array([[-a, -a], [a, -a], [a, a], [-a, a]])  # = np.c_[xi, eta]
        elif n_elm_gp == 9:  # ordre exacte 3
            a = 0.774596669241483
            return np.array(
                [
                    [-a, -a],
                    [a, -a],
                    [a, a],
                    [-a, a],
                    [0, -a],
                    [a, 0],
                    [0, a],
                    [-a, 0],
                    [0, 0],
                ]
            )
        elif n_elm_gp == 16:  # ordre exacte 5
            a = 0.339981043584856
            b = 0.861136311594053
            return np.array(
                [
                    [-b, -b],
                    [-a, -b],
                    [a, -b],
                    [b, -b],
                    [-b, -a],
                    [-a, -a],
                    [a, -a],
                    [b, -a],
                    [-b, a],
                    [-a, a],
                    [a, a],
                    [b, a],
                    [-b, b],
                    [-a, b],
                    [a, b],
                    [b, b],
                ]
            )
        else:
            assert 0, (
                "Number of gauss points "
                + str(n_elm_gp)
                + " unavailable for triangle element"
            )

    def get_gp_weight(self, n_elm_gp=None):
        if n_elm_gp is None:
            return self.w_pg

        if n_elm_gp == 1:
            return np.array([4.0])
        elif n_elm_gp == 4:  # ordre exacte 2
            return np.array([1.0, 1.0, 1.0, 1.0])
        elif n_elm_gp == 9:  # ordre exacte 3
            return np.array(
                [
                    25 / 81.0,
                    25 / 81.0,
                    25 / 81.0,
                    25 / 81.0,
                    40 / 81.0,
                    40 / 81.0,
                    40 / 81.0,
                    40 / 81.0,
                    64 / 81.0,
                ]
            )
        elif n_elm_gp == 16:  # ordre exacte 4
            w_a = 0.652145154862546
            w_b = 0.347854845137454
            return np.array(
                [
                    w_b**2,
                    w_a * w_b,
                    w_a * w_b,
                    w_b**2,
                    w_a * w_b,
                    w_a**2,
                    w_a**2,
                    w_a * w_b,
                    w_a * w_b,
                    w_a**2,
                    w_a**2,
                    w_a * w_b,
                    w_b**2,
                    w_a * w_b,
                    w_a * w_b,
                    w_b**2,
                ]
            )
        else:
            assert 0, (
                "Number of gauss points "
                + str(n_elm_gp)
                + " unavailable for triangle element"
            )


class Quad4(ElementQuadrangle):
    name = "quad4"
    default_n_gp = 4
    n_nodes = 4

    def __init__(self, n_elm_gp=4, **kargs):
        self.xi_nd = np.c_[[-1.0, 1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]]
        self.n_elm_gp = n_elm_gp
        ElementQuadrangle.__init__(self, n_elm_gp)

    # In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta)
    # vec_xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    # vec_xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
    def shape_function(self, vec_xi):
        xi = vec_xi[:, 0]
        eta = vec_xi[:, 1]
        return np.c_[
            0.25 * (1 - xi) * (1 - eta),
            0.25 * (1 + xi) * (1 - eta),
            0.25 * (1 + xi) * (1 + eta),
            0.25 * (1 - xi) * (1 + eta),
        ]

    def shape_function_derivative(self, vec_xi):
        return [
            np.array(
                [
                    [
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
                    ],
                ]
            )
            for xi in vec_xi
        ]


class Quad4r(Quad4):
    name = "quad4r"
    default_n_gp = 1
    n_nodes = 4

    def shape_function(self, vec_xi):
        # return center value every where (as if n_pg = 1)
        return 0.25 * np.ones((len(vec_xi), 4))

    def shape_function_derivative(self, vec_xi):
        # return center value every where (as if n_pg = 1)
        return [
            np.array(
                [
                    [-0.25, 0.25, 0.25, -0.25],
                    [-0.25, -0.25, 0.25, 0.25],
                ]
            )
            for xi in vec_xi
        ]


class Quad8(ElementQuadrangle):
    name = "quad8"
    default_n_gp = 9
    n_nodes = 8

    def __init__(self, n_elm_gp=9, **kargs):
        self.xi_nd = np.c_[
            [-1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0],
        ]
        self.n_elm_gp = n_elm_gp
        ElementQuadrangle.__init__(self, n_elm_gp)

    # In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta)
    # xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    # xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
    def shape_function(self, vec_xi):
        xi = vec_xi[:, 0]
        eta = vec_xi[:, 1]
        return np.c_[
            0.25 * (1 - xi) * (1 - eta) * (-1 - xi - eta),
            0.25 * (1 + xi) * (1 - eta) * (-1 + xi - eta),
            0.25 * (1 + xi) * (1 + eta) * (-1 + xi + eta),
            0.25 * (1 - xi) * (1 + eta) * (-1 - xi + eta),
            0.5 * (1 - xi**2) * (1 - eta),
            0.5 * (1 + xi) * (1 - eta**2),
            0.5 * (1 - xi**2) * (1 + eta),
            0.5 * (1 - xi) * (1 - eta**2),
        ]

    def shape_function_derivative(self, vec_xi):
        return [
            np.array(
                [
                    [
                        0.25 * (1 - xi[1]) * (2 * xi[0] + xi[1]),
                        0.25 * (1 - xi[1]) * (2 * xi[0] - xi[1]),
                        0.25 * (1 + xi[1]) * (2 * xi[0] + xi[1]),
                        0.25 * (-1 - xi[1]) * (-2 * xi[0] + xi[1]),
                        -xi[0] * (1 - xi[1]),
                        0.5 * (1 - xi[1] ** 2),
                        -xi[0] * (1 + xi[1]),
                        -0.5 * (1 - xi[1] ** 2),
                    ],
                    [
                        0.25 * (1 - xi[0]) * (2 * xi[1] + xi[0]),
                        0.25 * (-1 - xi[0]) * (xi[0] - 2 * xi[1]),
                        0.25 * (1 + xi[0]) * (2 * xi[1] + xi[0]),
                        0.25 * (1 - xi[0]) * (-xi[0] + 2 * xi[1]),
                        -0.5 * (1 - xi[0] ** 2),
                        -xi[1] * (1 + xi[0]),
                        0.5 * (1 - xi[0] ** 2),
                        -xi[1] * (1 - xi[0]),
                    ],
                ]
            )
            for xi in vec_xi
        ]


class Quad9(ElementQuadrangle):
    name = "quad9"
    default_n_gp = 9
    n_nodes = 9

    def __init__(self, n_elm_gp=9, **kargs):
        self.xi_nd = np.c_[
            [-1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0],
            [-1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0],
        ]
        self.n_elm_gp = n_elm_gp
        ElementQuadrangle.__init__(self, n_elm_gp)

    # In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta)
    # xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    # xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
    def shape_function(self, vec_xi):
        xi = vec_xi[:, 0]
        eta = vec_xi[:, 1]
        return np.c_[
            0.25 * xi * eta * (xi - 1) * (eta - 1),
            0.25 * xi * eta * (xi + 1) * (eta - 1),
            0.25 * xi * eta * (xi + 1) * (eta + 1),
            0.25 * xi * eta * (xi - 1) * (eta + 1),
            0.5 * (1 - xi**2) * eta * (eta - 1),
            0.5 * xi * (xi + 1) * (1 - eta**2),
            0.5 * (1 - xi**2) * eta * (eta + 1),
            0.5 * xi * (xi - 1) * (1 - eta**2),
            (1 - xi**2) * (1 - eta**2),
        ]

    def shape_function_derivative(self, vec_xi):
        return [
            np.array(
                [
                    [
                        0.25 * xi[1] * (xi[1] - 1) * (2 * xi[0] - 1),
                        0.25 * xi[1] * (xi[1] - 1) * (2 * xi[0] + 1),
                        0.25 * xi[1] * (xi[1] + 1) * (2 * xi[0] + 1),
                        0.25 * xi[1] * (xi[1] + 1) * (2 * xi[0] - 1),
                        -xi[0] * xi[1] * (xi[1] - 1),
                        0.5 * (2 * xi[0] + 1) * (1 - xi[1] ** 2),
                        -xi[0] * xi[1] * (xi[1] + 1),
                        0.5 * (2 * xi[0] - 1) * (1 - xi[1] ** 2),
                        -2 * xi[0] * (1 - xi[1] ** 2),
                    ],
                    [
                        0.25 * xi[0] * (xi[0] - 1) * (2 * xi[1] - 1),
                        0.25 * xi[0] * (xi[0] + 1) * (2 * xi[1] - 1),
                        0.25 * xi[0] * (xi[0] + 1) * (2 * xi[1] + 1),
                        0.25 * xi[0] * (xi[0] - 1) * (2 * xi[1] + 1),
                        0.5 * (1 - xi[0] ** 2) * (2 * xi[1] - 1),
                        -xi[0] * xi[1] * (xi[0] + 1),
                        0.5 * (1 - xi[0] ** 2) * (2 * xi[1] + 1),
                        -xi[0] * xi[1] * (xi[0] - 1),
                        -2 * xi[1] * (1 - xi[0] ** 2),
                    ],
                ]
            )
            for xi in vec_xi
        ]


# 9 nodes quad with 4 gauss points reduced interpolation
class Quad9r(Quad9):
    name = "quad9r"
    default_n_gp = 4
    n_nodes = 9

    def __init__(self, n_elm_gp=3, **kargs):
        # Define our "Master" 3 Gauss Points
        a = 1 / np.sqrt(3)
        self.xi_master = np.array([[-a, -a], [a, -a], [a, a], [-a, a]])
        self.w_master = np.array([1, 1, 1, 1])

        # Pre-calculate Shape Functions at the 4 Master points
        self.N_master = Quad9.shape_function(self, self.xi_master)
        self.dN_master = Quad9.shape_function_derivative(self, self.xi_master)
        super().__init__(n_elm_gp, **kargs)

    def _get_projection_weights(self, xi):
        """
        Each solver point is assigned to the nearest master point.
        We then must correct the weight, by scaling the return value by
        (W_master / W_solver).
        """
        # Find which master point is closest for each input point
        dist = np.linalg.norm(xi[:, None, :] - self.xi_master[None, :, :], axis=2)
        closest_master_idx = np.argmin(dist, axis=1)

        # We need the solver weights to do the correction
        # This assumes n_elm_gp was used to initialize the weights
        w_solver = self.get_gp_weight(len(xi))

        # Correction factor: (Master Weight) / (Solver Weight)
        # This ensures: sum(w_solver * (Value * Factor)) == Master_Weight * Value
        factors = np.zeros(len(xi))
        for j in range(3):
            mask = closest_master_idx == j
            # Sum of solver weights in this Voronoi zone
            zone_weight_sum = np.sum(w_solver[mask])
            # Scale factor for points in this zone
            factors[mask] = (
                self.w_master[j] / w_solver[mask] * (w_solver[mask] / zone_weight_sum)
            )

        return closest_master_idx, factors

    def shape_function(self, xi):
        indices, factors = self._get_projection_weights(xi)
        # Return the master value scaled by the weight correction factor
        return self.N_master[indices] * factors[:, None]

    def shape_function_derivative(self, xi):
        indices, factors = self._get_projection_weights(xi)
        # We must also scale the derivatives
        return [self.dN_master[idx] * f for idx, f in zip(indices, factors)]


#### Hourglass control shape function ####


class Quad4Hourglass(Quad4):
    name = "quad4hourglass"
    default_n_gp = 1
    n_nodes = 4

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

    def shape_function(self, vec_xi):
        h = 0.5 * np.array([[1, -1, 1, -1]])
        b = self._b_matrix

        return (
            h
            - (self.x_nd[:, :, 0] @ h.T) * b[0].T
            - (self.x_nd[:, :, 1] @ h.T) * b[1].T
        )[:, np.newaxis, :]
        # shape = (Nel, Nb_pg=1, Nddl=4)

    def _get_b_matrix(self):
        A = self.mesh.get_element_volumes(1)
        x = self.x_nd[:, :, 0].T
        y = self.x_nd[:, :, 1].T
        return (
            1
            / (2 * A)
            * np.array(
                [
                    [y[1] - y[3], y[2] - y[0], y[3] - y[1], y[0] - y[2]],
                    [x[3] - x[1], x[0] - x[2], x[1] - x[3], x[2] - x[0]],
                ]
            )
        )
