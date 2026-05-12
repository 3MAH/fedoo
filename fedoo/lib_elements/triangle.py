import numpy as np

from fedoo.lib_elements.element_base import Element2D


class ElementTriangle(Element2D):
    def __init__(self, n_elm_gp):
        if n_elm_gp == 0:  # if n_elm_gp == 0, we take the position of the nodes
            self.xi_pg = self.xi_nd
        else:
            self.xi_pg = self.get_gp_elm_coordinates(n_elm_gp)  # = np.c_[xi,eta]
            self.w_pg = self.get_gp_weight(n_elm_gp)

        self.shape_function_gp = self.shape_function(self.xi_pg)
        self.shape_function_derivative_gp = self.shape_function_derivative(self.xi_pg)

    def get_gp_elm_coordinates(self, n_elm_gp):
        if n_elm_gp == 1:
            return np.array([[1 / 3, 1 / 3]])
        elif n_elm_gp == 3:  # ordre exacte 2
            return np.array(
                [[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]]
            )  # = np.c_[xi, eta]
        elif n_elm_gp == 4:  # ordre exacte 3
            return np.array(
                [[1 / 3, 1 / 3], [1 / 5, 1 / 5], [3 / 5, 1 / 5], [1 / 5, 3 / 5]]
            )
        elif n_elm_gp == 7:  # ordre exacte 5
            a = (6.0 + np.sqrt(15)) / 21
            b = 4.0 / 7 - a
            return np.array(
                [
                    [1 / 3, 1 / 3],
                    [a, a],
                    [1 - 2 * a, a],
                    [a, 1 - 2 * a],
                    [b, b],
                    [1 - 2 * b, b],
                    [b, 1 - 2 * b],
                ]
            )
        elif n_elm_gp == 12:  # ordre exacte 6
            a = 0.063089014491502
            b = 0.249286745170910
            c = 0.310352451033785
            d = 0.053145049844816
            return np.array(
                [
                    [a, a],
                    [1 - 2 * a, a],
                    [a, 1 - 2 * a],
                    [b, b],
                    [1 - 2 * b, b],
                    [b, 1 - 2 * b],
                    [c, d],
                    [d, c],
                    [1 - (c + d), c],
                    [1 - (c + d), d],
                    [c, 1 - (c + d)],
                    [d, 1 - (c + d)],
                ]
            )
        elif n_elm_gp == 0:  # if n_elm_gp == 0, we take the position of the nodes
            self.xi_pg = self.xi_nd
        else:
            assert 0, (
                "Number of gauss points "
                + str(n_elm_gp)
                + " unavailable for triangle element"
            )

    def get_gp_weight(self, n_elm_gp):
        if n_elm_gp == 1:
            return np.array([1.0 / 2])
        elif n_elm_gp == 3:  # ordre exacte 2
            return np.array([1.0 / 6, 1.0 / 6, 1.0 / 6])
        elif n_elm_gp == 4:  # ordre exacte 3
            return np.array([-27.0 / 96, 25.0 / 96, 25.0 / 96, 25.0 / 96])
        elif n_elm_gp == 7:  # ordre exacte 5
            AA_pg = (155.0 + np.sqrt(15)) / 2400
            BB_pg = 31.0 / 240 - AA_pg
            return np.array([9.0 / 80, AA_pg, AA_pg, AA_pg, BB_pg, BB_pg, BB_pg])
        elif n_elm_gp == 12:  # ordre exacte 6
            w1 = 0.025422453185103
            w2 = 0.058393137863189
            w3 = 0.041425537809187
            return np.array([w1, w1, w1, w2, w2, w2, w3, w3, w3, w3, w3, w3])
        else:
            assert 0, (
                "Number of gauss points "
                + str(n_elm_gp)
                + " unavailable for triangle element"
            )


class Tri3(ElementTriangle):
    name = "tri3"
    default_n_gp = 3
    n_nodes = 3

    def __init__(self, n_elm_gp=3, **kargs):
        self.xi_nd = np.c_[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        self.n_elm_gp = n_elm_gp
        ElementTriangle.__init__(self, n_elm_gp)

    # In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta)
    # xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    # xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)

    def shape_function(self, xi):
        return np.c_[(1 - xi[:, 0] - xi[:, 1]), xi[:, 0], xi[:, 1]]

    def shape_function_derivative(self, xi):
        return [np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]) for x in xi]


class Tri3r(Tri3):
    name = "tri3r"
    default_n_gp = 1
    n_nodes = 3

    def shape_function(self, vec_xi):
        # return center value every where (as if n_pg = 1)
        return np.full((len(vec_xi), 3), 1 / 3)


class Tri3Bubble(Tri3):
    name = "tri3bubble"
    n_nodes = 4

    def shape_function(self, xi):
        return np.c_[
            1 - xi[:, 0] - xi[:, 1],
            xi[:, 0],
            xi[:, 1],
            (1 - xi[:, 0] - xi[:, 1]) * xi[:, 0] * xi[:, 1],
        ]

    def shape_function_derivative(self, xi):
        return [
            np.array(
                [
                    [-1.0, 1.0, 0.0, x[1] * (1 - 2 * x[0] - x[1])],
                    [-1.0, 0.0, 1.0, x[0] * (1 - 2 * x[1] - x[0])],
                ]
            )
            for x in xi
        ]


class Tri6(ElementTriangle):
    name = "tri6"
    default_n_gp = 4
    n_nodes = 6

    def __init__(self, n_elm_gp=4, **kargs):
        self.xi_nd = np.c_[
            [0.0, 1.0, 0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 1.0, 0.0, 0.5, 0.5]
        ]
        self.n_elm_gp = n_elm_gp
        ElementTriangle.__init__(self, n_elm_gp)

    # In the functions ShapeFunction and ShapeFunctionDerivative xi contains a list of point using reference element coordinates (xi, eta)
    # xi[:,0] -> list of values of xi for all points (gauss points in general but may be used with other points)
    # xi[:,1] -> list of values of eta for all points (gauss points in general but may be used with other points)
    def shape_function(self, xi):
        return np.c_[
            (1 - xi[:, 0] - xi[:, 1]) * (1 - 2 * xi[:, 0] - 2 * xi[:, 1]),
            xi[:, 0] * (2 * xi[:, 0] - 1),
            xi[:, 1] * (2 * xi[:, 1] - 1),
            4 * xi[:, 0] * (1 - xi[:, 0] - xi[:, 1]),
            4 * xi[:, 0] * xi[:, 1],
            4 * xi[:, 1] * (1 - xi[:, 0] - xi[:, 1]),
        ]

    def shape_function_derivative(self, xi):
        return [
            np.array(
                [
                    [
                        4 * (x[0] + x[1]) - 3,
                        4 * x[0] - 1,
                        0.0,
                        4 * (1 - 2 * x[0] - x[1]),
                        4 * x[1],
                        -4 * x[1],
                    ],
                    [
                        4 * (x[0] + x[1]) - 3,
                        0.0,
                        4 * x[1] - 1,
                        -4 * x[0],
                        4 * x[0],
                        4 * (1 - x[0] - 2 * x[1]),
                    ],
                ]
            )
            for x in xi
        ]


# 6 nodes triangle with 1 gp (very reduced)
# class Tri6vr(Tri6):
#     name = "tri6vr"
#     default_n_gp = 1
#     n_nodes = 6

#     def shape_function(self, vec_xi):
#         # return center value every where (as if n_pg = 1)
#         return np.full((len(vec_xi), 6), 1 / 3)


# 6 nodes triangle with 3 gauss points reduced interpolation 
class Tri6r(Tri6):
    name = "tri6r"
    default_n_gp = 3
    n_nodes = 6

    def __init__(self, n_elm_gp=3, **kargs):
        # Define our "Master" 3 Gauss Points
        self.xi_master = np.array([[1/6, 1/6], [2/3, 1/6], [1/6, 2/3]])
        self.w_master = np.array([1/6, 1/6, 1/6])
        
        # Pre-calculate Shape Functions at the 3 Master points
        self.N_master = Tri6.shape_function(self, self.xi_master)
        self.dN_master = Tri6.shape_function_derivative(self, self.xi_master)
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
            mask = (closest_master_idx == j)
            # Sum of solver weights in this Voronoi zone
            zone_weight_sum = np.sum(w_solver[mask])
            # Scale factor for points in this zone
            factors[mask] = self.w_master[j] / w_solver[mask] * (w_solver[mask] / zone_weight_sum)
            
        return closest_master_idx, factors

    def shape_function(self, xi):
        indices, factors = self._get_projection_weights(xi)
        # Return the master value scaled by the weight correction factor
        return self.N_master[indices] * factors[:, None]

    def shape_function_derivative(self, xi):
        indices, factors = self._get_projection_weights(xi)
        # We must also scale the derivatives
        return [self.dN_master[idx] * f for idx, f in zip(indices, factors)]
