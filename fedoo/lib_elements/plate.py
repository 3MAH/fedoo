from fedoo.lib_elements.element_list import CombinedElement
from fedoo.lib_elements.quadrangle import Quad4, Quad8, Quad9
from fedoo.lib_elements.triangle import Tri3, Tri6

import numpy as np

# ------------------------------------------------------------
# Reissner-Mindlin plate elements with full integration
# ------------------------------------------------------------

# tri3 plate element - prone to shear locking
ptri3 = CombinedElement("ptri3", "tri3", default_n_gp=3, local_csys=True)

# quad4 plate element - prone to shear locking
pquad4 = CombinedElement("pquad4", "quad4", default_n_gp=4, local_csys=True)

# tri6 plate element
ptri6 = CombinedElement("ptri6", "tri6", default_n_gp=7, local_csys=True)

# quad8 plate element
pquad8 = CombinedElement("pquad8", "quad8", default_n_gp=9, local_csys=True)

# quad9 plate element
pquad9 = CombinedElement("pquad9", "quad9", default_n_gp=9, local_csys=True)


# ------------------------------------------------------------
# Reissner-Mindlin plate elements with reduced integration
# ------------------------------------------------------------

# tri3 plate element - better than fully integrated, but still shear locking
ptri3sri = CombinedElement("ptri3sri", "tri3", default_n_gp=3, local_csys=True)
ptri3sri.set_variable_interpolation("_DispX", "tri3r")
ptri3sri.set_variable_interpolation("_DispY", "tri3r")
ptri3sri.set_variable_interpolation("_DispZ", "tri3r")
ptri3sri.set_variable_interpolation("_RotX", "tri3r")
ptri3sri.set_variable_interpolation("_RotY", "tri3r")
ptri3sri.set_variable_interpolation("_RotZ", "tri3r")

# quad4 plate element with reduced_integration - avoid most of shear locking
pquad4sri = CombinedElement("pquad4sri", "quad4", default_n_gp=4, local_csys=True)
pquad4sri.set_variable_interpolation("_DispX", "quad4r")
pquad4sri.set_variable_interpolation("_DispY", "quad4r")
pquad4sri.set_variable_interpolation("_DispZ", "quad4r")
pquad4sri.set_variable_interpolation("_RotX", "quad4r")
pquad4sri.set_variable_interpolation("_RotY", "quad4r")
pquad4sri.set_variable_interpolation("_RotZ", "quad4r")


# -----------------------------------------------------------------------------
# Reissner-Mindlin plate MITC elements (assumed shear strain interpolation)
# -----------------------------------------------------------------------------


class _BaseMITC:
    """A unified parent class for MITC shear strain interpolation."""

    axis_idx = 0  # Default: 0 for X-coords (RotY), 1 for Y-coords (RotX)
    sign = 1  # Default: 1 for RotY, -1 for RotX

    def __init__(self, n_elm_gp, **kargs):
        assembly = kargs.get("assembly", None)
        if not assembly:
            raise AttributeError(
                "The assembly must be provided to use MITC shell elements."
            )

        # Consolidate state variable extraction
        if "_NodeLocalPos" in assembly.sv:
            self.vec_x_stored = assembly.sv["_NodeLocalPos"]
        else:
            self.vec_x_stored = assembly.sv["_InitialNodeLocalPos"]

        # Continue up the cooperative inheritance chain to the geometric base class
        super().__init__(n_elm_gp)

    def shape_function_derivative(self, vec_xi):
        # 1. Automatically route to the correct geometric coordinate matrix (X or Y)
        coords = self.vec_x_stored[:, :, self.axis_idx]

        # 2. Delegate the specialized interpolation math to the specific topology
        B_xi, B_eta = self.compute_parametric_b(vec_xi, coords)

        # 3. Automatically apply the directional sign mapping and return
        return self.sign * np.stack([B_xi, B_eta], axis=2)


# Topology Specializations


class _Tri3MITC(_BaseMITC, Tri3):
    def compute_parametric_b(self, vec_xi, coords):
        xi, eta = vec_xi[:, 0], vec_xi[:, 1]
        d_dxi = coords[:, 1] - coords[:, 0]
        d_deta = coords[:, 2] - coords[:, 0]

        Nel, n_nodes = coords.shape[0], coords.shape[1]
        B_xi = np.zeros((Nel, len(xi), n_nodes))
        B_eta = np.zeros((Nel, len(xi), n_nodes))

        B_xi[:, :, 0] = (
            0.5 * d_dxi[:, None] * (1 - eta)[None, :]
            + 0.5 * d_deta[:, None] * eta[None, :]
        )
        B_xi[:, :, 1] = 0.5 * d_dxi[:, None] - 0.5 * d_deta[:, None] * eta[None, :]
        B_xi[:, :, 2] = 0.5 * d_dxi[:, None] * eta[None, :]

        B_eta[:, :, 0] = (
            0.5 * d_deta[:, None] * (1 - xi)[None, :]
            + 0.5 * d_dxi[:, None] * xi[None, :]
        )
        B_eta[:, :, 1] = 0.5 * d_deta[:, None] * xi[None, :]
        B_eta[:, :, 2] = 0.5 * d_deta[:, None] - 0.5 * d_dxi[:, None] * xi[None, :]
        return B_xi, B_eta


class _Quad4MITC(_BaseMITC, Quad4):
    def compute_parametric_b(self, vec_xi, coords):
        xi, eta = vec_xi[:, 0], vec_xi[:, 1]
        d_dxi_A = 0.5 * (coords[:, 1] - coords[:, 0])
        d_dxi_C = 0.5 * (coords[:, 2] - coords[:, 3])
        d_deta_D = 0.5 * (coords[:, 3] - coords[:, 0])
        d_deta_B = 0.5 * (coords[:, 2] - coords[:, 1])

        wA, wC, wD, wB = (
            0.25 * (1 - eta),
            0.25 * (1 + eta),
            0.25 * (1 - xi),
            0.25 * (1 + xi),
        )

        B_xi = np.zeros((coords.shape[0], len(xi), 4))
        B_eta = np.zeros((coords.shape[0], len(xi), 4))

        B_xi[:, :, 0] = B_xi[:, :, 1] = d_dxi_A[:, None] * wA[None, :]
        B_xi[:, :, 2] = B_xi[:, :, 3] = d_dxi_C[:, None] * wC[None, :]
        B_eta[:, :, 0] = B_eta[:, :, 3] = d_deta_D[:, None] * wD[None, :]
        B_eta[:, :, 1] = B_eta[:, :, 2] = d_deta_B[:, None] * wB[None, :]
        return B_xi, B_eta


class _Tri6MITC(_BaseMITC, Tri6):
    def _get_tri6_N_vectorized(self, vec_xi):
        xi, eta = vec_xi[:, 0], vec_xi[:, 1]
        return np.stack(
            [
                (1 - xi - eta) * (1 - 2 * xi - 2 * eta),
                xi * (2 * xi - 1),
                eta * (2 * eta - 1),
                4 * xi * (1 - xi - eta),
                4 * xi * eta,
                4 * eta * (1 - xi - eta),
            ],
            axis=1,
        )

    def _get_tri6_dN_at(self, xi, eta):
        dN_dxi = np.array(
            [
                4 * xi + 4 * eta - 3,
                4 * xi - 1,
                0,
                4 - 8 * xi - 4 * eta,
                4 * eta,
                -4 * eta,
            ]
        )
        dN_deta = np.array(
            [
                4 * xi + 4 * eta - 3,
                0,
                4 * eta - 1,
                -4 * xi,
                4 * xi,
                4 - 4 * xi - 8 * eta,
            ]
        )
        return dN_dxi, dN_deta

    def compute_parametric_b(self, vec_xi, coords):
        Nel = coords.shape[0]
        nodes_xi = np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]
        )

        d_dxi_nodes = np.zeros((Nel, 6))
        d_deta_nodes = np.zeros((Nel, 6))
        for k in range(6):
            dN_dxi, dN_deta = self._get_tri6_dN_at(nodes_xi[k, 0], nodes_xi[k, 1])
            d_dxi_nodes[:, k] = coords @ dN_dxi
            d_deta_nodes[:, k] = coords @ dN_deta

        N_gp_matrix = self._get_tri6_N_vectorized(vec_xi)
        B_xi = d_dxi_nodes[:, None, :] * N_gp_matrix[None, :, :]
        B_eta = d_deta_nodes[:, None, :] * N_gp_matrix[None, :, :]
        return B_xi, B_eta


class _BaseHighOrderQuadMITC(_BaseMITC):
    """
    Unified mathematical base shared by both MITC8 and MITC9.
    Uses dynamic polymorphism to handle 8-node or 9-node topologies.
    """

    def _get_asymmetric_6node_interpolation(self, vec_xi, direction="xi"):
        """Computes the asymmetric Q1,2 and Q2,1 interpolation spaces."""
        xi = vec_xi[:, 0]
        eta = vec_xi[:, 1]
        n_gp = len(xi)
        N = np.zeros((n_gp, 6))
        a = 1.0 / np.sqrt(3.0)

        if direction == "xi":
            # Space Q_{1,2}: Linear in xi, Quadratic in eta
            L1, L2 = 0.5 * (1.0 - xi / a), 0.5 * (1.0 + xi / a)
            Q1, Q2, Q3 = 0.5 * eta * (eta - 1.0), 1.0 - eta**2, 0.5 * eta * (eta + 1.0)
            N[:, 0], N[:, 1] = L1 * Q1, L2 * Q1
            N[:, 2], N[:, 3] = L1 * Q2, L2 * Q2
            N[:, 4], N[:, 5] = L1 * Q3, L2 * Q3

        elif direction == "eta":
            # Space Q_{2,1}: Quadratic in xi, Linear in eta
            Q1, Q2, Q3 = 0.5 * xi * (xi - 1.0), 1.0 - xi**2, 0.5 * xi * (xi + 1.0)
            L1, L2 = 0.5 * (1.0 - eta / a), 0.5 * (1.0 + eta / a)
            N[:, 0], N[:, 1] = Q1 * L1, Q1 * L2
            N[:, 2], N[:, 3] = Q2 * L1, Q2 * L2
            N[:, 4], N[:, 5] = Q3 * L1, Q3 * L2

        return N

    def compute_parametric_b(self, vec_xi, coords):
        # 1. Define the 6 asymmetric tying point grids
        a = 1.0 / np.sqrt(3.0)
        tying_xi = np.array(
            [[-a, -1.0], [a, -1.0], [-a, 0.0], [a, 0.0], [-a, 1.0], [a, 1.0]]
        )
        tying_eta = np.array(
            [[-1.0, -a], [-1.0, a], [0.0, -a], [0.0, a], [1.0, -a], [1.0, a]]
        )

        # 2. Polymorphic evaluation of native framework shape functions
        # This automatically resolves to the 8-node or 9-node parent methods
        dN_tying_xi = np.array(
            self.geom_shape_derivative(tying_xi)
        )  # Shape: (6, 2, n_nodes)
        dN_tying_eta = np.array(
            self.geom_shape_derivative(tying_eta)
        )  # Shape: (6, 2, n_nodes)

        N_node_at_tying_xi = self.geom_shape_function(tying_xi)  # Shape: (6, n_nodes)
        N_node_at_tying_eta = self.geom_shape_function(tying_eta)  # Shape: (6, n_nodes)

        # 3. Compute geometric Jacobian scaling metrics at the tying positions
        # coords shape: (Nel, n_nodes) -> Result shape: (Nel, 6)
        Jacobian_edge_xi = coords @ dN_tying_xi[:, 0, :].T
        Jacobian_edge_eta = coords @ dN_tying_eta[:, 1, :].T

        # 4. Get the asymmetric interpolation spaces mapping to the Gauss points
        N_xi_space = self._get_asymmetric_6node_interpolation(
            vec_xi, direction="xi"
        )  # (n_gp, 6)
        N_eta_space = self._get_asymmetric_6node_interpolation(
            vec_xi, direction="eta"
        )  # (n_gp, 6)

        # 5. Tensor contraction mapping nodes through tying points directly to Gauss points
        # g = gauss point index, k = tying point index (6), e = element index, i = node index
        B_xi = np.einsum(
            "gk,ek,ki->egi", N_xi_space, Jacobian_edge_xi, N_node_at_tying_xi
        )
        B_eta = np.einsum(
            "gk,ek,ki->egi", N_eta_space, Jacobian_edge_eta, N_node_at_tying_eta
        )

        return B_xi, B_eta


class _Quad8MITC(_BaseHighOrderQuadMITC, Quad8):
    def geom_shape_derivative(self, vec_xi):
        return Quad8.shape_function_derivative(self, vec_xi)

    def geom_shape_function(self, vec_xi):
        return Quad8.shape_function(self, vec_xi)


class _Quad9MITC(_BaseHighOrderQuadMITC, Quad9):
    def geom_shape_derivative(self, vec_xi):
        return Quad9.shape_function_derivative(self, vec_xi)

    def geom_shape_function(self, vec_xi):
        return Quad9.shape_function(self, vec_xi)


# Micro-Subclasses for shear strain interpolation associated to rot dofs


class _Tri3MITC_RotX(_Tri3MITC):
    name, axis_idx, sign = "_tri3mitc_rotx_shear", 1, -1


class _Tri3MITC_RotY(_Tri3MITC):
    name, axis_idx, sign = "_tri3mitc_roty_shear", 0, 1


class _Quad4MITC_RotX(_Quad4MITC):
    name, axis_idx, sign = "_quad4mitc_rotx_shear", 1, -1


class _Quad4MITC_RotY(_Quad4MITC):
    name, axis_idx, sign = "_quad4mitc_roty_shear", 0, 1


class _Tri6MITC_RotX(_Tri6MITC):
    name, axis_idx, sign = "_tri6mitc_rotx_shear", 1, -1


class _Tri6MITC_RotY(_Tri6MITC):
    name, axis_idx, sign = "_tri6mitc_roty_shear", 0, 1


class _Quad8MITC_RotX(_Quad8MITC):
    name, axis_idx, sign = "_quad8mitc_rotx_shear", 1, -1


class _Quad8MITC_RotY(_Quad8MITC):
    name, axis_idx, sign = "_quad8mitc_roty_shear", 0, 1


class _Quad9MITC_RotX(_Quad9MITC):
    name, axis_idx, sign = "_quad9mitc_rotx_shear", 1, -1


class _Quad9MITC_RotY(_Quad9MITC):
    name, axis_idx, sign = "_quad9mitc_roty_shear", 0, 1


# MITC Element Declarations

ptri3mitc = CombinedElement("ptri3mitc", "tri3", default_n_gp=3, local_csys=True)
ptri3mitc.set_variable_interpolation("_RotX", _Tri3MITC_RotX)
ptri3mitc.set_variable_interpolation("_RotY", _Tri3MITC_RotY)

pquad4mitc = CombinedElement("pquad4mitc", "quad4", default_n_gp=4, local_csys=True)
pquad4mitc.set_variable_interpolation("_RotX", _Quad4MITC_RotX)
pquad4mitc.set_variable_interpolation("_RotY", _Quad4MITC_RotY)

ptri6mitc = CombinedElement("ptri6mitc", "tri6", default_n_gp=7, local_csys=True)
ptri6mitc.set_variable_interpolation("_RotX", _Tri6MITC_RotX)
ptri6mitc.set_variable_interpolation("_RotY", _Tri6MITC_RotY)

pquad8mitc = CombinedElement("pquad8mitc", "quad8", default_n_gp=9, local_csys=True)
pquad8mitc.set_variable_interpolation("_RotX", _Quad8MITC_RotX)
pquad8mitc.set_variable_interpolation("_RotY", _Quad8MITC_RotY)

pquad9mitc = CombinedElement("pquad9mitc", "quad9", default_n_gp=9, local_csys=True)
pquad9mitc.set_variable_interpolation("_RotX", _Quad9MITC_RotX)
pquad9mitc.set_variable_interpolation("_RotY", _Quad9MITC_RotY)
