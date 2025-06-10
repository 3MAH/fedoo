import numpy as np
from scipy import sparse


from fedoo.lib_elements.element_base import Element


class FiniteDifference1D(Element):
    def __init__(
        self, n_elm_gp
    ):  # Points de gauss pour les éléments de référence 1D entre 0 et 1
        if n_elm_gp != 0:  # if n_elm_gp == 0, we take the position of the nodes
            assert 0, "Number of gauss points for Finite Difference elements should be set to 0"
        self.n_elm_gp = 0

    def ShapeFunction(xi_pg):
        assert 0, "No Shape Function for Finite Difference Method. Use lin2 or lin3 element instead"


class Parameter(FiniteDifference1D):
    # This finite difference element doesnt support any derivative
    def __init__(self, n_elm_gp=0, **kargs):
        FiniteDifference1D.__init__(self, n_elm_gp)

    def computeOperator(self, crd, elm):
        Nnd = np.shape(crd)[0]
        return {(0, 0): [sparse.identity(Nnd, "d", "csr")]}  # dictionnary


class Node(FiniteDifference1D):
    # Element to define a set of nodes
    # No integration and therefore n_elm_gp should be set to 0
    def __init__(self, n_elm_gp=0, **kargs):
        FiniteDifference1D.__init__(self, n_elm_gp)

    def computeOperator(self, crd, elm):
        # elm is a column vector (ie elm.shape = (Nel,1)) that contain the node numbers
        Nnd = crd.shape[0]
        Nel = elm.shape[0]
        row = np.arange(Nel)
        col = elm[:, 0]
        data = np.ones(Nel)
        return {
            (0, 0): [sparse.coo_matrix((data, (row, col)), shape=(Nel, Nnd)).tocsr()]
        }  # dictionnary


#        diag = np.zeros(np.shape(crd)[0])
#        diag[elm] = 1
#        Nnd = crd.shape[0]
#        return {0: [sparse.spdiags([diag], [0], Nnd, Nnd, 'csr' )]} #dictionnary


class ForwardFiniteDifference(FiniteDifference1D):
    # Only first order derivative
    # explicit method
    def __init__(self, n_elm_gp=0, **kargs):
        FiniteDifference1D.__init__(self, n_elm_gp)

    def computeOperator(self, crd, elm):
        Nnd = np.shape(crd)[0]
        data = {(0, 0): [sparse.identity(Nnd, "d", "csr")]}
        invLenght = 1 / (crd[elm[:, 1], 0] - crd[elm[:, 0], 0])  # 1/(element lenght)
        data[1, 0] = [sparse.spdiags([-invLenght, invLenght], [0, 1], Nnd, Nnd, "csr")]
        return data  # dictionnary


class ForwardFiniteDifferenceOrder2(FiniteDifference1D):
    # First order and 2nd order derivative
    def __init__(self, n_elm_gp=0, **kargs):
        FiniteDifference1D.__init__(self, n_elm_gp)

    def computeOperator(self, crd, elm):
        Nnd = np.shape(crd)[0]
        data = {(0, 0): [sparse.identity(Nnd, "d", "csr")]}
        invLenght = 1 / (crd[elm[:, 1], 0] - crd[elm[:, 0], 0])  # 1/(element lenght)
        data[1, 0] = [sparse.spdiags([-invLenght, invLenght], [0, 1], Nnd, Nnd, "csr")]
        invLenghtSquare = invLenght**2
        data[2, 0] = [
            sparse.spdiags(
                [invLenghtSquare, -2 * invLenghtSquare, invLenghtSquare],
                [-1, 0, 1],
                Nnd,
                Nnd,
                "csr",
            )
        ]
        return data  # dictionnary


class BackwardFiniteDifference(FiniteDifference1D):
    # This finite difference element doesnt support any derivative
    # explicit method
    def __init__(self, n_elm_gp=0, **kargs):
        FiniteDifference1D.__init__(self, n_elm_gp)

    def computeOperator(self, crd, elm):
        Nnd = np.shape(crd)[0]
        data = {(0, 0): [sparse.identity(Nnd, "d", "csr")]}
        invLenght = 1 / (crd[elm[:, 1], 0] - crd[elm[:, 0], 0])  # 1/(element lenght)
        data[1, 0] = [sparse.spdiags([-invLenght, invLenght], [-1, 0], Nnd, Nnd, "csr")]
        return data  # dictionnary


class BackwardFiniteDifferenceOrder2(FiniteDifference1D):
    # This finite difference element doesnt support any derivative
    # explicit method
    def __init__(self, n_elm_gp=0, **kargs):
        FiniteDifference1D.__init__(self, n_elm_gp)

    def computeOperator(self, crd, elm):
        Nnd = np.shape(crd)[0]
        data = {(0, 0): [sparse.identity(Nnd, "d", "csr")]}
        invLenght = 1 / (crd[elm[:, 1], 0] - crd[elm[:, 0], 0])  # 1/(element lenght)
        data[1, 0] = [sparse.spdiags([-invLenght, invLenght], [-1, 0], Nnd, Nnd, "csr")]
        invLenghtSquare = invLenght**2
        data[2, 0] = [
            sparse.spdiags(
                [invLenghtSquare, -2 * invLenghtSquare, invLenghtSquare],
                [-1, 0, 1],
                Nnd,
                Nnd,
                "csr",
            )
        ]
        return data  # dictionnary
