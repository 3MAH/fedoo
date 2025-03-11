# ===============================================================================
# Derivative and Differential operator
# ===============================================================================
# from fedoo.pgd.SeparatedArray import *
# from fedoo.core.modelingspace  import ModelingSpace
import numpy as np

from numbers import (
    Number,
)  # classe de base qui permet de tester si un type est numérique


class _Derivative:  # derivative operator used in DiffOp
    """
    Define a derivative operator.
    _Derivative(u,x,ordre,decentrement)

    Parameters
    ----------
    u (int) : the variable that is derived
    x (int) : derivative with respect to x
    ordre (int) : the order of the derivative (0 for no derivative)
    decentrement (int) : used only to define a decentrement when using finie diferences method
    """

    def __init__(self, u=0, x=0, ordre=0, decentrement=0, u_name=None):
        self.u = u  # u est la variable dérivée
        if ordre == 0:
            assert x == 0, "x should be set to 0 if derivative ordre is 0."
        self.x = x  # x est la coordonnée par rapport à qui on dérive
        self.u_name = u_name

        # x peut etre une liste. Dans ce cas on dérive par rapport à plusieurs coordonnées (ex : si x=[0,1] op = d_dx+d_dy pour la divergence)
        self.ordre = ordre  # ordre de la dérivée (0, 1 ou 2)
        self.decentrement = decentrement  # décentrement des dériviées pour différences finies uniquement

    def __eq__(self, other):
        if isinstance(other, _Derivative):
            return (self.u_name, self.x, self.ordre) == (
                other.u_name,
                other.x,
                other.ordre,
            )
        else:
            return False

    def __lt__(self, other):
        if isinstance(other, _Derivative):
            return (self.u_name, self.x, self.ordre) < (
                other.u_name,
                other.x,
                other.ordre,
            )
        elif np.isscalar(other):
            return False
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, _Derivative):
            return (self.u_name, self.x, self.ordre) <= (
                other.u_name,
                other.x,
                other.ordre,
            )
        elif np.isscalar(other):
            return False
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, _Derivative):
            return (self.u_name, self.x, self.ordre) > (
                other.u_name,
                other.x,
                other.ordre,
            )
        elif np.isscalar(other):
            return True
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, _Derivative):
            return (self.u_name, self.x, self.ordre) >= (
                other.u_name,
                other.x,
                other.ordre,
            )
        elif np.isscalar(other):
            return True
        else:
            return NotImplemented


class DiffOp:
    def __init__(self, u, x=0, ordre=0, decentrement=0, vir=0, u_name=None):
        self.mesh = None

        # mod_space = ModelingSpace.get_active()

        # if isinstance(u,str):
        #     u = mod_space.GetVariableRank(u)
        # if isinstance(x,str):
        #     x = mod_space.GetCoordinateRank(x)

        if isinstance(u, int):
            self.coef = [1]
            if vir == 0:
                self.op = [_Derivative(u, x, ordre, decentrement, u_name)]
                self.op_vir = [1]
            else:
                self.op_vir = [_Derivative(u, x, ordre, decentrement, u_name)]
                self.op = [1]
        elif isinstance(u, list) and isinstance(x, list) and isinstance(ordre, list):
            self.op = u
            self.op_vir = x
            self.coef = ordre
        else:
            raise NameError("Argument error")

    def __eq__(self, other):
        if np.isscalar(other):
            return False
        else:
            return self is other

    def __add__(self, A):
        if isinstance(A, DiffOp):
            return DiffOp(self.op + A.op, self.op_vir + A.op_vir, self.coef + A.coef)
        elif np.isscalar(A) and A == 0:
            return self
        else:  # A could be Number, np.ndarray, ...
            return DiffOp(self.op + [1], self.op_vir + [1], self.coef + [A])

    def __sub__(self, A):
        if isinstance(A, DiffOp):
            return DiffOp(self.op + A.op, self.op_vir + A.op_vir, self.coef + (-A).coef)
        elif np.isscalar(A) and A == 0:
            return self
        else:  # A could be Number, np.ndarray, ...
            return DiffOp(self.op + [1], self.op_vir + [1], self.coef + [-A])

    def __rsub__(self, A):
        if np.isscalar(A) and A == 0:
            return -self
        else:  # A could be Number, np.ndarray, ...
            return DiffOp([1] + self.op, [1] + self.op_vir, [A] + (-self).coef)

    def __neg__(self):
        return DiffOp(self.op, self.op_vir, [-cc for cc in self.coef])

    def __mul__(self, A):
        #        if isinstance(A, SeparatedArray) and A.norm() == 0: return 0
        if isinstance(A, DiffOp):
            res = DiffOp([], [], [])
            for ii in range(len(A.op)):
                for jj in range(len(self.op)):
                    if (
                        A.op_vir[ii] == 1 and self.op[jj] == 1
                    ):  # si A contient un opérateur réel et self un virtuel
                        res += DiffOp(
                            [A.op[ii]],
                            [self.op_vir[jj]],
                            [A.coef[ii] * self.coef[jj]],
                        )
                    elif A.op[ii] == 1 and self.op_vir[jj] == 1:  # si c'est l'inverse
                        res += DiffOp(
                            [self.op[ii]],
                            [A.op_vir[jj]],
                            [A.coef[ii] * self.coef[jj]],
                        )
                    else:
                        raise NameError("Impossible operation")
            return res
        else:  # isinstance(A, (Number, SeparatedArray)):
            if np.isscalar(A):
                if A == 0:
                    return 0
                if A == 1:
                    return self

            return DiffOp(self.op, self.op_vir, [A * cc for cc in self.coef])

    def __radd__(self, A):
        return self + A

    def __rmul__(self, A):
        return self * A

    def __div__(self, A):
        return self * (1 / A)

    def __getitem__(self, item):
        return (self.op[item], self.op_vir[item], self.coef[item])

    def __len__(self):
        return len(self.op)

    def __str__(self):
        res = ""
        for ii in range(len(self.op)):
            if np.isscalar(self.coef[ii]):
                coef_str = str(self.coef[ii]) + " "
            else:
                coef_str = "f "

            if self.op[ii] == 1:
                op_str = ""
            elif self.op[ii].ordre == 0:
                op_str = "u" + str(self.op[ii].u)
            else:
                op_str = "du" + str(self.op[ii].u) + "/dx" + str(self.op[ii].x)

            if self.op_vir[ii] == 1:
                op_vir_str = ""
            elif self.op_vir[ii].ordre == 0:
                op_vir_str = "v" + str(self.op_vir[ii].u)
            else:
                op_vir_str = (
                    "dv" + str(self.op_vir[ii].u) + "/dx" + str(self.op_vir[ii].x)
                )
            if ii != 0:
                res += " + "
            res += coef_str + op_vir_str + " " + op_str
        return res

    def _getitem_sorting(self, item):
        # coef not used for sorting: only return op and op_vir
        return (self.op[item], self.op_vir[item])

    def sort(self):
        sorted_indices = sorted(range(len(self)), key=self._getitem_sorting)
        same_as_next = [
            self._getitem_sorting(sorted_indices[i])
            == self._getitem_sorting(sorted_indices[i + 1])
            for i in range(0, len(self) - 1)
        ]
        self.coef = [self.coef[i] for i in sorted_indices]
        self.op = [self.op[i] for i in sorted_indices]
        self.op_vir = [self.op_vir[i] for i in sorted_indices]

        return same_as_next, sorted_indices

    # def sort2(self):
    #     nn = 50  # should be higher than the max number of variables, orders and coordinates
    #     intForSort = []
    #     for ii in range(len(self.op)):
    #         if self.op[ii] != 1 and self.op_vir != 1:
    #             intForSort.append(
    #                 self.op_vir[ii].ordre
    #                 + nn * self.op_vir[ii].x
    #                 + nn**2 * self.op_vir[ii].u
    #                 + nn**3 * self.op[ii].ordre
    #                 + nn**4 * self.op[ii].x
    #                 + nn**5 * self.op[ii].u
    #             )
    #         elif self.op[ii] == 1:
    #             if self.op_vir == 1:
    #                 intForSort.append(-1)
    #             else:
    #                 intForSort.append(
    #                     nn**6
    #                     + nn**6 * self.op_vir[ii].ordre
    #                     + nn**7 * self.op_vir[ii].x
    #                     + nn**8 * self.op_vir[ii].u
    #                 )
    #         else:  # self.op_vir[ii] = 1
    #             intForSort.append(
    #                 nn**9
    #                 + nn**9 * self.op[ii].ordre
    #                 + nn**10 * self.op[ii].x
    #                 + nn**11 * self.op[ii].u
    #             )

    #     sorted_indices = np.array(intForSort).argsort()
    #     same_as_next = [
    #         intForSort[sorted_indices[i]] == intForSort[sorted_indices[i + 1]]
    #         for i in range(len(self)-1)
    #     ]

    #     self.coef = [self.coef[i] for i in sorted_indices]
    #     self.op = [self.op[i] for i in sorted_indices]
    #     self.op_vir = [self.op_vir[i] for i in sorted_indices]

    #     return same_as_next, sorted_indices
    #     # return [intForSort[i] for i in sorted_indices], sorted_indices

    def nvar(self):
        return max([op.u for op in self.op]) + 1

    #    def reduction(self, **kwargs):
    #        for gg in self.coef:
    #            if isinstance(gg,SeparatedArray):
    #                gg.reduction(**kwargs)

    @property
    def virtual(self):  # retourne l'opérateur virtuel
        return DiffOp(self.op_vir, self.op, self.coef)
