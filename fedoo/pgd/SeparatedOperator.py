from fedoo.pgd.SeparatedArray import SeparatedArray
import scipy as sp
from scipy import linalg
from numbers import Number


class SeparatedOperator:  # Opérateur discret avec galerkin
    """
    Discrete separated operator

    This can be instantiated in two ways:
        SeparatedOperator(op)
            if op is a SeparatedOperator object, return a copy of op
        SeparatedOperator([[Array11, Array12,...], [Array21,22, ...], ....])
            Create an SeparatedOperator object from a list of list of sparse arrays
            (lil or csr format).
            The lenght of the list is the number of separated terms
            Each element of the list is another list whose lenght is the
            number of dimensions. The arrays in these lists are the
            discrete operators over each dimension.


    Attributes
    ----------
    data : list of list of sparse arrays (format lil or csr)
        Data for each terms and each separated space.

    Principal methods
    ----------
    get_Dimension(): return the number of separated spaces
    NumberOfOperators(): return the number of terms in the sum of separated operators
    GetShape(dd): return the number of column in the operator related to the space dd
    GetShapeRow(dd): return the number of row in the operator related to the space dd

    Notes
    -----
    SeparatedOperator object can be used in arithmetic operations: it support
    addition (sum of the terms of two operators), and multiplication by a FoncD


    Data Structure
        The operators are contained in the data attributes.
    """

    def __add__(self, A):  # somme
        if A.__class__ == SeparatedOperator:
            if sp.sum(sp.array(A.GetShape()) - sp.array(self.GetShape())) == 0:
                return SeparatedOperator(self.data + A.data)
            else:
                raise NameError("Dimensions doesnt match")
        elif A == 0:
            return self
        else:
            return NotImplemented

    def __radd__(self, A):  # somme
        if A == 0:
            return self
        else:
            return NotImplemented

    def __getitem__(self, key):
        return self.data[key]

    def __init__(self, AA=[]):
        if isinstance(AA, SeparatedOperator):
            self.data = [[M.copy() for M in listM] for listM in AA]
        else:
            self.data = AA

    def __mul__(self, FF):
        res = self.dot(FF)
        if res == NotImplemented:
            return self.tensorProd(FF)
        return res

    def __rmul__(self, a):  # multiplication à droite
        return self * a

    def __setitem__(self, key, value):
        self.data[key] = value

    def tensorProd(
        self, op, modify=False
    ):  # à vérifier dans le cas où le nombre de terme est différent de 1 pour les deux objets SeparatedOperator
        """
        SeparatedOperator.tensorProd(op, modify = False)

        If op is a SeparatedOperator: Return the tensor product of two SeparatedOperator objects
            The returned object is a SeparatedOperator of dimension self.dim+op.dim

        if modify == True: the object is modified
        """
        if isinstance(op, SeparatedOperator):
            res = [self.data[i] + op.data[0] for i in range(self.NumberOfOperators())]
            for j in range(1, op.NumberOfOperators()):
                res = res + [
                    self.data[i] + op.data[j] for i in range(self.NumberOfOperators())
                ]
        else:
            return NotImplemented
        if modify == True:
            self.data = res
            return self
        else:
            return SeparatedOperator(res)

    def dot(self, FF):
        if self.data == []:
            return 0
        if isinstance(FF, SeparatedArray):
            if FF.dim == self.get_dimension():
                ntA = self.NumberOfOperators()
                nf = FF.nbTerm()
                CC = [
                    sp.c_[sp.zeros((self.GetShapeRow(dd), ntA * nf))]
                    for dd in range(self.get_dimension())
                ]
                colone = 0
                for kk in range(ntA):
                    for dd in range(self.get_dimension()):
                        CC[dd][:, colone : colone + nf] = (
                            self.data[kk][dd] * FF.data[dd]
                        )
                    colone += nf
            else:
                raise NameError("Dimensions doesnt match")
            return SeparatedArray(CC)
        elif isinstance(FF, Number):
            if FF == 0:
                return 0
            new_op = SeparatedOperator(self)
            ntA = new_op.NumberOfOperators()
            for kk in range(ntA):
                new_op.data[kk][0] *= FF
            return new_op
        else:
            return NotImplemented

    def append(self, Op):
        self.data.append(Op)

    def copy(self):
        """
        Return a copy of the object
        """
        return self.__class__(self)

    def NumberOfOperators(self):
        """
        Return the number of terms in the sum of separated operators
        """
        return len(self.data)

    def get_dimension(self):
        if self.data == []:
            return 0
        else:
            return len(self.data[0])

    def GetShape(self, dd=None):
        """
        SeparatedOperator.GetShape(dd)
        Number of columns in the operators related to each separated space

        dd (int): the id of the dimension
        If dd is not defined, return a list containing the number of columns for all the separated spaces

        For usual PGD applications the arrays defined in the data attributes are square.
        If it is not the case, the method GetShapeRow gives the number of lines of
        operators for each dimension.
        """
        if dd is None:
            return [sp.shape(data_dd)[1] for data_dd in self.data[0]]
        else:
            return sp.shape(self.data[0][dd])[1]

    def GetShapeRow(self, dd=None):
        """
        See the documentation of SeparatedOperator.GetShape
        """
        if dd is None:
            return [sp.shape(data_dd)[0] for data_dd in self.data[0]]
        else:
            return sp.shape(self.data[0][dd])[0]

    def tocsr(self):
        """
        Convert in the lil format all the sparse arrays
        that are in the data attribute
        (see the scipy documentation for more informations)
        """
        for dd in range(self.get_dimension()):
            for kk in range(self.NumberOfOperators()):
                self.data[kk][dd] = self.data[kk][dd].tocsr()

    def tolil(self):
        """
        Convert in the csr format all the sparse arrays
        that are in the data attribute
        (see the scipy documentation for more informations)
        """
        for dd in range(self.get_dimension()):
            for kk in range(self.NumberOfOperators()):
                self.data[kk][dd] = self.data[kk][dd].tolil()
