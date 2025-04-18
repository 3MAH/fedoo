# -*- coding: utf-8 -*-
import numpy as np
from copy import deepcopy

try:
    from simcoon import simmit as sim

    USE_SIMCOON = True
except ImportError:
    USE_SIMCOON = False

# class StressTensorArray(np.ndarray):

#     def __new__(cls, input_array):
#         # Input array is an already formed ndarray instance
#         # We first cast to be our class type
#         if len(input_array) != 6: raise NameError('lenght for StressTensorArray object must be 6')
#         obj = np.asarray(input_array).view(cls)

#         # # add the new attribute to the created instance
#         # obj.info = info
#         # Finally, we must return the newly created object:
#         return obj

#     def vtk_format(self):
#         """
#         Return a array adapted to export symetric tensor data in a vtk file
#         See the utilities.ExportData class for more details
#         """
#         return np.vstack((self[0:4].reshape(4,-1), self[5], self[4])).astype(float)

#     def to_tensor(self):
#         return np.array([[self[0],self[3],self[4]], [self[3], self[1], self[5]], [self[4], self[5], self[2]]])

#     def von_mises(self):
#         """
#         Return the Von Mises stress
#         """
#         return np.sqrt( 0.5 * ((self[0]-self[1])**2 + (self[1]-self[2])**2 + (self[0]-self[2])**2 \
#                          + 6 * (self[3]**2 + self[4]**2 + self[5]**2) ) )


# def __array_finalize__(self, obj):
#     # see InfoArray.__array_finalize__ for comments
#     if obj is None: return
#     self.info = getattr(obj, 'info', None)


class _SymetricTensorList(list):  # base class for StressTensorList and StrainTensorList
    def __init__(self, l):
        if len(l) != 6:
            raise NameError(
                "list lenght for " + str(self.__class__.__name__) + " object must be 6"
            )
        if isinstance(l, np.ndarray):
            self.array = l
        else:
            self.array = (
                None  # if object build from an array, keep it in memory to avoid copy
            )
        list.__init__(self, l)

    def __add__(self, tensor_list):
        if np.isscalar(tensor_list) and tensor_list == 0:
            return self
        return self.__class__(self.asarray() + tensor_list.asarray())

    def __sub__(self, tensor_list):
        if np.isscalar(tensor_list) and tensor_list == 0:
            return self
        return self.__class__(self.asarray() - tensor_list.asarray())

    def __deepcopy__(self, memo=None):
        if self.array is None:
            return deepcopy(self, memo)
        else:
            return self.__class__(self.array.copy())

    def vtk_format(self):
        """
        Return a array adapted to export symetric tensor data in a vtk file
        See the utilities.ExportData class for more details
        """
        try:
            return np.vstack([self[i] for i in [0, 1, 2, 3, 5, 4]]).astype(float)
        except:
            self.fill_zeros()
            return np.vstack([self[i] for i in [0, 1, 2, 3, 5, 4]]).astype(float)

    def to_tensor(self):
        return np.array(
            [
                [self[0], self[3], self[4]],
                [self[3], self[1], self[5]],
                [self[4], self[5], self[2]],
            ]
        )

    def asarray(self):
        if self.array is None:
            try:
                return np.array(self)
            except ValueError:  # fill zeros first
                for i in range(6):
                    if not (np.isscalar(self[i])):
                        N = len(self[i])  # number of stress values
                        break

                res = np.empty((6, N))
                for i in range(6):
                    if np.isscalar(self[i]) and self[i] == 0:
                        res[i] = np.zeros(N)
                    else:
                        res[i] = self[i]
                return res
        else:
            return self.array

    def deviatoric(self):
        """
        Return the deviatoric part of the Tensor using voigt form
        """
        return self.__class__(
            [
                2 / 3 * self[0] - 1 / 3 * self[1] - 1 / 3 * self[2],
                -1 / 3 * self[0] + 2 / 3 * self[1] - 1 / 3 * self[2],
                -1 / 3 * self[0] - 1 / 3 * self[1] + 2 / 3 * self[2],
                self[3],
                self[4],
                self[5],
            ]
        )

    def trace(self):
        return self[0] + self[1] + self[2]

    def hydrostatic(self):
        """
        Return the hydrostatic part of the Tensor using void form
        """
        trace = (1 / 3) * self.trace()
        return self.__class__([trace, trace, trace, 0, 0, 0])

    def diagonalize(self):
        """
        Return the principal value and principal directions of the tensor for all given points
        Return eigenvalues, eigenvectors
        The line of eigenvalues are the values of the principal stresses for all points.
        eigenvectors[i] is the principal direction associated to the ith principal value.
        The line of eigenvectors[i] are component of the vector, for all points.
        """
        full_tensor = self.to_tensor().transpose(2, 0, 1)
        eigenvalues, eigenvectors = np.linalg.eig(full_tensor)
        return eigenvalues, eigenvectors.transpose(2, 0, 1)

    def fill_zeros(self):
        for i in range(6):
            if not (np.isscalar(self[i])):
                N = len(self[i])  # number of stress values
                break
        for i in range(6):
            if np.isscalar(self[i]) and self[i] == 0:
                self[i] = np.zeros(N)

    def convert(self, assemb, convert_from=None, convert_to="GaussPoint"):
        return self.__class__(
            [assemb.convert_data(S, convert_from, convert_to) for S in self]
        )


class StressTensorList(_SymetricTensorList):
    def cauchy_to_pk2(self, F):
        if USE_SIMCOON:
            return StressTensorList(
                sim.stress_convert(self.asarray(), F, "Cauchy2PKII", copy=False)
            )
        else:
            raise NameError("Install simcoon to allow conversion from cauchy to pk2")

    def pk2_to_cauchy(self, F):
        if USE_SIMCOON:
            return StressTensorList(
                sim.stress_convert(self.asarray(), F, "PKII2Cauchy", copy=False)
            )
        else:
            pk2 = self.to_tensor().transpose(2, 0, 1)

            #            GradX = [[Assembly.get_all()['Assembling'].get_node_results(GradOp[i][j], Mesh.get_all()[meshname].nodes.T.reshape(-1)+Problem.get_disp()) for j in range(3)] for i in range(3)]
            F = np.transpose(np.array(F)[:, :, :], (2, 0, 1))
            J = np.linalg.det(F)
            FT = F.transpose(0, 2, 1)

            cauchy = (1 / J).reshape(-1, 1, 1) * (F @ pk2 @ FT)
            return StressTensorList(
                [
                    cauchy[:, 0, 0],
                    cauchy[:, 1, 1],
                    cauchy[:, 2, 2],
                    cauchy[:, 0, 1],
                    cauchy[:, 0, 2],
                    cauchy[:, 1, 2],
                ]
            )

    def cauchy_to_pk1(self, F):
        if USE_SIMCOON:
            return StressTensorList(
                sim.stress_convert(self.asarray(), F, "Cauchy2PKI", copy=False)
            )
        else:
            raise NameError("Install simcoon to allow conversion from cauchy to pk1")

    def pk1_to_cauchy(self, F):
        if USE_SIMCOON:
            return StressTensorList(
                sim.stress_convert(self.asarray(), F, "PKI2Cauchy", copy=False)
            )
        else:
            raise NameError("Install simcoon to allow conversion from pk1 to cauchy")

    def von_mises(self):
        """
        Return the vonMises stress
        """
        # sim.Mises_stress(self.asarray()) # not vectorized for now
        return np.sqrt(
            0.5
            * (
                (self[0] - self[1]) ** 2
                + (self[1] - self[2]) ** 2
                + (self[0] - self[2]) ** 2
                + 6 * (self[3] ** 2 + self[4] ** 2 + self[5] ** 2)
            )
        )

    def pressure(self):
        return (-1 / 3) * self.trace()

    def to_strain(self):
        return StrainTensorList(self[:3] + [self[i] * 2 for i in [3, 4, 5]])

    def to_stress(self):
        return self


class StrainTensorList(_SymetricTensorList):
    def vtk_format(self):
        """
        Return a array adapted to export symetric tensor data in a vtk file
        See the utilities.ExportData class for more details
        """

        try:
            return np.vstack(self[:3] + [self[i] / 2 for i in [3, 5, 4]]).astype(float)
        except:
            self.fill_zeros()
            return np.vstack(self[:3] + [self[i] / 2 for i in [3, 5, 4]]).astype(float)

    def to_tensor(self):
        return np.array(
            [
                [self[0], self[3] / 2, self[4] / 2],
                [self[3] / 2, self[1], self[5] / 2],
                [self[4] / 2, self[5] / 2, self[2]],
            ]
        )

    def to_stress(self):
        return StressTensorList(self[:3] + [self[i] / 2 for i in [3, 4, 5]])

    def to_strain(self):
        return self
