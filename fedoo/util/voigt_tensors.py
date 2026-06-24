"""Symmetric tensor objects based on the voigt notations."""

import numpy as np
import simcoon as sim


class _SymetricTensorList(list):  # base class for StressTensorList and StrainTensorList
    """Base class for handling collections of symmetric tensors in Voigt notation.

    The 6 components are stored as either scalars or NumPy arrays (one value per point).
    Order: [11, 22, 33, 12, 13, 23] (standard Voigt).
    """

    def __init__(self, l):
        if len(l) != 6:
            raise ValueError(
                "list length for " + str(self.__class__.__name__) + " object must be 6"
            )
        if isinstance(l, np.ndarray):
            self.array = l
        else:
            self.array = (
                None  # if object build from an array, keep it in memory to avoid copy
            )
        list.__init__(self, l)

    def __getitem__(self, item):
        if isinstance(item, str):
            if item == "vm":
                return self.von_mises()
            elif item == "pressure":
                return self.pressure()
            elif item in ["I", "II", "III"]:
                return self.eigvalues()[{"I": 2, "II": 1, "III": 0}.get(item)]
            elif item in ["XX", "YY", "ZZ", "XY", "XZ", "YZ"]:
                return self[
                    {
                        "XX": 0,
                        "YY": 1,
                        "ZZ": 2,
                        "XY": 3,
                        "XZ": 4,
                        "YZ": 5,
                    }.get(item)
                ]
            elif item == "eigvalues":
                return self.eigvalues()
            else:
                raise TypeError(f"unknown component '{item}'")
        else:
            return super().__getitem__(item)

    def __add__(self, tensor_list):
        if np.isscalar(tensor_list) and tensor_list == 0:
            return self
        return self.__class__(self.asarray() + tensor_list.asarray())

    def __sub__(self, tensor_list):
        if np.isscalar(tensor_list) and tensor_list == 0:
            return self
        return self.__class__(self.asarray() - tensor_list.asarray())

    def __copy__(self):
        return self.copy()

    def __deepcopy__(self, memo=None):
        return self.copy(asarray=False)

    def vtk_format(self):
        """Return a array adapted to export symetric tensor data in a vtk file."""
        try:
            return np.vstack([self[i] for i in [0, 1, 2, 3, 5, 4]]).astype(float)
        except:
            self.fill_zeros()
            return np.vstack([self[i] for i in [0, 1, 2, 3, 5, 4]]).astype(float)

    def to_tensor(self):
        """Reconstruct the 3x3 symmetric tensor(s).

        Returns
        -------
        numpy.ndarray
            Array of shape (3, 3, n_points).
        """
        return np.array(
            [
                [self[0], self[3], self[4]],
                [self[3], self[1], self[5]],
                [self[4], self[5], self[2]],
            ]
        )

    def asarray(self, copy=False):
        """Return the data as a single NumPy array of shape (6, n_points).

        Parameters
        ----------
        copy : bool, default False
            If True, returns a copy of the underlying array. If False,
            returns a view if possible or builds a new array from components.

        Returns
        -------
        numpy.ndarray
            Array of shape (6, n_points).
        """
        if self.array is None:
            try:
                return np.array(self)
            except ValueError:  # fill zeros first
                res = np.empty((6, self.n_points))
                for i in range(6):
                    res[i] = self[i]
                return res
        else:
            if copy:
                return self.array.copy()
            else:
                return self.array

    def deviatoric(self):
        """Deviatoric part of the Tensor using voigt form."""
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
        """Trace of the symmetric tensor."""
        return self[0] + self[1] + self[2]

    def hydrostatic(self):
        """Hydrostatic part of the Tensor using void form."""
        trace = (1 / 3) * self.trace()
        return self.__class__([trace, trace, trace, 0, 0, 0])

    def diagonalize(self):
        """Compute principal values and directions of the tensor for all points.

        Returns
        -------
        A tuple (eigenvalues, eigenvectors):

        eigenvalues : (3, n_points) numpy.ndarray
            eigenvalues[i] gives ith principal values arranged in ascending
            order for all points
        eigenvectors : (3, 3, n_points) numpy.ndarray
            eigenvectors[i, j] gives the jth component of the principal
            direction associated to the ith principal value.
        """
        full_tensor = self.to_tensor().transpose(2, 0, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(full_tensor)
        return eigenvalues.T, eigenvectors.transpose(2, 1, 0)

    def eigvalues(self):
        """Return the principal values of the tensor for all points.

        Returns
        -------
        eigenvalues : (3, n_points) numpy.ndarray
            eigenvalues[i] gives ith principal values arranged in ascending
            order for all points
        """
        full_tensor = self.to_tensor().transpose(2, 0, 1)
        return np.linalg.eigvalsh(full_tensor).T

    def fill_zeros(self):
        """Replace null scalar components with arrays of zeros.

        This method synchronizes the memory layout by ensuring all components
        have the same length (n_points), which is necessary for some
        vectorized operations or exports.
        """
        n = self.n_points
        for i in range(6):
            if np.isscalar(self[i]) and self[i] == 0:
                self[i] = np.zeros(n)

    def convert(self, assemb, convert_from=None, convert_to="GaussPoint", method=None):
        return self.__class__(
            [assemb.convert_data(S, convert_from, convert_to, method) for S in self]
        )

    def copy(self, asarray=False):
        """Create a copy of the tensor list.

        Parameters
        ----------
        asarray : bool, default False
            If True, the copy is backed by a single unified NumPy array.
            If False, preserves the hybrid scalar/array storage for memory efficiency.

        Returns
        -------
        _SymetricTensorList
            A new instance of the same class.
        """
        if self.array is None:
            if asarray:
                return self.__class__(self.asarray(copy=True))
            else:
                return self.__class__(
                    [comp if np.isscalar(comp) else comp.copy() for comp in self]
                )
        else:
            return self.__class__(self.array.copy())

    @property
    def n_points(self):
        """Number of points where stress values are given."""
        for i in range(6):
            if not (np.isscalar(self[i])):
                return len(self[i])  # number of stress values
        return 1

    @property
    def shape(self):
        return (6, self.n_points)


class StressTensorList(_SymetricTensorList):
    """A list of symmetric stress tensors.

    Supports operations like Von Mises stress, hydrostatic pressure,
    and PK2/Cauchy conversions.
    """

    def cauchy_to_pk2(self, F):
        """Convert Cauchy stress tensors to Second Piola-Kirchhoff (PK2) stress tensors.

        Requires the deformation gradient tensor F.

        Parameters
        ----------
        F : array_like
            Deformation gradient tensor of shape (3, 3, n_points).

        Returns
        -------
        StressTensorList
            The converted stress tensors in PK2 formulation.
        """
        return StressTensorList(
            sim.stress_convert(self.asarray(), F, "Cauchy2PKII", copy=False)
        )

    def pk2_to_cauchy(self, F):
        """Convert Second Piola-Kirchhoff (PK2) stress tensors to Cauchy stress tensors.

        Parameters
        ----------
        F : array_like
            Deformation gradient tensor of shape (3, 3, n_points).

        Returns
        -------
        StressTensorList
            The converted stress tensors in Cauchy formulation.
        """
        return StressTensorList(
            sim.stress_convert(self.asarray(), F, "PKII2Cauchy", copy=False)
        )

    def cauchy_to_pk1(self, F):
        """Convert Cauchy stress tensors to First Piola-Kirchhoff (PK1) stress tensors.

        Requires the deformation gradient tensor F.

        Parameters
        ----------
        F : array_like
            Deformation gradient tensor of shape (3, 3, n_points).

        Returns
        -------
        StressTensorList
            The converted stress tensors in PK2 formulation.
        """
        return StressTensorList(
            sim.stress_convert(self.asarray(), F, "Cauchy2PKI", copy=False)
        )

    def pk1_to_cauchy(self, F):
        """Convert First Piola-Kirchhoff (PK1) stress tensors to Cauchy stress tensors.

        Parameters
        ----------
        F : array_like
            Deformation gradient tensor of shape (3, 3, n_points).

        Returns
        -------
        StressTensorList
            The converted stress tensors in Cauchy formulation.
        """
        return StressTensorList(
            sim.stress_convert(self.asarray(), F, "PKI2Cauchy", copy=False)
        )

    def von_mises(self):
        """Calculate the Von Mises equivalent stress.

        The calculation is vectorized for all points.

        Returns
        -------
        float or numpy.ndarray
            The Von Mises stress value(s). Returns a scalar if n_points=1,
            otherwise an array of shape (n_points,).
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
        """Calculate the hydrostatic pressure.

        Defined as -1/3 * trace(sigma).

        Returns
        -------
        float or numpy.ndarray
            The pressure value(s) for all points.
        """
        return (-1 / 3) * self.trace()

    def to_strain(self):
        """Convert current object to StrainTensorList."""
        return StrainTensorList(self[:3] + [self[i] * 2 for i in [3, 4, 5]])

    def to_stress(self):
        """Convert current object to StressTensorList."""
        return self


class StrainTensorList(_SymetricTensorList):
    """A list of symmetric strain tensors.

    Handles specific Voigt scaling for VTK export and tensor reconstruction.
    """

    def vtk_format(self):
        """Format the symmetric tensor data for VTK export.

        Rearranges components to match the VTK symmetric tensor convention
        (XX, YY, ZZ, XY, YZ, XZ).

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (6, n_points) with float type.
        """
        try:
            return np.vstack(self[:3] + [self[i] / 2 for i in [3, 5, 4]]).astype(float)
        except:
            self.fill_zeros()
            return np.vstack(self[:3] + [self[i] / 2 for i in [3, 5, 4]]).astype(float)

    def to_tensor(self):
        """Reconstruct the 3x3 symmetric tensor(s).

        Returns
        -------
        numpy.ndarray
            Array of shape (3, 3, n_points).
        """
        return np.array(
            [
                [self[0], self[3] / 2, self[4] / 2],
                [self[3] / 2, self[1], self[5] / 2],
                [self[4] / 2, self[5] / 2, self[2]],
            ]
        )

    def to_stress(self):
        """Convert current object to StressTensorList."""
        return StressTensorList(self[:3] + [self[i] / 2 for i in [3, 4, 5]])

    def to_strain(self):
        """Convert current object to StrainTensorList."""
        return self
