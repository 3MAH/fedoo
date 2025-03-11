# derive de ConstitutiveLaw
# simcoon compatible

from fedoo.core.mechanical3d import Mechanical3D
from fedoo.constitutivelaw.elastic_anisotropic import ElasticAnisotropic

import numpy as np


class ElasticIsotrop(ElasticAnisotropic):
    """
    A simple linear elastic isotropic constitutive law defined from a Yound Modulus and a Poisson Ratio.

    The constitutive Law should be associated with :mod:`fedoo.weakform.InternalForce`

    Parameters
    ----------
    E : scalars or arrays of gauss point values.
        The Young Modulus of the elastic isotropic material
    nu : scalars or arrays of gauss point values.
        The Poisson Ratio of the elastic isotropic material
    name : str, optional
        The name of the constitutive law
    """

    def __init__(self, E, nu, name=""):
        Mechanical3D.__init__(self, name)  # heritage
        self.E = E
        """Young Modulus of the material"""
        self.nu = nu
        """Poisson Ratio of the material """

    def get_tangent_matrix(
        self, assembly, dimension=None
    ):  # Tangent Matrix in lobal coordinate system (no change of basis)
        if dimension is None:
            dimension = assembly.space.get_dimension()

        E = self.E
        nu = self.nu

        # the returned stiffness matrix is 6x6 even in 2D
        if np.isscalar(E) and np.isscalar(nu):
            H = np.zeros((6, 6), dtype=float)
        else:
            H = np.zeros((6, 6), dtype="object")

        if dimension == "2Dstress":
            # for 2D plane stress problems
            H[0, 0] = H[1, 1] = E / (1 - nu**2)
            H[0, 1] = nu * E / (1 - nu**2)
            H[3, 3] = 0.5 * E / (1 + nu)
            H[1, 0] = H[0, 1]  # symétrie
        else:
            H[0, 0] = H[1, 1] = H[2, 2] = E * (
                1.0 / (1 + nu) + nu / ((1.0 + nu) * (1 - 2 * nu))
            )  # H1 = 2*mu+lamb
            H[0, 1] = H[0, 2] = H[1, 2] = E * (
                nu / ((1 + nu) * (1 - 2 * nu))
            )  # H2 = lamb
            H[3, 3] = H[4, 4] = H[5, 5] = 0.5 * E / (1 + nu)  # H3 = mu
            H[1, 0] = H[0, 1]
            H[2, 0] = H[0, 2]
            H[2, 1] = H[1, 2]  # symétrie

        return H

    @property
    def G(self):
        """Shear modulus of the material"""
        return self.E / (1 + self.nu) / 2


if __name__ == "__main__":
    law = ElasticIsotrop(5e9, 0.3)
