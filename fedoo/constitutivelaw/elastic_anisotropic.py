# derive de ConstitutiveLaw
# compatible with the simcoon strain and stress notation

from fedoo.core.mechanical3d import Mechanical3D
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList

import numpy as np


class ElasticAnisotropic(Mechanical3D):
    """
    Linear full Anistropic constitutive law defined from the rigidity matrix H.

    The constitutive Law should be associated with :mod:`fedoo.weakform.InternalForce`

    Parameters
    ----------
    H : list of list or an array (shape=(6,6)) of scalars or arrays of gauss point values.
        The rigidity matrix.
        If H is a list of gauss point values, the shape shoud be H.shape = (6,6,NumberOfGaussPoints)
    name : str, optional
        The name of the constitutive law
    """

    def __init__(self, H, name=""):
        Mechanical3D.__init__(self, name)  # heritage

        self._H = H
        # self._stress = 0
        # self._grad_disp = 0

    def initialize(self, assembly, pb):
        assembly.sv["TangentMatrix"] = self.get_tangent_matrix(assembly)

    def update(self, assembly, pb):
        # linear problem = no need to recompute tangent matrix if it has already been computed
        if not (assembly._nlgeom) and "TangentMatrix" in assembly.sv:
            H = assembly.sv["TangentMatrix"]
        else:
            H = self.get_tangent_matrix(assembly)
            assembly.sv["TangentMatrix"] = H

        if "DStrain" in assembly.sv:
            total_strain = assembly.sv["Strain"] + assembly.sv["DStrain"]
        else:
            total_strain = assembly.sv["Strain"]

        assembly.sv["Stress"] = StressTensorList(
            [
                sum(
                    [total_strain[j] * assembly.convert_data(H[i][j]) for j in range(6)]
                )
                for i in range(6)
            ]
        )  # H[i][j] are converted to gauss point excepted if scalar

    def get_stress_from_strain(self, assembly, strain_tensor):
        H = self.get_tangent_matrix(assembly)

        sigma = StressTensorList(
            [sum([strain_tensor[j] * H[i][j] for j in range(6)]) for i in range(6)]
        )

        return sigma  # list of 6 objets

    def get_tangent_matrix(
        self, assembly, dimension=None
    ):  # Tangent Matrix in lobal coordinate system (no change of basis)
        if dimension is None:
            dimension = assembly.space.get_dimension()

        H = self.local2global_H(self._H)
        if dimension == "2Dstress":
            return self.get_H_plane_stress(H)
        else:
            return H

    def get_elastic_matrix(self, dimension="3D"):
        return self.get_tangent_matrix(None, dimension)

    # def ComputeStrain(self, assembly, pb, nlgeom, type_output='GaussPoint'):
    #     displacement = pb.get_dof_solution()
    #     if np.isscalar(displacement) and displacement == 0:
    #         return 0 #if displacement = 0, Strain = 0
    #     else:
    #         return assembly.get_strain(displacement, type_output)
