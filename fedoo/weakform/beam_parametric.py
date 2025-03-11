#### move to the PGD directory

from fedoo.core.base import ConstitutiveLaw
from fedoo.core.weakform import WeakFormBase
from fedoo.pgd.SeparatedArray import SeparatedOnes

import numpy as np


class ParametricBeam(WeakFormBase):
    """
    ParametricBernoulliBeam(E=None, nu = None, S=None, Jx=None, Iyy=None, Izz = None, R = None, name = "").
    * Same has BernoulliBeam but with the possibility to set each parameter as a coordinate of the problem.
    * Mainly usefull for parametric problem with the Proper Generalized Decomposition
    """

    def __init__(
        self,
        E=None,
        nu=None,
        S=None,
        Jx=None,
        Iyy=None,
        Izz=None,
        R=None,
        k=0,
        name="",
        space=None,
    ):
        """
        Weak formulation dedicated to treat parametric problems using Bernoulli beams for isotropic materials

        Arugments
        ----------
        name: str
            name of the weak formulation
        List of other optional parameters
            E: Young Modulus
            nu: Poisson Ratio
            S: Section area
            Jx: Torsion constant
            Iyy, Izz: Second moment of area, if Izz is not specified, Iyy = Izz is assumed
            k is a scalar. k=0 for no shear effect. For other values, k is the shear area coefficient
        When the differntial operator is generated (using PGD.Assembly) the parameters are searched among the Coordinatename defining the associated mesh.
        If a parameter is not found , a Numeric value should be specified in argument.

        In the particular case of cylindrical beam, the radius R can be specified instead of S, Jx, Iyy and Izz.
            S = pi * R**2
            Jx = pi * R**4/2
            Iyy = Izz = pi * R**4/4
        """

        WeakFormBase.__init__(self, name, space)

        self.space.new_variable("DispX")
        self.space.new_variable("DispY")
        if self.space.ndim == 3:
            self.space.new_variable("DispZ")
            self.space.new_variable("RotX")  # torsion rotation
            self.space.new_variable("RotY")
            self.space.new_variable("RotZ")
            self.space.new_vector("Disp", ("DispX", "DispY", "DispZ"))
            self.space.new_vector("Rot", ("RotX", "RotY", "RotZ"))
        else:  # if get_Dimension() == '2Dplane':
            self.space.new_variable("RotZ")
            self.space.new_vector("Disp", ("DispX", "DispY"))
            self.space.new_vector("Rot", ("RotZ"))
        # elif get_Dimension() == '2Dstress':
        #     assert 0, "No 2Dstress model for a beam kinematic. Choose '2Dplane' instead."

        if R is not None:
            S = np.pi * R**2
            Jx = np.pi * R**4 / 2
            Iyy = Izz = np.pi * R**4 / 4

        self.__parameters = {
            "E": E,
            "nu": nu,
            "S": S,
            "Jx": Jx,
            "Iyy": Iyy,
            "Izz": Izz,
            "k": k,
        }

    def __GetKe(self, mesh):
        NN = mesh.n_nodes  # number of nodes for every submeshes
        E_S = SeparatedOnes(NN)
        G_Jx = SeparatedOnes(NN)  # G = E/(1+nu)/2
        E_Iyy = SeparatedOnes(NN)
        E_Izz = SeparatedOnes(NN)
        if self.__parameters["k"] != 0:
            kG_S = self.__parameters["k"] * SeparatedOnes(NN)  # G = E/(1+nu)/2
        else:
            kG_S = 0

        for param in ["E", "nu", "R", "S", "Jx", "Iyy", "Izz"]:
            if mesh.FindCoordinatename(param) is not None:
                self.space.new_coordinate(param)
                id_mesh = mesh.FindCoordinatename(param)
                mesh._SetSpecificVariableRank(
                    id_mesh, "default", 0
                )  # all the variables use the same function for the submeshes related to parametric coordinates
                col = mesh.GetListMesh()[id_mesh].crd_name.index(param)

                if param == "R":
                    E_S.data[id_mesh][:, 0] = E_S.data[id_mesh][:, 0] * (
                        mesh.GetListMesh()[id_mesh].nodes[:, col] ** 2 * np.pi
                    )
                    G_Jx.data[id_mesh][:, 0] = G_Jx.data[id_mesh][:, 0] * (
                        mesh.GetListMesh()[id_mesh].nodes[:, col] ** 4 * (np.pi / 2)
                    )
                    E_Iyy.data[id_mesh][:, 0] = E_Iyy.data[id_mesh][:, 0] * (
                        mesh.GetListMesh()[id_mesh].nodes[:, col] ** 4 * (np.pi / 4)
                    )
                    E_Izz = E_Iyy
                    if not (isinstance(kG_S, int) and kG_S == 0):
                        kG_S.data[id_mesh][:, 0] = kG_S.data[id_mesh][:, 0] * (
                            mesh.GetListMesh()[id_mesh].nodes[:, col] ** 2 * np.pi
                        )
                    break

                if param in ["E", "S"]:
                    E_S.data[id_mesh][:, 0] = (
                        E_S.data[id_mesh][:, 0]
                        * mesh.GetListMesh()[id_mesh].nodes[:, col]
                    )
                    if not (isinstance(kG_S, int) and kG_S == 0):
                        kG_S.data[id_mesh][:, 0] = (
                            kG_S.data[id_mesh][:, 0]
                            * mesh.GetListMesh()[id_mesh].nodes[:, col]
                        )
                    if param == "E":
                        G_Jx.data[id_mesh][:, 0] = (
                            G_Jx.data[id_mesh][:, 0]
                            * mesh.GetListMesh()[id_mesh].nodes[:, col]
                        )
                if param == "Jx":
                    G_Jx.data[id_mesh][:, 0] = (
                        G_Jx.data[id_mesh][:, 0]
                        * mesh.GetListMesh()[id_mesh].nodes[:, col]
                    )
                if param == "nu":
                    G_Jx.data[id_mesh][:, 0] = G_Jx.data[id_mesh][:, 0] * (
                        0.5 / (1 + mesh.GetListMesh()[id_mesh].nodes[:, col])
                    )
                    if not (isinstance(kG_S, int) and kG_S == 0):
                        kG_S.data[id_mesh][:, 0] = kG_S.data[id_mesh][:, 0] * (
                            0.5 / (1 + mesh.GetListMesh()[id_mesh].nodes[:, col])
                        )
                if param in ["E", "Iyy"]:
                    E_Iyy.data[id_mesh][:, 0] = (
                        E_Iyy.data[id_mesh][:, 0]
                        * mesh.GetListMesh()[id_mesh].nodes[:, col]
                    )
                if param in ["E", "Izz"]:
                    E_Izz.data[id_mesh][:, 0] = (
                        E_Izz.data[id_mesh][:, 0]
                        * mesh.GetListMesh()[id_mesh].nodes[:, col]
                    )

            elif self.__parameters[param] is not None:
                if param in ["E", "S"]:
                    E_S = E_S * self.__parameters[param]
                    if not (isinstance(kG_S, int) and kG_S == 0):
                        kG_S = kG_S * self.__parameters[param]
                if param in ["E", "Jx"]:
                    G_Jx = G_Jx * self.__parameters[param]
                if param == "nu":
                    G_Jx = G_Jx * (0.5 / (1 + self.__parameters["nu"]))
                    if not (isinstance(kG_S, int) and kG_S == 0):
                        kG_S = kG_S * (0.5 / (1 + self.__parameters["nu"]))
                if param in ["E", "Iyy"]:
                    E_Iyy = E_Iyy * self.__parameters[param]
                if param in ["E", "Izz"]:
                    E_Izz = E_Izz * self.__parameters[param]

            elif param != "R":
                if param == "Izz":
                    E_Izz = E_Iyy
                else:
                    assert 0, "The parameter " + param + " is not defined"

        return [E_S, kG_S, kG_S, G_Jx, E_Iyy, E_Izz]

    def get_weak_equation(self, assembly, pb):
        mesh = assembly.mesh

        Ke = self.__GetKe(mesh)
        eps = self.space.op_beam_strain()

        return sum(
            [eps[i].virtual * eps[i] * Ke[i] if eps[i] != 0 else 0 for i in range(6)]
        )

    def GetGeneralizedStress(self, mesh):
        Ke = self.__GetKe(mesh)
        eps = self.space.op_beam_strain()

        return [eps[i] * Ke[i] for i in range(6)]


def ParametricBernoulliBeam(
    E=None, nu=None, S=None, Jx=None, Iyy=None, Izz=None, R=None, name=""
):
    # same as ParametricBeam with k=0 (no shear effect)
    return ParametricBeam(E=E, nu=nu, S=S, Jx=Jx, Iyy=Iyy, Izz=Izz, R=R, k=0, name=name)
