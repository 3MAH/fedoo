# derive de ConstitutiveLaw
# compatible with the simcoon strain and stress notation

from fedoo.core.mechanical3d import Mechanical3D
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList
import warnings

try:
    from simcoon import simmit as sim

    try:
        from simcoon import __version__

        USE_SIMCOON = True
    except ImportError:
        warnings.warn("Simcoon version is to old. Simcoon ignored.")
        USE_SIMCOON = False

except ImportError:
    USE_SIMCOON = False

import numpy as np


class Simcoon(Mechanical3D):
    """Constitutive laws from the simcoon library.

    The constitutive Law should be associated with
    :mod:`fedoo.weakform.StressEquilibrium`

    Parameters
    ----------
    umat_name: str
        Name of the constitutive law.
    props: numpy.array
        The constitive laws properties
    name : str
        The name of the constitutive law
    """

    def __init__(self, umat_name, props, name=""):
        if not (USE_SIMCOON):
            raise NameError(
                "Simcoon library need to be installed for using the constitutive laws"
            )

        # props is a nparray containing all the material variables
        # nstatev is a nparray containing all the material variables
        Mechanical3D.__init__(self, name)  # heritage
        # self._statev_initial = statev #statev may be an int or an array
        # self.__useElasticModulus = True ??

        # self.__currentGradDisp = self.__initialGradDisp = 0

        # ndi = nshr = 3 #compute the 3D constitutive law even for 2D law
        self.umat_name = umat_name
        self.props = np.asfortranarray(
            np.c_[props]
        )  # if props is 1d -> take it as column.
        # ensure a fortran order for compatibility with armadillo

        self.use_elastic_lt = True
        # option to use the elastic tangeant matrix
        # (in principle = initial tangent matrix)
        # at the begining of each time step

        # _Lt_from_F attribute is set to True if the tangent matrix is related
        # to F instead of log epsilon, ie for hyper elastic materials
        if umat_name == "ELISO":
            self.n_statev = 1
            self.props_label = {"E": 0, "nu": 1, "alpha": 2}
            self.statev_label = {"T": 0}
        elif umat_name == "ELIST":
            self.n_statev = 1
            self.props_label = {
                "axis": 0,
                "EL": 1,
                "ET": 2,
                "nuTL": 3,
                "nuTT": 4,
                "GLT": 5,
                "alphaL": 6,
                "alphaT": 7,
            }
            self.statev_label = {"T": 0}
        elif umat_name == "ELORT":
            self.n_statev = 1
            self.props_label = {
                "Ex": 0,
                "Ey": 1,
                "Ez": 2,
                "nuxy": 3,
                "nuxz": 4,
                "nuyz": 5,
                "Gxy": 6,
                "Gxz": 7,
                "Gyz": 8,
                "alphax": 9,
                "alphay": 10,
                "alphaz": 11,
            }
            self.statev_label = {"T": 0}
        elif umat_name == "EPICP":
            self.n_statev = 8
            self.props_label = {
                "E": 0,
                "nu": 1,
                "alpha": 2,
                "sigmaY": 3,
                "k": 4,
                "m": 5,
            }  # powerlaw sigma_e = sigmaY + k * eps_p^m
            self.statev_label = {"T": 0, "P": 1, "EP": slice(2, 8)}
        elif umat_name == "EPKCP":
            self.n_statev = 14
            self.props_label = {
                "E": 0,
                "nu": 1,
                "alpha": 2,
                "sigmaY": 3,
                "K": 4,
                "m": 5,
                "h": 6,
            }
            # powerlaw sigma_e = sigmaY + k * eps_p^m -
            # h=linear kinematical hardening
            self.statev_label = {
                "T": 0,
                "P": 1,
                "EP": slice(2, 8),
                "X": slice(8, 14),
            }  # X=backstress
        elif umat_name == "EPCHA":
            self.n_statev = 33
            self.props_label = {
                "E": 0,
                "nu": 1,
                "alpha": 2,
                "sigmaY": 3,
                "Q": 4,
                "b": 5,
                "C_1": 6,
                "D_1": 7,
                "C_2": 8,
                "D_2": 9,
            }
            self.statev_label = {
                "T": 0,
                "P": 1,
                "EP": slice(2, 8),
                "a1": slice(8, 14),
                "a2": slice(14, 20),
                "X1": slice(20, 26),
                "X2": slice(26, 32),
                "Hp": 33,
            }
        elif umat_name == "EPHIL":
            self.n_statev = 8
            self.props_label = {
                "E": 0,
                "nu": 1,
                "alpha": 2,
                "sigmaY": 3,
                "K": 4,
                "m": 5,
                "F_hill": 6,
                "G_hill": 7,
                "H_hill": 8,
                "L_hill": 9,
                "M_hill": 10,
                "N_hill": 11,
            }
            self.statev_label = {"T": 0, "P": 1, "EP": slice(2, 8)}
        elif umat_name == "EPHAC":
            self.n_statev = 33
            self.props_label = {
                "E": 0,
                "nu": 1,
                "G": 2,
                "alpha": 3,
                "sigmaY": 4,
                "Q": 5,
                "b": 6,
                "C_1": 7,
                "D_1": 8,
                "C_2": 9,
                "D_2": 10,
                "F_hill": 11,
                "G_hill": 12,
                "H_hill": 13,
                "L_hill": 14,
                "M_hill": 15,
                "N_hill": 16,
            }
            self.statev_label = {
                "T": 0,
                "P": 1,
                "EP": slice(2, 8),
                "a1": slice(8, 14),
                "a2": slice(14, 20),
                "X1": slice(20, 26),
                "X2": slice(26, 32),
                "Hp": 33,
            }
        elif umat_name == "EPANI":
            self.n_statev = 33
            self.props_label = {
                "E": 0,
                "nu": 1,
                "G": 2,
                "alpha": 3,
                "sigmaY": 4,
                "Q": 5,
                "b": 6,
                "C_1": 7,
                "D_1": 8,
                "C_2": 9,
                "D_2": 10,
                "P11": 11,
                "P22": 12,
                "P33": 13,
                "P12": 14,
                "P13": 15,
                "P23": 16,
                "P44": 17,
                "P55": 18,
                "P66": 19,
            }
            self.statev_label = {
                "T": 0,
                "P": 1,
                "EP": slice(2, 8),
                "a1": slice(8, 14),
                "a2": slice(14, 20),
                "X1": slice(20, 26),
                "X2": slice(26, 32),
                "Hp": 33,
            }
        elif umat_name == "EPDFA":
            self.n_statev = 33
            self.props_label = {
                "E": 0,
                "nu": 1,
                "G": 2,
                "alpha": 3,
                "sigmaY": 4,
                "Q": 5,
                "b": 6,
                "F_dfa": 11,
                "G_dfa": 12,
                "H_dfa": 13,
                "L_dfa": 14,
                "M_dfa": 15,
                "N_dfa": 16,
                "K_dfa": 17,
            }
            self.statev_label = {
                "T": 0,
                "P": 1,
                "EP": slice(2, 8),
                "a1": slice(8, 14),
                "a2": slice(14, 20),
                "X1": slice(20, 26),
                "X2": slice(26, 32),
                "Hp": 33,
            }
        elif umat_name == "EPHIN":
            n_plas = self.props[0, 3]
            # should be the same for all gauss_points. If not, needs several
            # assemblies
            self.n_statev = 7 + n_plas * 7
            self.props_label = {
                "E": 0,
                "nu": 1,
                "alpha": 2,
            }
            # several plastic laws i "sigmaY":4+i*9, "k":4+i*9+1, "m":4+i*9+2,
            # "F_hill":4+i*9+3, "G_hill":4+i*9+4, "H_hill":4+i*9+5,
            # "L_hill":4+i*9+6, "M_hill":4+i*9+7, "N_hill":4+i*9+8
            self.statev_label = {
                "T": 0,
                "EP": slice(1, 7),
            }  # Pi:i*7+7, EPi:slice(i*7+8,i*7+14)
        elif umat_name == "SMAUT":
            self.n_statev = 17
            self.props_label = {
                "flagT": 0,
                "E_A": 1,
                "E_M": 2,
                "nu_A": 3,
                "nu_M": 4,
                "alphaA": 5,
                "alphaM": 6,
                "Hmin": 7,
                "Hmax": 8,
                "k1": 9,
                "sigmacrit": 10,
                "C_A": 11,
                "C_M": 12,
                "Ms0": 13,
                "Mf0": 14,
                "As0": 15,
                "Af0": 16,
                "n1": 17,
                "n2": 18,
                "n3": 19,
                "n4": 20,
                "sigmacaliber": 21,
                "b_prager": 22,
                "n_prager": 23,
                "c_lambda": 24,
                "p0_lambda": 25,
                "n_lambda": 26,
                "alpha_lambda": 27,
            }
            self.statev_label = {
                "T_init": 0,
                "xi": 1,
                "ET": slice(2, 8),
                "xi_F": 8,
                "xi_R": 9,
                "rhoDs0": 10,
                "rhoDE0": 11,
                "D": 12,
                "a1": 13,
                "a2": 14,
                "a3": 15,
                "Y0t": 16,
            }
        elif umat_name == "LLDM0":
            self.n_statev = 10
            self.props_label = {
                "axis": 0,
                "EL": 1,
                "ET": 2,
                "nuTL": 3,
                "nuTT": 4,
                "GLT": 5,
                "alphaL": 6,
                "alphaT": 7,
            }
            self.statev_label = {
                "T": 0,
                "d_22": 1,
                "d_12": 2,
                "p_ts": 3,
                "EP": slice(4, 10),
            }
        elif umat_name == "ZENER":
            self.n_statev = 8
            self.props_label = {
                "E0": 0,
                "nu0": 1,
                "alpha": 2,
                "E1": 3,
                "nu1": 4,
                "etaB1": 5,
                "etaS1": 6,
            }
            self.statev_label = {"T": 0, "v": 1, "EV": slice(2, 8)}
        elif umat_name == "ZENNK":
            n_kelvin = self.props[0, 3]
            # should be the same for all gauss_points. If not, needs several
            # assemblies
            self.n_statev = 7 + 7 * n_kelvin
            self.props_label = {
                "E0": 0,
                "nu0": 1,
                "alpha": 2,
            }  # Ei":4+i*4,"nui":5+i*4,"etaBi":6+i*4,"etaSi":7+i*4
            self.statev_label = {
                "T": 0,
                "EV": slice(1, 7),
            }  # vi: i*7+7, EVi: slice(i*7+8,i*7+14)
        elif umat_name == "PRONK":
            n_prony = self.props[0, 3]
            # should be the same for all gauss_points. If not, needs several
            # assemblies
            self.n_statev = 7 + 7 * n_prony
            self.props_label = {
                "E0": 0,
                "nu0": 1,
                "alpha": 2,
            }  # Ei":4+i*4,"nui":5+i*4,"etaBi":6+i*4,"etaSi":7+i*4
            self.statev_label = {
                "T": 0,
                "EV_tilde": slice(1, 7),
            }  # vi: i*7+7, EVi: slice(i*7+8,i*7+14)
        elif umat_name == "SMAMO":
            nvariants = self.props[0, 7]
            # should be the same for all gauss_points. If not, needs several
            # assemblies
            self.n_statev = nvariants + 8
            self.props_label = {}
            self.statev_label = {}
        elif umat_name == "SMAMC":
            nvariants = self.props[0, 8]
            # should be the same for all gauss_points. If not, needs several
            # assemblies
            self.n_statev = nvariants + 8
            self.props_label = {}
            self.statev_label = {}
        elif umat_name == "NEOHC":
            self.n_statev = 1
            self.props_label = {
                "mu": 0,
                "kappa": 1,
            }
            self.statev_label = {"T": 0}
            self._Lt_from_F = True
        elif umat_name == "MOORI":
            self.n_statev = 1
            self.props_label = {
                "C_10": 0,
                "C_01": 1,
                "kappa": 2,
            }
            self.statev_label = {"T": 0}
            self._Lt_from_F = True
        elif umat_name == "YEOHH":
            self.n_statev = 1
            self.props_label = {
                "C_10": 0,
                "C_20": 1,
                "C_30": 2,
                "kappa": 3,
            }
            self.statev_label = {"T": 0}
            self._Lt_from_F = True
        elif umat_name == "ISHAH":
            self.n_statev = 1
            self.props_label = {
                "C_10": 0,
                "C_20": 1,
                "C_01": 2,
                "kappa": 3,
            }
            self.statev_label = {"T": 0}
            self._Lt_from_F = True
        elif umat_name == "GETHH":
            self.n_statev = 1
            self.props_label = {
                "C_1": 0,
                "C_2": 1,
                "kappa": 2,
            }
            self.statev_label = {"T": 0}
            self._Lt_from_F = True
        elif umat_name == "SWANH":
            self.n_statev = 1
            self.props_label = {
                "N_Swanson": 0,
                "kappa": 1,
                # Nb of Swanson parameters : 2+i*4, i being the number of
                # Swanson modes
                # (A, B, alpha, beta) are vectors of size N_Swanson
            }
            self.statev_label = {"T": 0}
            self._Lt_from_F = True
        else:
            raise ValueError("Invalid umat_name: Expected a valid 5 char string.")

    def initialize(self, assembly, pb):
        if "Statev" not in assembly.sv:
            # initialize data with the right shapes
            assembly.sv["Statev"] = np.zeros(
                (self.n_statev, assembly.n_gauss_points), order="F"
            )  # initialize all statev to 0

            # initialize all DR to np.eye(3)
            DR = np.empty((3, 3, assembly.n_gauss_points), order="F")
            DR[...] = np.eye(3).reshape(3, 3, 1)
            assembly.sv["DR"] = DR

            if assembly._nlgeom:
                F = np.empty((3, 3, assembly.n_gauss_points), order="F")
                F[...] = np.eye(3).reshape(3, 3, 1)
                assembly.sv["F"] = F
            else:
                F = np.array([])

            if "Temp" in assembly.sv:
                temp = assembly.sv["Temp"]
            else:
                temp = None

            assembly.sv["Wm"] = np.zeros((4, assembly.n_gauss_points), order="F")
            # assembly.sv["Stress"] = StressTensorList(
            #     np.zeros((6, assembly.n_gauss_points), order="F")
            # )

            # Launch the UMAT to compute the elastic matrix in "TangentMatrix"
            zeros_6 = np.zeros((6, assembly.n_gauss_points), order="F")

            if assembly.space.get_dimension() == "2Dstress":
                if self._Lt_from_F:
                    raise NotImplementedError(
                        "Simcoon hyperelastic law are not compatible with "
                        "2D plane stress assumption"
                    )
                ndi = 2
            else:
                ndi = 3

            (sigma, statev, wm, assembly.sv["TangentMatrix"]) = sim.umat(
                self.umat_name,
                zeros_6,
                zeros_6,
                F,
                F,
                zeros_6,
                DR,
                self.props,
                assembly.sv["Statev"],
                0,
                0,
                assembly.sv["Wm"],
                temp,
                ndi=ndi,
            )

            if ndi == 2:  # plane stress assumption
                assembly.sv["TangentMatrix"] = self.get_tangent_matrix(
                    assembly, "2Dstress"
                )

            if self.use_elastic_lt:
                assembly.sv["ElasticMatrix"] = assembly.sv["TangentMatrix"]

    def update(self, assembly, pb):
        if "DStrain" in assembly.sv:
            de = assembly.sv["DStrain"]
        else:
            de = assembly.sv["Strain"] - assembly.sv_start["Strain"]

        if "Temp" in assembly.sv:
            temp = assembly.sv["Temp"]
        else:
            temp = None

        if assembly._nlgeom:
            F0 = assembly.sv_start["F"]
            F1 = assembly.sv["F"]
        else:
            F0 = F1 = np.array([])

        if assembly.space.get_dimension() == "2Dstress":
            ndi = 2
        else:
            ndi = 3

        (
            stress,
            assembly.sv["Statev"],
            assembly.sv["Wm"],
            assembly.sv["TangentMatrix"],
        ) = sim.umat(
            self.umat_name,
            assembly.sv_start["Strain"].array,
            de.array,
            F0,
            F1,
            assembly.sv_start["Stress"].array,
            assembly.sv["DR"],
            self.props,
            assembly.sv_start["Statev"],
            pb.time,
            pb.dtime,
            assembly.sv_start["Wm"],
            temp,
            ndi=ndi,
        )

        if ndi == 2:  # plane stress assumption
            assembly.sv["TangentMatrix"] = self.get_tangent_matrix(assembly, "2Dstress")

        assembly.sv["Stress"] = StressTensorList(stress)
        # to check the symetriy of the tangentmatrix :
        # print(
        #     np.abs(
        #         assembly.sv["TangentMatrix"]
        #         - assembly.sv["TangentMatrix"].transpose((1, 0, 2))
        #     ).max()
        # )

    def set_start(self, assembly, pb):
        if self.use_elastic_lt:
            assembly.sv["TangentMatrix"] = assembly.sv["ElasticMatrix"]

    def get_tangent_matrix(self, assembly, dimension=None):
        if dimension is None:
            dimension = assembly.space.get_dimension()

        H = self.local2global_H(assembly.sv["TangentMatrix"])
        if dimension == "2Dstress":
            return self.get_H_plane_stress(H)
        else:
            return H

    # def get_elastic_matrix(self, dimension = "3D"):
    #     return self.get_tangent_matrix(None,dimension)
