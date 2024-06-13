# derive de ConstitutiveLaw
# compatible with the simcoon strain and stress notation

from fedoo.core.mechanical3d import Mechanical3D
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList

try:
    from simcoon import simmit as sim

    USE_SIMCOON = True
except ImportError:
    USE_SIMCOON = False

import numpy as np


class Simcoon(Mechanical3D):
    # """
    # Linear full Anistropic constitutive law defined from the rigidity matrix H.

    # The constitutive Law should be associated with :mod:`fedoo.weakform.InternalForce`

    # Parameters
    # ----------
    # H : list of list or an array (shape=(6,6)) of scalars or arrays of gauss point values.
    #     The rigidity matrix.
    #     If H is a list of gauss point values, the shape shoud be H.shape = (6,6,NumberOfGaussPoints)
    # name : str, optional
    #     The name of the constitutive law
    # """

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

        self.use_elastic_lt = True  # option to use the elastic tangeant matrix (in principle = initial tangent matrix) at the begining of each time step

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
                "k": 4,
                "m": 5,
                "h": 6,
            }  # powerlaw sigma_e = sigmaY + k * eps_p^m - #h=linear kinematical hardening
            self.statev_label = {
                "T": 0,
                "P": 1,
                "EP": slice(2, 8),
                "X": slice(8, 14),
            }  # X=backstress
        elif umat_name == "EPCHA":
            self.n_statev = 32
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
            }
        elif umat_name == "SMAUT":
            self.n_statev = 16
            self.props_label = {}
            self.statev_label = {}
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
            n_kelvin = self.props[
                0, 3
            ]  # should be the same for all gauss_points. If not, needs several assemblies
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
            n_prony = self.props[
                0, 3
            ]  # should be the same for all gauss_points. If not, needs several assemblies
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
        elif umat_name == "EPHIC":
            self.n_statev = 8
            self.props_label = {
                "E": 0,
                "nu": 1,
                "alpha": 2,
                "sigmaY": 3,
                "k": 4,
                "m": 5,
                "F_hill": 6,
                "G_hill": 7,
                "H_hill": 8,
                "L_hill": 9,
                "M_hill": 10,
                "N_hill": 11,
            }
            self.statev_label = {"T": 0, "P": 1, "EP": slice(2, 8)}
        elif umat_name == "EPHIN":
            n_plas = self.props[
                0, 3
            ]  # should be the same for all gauss_points. If not, needs several assemblies
            self.n_statev = 7 + n_plas * 7
            self.props_label = {
                "E": 0,
                "nu": 1,
                "alpha": 2,
            }  # several plastic laws i "sigmaY":4+i*9, "k":4+i*9+1, "m":4+i*9+2, "F_hill":4+i*9+3, "G_hill":4+i*9+4, "H_hill":4+i*9+5, "L_hill":4+i*9+6, "M_hill":4+i*9+7, "N_hill":4+i*9+8
            self.statev_label = {
                "T": 0,
                "EP": slice(1, 7),
            }  # Pi:i*7+7, EPi:slice(i*7+8,i*7+14)
        elif umat_name == "SMAMO":
            nvariants = self.props[
                0, 7
            ]  # should be the same for all gauss_points. If not, needs several assemblies
            self.n_statev = nvariants + 8
            self.props_label = {}
            self.statev_label = {}
        elif umat_name == "SMAMC":
            nvariants = self.props[
                0, 8
            ]  # should be the same for all gauss_points. If not, needs several assemblies
            self.n_statev = nvariants + 8
            self.props_label = {}
            self.statev_label = {}

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

            assembly.sv["Wm"] = np.zeros((4, assembly.n_gauss_points), order="F")
            # assembly.sv["Stress"] = StressTensorList(
            #     np.zeros((6, assembly.n_gauss_points), order="F")
            # )

            # Launch the UMAT to compute the elastic matrix in "TangentMatrix"
            if self.props.shape[1] == 1:
                zeros_6 = np.zeros(6)
                (sigma, statev, wm, assembly.sv["TangentMatrix"]) = sim.umat(
                    self.umat_name,
                    zeros_6,
                    zeros_6,
                    zeros_6,
                    np.eye(3, order="F"),
                    self.props[:, 0],
                    np.zeros(self.n_statev),
                    0,
                    0,
                    np.zeros(4),
                )
            else:
                zeros_6 = np.zeros((6, assembly.n_gauss_points), order="F")
                (sigma, statev, wm, assembly.sv["TangentMatrix"]) = sim.umat(
                    self.umat_name,
                    zeros_6,
                    zeros_6,
                    zeros_6,
                    DR,
                    self.props,
                    assembly.sv["Statev"],
                    0,
                    0,
                    assembly.sv["Wm"],
                )

            if self.use_elastic_lt:
                assembly.sv["ElasticMatrix"] = assembly.sv["TangentMatrix"]

    def update(self, assembly, pb):
        if "DStrain" in assembly.sv:
            de = assembly.sv["DStrain"]
        else:
            de = assembly.sv["Strain"] - assembly.sv_start["Strain"]

        # if 'Stress' not in assembly.sv or assembly.sv['Stress'] is 0:
        #     assembly.sv['Stress'] = StressTensorList(np.zeros((6, assembly.n_gauss_points), order='F'))

        # if assembly.sv_start['Strain'] is 0:
        #     assembly.sv_start['Strain'] = StrainTensorList(np.zeros((6, assembly.n_gauss_points), order='F'))

        if "Temp" in assembly.sv:
            temp = assembly.sv["Temp"]
        else:
            temp = None

        try:
            (
                stress,
                assembly.sv["Statev"],
                assembly.sv["Wm"],
                assembly.sv["TangentMatrix"],
            ) = sim.umat(
                self.umat_name,
                assembly.sv_start["Strain"].array,
                de.array,
                assembly.sv_start["Stress"].array,
                assembly.sv["DR"],
                self.props,
                assembly.sv_start["Statev"],
                pb.time,
                pb.dtime,
                assembly.sv_start["Wm"],
                temp,
            )
        except:  # for compatibility with old version of simcoon
            (
                stress,
                assembly.sv["Statev"],
                assembly.sv["Wm"],
                assembly.sv["TangentMatrix"],
            ) = sim.umat(
                self.umat_name,
                assembly.sv_start["Strain"].array,
                de.array,
                assembly.sv_start["Stress"].array,
                assembly.sv["DR"],
                self.props,
                assembly.sv_start["Statev"],
                pb.time,
                pb.dtime,
                assembly.sv_start["Wm"],
            )

        # work only in global local frame

        #### TEST ######
        # assembly.sv['TangentMatrix'][:,:3] = assembly.sv['TangentMatrix'][:,:3] + stress.reshape(6,1,-1)
        # F = np.transpose(assembly.sv['F'], (2,0,1))
        # J = np.linalg.det(F)
        # print(J.min())
        # assembly.sv['TangentMatrix'] = (1/J)*assembly.sv['TangentMatrix']
        # stress /= J
        # assembly.sv['TangentMatrix'] = 5*assembly.sv['TangentMatrix']
        #### END TEST #####

        assembly.sv["Stress"] = StressTensorList(stress)
        # to check the symetriy of the tangentmatrix :
        # print(np.abs(assembly.sv['TangentMatrix'] - assembly.sv['TangentMatrix'].transpose((1,0,2))).max())

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
