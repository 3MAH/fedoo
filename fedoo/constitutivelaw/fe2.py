# derive de ConstitutiveLaw
# This law should be used with an StressEquilibrium WeakForm

from fedoo.core.mechanical3d import Mechanical3D
from fedoo.weakform.stress_equilibrium import StressEquilibrium
from fedoo.core.assembly import Assembly
from fedoo.problem.non_linear import NonLinear
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList
from fedoo.constraint.periodic_bc import (
    PeriodicBC,
)  # , DefinePeriodicBoundaryConditionNonPerioMesh
from fedoo.homogen.tangent_stiffness import (
    get_tangent_stiffness,
    get_homogenized_stiffness,
)
import numpy as np
import multiprocessing


class FE2(Mechanical3D):
    """
    ConstitutiveLaw that solve a Finite Element Problem at each point of gauss
    in the contexte of the so called "FE²" method.

    Parameters
    ----------
    assemb: Assembly or Assembly name (str), or list of Assembly (with len(list) = number of integration points).
        Assembly that correspond to the microscopic problem
    name: str, optional
        The name of the constitutive law
    """

    def __init__(self, assemb, name=""):
        # props is a nparray containing all the material variables
        # nstatev is a nparray containing all the material variables
        if isinstance(assemb, str):
            assemb = Assembly.get_all()[assemb]
        Mechanical3D.__init__(self, name)  # heritage

        if isinstance(assemb, list):
            self.__assembly = [
                Assembly.get_all()[a] if isinstance(a, str) else a for a in assemb
            ]
            self.__mesh = [a.mesh for a in self.__assembly]
        else:
            self.__mesh = assemb.mesh
            self.__assembly = assemb

        self.list_problem = None

        self.use_elastic_lt = True  # option to use the elastic tangeant matrix (in principle = initial tangent matrix) at the begining of each time step

        # self.__currentGradDisp = self.__initialGradDisp = 0

    # def get_pk2(self):
    #     return StressTensorList(self.__stress)

    # # def get_kirchhoff(self):
    # #     return StressTensorList(self.Kirchhoff.T)

    # # def get_cauchy(self):
    # #     return StressTensorList(self.Cauchy.T)

    # def get_strain(self, **kargs):
    #     return StrainTensorList(self.__strain)

    # # def get_statev(self):
    # #     return self.statev.T

    # def get_stress(self, **kargs): #same as GetPKII (used for small def)
    #     return StressTensorList(self.__stress)

    # # def GetHelas (self):
    # #     # if self.__L is None:
    # #     #     self.RunUmat(np.eye(3).T.reshape(1,3,3), np.eye(3).T.reshape(1,3,3), time=0., dtime=1.)

    # #     return np.squeeze(self.L.transpose(1,2,0))

    # def get_wm(self):
    #     return self.__Wm

    # def get_disp_grad(self):
    #     if np.isscalar(self.__currentGradDisp) and self.__currentGradDisp == 0: return 0
    #     else: return self.__currentGradDisp

    # def get_tangent_matrix(self):

    #     H = np.squeeze(self.Lt.transpose(1,2,0))
    #     return H

    # def NewTimeIncrement(self):
    #     # self.set_start() #in set_start -> set tangeant matrix to elastic

    #     #save variable at the begining of the Time increment
    #     self.__initialGradDisp = self.__currentGradDisp
    #     self.Lt = self.L.copy()

    # def to_start(self):
    #     # self.to_start()
    #     self.__currentGradDisp = self.__initialGradDisp
    #     self.Lt = self.L.copy()

    # def reset(self):
    #     """
    #     reset the constitutive law (time history)
    #     """
    #     #a modifier
    #     self.__currentGradDisp = self.__initialGradDisp = 0
    #     # self.__Statev = None
    #     self.__currentStress = None #lissStressTensor object describing the last computed stress (GetStress method)
    #     # self.__currentGradDisp = 0
    #     # self.__F0 = None

    def initialize(self, assembly, pb):
        if self.list_problem is None:  # only initialize once
            nb_points = assembly.n_gauss_points

            # Definition of the set of nodes for boundary conditions
            if not (isinstance(self.__mesh, list)):
                self.list_mesh = [self.__mesh for i in range(nb_points)]
                self.list_assembly = [self.__assembly.copy() for i in range(nb_points)]
            else:
                self.list_mesh = self.__mesh
                self.list_assembly = self.__assembly

            self.list_problem = []
            self._list_volume = np.empty(nb_points)
            self._list_center = np.empty(nb_points, dtype=int)
            # self.L = np.empty((nb_points,6,6))
            assembly.sv["TangentMatrix"] = np.empty((6, 6, nb_points))

            print("-- Initialize micro problems --")
            for i in range(nb_points):
                print("\r", str(i + 1), "/", str(nb_points), end="")
                crd = self.list_mesh[i].nodes
                type_el = self.list_mesh[i].elm_type
                xmax = np.max(crd[:, 0])
                xmin = np.min(crd[:, 0])
                ymax = np.max(crd[:, 1])
                ymin = np.min(crd[:, 1])
                zmax = np.max(crd[:, 2])
                zmin = np.min(crd[:, 2])

                crd_center = (
                    np.array([xmin, ymin, zmin]) + np.array([xmax, ymax, zmax])
                ) / 2
                self._list_volume[i] = (
                    (xmax - xmin) * (ymax - ymin) * (zmax - zmin)
                )  # total volume of the domain

                if "_StrainNodes" in self.list_mesh[i].node_sets:
                    strain_nodes = self.list_mesh[i].node_sets["_StrainNodes"]
                else:
                    strain_nodes = self.list_mesh[i].add_nodes(
                        crd_center, 2
                    )  # add virtual nodes for macro strain
                    self.list_mesh[i].add_node_set(strain_nodes, "_StrainNodes")

                self._list_center[i] = np.linalg.norm(
                    crd[:-2] - crd_center, axis=1
                ).argmin()
                # list_material.append(self.__constitutivelaw.copy())

                # Type of problem
                self.list_problem.append(
                    NonLinear(self.list_assembly[i], name="_fe2_cell_" + str(i))
                )
                pb_micro = self.list_problem[-1]
                meshperio = True

                # Shall add other conditions later on
                pb_micro.bc.add(
                    PeriodicBC(
                        [
                            strain_nodes[0],
                            strain_nodes[0],
                            strain_nodes[0],
                            strain_nodes[1],
                            strain_nodes[1],
                            strain_nodes[1],
                        ],
                        ["DispX", "DispY", "DispZ", "DispX", "DispY", "DispZ"],
                        dim=3,
                        meshperio=meshperio,
                        name="_fe2_cell_" + str(i),
                    )
                )

                pb_micro.bc.add("Dirichlet", [self._list_center[i]], "Disp", 0)
                # self.list_assembly[i].initialize()
                assembly.sv["TangentMatrix"][:, :, i] = get_homogenized_stiffness(
                    self.list_assembly[i]
                )

            pb.make_active()
            if self.use_elastic_lt:
                assembly.sv["ElasticMatrix"] = assembly.sv["TangentMatrix"].copy()

            assembly.sv["Strain"] = StrainTensorList(np.zeros((6, nb_points)))
            assembly.sv["Stress"] = StressTensorList(np.zeros((6, nb_points)))
            assembly.sv["Wm"] = np.zeros((4, nb_points))

            print("")

    def set_start(self, assembly, pb):
        if self.use_elastic_lt:
            assembly.sv["TangentMatrix"] = assembly.sv["ElasticMatrix"]

    def _update_pb(self, id_pb, assembly_macro, pb_macro):
        strain = assembly_macro.sv["Strain"]
        strain_start = assembly_macro.sv_start["Strain"]
        nb_points = len(self.list_problem)
        pb = self.list_problem[id_pb]

        print("\r", str(id_pb + 1), "/", str(nb_points), end="")
        strain_nodes = self.list_mesh[id_pb].node_sets["_StrainNodes"]

        pb.bc.remove("Strain")
        pb.bc.add(
            "Dirichlet",
            [strain_nodes[0]],
            "DispX",
            strain[0][id_pb],
            start_value=strain_start[0][id_pb],
            name="Strain",
        )  # EpsXX
        pb.bc.add(
            "Dirichlet",
            [strain_nodes[0]],
            "DispY",
            strain[1][id_pb],
            start_value=strain_start[1][id_pb],
            name="Strain",
        )  # EpsYY
        pb.bc.add(
            "Dirichlet",
            [strain_nodes[0]],
            "DispZ",
            strain[2][id_pb],
            start_value=strain_start[2][id_pb],
            name="Strain",
        )  # EpsZZ
        pb.bc.add(
            "Dirichlet",
            [strain_nodes[1]],
            "DispX",
            strain[3][id_pb],
            start_value=strain_start[3][id_pb],
            name="Strain",
        )  # EpsXY
        pb.bc.add(
            "Dirichlet",
            [strain_nodes[1]],
            "DispY",
            strain[4][id_pb],
            start_value=strain_start[4][id_pb],
            name="Strain",
        )  # EpsXZ
        pb.bc.add(
            "Dirichlet",
            [strain_nodes[1]],
            "DispZ",
            strain[5][id_pb],
            start_value=strain_start[5][id_pb],
            name="Strain",
        )  # EpsYZ

        pb.nlsolve(
            dt=pb_macro.dtime,
            tmax=pb_macro.dtime,
            update_dt=True,
            tol_nr=0.05,
            print_info=0,
        )

        assembly_macro.sv["TangentMatrix"][:, :, id_pb] = get_tangent_stiffness(pb.name)

        stress_field = self.list_assembly[id_pb].sv["Stress"]  # computed micro stress
        # integrate micro stress to get the macro one
        assembly_macro.sv["Stress"].asarray()[:, id_pb] = np.array(
            [
                1
                / self._list_volume[id_pb]
                * self.list_assembly[id_pb].integrate_field(stress_field[i])
                for i in range(6)
            ]
        )

        Wm_field = self.list_assembly[id_pb].sv["Wm"]
        assembly_macro.sv["Wm"][:, id_pb] = (
            1 / self._list_volume[id_pb]
        ) * self.list_assembly[id_pb].integrate_field(Wm_field)

    def update(self, assembly, pb):
        displacement = pb.get_dof_solution()

        # resolution of the micro problem at each gauss points
        nb_points = len(self.list_problem)

        print("-- Update micro cells --")

        # with multiprocessing.Pool(4) as pool:
        #     pool.map(self._update_pb, range(nb_points))

        for id_pb in range(nb_points):
            self._update_pb(id_pb, assembly, pb)

        print("")
