"""FE² constitutive law."""

import numpy as np

from fedoo.constraint.periodic_bc import PeriodicBC
from fedoo.core.assembly import Assembly
from fedoo.core.mechanical3d import Mechanical3D
from fedoo.homogen.tangent_stiffness import (
    get_homogenized_stiffness,
    get_tangent_stiffness,
)
from fedoo.problem.non_linear import NonLinear, NonlinearConvergenceError
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList


class FE2(Mechanical3D):
    """FE² constitutive law based on periodic microscopic problems.

    A microscopic finite-element problem is solved at each macroscopic
    integration point. The macroscopic strain is prescribed through the
    problem-level ``MeanStrain`` degrees of freedom created by
    :class:`PeriodicBC`.

    At each update of the macroscopic problem, every microscopic problem
    starts again from its state at the beginning of the macroscopic
    increment and is loaded up to the current macroscopic strain. The
    microscopic states are committed only when the macroscopic increment has
    converged. The homogenized tangent matrix is computed at the converged
    microscopic state.

    Parameters
    ----------
    assemb : Assembly, str, or list of Assembly
        Microscopic assembly, its registered name, or one microscopic
        assembly per macroscopic integration point.
    name : str, optional
        Name of the constitutive law.
    micro_nr_options : mapping, optional
        Options forwarded to each microscopic problem's
        :meth:`~fedoo.problem.non_linear.NonLinear.set_nr_criterion`. FE²
        defaults to ``tol=0.05`` and ``dt_increase_niter=0`` (once a substep
        is cut, it is kept at the smaller size).

    Attributes
    ----------
    use_elastic_lt : bool, default=True
        If True, the elastic homogenized matrix is used for the prediction of
        each macroscopic increment.

    Notes
    -----
    With an algorithmic microscopic tangent and a single substep, the
    homogenized tangent linearizes the macroscopic stress response. If a
    microscopic problem requires several substeps, the tangent linearizes
    only the final substep. A cell that cannot converge above the minimum
    substep size (1e-6 of the macro increment) rejects the macro trial.

    The outputs added to a microscopic problem are written once per converged
    macroscopic increment.
    """

    def __init__(self, assemb, name="", *, micro_nr_options=None):
        if isinstance(assemb, str):
            assemb = Assembly.get_all()[assemb]
        super().__init__(name=name)

        if isinstance(assemb, list):
            self.__assembly = [
                Assembly.get_all()[item] if isinstance(item, str) else item
                for item in assemb
            ]
            self.__mesh = [item.mesh for item in self.__assembly]
        else:
            self.__assembly = assemb
            self.__mesh = assemb.mesh

        self.list_problem = None
        self.use_elastic_lt = True
        self._micro_nr_options = {"tol": 0.05, "dt_increase_niter": 0}
        if micro_nr_options is not None:
            self._micro_nr_options.update(micro_nr_options)

    def initialize(self, assembly, pb):
        if self.list_problem is not None:
            return

        nb_points = assembly.n_gauss_points
        if isinstance(self.__mesh, list):
            if len(self.__assembly) != nb_points:
                raise ValueError(
                    "A list of microscopic assemblies must contain one "
                    "assembly per macroscopic integration point."
                )
            self.list_mesh = self.__mesh
            self.list_assembly = self.__assembly
        else:
            self.list_mesh = [self.__mesh for _ in range(nb_points)]
            self.list_assembly = [self.__assembly.copy() for _ in range(nb_points)]

        self.list_problem = []
        self._strain_bc = [None] * nb_points  # strain condition of each cell
        self._committed = [None] * nb_points  # state at the last macro commit
        self._has_trial = False  # True if the cells hold an uncommitted trial
        self._list_volume = np.empty(nb_points)
        assembly.sv["TangentMatrix"] = np.empty((6, 6, nb_points))

        print("-- Initialize micro problems --")
        for index in range(nb_points):
            print("\r", index + 1, "/", nb_points, end="")
            coordinates = self.list_mesh[index].nodes
            lower = np.min(coordinates, axis=0)
            upper = np.max(coordinates, axis=0)
            box_center = (lower + upper) / 2
            self._list_volume[index] = np.prod(upper - lower)
            center_node = np.linalg.norm(coordinates - box_center, axis=1).argmin()

            micro_problem = NonLinear(
                self.list_assembly[index],
                name=f"_fe2_cell_{index}",
            )
            micro_problem.set_nr_criterion(**self._micro_nr_options)
            self.list_problem.append(micro_problem)
            micro_problem.bc.add(
                PeriodicBC(
                    "small_strain",
                    dim=3,
                    name=f"_fe2_cell_{index}",
                )
            )
            micro_problem.bc.add("Dirichlet", [center_node], "Disp", 0)
            assembly.sv["TangentMatrix"][:, :, index] = get_homogenized_stiffness(
                self.list_assembly[index]
            )

        pb.make_active()
        if self.use_elastic_lt:
            assembly.sv["ElasticMatrix"] = assembly.sv["TangentMatrix"].copy()

        assembly.sv["Strain"] = StrainTensorList(np.zeros((6, nb_points)))
        assembly.sv["Stress"] = StressTensorList(np.zeros((6, nb_points)))
        assembly.sv["Wm"] = np.zeros((4, nb_points))
        print("")

    def set_start(self, assembly, pb):
        if self._has_trial:
            # the macroscopic increment has converged: commit the cells
            for index, micro_problem in enumerate(self.list_problem):
                micro_problem.set_start(save_results=True)
                self._committed[index] = self._save_state(index)
            self._has_trial = False
        if self.use_elastic_lt:
            assembly.sv["TangentMatrix"] = assembly.sv["ElasticMatrix"].copy()

    def to_start(self, assembly, pb):
        self._has_trial = False
        for index, micro_problem in enumerate(self.list_problem):
            if self._committed[index] is not None:
                micro_problem.time = pb.time
                self._restore_state(index)

    def _save_state(self, index):
        """Return the committed state of a microscopic problem."""
        micro_problem = self.list_problem[index]
        disp = micro_problem._U
        if not np.isscalar(disp):
            disp = disp.copy()
        return disp, [
            leaf.sv_start.copy() if hasattr(leaf, "sv_start") else None
            for leaf in micro_problem.assembly.iter_leaf()
        ]

    def _restore_state(self, index):
        """Bring a microscopic problem back to its last committed state."""
        micro_problem = self.list_problem[index]
        if not micro_problem._initialized:
            micro_problem.initialize()
            micro_problem.set_start()
            self._committed[index] = self._save_state(index)
            return
        disp, list_sv = self._committed[index]
        micro_problem._U = disp if np.isscalar(disp) else disp.copy()
        for leaf, sv in zip(micro_problem.assembly.iter_leaf(), list_sv):
            if sv is not None:
                leaf.sv_start = sv.copy()
        micro_problem.to_start()

    def _update_pb(self, index, assembly_macro, pb_macro):
        strain = assembly_macro.sv["Strain"].asarray()[:, index]
        strain_start = assembly_macro.sv_start["Strain"].asarray()[:, index]
        micro_problem = self.list_problem[index]

        print("\r", index + 1, "/", len(self.list_problem), end="")
        span = pb_macro.dtime if pb_macro.dtime > 0 else 1.0
        micro_problem.time = pb_macro.time
        micro_problem.dtime = span
        self._restore_state(index)
        if self._strain_bc[index] is not None:
            micro_problem.bc.remove(self._strain_bc[index])
        self._strain_bc[index] = micro_problem.bc.add(
            "Dirichlet", "MeanStrain", strain, start_value=strain_start
        )
        try:
            micro_problem.nlsolve(
                t0=pb_macro.time,
                tmax=pb_macro.time + span,
                dt=span,
                dt_min=span * 1e-6,
                print_info=0,
                _trial=True,
            )
        except NonlinearConvergenceError as error:
            raise NonlinearConvergenceError(
                f"FE2: microscopic problem at integration point {index} did not converge."
            ) from error

        # tangent matrix of the cell at its converged state
        micro_problem.update(compute="matrix", updateWeakForm=False)
        micro_problem._update_a()
        assembly_macro.sv["TangentMatrix"][:, :, index] = get_tangent_stiffness(
            micro_problem
        )

        micro_assembly = self.list_assembly[index]
        stress_field = micro_assembly.sv["Stress"]
        assembly_macro.sv["Stress"].asarray()[:, index] = np.array(
            [
                micro_assembly.integrate_field(component) / self._list_volume[index]
                for component in stress_field
            ]
        )

        energy_field = micro_assembly.sv.get("Wm")
        if energy_field is not None:
            assembly_macro.sv["Wm"][:, index] = (
                micro_assembly.integrate_field(energy_field) / self._list_volume[index]
            )

    def update(self, assembly, pb):
        print("-- Update micro cells --")
        self._has_trial = False  # stays False if a cell fails below
        # new arrays: the start state of the increment shares the former ones
        assembly.sv["Stress"] = StressTensorList(assembly.sv["Stress"].asarray().copy())
        assembly.sv["TangentMatrix"] = assembly.sv["TangentMatrix"].copy()
        assembly.sv["Wm"] = assembly.sv["Wm"].copy()
        for index in range(len(self.list_problem)):
            self._update_pb(index, assembly, pb)
        self._has_trial = True
        print("")
