"""FE² constitutive law."""

import numpy as np

from fedoo.constraint.periodic_bc import PeriodicBC
from fedoo.core.assembly import Assembly
from fedoo.core.mechanical3d import Mechanical3D
from fedoo.homogen.tangent_stiffness import (
    get_homogenized_stiffness,
    get_tangent_stiffness,
)
from fedoo.problem.non_linear import NonLinear
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList


def _leaf_assemblies(assembly):
    """Return the assemblies that hold state variables under an assembly."""
    if hasattr(assembly, "list_assembly"):
        return [
            leaf for item in assembly.list_assembly for leaf in _leaf_assemblies(item)
        ]
    return [assembly]


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

    Attributes
    ----------
    tol_nr : float, default=0.05
        Tolerance of the Newton-Raphson algorithm of the microscopic
        problems. The macroscopic stress and tangent carry an error of about
        a hundredth of this value: take it at least ten times smaller than
        the tolerance of the macroscopic problem.
    use_elastic_lt : bool, default=True
        If True, the elastic homogenized matrix is used for the prediction of
        each macroscopic increment.

    Notes
    -----
    The homogenized tangent is the exact derivative of the macroscopic stress
    when the tangent of the microscopic laws is the algorithmic one (for a
    :py:class:`fedoo.constitutivelaw.Simcoon` law, ``tangent_mode=2``). With a
    continuum tangent the macroscopic Newton-Raphson iterations still
    converge, at a linear rate.

    If a microscopic problem does not converge, its increment is cut (by 4,
    down to 1e-6 of the macroscopic increment), then a ``RuntimeError`` is
    raised. The cut is a fallback: the tangent then linearizes the last
    sub-increment only, and the macroscopic stress is not a smooth function
    of the strain where the cut switches on.

    The outputs added to a microscopic problem are written once per converged
    macroscopic increment.
    """

    def __init__(self, assemb, name=""):
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
        self.tol_nr = 0.05

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
        # the trial states are dropped by the next update
        self._has_trial = False

    def _save_state(self, index):
        """Return the committed state of a microscopic problem."""
        micro_problem = self.list_problem[index]
        disp = micro_problem._U
        if not np.isscalar(disp):
            disp = disp.copy()
        leaves = _leaf_assemblies(self.list_assembly[index])
        return disp, [dict(getattr(leaf, "sv_start", {})) for leaf in leaves]

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
        for leaf, sv in zip(_leaf_assemblies(self.list_assembly[index]), list_sv):
            if hasattr(leaf, "sv_start"):
                leaf.sv_start = dict(sv)
        micro_problem.to_start()

    def _solve_micro_increment(self, index, pb_macro):
        """Load a microscopic problem from its committed state to the trial.

        The increment is cut when the Newton-Raphson algorithm fails. The
        sub-increments are committed in the microscopic problem (the state at
        the last macroscopic commit is kept apart), the last one is not.
        """
        micro_problem = self.list_problem[index]
        span = pb_macro.dtime if pb_macro.dtime > 0 else 1.0
        micro_problem.t0 = pb_macro.time
        micro_problem.tmax = pb_macro.time + span
        done, step = 0.0, 1.0
        while done < 1 - 1e-8:
            step = min(step, 1 - done)
            micro_problem.time = micro_problem.t0 + done * span
            micro_problem.dtime = step * span
            converged = micro_problem.solve_time_increment(tol_nr=self.tol_nr)[0]
            if converged:
                done += step
                if done < 1 - 1e-8:
                    micro_problem.set_start()
                continue
            micro_problem.to_start()
            step *= 0.25
            if step < 1e-6:
                raise RuntimeError(
                    "FE2: the microscopic problem of the integration point "
                    f"{index} has not converged."
                )

    def _update_pb(self, index, assembly_macro, pb_macro):
        strain = assembly_macro.sv["Strain"].asarray()[:, index]
        strain_start = assembly_macro.sv_start["Strain"].asarray()[:, index]
        micro_problem = self.list_problem[index]

        print("\r", index + 1, "/", len(self.list_problem), end="")
        self._restore_state(index)
        if self._strain_bc[index] is not None:
            micro_problem.bc.remove(self._strain_bc[index])
        self._strain_bc[index] = micro_problem.bc.add(
            "Dirichlet", "MeanStrain", strain, start_value=strain_start
        )
        micro_problem.init_bc_start_value()
        if not np.array_equal(strain, strain_start):
            self._solve_micro_increment(index, pb_macro)

        # tangent matrix of the cell at its converged state
        micro_problem.update(compute="matrix", updateWeakForm=False)
        micro_problem._update_a()
        assembly_macro.sv["TangentMatrix"][:, :, index] = get_tangent_stiffness(
            micro_problem.name
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
