# doesn't seem to work. Not imported by default.
# use with hex8sri elements


from fedoo.core.weakform import WeakFormBase, WeakFormSum
from fedoo.core.base import AssemblyBase
from fedoo.weakform.stress_equilibrium import StressEquilibrium
import numpy as np


class StressEquilibriumBbar(StressEquilibrium):
    """Mechanical equilibrium equation for solids with the B-bar method.

    This method is still experimental ! Use with caution. The fbar method
    from the standard StressEquilibrium should be prefered.

    The main point to consider are:
      * This weak form is the same as the standard StressEquilibrium but
        allow to use the Bbar method.
      * This weak form can be used for solid in 3D or using a 2D plane
        assumption (plane strain or plane stress).
      * Include initial stress for non linear problems or if defined in
        the associated assembly.
      * This weak form accepts geometrical non linearities if simcoon is
        installed. (nlgeom should be in {True, 'UL', 'TL'}. In this case
        the initial displacement is also
        considered.

    Parameters
    ----------
    constitutivelaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Material Constitutive Law (:mod:`fedoo.constitutivelaw`)
    name: str
        name of the WeakForm
    nlgeom: bool, 'UL' or 'TL', optional
        If True, the geometrical non linearities are activate based on the
        updated lagrangian method. This parameters is used only in the
        context of NonLinearProblems such as
        :mod:`fedoo.problem.NonLinearStatic` or
        :mod:`fedoo.problem.NonLinearNewmark`.
        If nlgeom == 'UL' the updated lagrangian method is used (same as True).
        If nlgeom == 'TL' the total lagrangian method is used.
        If not defined, the problem.nlgeom parameter is used instead.
    space: ModelingSpace
        Modeling space associated to the weakform. If None is specified,
        the active ModelingSpace is considered.
    """

    def __init__(self, constitutivelaw, name="", nlgeom=None, space=None):
        super().__init__(constitutivelaw, name, nlgeom, space)
        self.assembly_options["elm_type", "quad4"] = "quad4sri"
        self.assembly_options["elm_type", "hex8"] = "hex8sri"

    def get_weak_equation(self, assembly, pb):
        """Get the weak equation related to the current problem state."""
        if assembly._nlgeom == "TL":  # add initial displacement effect
            if self.space.is_axisymmetric:
                raise NotImplementedError(
                    "'2Daxi' ModelingSpace is not implemented with the total "
                    "lagrangian formulation for the b-bar / sub-integration "
                    "weak form. Use updated lagrangian instead."
                )
            eps = self.space.op_strain(assembly.sv["DispGradient"])
            initial_stress = assembly.sv["PK2"]
        else:
            eps = self.space.op_strain()
            if assembly.elm_type in ["hex8sri", "quad4sri"]:
                n_normal_components = self.space.ndim
                eps_vol = sum(eps[:n_normal_components])
                eps_vol_reduced = self.space.derivative(
                    "_DispX", "X"
                ) + self.space.derivative("_DispY", "Y")
                if n_normal_components == 3:
                    eps_vol_reduced += self.space.derivative("_DispZ", "Z")

                correction = (eps_vol_reduced - eps_vol) / n_normal_components
                for i in range(n_normal_components):
                    eps[i] = eps[i] + correction

            initial_stress = assembly.sv[
                "Stress"
            ]  # Stress = Cauchy for updated lagrangian method

            if self.space.is_axisymmetric:
                rr = assembly.sv["_R_gausspoints"]

                # nlgeom = False
                eps[2] = self.space.variable("DispX") * np.divide(
                    1, rr, out=np.zeros_like(rr), where=rr != 0
                )  # put zero if X==0 (division by 0)
                # eps[2] = self.space.variable('DispX') * (1/rr)

        H = assembly.sv["TangentMatrix"]

        sigma = [
            sum([0 if eps[j] == 0 else eps[j] * H[i][j] for j in range(6)])
            for i in range(6)
        ]

        DiffOp = sum(
            [0 if eps[i] == 0 else eps[i].virtual * sigma[i] for i in range(6)]
        )

        if not np.array_equal(initial_stress, 0):
            # this term doesnt seem to improve convergence !
            if self.geometric_stiffness:
                DiffOp = DiffOp + sum(
                    [
                        0
                        if self._nl_strain_op_vir[i] == 0
                        else self._nl_strain_op_vir[i] * initial_stress[i]
                        for i in range(6)
                    ]
                )

            DiffOp = DiffOp + sum(
                [
                    0 if eps[i] == 0 else eps[i].virtual * initial_stress[i]
                    for i in range(6)
                ]
            )

        if self.space.is_axisymmetric:
            DiffOp = DiffOp * ((2 * np.pi) * rr)

        return DiffOp

    def initialize(self, assembly, pb):
        """Initialize the weakform at the begining of a problem."""
        super().initialize(assembly, pb)
        if assembly.elm_type in ["hex8sri", "quad4sri"]:
            if assembly._nlgeom == "TL":
                raise NotImplementedError(
                    f"{assembly.elm_type} not \
                                          implemented with total lagrangian \
                                          formulation"
                )
            self.space.variable_alias("_DispX", "DispX")
            self.space.variable_alias("_DispY", "DispY")
            if self.space.ndim == 3:
                self.space.variable_alias("_DispZ", "DispZ")


class StressEquilibriumFbar(StressEquilibrium):
    """Weak formulation of the mechanical equilibrium equation for solids.

    This method is still experimental ! Use with caution. The fbar method
    can be used with the the standard StressEquilibrium weak form.

    The main point to consider are:
      * This weak form is the same as the standard StressEquilibrium but
        allow to use the consistant tangent matrix with the Fbar method.
      * This weak form can be used for solid in 3D or using a 2D plane
        assumption (plane strain or plane stress).
      * Include initial stress for non linear problems or if defined in
        the associated assembly.
      * This weak form accepts geometrical non linearities if simcoon is
        installed. (nlgeom should be in {True, 'UL', 'TL'}. In this case
        the initial displacement is also
        considered.

    Parameters
    ----------
    constitutivelaw: ConstitutiveLaw name (str) or ConstitutiveLaw object
        Material Constitutive Law (:mod:`fedoo.constitutivelaw`)
    name: str
        name of the WeakForm
    nlgeom: bool, 'UL' or 'TL', optional
        If True, the geometrical non linearities are activate based on the
        updated lagrangian method. This parameters is used only in the
        context of NonLinearProblems such as
        :mod:`fedoo.problem.NonLinearStatic` or
        :mod:`fedoo.problem.NonLinearNewmark`.
        If nlgeom == 'UL' the updated lagrangian method is used (same as True).
        If nlgeom == 'TL' the total lagrangian method is used.
        If not defined, the problem.nlgeom parameter is used instead.
    space: ModelingSpace
        Modeling space associated to the weakform. If None is specified,
        the active ModelingSpace is considered.
    """

    def __init__(self, constitutivelaw, name="", nlgeom=None, space=None):
        super().__init__(constitutivelaw, name, nlgeom, space)
        if self.space.is_axisymmetric:
            raise NotImplementedError(
                "StressEquilibriumFbar is not implemented in '2Daxi': the "
                "F-bar volumetric correction requires the hoop component "
                "of the strain at the element centroid, which is not yet "
                "wired through _op_strain_center. Use StressEquilibrium "
                "(standard) or StressEquilibriumMixed in 2Daxi instead."
            )
        self.fbar = True
        self.assembly_options["elm_type", "quad4"] = "quad4sri"
        self.assembly_options["elm_type", "hex8"] = "hex8sri"

    def get_weak_equation(self, assembly, pb):
        """Get the weak equation related to the current problem state."""
        if assembly._nlgeom == "TL":  # add initial displacement effect
            if self.space.is_axisymmetric:
                raise NotImplementedError(
                    "'2Daxi' ModelingSpace is not implemented with the total "
                    "lagrangian formulation for the f-bar weak form. "
                    "Use updated lagrangian instead."
                )
            eps = self.space.op_strain(assembly.sv["DispGradient"])
            initial_stress = assembly.sv["PK2"]
        else:
            eps = self.space.op_strain()
            initial_stress = assembly.sv[
                "Stress"
            ]  # Stress = Cauchy for updated lagrangian method

            if self.space.is_axisymmetric:
                rr = assembly.sv["_R_gausspoints"]

                # nlgeom = False
                eps[2] = self.space.variable("DispX") * np.divide(
                    1, rr, out=np.zeros_like(rr), where=rr != 0
                )  # put zero if X==0 (division by 0)
                # eps[2] = self.space.variable('DispX') * (1/rr)

        H = assembly.sv["TangentMatrix"]

        sigma = [
            sum([0 if eps[j] == 0 else eps[j] * H[i][j] for j in range(6)])
            for i in range(6)
        ]

        DiffOp = sum(
            [0 if eps[i] == 0 else eps[i].virtual * sigma[i] for i in range(6)]
        )

        if not (np.isscalar(initial_stress) and initial_stress == 0):
            # this term doesnt seem to improve convergence in general !
            if self.geometric_stiffness:
                DiffOp = DiffOp + sum(
                    [
                        0
                        if self._nl_strain_op_vir[i] == 0
                        else self._nl_strain_op_vir[i] * initial_stress[i]
                        for i in range(6)
                    ]
                )

            DiffOp = DiffOp + sum(
                [
                    0 if eps[i] == 0 else eps[i].virtual * initial_stress[i]
                    for i in range(6)
                ]
            )

        #
        # Fbar additional term fro
        #
        if self.fbar:
            # ref: DESIGN OF SIMPLE LOW ORDER FINITE ELEMENTS FOR LARGE STRAIN
            # ANALYSIS OF NEARLY INCOMPRESSIBLE SOLIDS, Neto et al,
            # International Journal of Solids and Structures

            q = [
                (1 / 3.0) * (H[i][0] + H[i][1] + H[i][2])
                - (2 / 3.0) * initial_stress[i]
                for i in range(6)
            ]

            # q = np.column_stack([q, q, q, 0, 0, 0])
            eps0 = self._op_strain_center()
            # eps0 = [0,0,0] -> eps operator at the element centroid

            DiffOp = DiffOp + sum(
                [
                    (
                        0
                        if eps[i] == 0
                        else eps[i].virtual
                        * q[i]
                        * (eps0[0] - eps[0] + eps0[1] - eps[1] + eps0[2] - eps[2])
                    )
                    for i in range(6)
                ]
            )

        if self.space.is_axisymmetric:
            DiffOp = DiffOp * ((2 * np.pi) * rr)

        return DiffOp

    def _op_strain_center(self):
        du_dx = self.space.derivative("_DispX", "X")
        dv_dy = self.space.derivative("_DispY", "Y")
        du_dy = self.space.derivative("_DispX", "Y")
        dv_dx = self.space.derivative("_DispY", "X")

        if self.space.ndim == 2:
            eps = [du_dx, dv_dy, 0, du_dy + dv_dx, 0, 0]

        else:  # assume ndim == 3
            dw_dz = self.space.derivative("_DispZ", "Z")
            du_dz = self.space.derivative("_DispX", "Z")
            dv_dz = self.space.derivative("_DispY", "Z")
            dw_dx = self.space.derivative("_DispZ", "X")
            dw_dy = self.space.derivative("_DispZ", "Y")
            eps = [
                du_dx,
                dv_dy,
                dw_dz,
                du_dy + dv_dx,
                du_dz + dw_dx,
                dv_dz + dw_dy,
            ]
        return eps

    def initialize(self, assembly, pb):
        """Initialize the weakform at the begining of a problem."""
        super().initialize(assembly, pb)
        self.assembly_options["assume_sym"] = False

        if assembly.elm_type in ["hex8sri", "quad4sri"]:
            if assembly._nlgeom == "TL":
                raise NotImplementedError(
                    f"{assembly.elm_type} not \
                                          implemented with total lagrangian \
                                          formulation"
                )
            self.space.variable_alias("_DispX", "DispX")
            self.space.variable_alias("_DispY", "DispY")
            if self.space.ndim == 3:
                self.space.variable_alias("_DispZ", "DispZ")
                self.space.new_vector("_Disp", ("_DispX", "_DispY", "_DispZ"))
            else:
                self.space.new_vector("_Disp", ("_DispX", "_DispY"))
        else:
            raise ValueError(
                "Incompatible element: StressEquilibirumFbar "
                "should used either 'hex8sri' or 'quad4sri' elements."
            )


class HourglassStiffness(WeakFormBase):
    """Hourglass stiffness weak formulation for reduced integration elements.

    This WeakForm should be added to a StressEquilibrium WeakForm to control
    the hourglass deformation modes associated to reduced integration.
    In most cases, the use of :py:class:`StressEquilibriumRI` that combines both
    HourglassStiffness and StressEquilibrium is prefered.

    This weakform should be used only for 'hex8' or 'quad4' elements with
    one integration point (n_elm_gp = 1 in the assembly).
    It is based on the classical method proposed by Flanagan and Belytschko in
    1981. In this method, the hourglass stiffness is normalized
    using the material tangent properties. The material properties are
    automatically extracted. If no StressEquilibrium object is found
    (including material properties), this will produce an error.


    Parameters
    ----------
    stiffness_coef: float, default: 0.01
        Coefficient to control the hourglass stiffness. This coefficient is a
        compromise between a sufficient stiffness to suppress hourglass modes
        and a not too high stiffness to avoid additionnal flexural stiffness.
    name: str
        name of the WeakForm
    nlgeom: bool, 'UL' or 'TL', optional
        If nlgeom is False, the stiffness is considered as linear based on the
        initial configuration. If not, the geometry and material properties
        are udpated at each iteration. The use of nlgeom = False, generally
        produces accurate results, even for finite strain problems.
    space: ModelingSpace
        Modeling space associated to the weakform. If None is specified,
        the active ModelingSpace is considered.

    Example
    --------
    Define a reduced integration weakform with default values for a 2D problem.

      >>> import fedoo as fd
      >>> fd.ModelingSpace("2Dstress")
      >>> material = fd.constitutivelaw.ElasticIsotrop(100e3, 0.3)
      >>> wf = fd.StressEquilibrium(material)
      >>> # force the use of reduced integration
      >>> wf.assembly_options["n_elm_gp"] = 1
      >>> wf = wf + fd.HourglassStiffness()
    """

    def __init__(self, stiffness_coef=0.01, name="", nlgeom=False, space=None):
        WeakFormBase.__init__(self, name, space)
        if self.space.is_axisymmetric:
            raise NotImplementedError(
                "HourglassStiffness is not implemented in '2Daxi': the "
                "2*pi*r integration weight on the stabilization term has "
                "not been validated. Use full integration (no hourglass "
                "stabilization needed) in 2Daxi."
            )
        self.assembly_options["n_elm_gp"] = 1
        self.assembly_options["elm_type", "quad4"] = "quad4hourglass"
        self.assembly_options["elm_type", "hex8"] = "hex8hourglass"
        self.stiffness_coef = stiffness_coef
        self.stress_equilibrium_assembly = None
        # can be a an assembly or None
        # (in this case, the get_bulk_modulus method is used) or None
        # (if None the associated assembly is searched to use get_bulk_modulus method)
        self.compute_stiffness_only_once = None

        self.nlgeom = nlgeom
        """Method used to treat the geometric non linearities.
            * Set to False if geometric non linarities are ignored.
            * Set to True or 'UL' to use the updated lagrangian method
              (update the mesh)
            * Set to 'TL' to use the total lagrangian method (base on the
              initial mesh with initial displacement effet)
        """

    def get_weak_equation(self, assembly, pb):
        """Get the weak equation related to the current problem state."""

        ### possible improvement : use the rigid body rotation to compute a
        ### covariant hourlgass

        # disp dof using the hourglass shape function give the hourglass dof
        # if assembly.elm_type == 'quad4hourglass':
        op_dq = [assembly.space.op_disp()]
        ndim = len(op_dq[0])
        if assembly.elm_type == "hex8hourglass":
            n_hourglass_mode = 4
            for i in range(1, n_hourglass_mode):
                op_dq.append(assembly.space.op_disp())
                for j in range(ndim):
                    op_dq[i][j].op[0].x = i
        elif assembly.elm_type == "quad4hourglass":
            n_hourglass_mode = 1
        else:
            raise ValueError(
                'elm_type should be "quad4hourglass" or '
                '"hex8hourglass" for HourlgassStiffness weakform'
            )

        if np.array_equal(pb.get_dof_solution(), 0):
            q0 = [[0, 0, 0] for dq in op_dq]
        else:
            q0 = [
                [
                    assembly.get_gp_results(dq[i], pb.get_dof_solution())
                    for i in range(ndim)
                ]
                for dq in op_dq
            ]

        if self.compute_stiffness_only_once and hasattr(
            assembly, "_hourglass_stiffness"
        ):
            hourglass_stiffness = assembly._hourglass_stiffness
        else:
            try:
                b = assembly._b_matrix.transpose(0, 2, 1)
            except AttributeError:
                assembly.compute_elementary_operators()
                b = assembly._b_matrix.transpose(0, 2, 1)
            # A = assembly.mesh.get_element_volumes()
            # coef = A * sum([(b[i] ** 2).sum(axis = 1) for i in range(ndim)])
            # coef = (1/A) * sum([(b[i] ** 2).sum(axis = 1) for i in range(ndim)])
            coef = sum([(b[i] ** 2).sum(axis=1) for i in range(ndim)])

            if isinstance(assembly.stress_equilibrium_assembly, AssemblyBase):
                pwave_modulus = self.get_p_wave_modulus(
                    assembly.stress_equilibrium_assembly
                )
            else:
                raise TypeError(
                    "HourglassStiffness could not find the StressEquilibrium "
                    "assembly required to compute the p-wave modulus. Set the "
                    "stress_equilibrium_assembly attribute of the "
                    "HourglassStiffness weakform explicitly."
                )

            hourglass_stiffness = 1 / ndim * self.stiffness_coef * pwave_modulus * coef
            # Formulation from Flanagan, D.P. and Belytschko, T. (1981)

            if self.compute_stiffness_only_once:
                assembly._hourglass_stiffness = hourglass_stiffness

        DiffOp = 0
        for hg_mode in range(n_hourglass_mode):
            DiffOp += sum(
                [
                    op_dq[hg_mode][i].virtual * (op_dq[hg_mode][i] + q0[hg_mode][i])
                    for i in range(ndim)
                ]
            )

        DiffOp = DiffOp * hourglass_stiffness

        # if self.space.is_axisymmetric:
        #     # not sur it works
        #     DiffOp = DiffOp * ((2 * np.pi) * rr)

        return DiffOp

    def initialize(self, assembly, pb):
        """Initialize the weakform at the begining of a problem."""
        super().initialize(assembly, pb)
        self._initialize_nlgeom(assembly, pb)
        if self.compute_stiffness_only_once is None:
            is_nonlinear_problem = pb.__class__.__name__.lower().startswith("nonlinear")
            self.compute_stiffness_only_once = not (
                assembly._nlgeom or is_nonlinear_problem
            )

        assembly.stress_equilibrium_assembly = (
            self.stress_equilibrium_assembly
        )  # pas propre, on devrait définir un autre dictionnaire dans assembly
        if self.stress_equilibrium_assembly is None:
            # extract the assembly associated to StressEquilibrium.
            # required to compute the tangent bulk modulus
            # The StressEquilibrium weakform may be wrapped inside a
            # WeakFormSum (e.g. compiled by a problem-level time integrator),
            # so the members of a sum are probed as well.
            list_assembly = assembly.associated_assembly_sum.list_assembly
            for a in list_assembly:
                if a.elm_type not in ["quad4", "hex8"]:
                    continue
                wf = a.weakform
                list_wf = getattr(wf, "list_weakform", [wf])
                if any(isinstance(w, StressEquilibrium) for w in list_wf):
                    assembly.stress_equilibrium_assembly = a
                    break

    def update(self, assembly, pb):
        """Update the weakform to the current state.

        This method is applyed before the update of constutive law (stress and
        stiffness matrix).
        """
        if assembly._nlgeom == "UL":
            # if updated lagragian method
            # -> update the mesh and recompute elementary op
            assembly.set_disp(pb.get_disp())

    def to_start(self, assembly, pb):
        """Reset the current time increment."""
        if assembly._nlgeom == "UL":
            # if updated lagrangian method -> reset the mesh to the begining
            # of the increment
            assembly.set_disp(pb.get_disp())

    def get_p_wave_modulus(self, assembly):
        # M = E*(1-nu)/((1+nu)*(1-2*nu)) = K+4/3 * mu
        #   = lambda + 2 * mu = p wave modulus
        Lt = assembly.sv["TangentMatrix"]
        bulk_modulus = (1 / 9) * sum([Lt[i, j] for i in range(3) for j in range(3)])
        # bulk_modulus = E/(3*(1-2*nu)) for elastic isotropic law
        shear_modulus = (1 / 3) * sum(
            [Lt[i, j] for i in range(3, 6) for j in range(3, 6)]
        )
        return bulk_modulus + 4 / 3 * shear_modulus


class StressEquilibriumRI(WeakFormSum):
    """Stress-equilibrium weak form with reduced integration.

    This weak form combines :class:`StressEquilibrium` with
    :class:`HourglassStiffness` to control the hourglass deformation modes
    associated with reduced integration.

    It should only be used with ``hex8`` or ``quad4`` elements. Unlike the
    standard :class:`StressEquilibrium` weak form, it uses one Gauss point by
    default and adds the hourglass stabilization proposed by Flanagan and
    Belytschko.

    Parameters
    ----------
    constitutivelaw : ConstitutiveLaw or str
        Constitutive law used by the stress-equilibrium term.
    hourglass_stiffness : float, default=0.01
        Coefficient controlling the hourglass stiffness. It should be large
        enough to suppress hourglass modes without introducing excessive
        artificial flexural stiffness.
    name : str, optional
        Name of the weak form.
    nlgeom : bool or {'UL', 'TL'}, optional
        Geometric-nonlinearity formulation used by the stress-equilibrium
        term. ``True`` and ``"UL"`` select the updated Lagrangian method;
        ``"TL"`` selects the total Lagrangian method.
    nlgeom_hourglass : bool or {'UL', 'TL'}, default=False
        Geometric-nonlinearity formulation used by the hourglass term.
        Keeping this false generally gives accurate results, including for
        finite-strain problems.
    space : ModelingSpace, optional
        Modeling space. Defaults to the active modeling space.

    Example
    -------
    Define a reduced-integration weak form for a 2D problem:

    >>> import fedoo as fd
    >>> fd.ModelingSpace("2Dstress")
    >>> material = fd.constitutivelaw.ElasticIsotrop(100e3, 0.3)
    >>> wf = fd.weakform.StressEquilibriumRI(material)
    """

    def __init__(
        self,
        constitutivelaw,
        hourglass_stiffness=0.01,
        name="",
        nlgeom=None,
        nlgeom_hourglass=False,
        space=None,
    ):
        equilibrium = StressEquilibrium(constitutivelaw, nlgeom=nlgeom, space=space)
        equilibrium.assembly_options["n_elm_gp", "quad4"] = 1
        equilibrium.assembly_options["n_elm_gp", "hex8"] = 1
        hourglass = HourglassStiffness(
            hourglass_stiffness, nlgeom=nlgeom_hourglass, space=space
        )
        super().__init__([equilibrium, hourglass], name)

    @property
    def corate(self):
        return self.list_weakform[0].corate

    @corate.setter
    def corate(self, value):
        self.list_weakform[0].corate = value
