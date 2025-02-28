# doesn't seem to work. Not imported by default.
# use with hex8sri elements

from fedoo.weakform.stress_equilibrium import StressEquilibrium
import numpy as np


class StressEquilibriumBbar(StressEquilibrium):
    """Weak formulation of the mechanical equilibrium equation for solids.

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

    def get_weak_equation(self, assembly, pb):
        """Get the weak equation related to the current problem state."""
        if assembly._nlgeom == "TL":  # add initial displacement effect
            eps = self.space.op_strain(assembly.sv["DispGradient"])
            initial_stress = assembly.sv["PK2"]
        else:
            eps = self.space.op_strain()
            if assembly.elm_type in ["hex8sri"]:
                eps[0] = (
                    eps[0]
                    - self.space.derivative("DispX", "X")
                    + self.space.derivative("_DispX", "X")
                )
                eps[1] = (
                    eps[1]
                    - self.space.derivative("DispY", "Y")
                    + self.space.derivative("_DispY", "Y")
                )
                eps[2] = (
                    eps[2]
                    - self.space.derivative("DispZ", "Z")
                    + self.space.derivative("_DispZ", "Z")
                )

            initial_stress = assembly.sv[
                "Stress"
            ]  # Stress = Cauchy for updated lagrangian method

            if self.space._dimension == "2Daxi":
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
            # if assembly._nlgeom:
            #     DiffOp = DiffOp + \
            #         sum([0 if self._nl_strain_op_vir[i] == 0 else
            #              self._nl_strain_op_vir[i] * initial_stress[i]
            #              for i in range(6)])

            DiffOp = DiffOp + sum(
                [
                    0 if eps[i] == 0 else eps[i].virtual * initial_stress[i]
                    for i in range(6)
                ]
            )

        if self.space._dimension == "2Daxi":
            DiffOp = DiffOp * ((2 * np.pi) * rr)

        return DiffOp

    def initialize(self, assembly, pb):
        """Initialize the weakform at the begining of a problem."""
        super().initialize(assembly, pb)
        if assembly.elm_type in ["hex8sri"]:
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
