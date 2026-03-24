"""Weak form for phase-field damage evolution equation.

Solves the reaction-diffusion equation for the damage variable d:
    integral [r*d*delta_d + k*grad(d)*grad(delta_d)] dOmega = integral f*delta_d dOmega

where:
    k = diffusion coefficient (depends on Gc, l0, model)
    r = reaction coefficient (depends on history variable H)
    f = source term (depends on H)
"""

from fedoo.core.base import ConstitutiveLaw
from fedoo.core.weakform import WeakFormBase

import numpy as np


class PhaseFieldEvolution(WeakFormBase):
    """Weak formulation for the phase-field damage evolution equation.

    This is a reaction-diffusion equation in the damage variable d,
    following the SteadyHeatEquation pattern. The coefficients r, k, f
    are computed from the phase-field constitutive law.

    Parameters
    ----------
    phasefield_cl : PhaseFieldDamage or str
        Phase-field constitutive law that provides Gc, l0, model,
        diffusion_coeff, get_reaction_coeff, and get_source_coeff.
    name : str, optional
        Name of the weak form.
    space : ModelingSpace, optional
        Modeling space. If None, the active ModelingSpace is used.
    """

    def __init__(self, phasefield_cl, name=None, space=None):
        if isinstance(phasefield_cl, str):
            phasefield_cl = ConstitutiveLaw.get_all()[phasefield_cl]

        if name is None:
            name = phasefield_cl.name

        WeakFormBase.__init__(self, name, space)

        self.space.new_variable("Damage")

        if self.space.ndim == 3:
            self.__op_grad_d = [
                self.space.derivative("Damage", "X"),
                self.space.derivative("Damage", "Y"),
                self.space.derivative("Damage", "Z"),
            ]
        else:  # 2D
            self.__op_grad_d = [
                self.space.derivative("Damage", "X"),
                self.space.derivative("Damage", "Y"),
                0,
            ]

        self.__op_grad_d_vir = [0 if op == 0 else op.virtual for op in self.__op_grad_d]

        self.__op_d = self.space.variable("Damage")
        self.__op_d_vir = self.__op_d.virtual

        self.phasefield_cl = phasefield_cl
        self.constitutivelaw = None  # no separate CL for the damage equation

        self.assembly_options["assume_sym"] = True

    def initialize(self, assembly, pb):
        n_gp = assembly.n_gauss_points
        if not (np.isscalar(pb.get_dof_solution())):
            assembly.sv["DamageGradient"] = [
                0 if op == 0 else assembly.get_gp_results(op, pb.get_dof_solution())
                for op in self.__op_grad_d
            ]
        else:
            assembly.sv["DamageGradient"] = [0, 0, 0]

        # Initialize damage at gauss points
        assembly.sv["Damage_GP"] = np.zeros(n_gp)

    def update(self, assembly, pb):
        assembly.sv["DamageGradient"] = [
            0 if op == 0 else assembly.get_gp_results(op, pb.get_dof_solution())
            for op in self.__op_grad_d
        ]

        # Update damage at gauss points from current solution
        sol = pb.get_dof_solution()
        if not (np.isscalar(sol) and sol == 0):
            assembly.sv["Damage_GP"] = assembly.convert_data(
                sol, convert_from="Node", convert_to="GaussPoint"
            )

    def get_weak_equation(self, assembly, pb):
        cl = self.phasefield_cl

        # Get history variable from the mechanical assembly's state
        H = cl._H
        if H is None:
            H = np.zeros(assembly.n_gauss_points)

        # Convert H from mechanical GP to damage GP if needed
        # (same mesh assumed, so direct use is fine)

        # Coefficients
        k = cl.diffusion_coeff  # scalar
        r = cl.get_reaction_coeff(H)  # array (n_gp)
        f = cl.get_source_coeff(H)  # array (n_gp)

        # Convert r and f to gauss point data for the damage assembly
        # They come from the mechanical assembly's GP, but if same mesh
        # is used, the GP layout is identical.

        # --- Diffusion term: k * grad(d_vir) . grad(d) ---
        diff_op = sum(
            0
            if self.__op_grad_d_vir[i] == 0
            else self.__op_grad_d_vir[i] * self.__op_grad_d[i] * k
            for i in range(3)
        )

        # --- Reaction term: r * d_vir * d ---
        diff_op = diff_op + self.__op_d_vir * self.__op_d * r

        # --- Source term (RHS): -f * d_vir ---
        # Negative because it goes to the RHS: K*d = f becomes K*d - f = 0
        # In fedoo's incremental form, the source is an initial state contribution
        diff_op = diff_op - self.__op_d_vir * f

        # --- Add contribution from current damage state (incremental) ---
        d_grad = assembly.sv["DamageGradient"]
        diff_op = diff_op + sum(
            0
            if (self.__op_grad_d_vir[i] == 0 or np.array_equal(d_grad[i], 0))
            else self.__op_grad_d_vir[i] * d_grad[i] * k
            for i in range(3)
        )

        d_gp = assembly.sv.get("Damage_GP", 0)
        if not (np.isscalar(d_gp) and d_gp == 0):
            diff_op = diff_op + self.__op_d_vir * d_gp * r

        return diff_op

    def reset(self):
        pass

    def to_start(self):
        pass
