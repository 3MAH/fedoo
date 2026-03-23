import numpy as np
from fedoo.core.weakform import WeakFormBase
from fedoo.weakform.inertia import Inertia


class ArtificialDamping(WeakFormBase):
    """
    Weak formulation for artificial viscous stabilization.

    This class provides an artifical damping to stabilize unstable increments
    in both static and dynamic problems. It introduces a velocity-dependent
    damping force that regularizes the system when the stiffness matrix is
    singular or non-positive definite (e.g., during buckling, snap-through, or
    material softening).

    The damping force is defined as:
    .. math::
        F_{stab} = c_{stab} \cdot M^* \cdot v

    where :math:`M^*` is an artificial mass matrix (unit-density volume
    integrator) and :math:`v` is the velocity computed from displacement:
    :math:`\Delta u / \Delta t`.

    Parameters
    ----------
    c_stab : float, default=2e-4
        The stabilization coefficient. If ``energy_fraction`` is True, this is
        interpreted as the target ratio of dissipated stabilization energy to
        external work.
    energy_fraction : bool, default=True
        If True, the coefficient ``c_stab`` is automatically adapted at each
        increment to maintain a target energy ratio, ensuring the stabilization
        remains "invisible" to the physical results.
    variables : str | list[str], optional
        The variables or vectors (e.g., 'Disp', 'Rot') to which damping is
        applied. By default, all displacement and rotational variables in the
        space are included.
    mat_lumping : bool, default=True
        If True, the stabilization matrix is lumped (diagonalized). This is
        recommended as it ensures that stabilization forces are
        localized and independent for each degree of freedom, improving
        numerical robustness.
    name : str, optional
        Name of the WeakForm instance.
    space : ModelingSpace, optional
        Modeling space for the weakform.

    Notes
    -----
    * The default mass matrix doesn't add rotational damping.

    Examples
    --------
    .. code-block:: python

        wf = fd.weakform.StressEquilibrium(material)
        wf += fd.weakform.ArtificialDamping(c_stab=0.05)
    """

    def __init__(
        self,
        c_stab=2e-4,
        energy_fraction=True,
        variables=None,
        mat_lumping=True,
        name="",
        space=None,
    ):
        super().__init__(name, space)
        self.damped_variables = variables
        self.c_stab = c_stab
        self.energy_fraction = energy_fraction

        if mat_lumping:
            self.assembly_options["mat_lumping"] = True

    def initialize(self, assembly, pb):
        if self.damped_variables is None:
            self.damped_variables = [
                vec for vec in ["Disp", "Rot"] if vec in self.space.list_vectors()
            ]
        if isinstance(self.damped_variables, str):
            self.damped_variables = [self.damped_variables]
        self.damped_variables = [
            item
            for var in self.damped_variables
            for item in (
                self.space.get_vector(var)
                if var in self.space.list_vectors()
                else [var]
            )
        ]

        # Initialize the global stabilization factor
        if self.energy_fraction:
            self.target_ratio = self.c_stab  # alias
            # Start with a very small global fraction
            self._c_stab = 1e-3 * self.target_ratio
            self._c_stab_initialized = False

    def set_start(self, assembly, pb):
        """Update historical variables and adapt c_stab based on energy ratio."""
        if self.energy_fraction:
            dt = pb.dtime

            # 1. Skip if it's the very first initialization or a zero-time step
            if dt == 0:
                return

            # 2. Get the converged displacement increment from the PREVIOUS step
            # note: pb._dU = 0 here so we need te get the saved value in sv
            if "_DeltaDisp" not in assembly.sv:
                return
            delta_u = assembly.sv["_DeltaDisp"]

            # 3. Calculate Incremental External Work (dW_ext = du * F_ext)
            f_ext = pb.get_ext_forces(include_mpc=False).ravel()
            delta_W_ext = np.dot(delta_u, f_ext)

            # 4. Calculate current Damping Energy (dE_damp = du * F_damp)
            # We need the spatial matrix M* to calculate the force

            # F_damp = c_stab * M* * (delta_u / dt)
            # M  = assembly.get_global_matrix()
            # delta_E_damp = self._c_stab/dt * (delta_u @ M @ delta_u)
            delta_E_damp = delta_u @ assembly.get_global_vector()

            # 5. Adaptive Adjustment
            # We only adjust if there is significant external work being done
            # Otherwise, we keep the current c_stab to avoid division by zero
            if abs(delta_W_ext) > 1e-10:
                current_ratio = abs(delta_E_damp / delta_W_ext)

                # If current_ratio is 0 (no movement), we don't change anything
                if current_ratio > 0:
                    # Calculate adjustment: c_new = c_old * (target / current)
                    adjustment = self.target_ratio / current_ratio

                    if self._c_stab_initialized:
                        # Safeguard: Don't let c_stab change by more than a factor of 10
                        # in a single step to maintain numerical stability.
                        self._c_stab *= np.clip(adjustment, 0.1, 10.0)
                    else:
                        self._c_stab *= adjustment

        # 6. Reset the accumulated displacement for the new increment
        assembly.sv["_DeltaDisp"] = 0

    def update(self, assembly, pb):
        assembly.sv["_DeltaDisp"] = pb._dU  # alias required for set_start

    def get_weak_equation(self, assembly, pb):
        dt = pb.dtime

        # If dt is 0, we can't compute pseudo-velocity. Return 0 to avoid division by zero.
        if dt == 0:
            return 0

        # 1. Retrieve the current displacement increment
        delta_u = pb._dU

        # 2. Compute Pseudo-Velocity
        # v_pseudo = delta_u / dt

        # 3. Weak equation of the virtual mass matrix (M*)
        op_var = [self.space.variable(var) for var in self.damped_variables]
        op_var_vir = [op.virtual if op != 0 else 0 for op in op_var]

        # 4. Calculate Tangent Contribution: (c_stab / dt) * M*
        tangent_matrix = sum(
            [a * b * (self._c_stab / dt) for (a, b) in zip(op_var_vir, op_var)]
        )

        # 4. Calculate Residual Contribution: c_stab * M* * v_pseudo
        if not np.array_equal(delta_u, 0):
            # Scale by the stabilization coefficient
            # v_pseudo[self._variables_id] *= self._vec_c_stab[:,np.newaxis]

            # Apply the matrix operator to the pseudo-velocity array
            damping_force = assembly.operator_apply(tangent_matrix, delta_u)

            # Axisymmetric correction
            # if self.space is not None and getattr(self.space, "_dimension", "") == "2Daxi":
            #     rr = assembly.sv["_R_gausspoints"]
            #     damping_force = damping_force * ((2 * np.pi) * rr)

            return tangent_matrix + damping_force

        return tangent_matrix
