import numpy as np
from fedoo.core.weakform import WeakFormBase
from fedoo.weakform.inertia import Inertia
# Assuming WeakFormBase is imported


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
    mass_wf : WeakForm, optional
        The artificial mass matrix used for damping (the :math:`M^*` matrix).
        Defaults to ``Inertia(1)`` (mass matrix with unit density), which
        provides a volume-weighted distribution ensuring mesh-independent
        results.
    energy_fraction : bool, default=True
        If True, the coefficient ``c_stab`` is automatically adapted at each
        increment to maintain a target energy ratio, ensuring the stabilization
        remains "invisible" to the physical results.
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
        mass_wf=None,
        energy_fraction=True,
        mat_lumping=True,
        name="",
        space=None,
    ):
        super().__init__(name, space)
        # The base weak form representing the spatial distribution (M*)
        if mass_wf is None:
            mass_wf = Inertia(1)
        self.mass_wf = mass_wf

        # c_stab is the stabilization coefficient.
        # if energy_fraction, the coefficient is interpreted as a enery ratio
        if energy_fraction:
            self.target_ratio = c_stab
            self.c_stab = 1e-3 * c_stab  # we start with a low energy ratio
            self.c_stab_initialized = False

        self.energy_fraction = energy_fraction
        if mat_lumping:
            self.mass_wf.assembly_options["mat_lumping"] = True

    def set_start(self, assembly, pb):
        """Update historical variables and adapt c_stab based on energy ratio."""
        if self.energy_fraction:
            dt = pb.dtime
            # if not (np.isscalar(pb.get_disp()) and pb.get_disp() == 0):
            #     assembly.sv["_DeltaDisp"] = 0
            #     return

            # 1. Skip if it's the very first initialization or a zero-time step
            if dt == 0:
                return

            # 2. Get the converged displacement increment from the PREVIOUS step
            # Note: ravel() ensures we can perform dot products easily
            if "_DeltaDisp" not in assembly.sv:
                return
            delta_u = assembly.sv["_DeltaDisp"].ravel()

            # 3. Calculate Incremental External Work (dW_ext = du * F_ext)
            f_ext = pb.get_ext_forces(include_mpc=False).ravel()
            delta_W_ext = np.dot(delta_u, f_ext)

            # 4. Calculate current Damping Energy (dE_damp = du * F_damp)
            # We need the spatial matrix M* to calculate the force

            # F_damp = c_stab * M* * (delta_u / dt)
            # M  = assembly.get_global_matrix()
            # delta_E_damp = self.c_stab/dt * (delta_u @ M @ delta_u)
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

                    if self.c_stab_initialized:
                        # Safeguard: Don't let c_stab change by more than a factor of 10
                        # in a single step to maintain numerical stability.
                        self.c_stab *= np.clip(adjustment, 0.1, 10.0)
                    else:
                        self.c_stab *= adjustment

        # 6. Reset the accumulated displacement for the new increment
        assembly.sv["_DeltaDisp"] = np.zeros_like(assembly.sv["_DeltaDisp"])

    def update(self, assembly, pb):
        # assembly.sv["_DeltaDisp"] = pb._get_vect_component(pb._dU, "Disp")
        assembly.sv["_DeltaDisp"] = pb._dU.reshape(-1, pb.mesh.n_nodes)

    def get_weak_equation(self, assembly, pb):
        dt = pb.dtime

        # If dt is 0, we can't compute pseudo-velocity. Return 0 to avoid division by zero.
        if dt == 0 or self.c_stab == 0.0:
            return 0

        # 1. Retrieve the current displacement increment
        if "_DeltaDisp" not in assembly.sv:
            return 0
        delta_u = assembly.sv["_DeltaDisp"]

        # 2. Compute Pseudo-Velocity
        v_pseudo = delta_u / dt

        # 3. Get the Base Matrix (M*) from the input weakform
        base_wf = self.mass_wf.get_weak_equation(assembly, pb)

        if hasattr(base_wf, "split_mat_vec"):
            mat, vec = base_wf.split_mat_vec()
        else:
            mat = base_wf
            vec = 0

        # 4. Calculate Tangent Contribution: (c_stab / dt) * M*
        tangent_matrix = mat * (self.c_stab / dt)

        # 5. Calculate Residual Contribution: c_stab * M* * v_pseudo
        if not np.array_equal(v_pseudo, 0):
            # Scale by the stabilization coefficient
            v_pseudo *= self.c_stab

            # Apply the matrix operator to the pseudo-velocity array
            damping_force = assembly.operator_apply(mat, v_pseudo.ravel())

            # Axisymmetric correction if your framework requires it here
            # if self.space is not None and getattr(self.space, "_dimension", "") == "2Daxi":
            #     rr = assembly.sv["_R_gausspoints"]
            #     damping_force = damping_force * ((2 * np.pi) * rr)

            return tangent_matrix + vec + damping_force

        return tangent_matrix + vec
