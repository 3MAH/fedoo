# derive de ConstitutiveLaw

from fedoo.constitutivelaw.spring import Spring
from fedoo.core.base import ConstitutiveLaw
from fedoo.core.base import AssemblyBase
import numpy as np


class CohesiveLaw(Spring):
    """
    Bilinear cohesive Law based on the Crisfield model

    This constitutive Law should be associated with
    :mod:`fedoo.weakform.InterfaceForce`

    Parameters
    ----------
    GIc: scalar
        Toughness in Mode-I
    SImax: scalar
        Maximal failure stress in Mode-I
    KI: scalar
        Initial interface rigidity before damage
    GIIc: scalar
        Toughness in Mode-II
    SIImax: scalar
        Maximal failure stress in Mode-II
    KII: scalar
        Initial interface rigidity before damage
    axis: int
        axis should be eiter 0,1 or 2 (default). It define the normal
        direction to the failure plane the is used for mode identification.
        The axis is defined in local coordinate system.
    name: str, optional
        The name of the constitutive law
    tangent_mode: {"secant", "consistent"}, default="secant"
        Tangent stiffness used by the nonlinear solver. The secant tangent uses
        the current damaged stiffness and is the more robust default. The
        consistent tangent includes the derivative of damage during active
        loading and can be selected explicitly.
    """

    # Use with WeakForm.InterfaceForce
    def __init__(
        self,
        GIc=0.3,
        SImax=60,
        KI=1e4,
        GIIc=1.6,
        SIImax=None,
        KII=5e4,
        axis=2,
        name="",
        tangent_mode="secant",
    ):
        # GIc la ténacité (l'énergie à la rupture = l'aire sous la courbe du modèle en N/mm)
        #        SImax = 60.  # la contrainte normale maximale de l'interface (MPa)
        #        KI = 1e4          # la raideur des éléments cohésive (la pente du modèle en N/mm3)
        #
        #        # Mode II (12)
        #        G_IIc = 1.6
        #        KII = 5e4

        #
        ##----------------------- la puissance du critère de propagation (cas de critère de Power Law)---------------------------
        #        alpha = 2.
        #
        # axis = axe dans le repère local perpendiculaire au plan (will be deprecated because = 2 by convention)

        ConstitutiveLaw.__init__(self, name)  # heritage
        if axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1, or 2.")

        tangent_mode = tangent_mode.lower()
        if tangent_mode not in ("consistent", "secant"):
            raise ValueError("tangent_mode must be either 'consistent' or 'secant'.")
        self.tangent_mode = tangent_mode

        self.parameters = {
            "GIc": GIc,
            "SImax": SImax,
            "KI": KI,
            "GIIc": GIIc,
            "SIImax": SIImax,
            "KII": KII,
            "axis": axis,
        }

        delta_0_I = SImax / KI
        delta_m_I = 2 * GIc / SImax
        if SIImax is None:
            SIImax_check = SImax * np.sqrt(GIIc / GIc)
        else:
            SIImax_check = SIImax
        delta_0_II = SIImax_check / KII
        delta_m_II = 2 * GIIc / SIImax_check
        if delta_m_I <= delta_0_I or delta_m_II <= delta_0_II:
            raise ValueError(
                "Cohesive parameters must satisfy delta_m > delta_0 "
                "in both fracture modes."
            )

    def initialize(self, assembly, pb):
        assembly.sv["InterfaceStress"] = 0  # Interface Stress
        assembly.sv["DamageVariable"] = 0  # damage variable
        assembly.sv["DamageVariableOpening"] = (
            0  # DamageVariableOpening is used for the opening mode (mode I). It is equal to DamageVariable in traction and equal to 0 in compression (soft contact law)
        )
        assembly.sv["DamageVariableIrreversible"] = (
            0  # irreversible damage variable used for time evolution
        )
        assembly.sv["TangentMatrix"] = self.get_K(assembly)

    def get_secant_matrix(self, assembly):
        """Return the current damaged secant stiffness in local coordinates."""
        Umd = 1 - assembly.sv["DamageVariable"]
        UmdI = 1 - assembly.sv["DamageVariableOpening"]

        axis = self.parameters["axis"]

        Kt = Umd * self.parameters["KII"]
        Kn = UmdI * self.parameters["KI"]
        Kdiag = [Kt if i != axis else Kn for i in range(3)]
        return [[Kdiag[0], 0, 0], [0, Kdiag[1], 0], [0, 0, Kdiag[2]]]

    def get_tangent_matrix(self, assembly, delta=None, damage_gradient=None):
        """Return the selected tangent stiffness in local coordinates.

        The consistent correction is active only while damage grows beyond
        its committed value. During unloading, reloading below the historical
        maximum, and after complete failure, ``damage_gradient`` is zero and
        this method returns the damaged secant stiffness.
        """
        secant = self.get_secant_matrix(assembly)
        if self.tangent_mode == "secant" or delta is None:
            return secant

        if damage_gradient is None:
            _, _, damage_gradient = self._compute_damage(assembly, delta)

        delta = np.asarray(np.broadcast_arrays(*delta), dtype=float)
        opening = delta[self.parameters["axis"]] > 0
        axis = self.parameters["axis"]
        stiffness = [
            self.parameters["KII"] if i != axis else self.parameters["KI"]
            for i in range(3)
        ]

        tangent = [[None for _ in range(3)] for _ in range(3)]
        for i in range(3):
            row_gradient = damage_gradient
            if i == axis:
                # Normal contact remains elastic in compression.
                row_gradient = damage_gradient * opening
            for j in range(3):
                correction = stiffness[i] * delta[i] * row_gradient[j]
                tangent[i][j] = secant[i][j] - correction
        return tangent

    def get_K(self, assembly, delta=None, damage_gradient=None):
        return self.local2global_K(
            self.get_tangent_matrix(assembly, delta, damage_gradient)
        )

    def set_damage(self, assembly, value, irreversible=True):
        """
        Initialize the damage variable to a certain value: array for multi-point initialization or scalar.
        The damage is considered as irreversible by default.
        Use Irreversible = False for reversible damage.
        The damage should be udpated with CohesiveLaw.updateDamageVariable
        to determine if the crack is opening or closing. If not, no contact will be considered.
        """
        assembly.sv["DamageVariable"] = assembly.sv["DamageVariableOpening"] = value
        if irreversible:
            self.update_irreversible_damage(assembly)

    def update_irreversible_damage(self, assembly):
        if (
            np.isscalar(assembly.sv["DamageVariable"])
            and assembly.sv["DamageVariable"] == 0
        ):
            assembly.sv["DamageVariableIrreversible"] = 0
        else:
            assembly.sv["DamageVariableIrreversible"] = assembly.sv[
                "DamageVariable"
            ].copy()

    def update_damage(self, assembly, U, irreversible=False, type_data="PG"):
        if isinstance(assembly, str):
            assembly = AssemblyBase.get_all()[assembly]

        # In an updated-Lagrangian analysis, the current assembly carries the
        # deformed interface geometry and its updated local coordinate system.
        # For a geometrically linear analysis, assembly.current is assembly.
        result_assembly = getattr(assembly, "current", assembly)
        op_delta = result_assembly.space.op_disp()
        if type_data == "Node":
            delta = [result_assembly.get_node_results(op, U) for op in op_delta]
        else:
            delta = [result_assembly.get_gp_results(op, U) for op in op_delta]

        self._update_damage(assembly, delta)

        if irreversible:
            assembly.sv["DamageVariableIrreversible"] = assembly.sv[
                "DamageVariable"
            ].copy()

    def _compute_damage(self, assembly, delta):
        """Compute damage and its derivative with respect to separation.

        The derivative is the algorithmic derivative: it is nonzero only when
        trial damage exceeds the irreversible damage stored at the start of the
        increment. Unloading and elastic reloading therefore use the damaged
        secant stiffness.
        """
        alpha = 2.0
        delta = np.asarray(np.broadcast_arrays(*delta), dtype=float)
        if delta.shape[0] != 3:
            raise ValueError("CohesiveLaw expects three separation components.")

        state_shape = delta.shape[1:]
        delta = delta.reshape(3, -1)
        n_points = delta.shape[1]
        axis = self.parameters["axis"]
        tangential_axes = [i for i in range(3) if i != axis]

        delta_n = delta[axis]
        delta_t_vector = delta[tangential_axes]
        delta_t = np.sqrt(np.sum(delta_t_vector**2, axis=0))

        delta_0_I = self.parameters["SImax"] / self.parameters["KI"]
        SIImax = self.parameters["SIImax"]
        if SIImax is None:
            SIImax = self.parameters["SImax"] * np.sqrt(
                self.parameters["GIIc"] / self.parameters["GIc"]
            )
        delta_0_II = SIImax / self.parameters["KII"]
        delta_m_II = 2.0 * self.parameters["GIIc"] / SIImax

        opening = delta_n > 0.0
        compression = ~opening
        t0 = np.empty(n_points)
        tm = np.empty(n_points)
        equivalent_delta = np.empty(n_points)
        beta = np.zeros(n_points)
        t0_beta = np.zeros(n_points)
        tm_beta = np.zeros(n_points)

        if np.any(opening):
            beta[opening] = delta_t[opening] / delta_n[opening]
            beta_opening = beta[opening]

            denominator = delta_0_II**2 + (beta_opening * delta_0_I) ** 2
            t0[opening] = (
                delta_0_II * delta_0_I * np.sqrt((1.0 + beta_opening**2) / denominator)
            )

            mode_I_term = self.parameters["KI"] / self.parameters["GIc"]
            mode_II_term = (
                self.parameters["KII"] * beta_opening**2 / self.parameters["GIIc"]
            )
            power_sum = mode_I_term**alpha + mode_II_term**alpha
            tm[opening] = (
                2.0
                * (1.0 + beta_opening) ** 2
                / t0[opening]
                * power_sum ** (-1.0 / alpha)
            )
            equivalent_delta[opening] = np.sqrt(
                delta_t[opening] ** 2 + delta_n[opening] ** 2
            )

            t0_beta[opening] = t0[opening] * (
                beta_opening / (1.0 + beta_opening**2)
                - beta_opening * delta_0_I**2 / denominator
            )
            power_sum_beta = (
                alpha
                * mode_II_term ** (alpha - 1.0)
                * (2.0 * self.parameters["KII"] * beta_opening)
                / self.parameters["GIIc"]
            )
            tm_beta[opening] = tm[opening] * (
                2.0 / (1.0 + beta_opening)
                - t0_beta[opening] / t0[opening]
                - power_sum_beta / (alpha * power_sum)
            )

        t0[compression] = delta_0_II
        tm[compression] = delta_m_II
        equivalent_delta[compression] = delta_t[compression]

        trial_damage = (equivalent_delta >= tm).astype(float)
        softening = (equivalent_delta > t0) & (equivalent_delta < tm)
        softening_factor = np.zeros(n_points)
        softening_factor[softening] = tm[softening] / (tm[softening] - t0[softening])
        trial_damage[softening] = softening_factor[softening] * (
            1.0 - t0[softening] / equivalent_delta[softening]
        )

        irreversible = np.asarray(
            assembly.sv["DamageVariableIrreversible"], dtype=float
        )
        irreversible = np.broadcast_to(irreversible, state_shape).reshape(-1)
        damage = np.maximum(irreversible, trial_damage)

        # max(d_irreversible, d_trial) makes this derivative vanish unless
        # trial damage is actively increasing.
        active = softening & (trial_damage > irreversible)
        damage_gradient = np.zeros((3, n_points))

        active_compression = active & compression
        if np.any(active_compression):
            r = equivalent_delta[active_compression]
            damage_r = (
                softening_factor[active_compression] * t0[active_compression] / r**2
            )
            for component, tangential_axis in enumerate(tangential_axes):
                damage_gradient[tangential_axis, active_compression] = (
                    damage_r * delta_t_vector[component, active_compression] / r
                )

        active_opening = active & opening
        if np.any(active_opening):
            r = equivalent_delta[active_opening]
            t0_active = t0[active_opening]
            tm_active = tm[active_opening]
            factor = softening_factor[active_opening]

            factor_beta = (
                tm_active * t0_beta[active_opening]
                - tm_beta[active_opening] * t0_active
            ) / (tm_active - t0_active) ** 2
            damage_r = factor * t0_active / r**2
            damage_beta = factor_beta * (1.0 - t0_active / r) - (
                factor * t0_beta[active_opening] / r
            )

            delta_n_active = delta_n[active_opening]
            beta_active = beta[active_opening]
            damage_gradient[axis, active_opening] = (
                damage_r * delta_n_active / r
                - damage_beta * beta_active / delta_n_active
            )

            delta_t_active = delta_t[active_opening]
            nonzero_tangent = delta_t_active > 0.0
            for component, tangential_axis in enumerate(tangential_axes):
                gradient = damage_r * (delta_t_vector[component, active_opening] / r)
                gradient[nonzero_tangent] += damage_beta[nonzero_tangent] * (
                    delta_t_vector[component, active_opening][nonzero_tangent]
                    / (
                        delta_t_active[nonzero_tangent]
                        * delta_n_active[nonzero_tangent]
                    )
                )
                damage_gradient[tangential_axis, active_opening] = gradient

        return (
            damage.reshape(state_shape),
            (opening * damage).reshape(state_shape),
            damage_gradient.reshape((3,) + state_shape),
        )

    def _update_damage(self, assembly, delta):
        damage, damage_opening, damage_gradient = self._compute_damage(assembly, delta)
        assembly.sv["DamageVariable"] = damage
        assembly.sv["DamageVariableOpening"] = damage_opening
        return damage_gradient

    def reset(self):
        pass

    def set_start(self, assembly, pb):
        # Commit damage at the end of the converged increment. At the start of
        # the next increment, the exact algorithmic response is unloading from
        # that state, so the predictor matrix must be the damaged secant rather
        # than the negative active-softening tangent from the previous step.
        self.update_irreversible_damage(assembly)
        assembly.sv["TangentMatrix"] = self.local2global_K(
            self.get_secant_matrix(assembly)
        )

    # def to_start(self, assembly, pb):
    #     #Damage variable will be recompute. NPOthing to be done here (to be checked)
    #     pass

    def update(self, assembly, pb):
        displacement = pb.get_dof_solution()

        if np.isscalar(displacement) and displacement == 0:
            assembly.sv["InterfaceStress"] = assembly.sv["RelativeDisp"] = 0
            K = self.get_K(assembly)
        else:
            # InterfaceForce updates assembly.current before this constitutive
            # update. Evaluate the jump with its current local frame so damage,
            # traction, and the assembled tangent use the same configuration.
            result_assembly = getattr(assembly, "current", assembly)
            op_delta = result_assembly.space.op_disp()
            delta = [
                result_assembly.get_gp_results(op, displacement) for op in op_delta
            ]
            assembly.sv["RelativeDisp"] = delta

            damage_gradient = self._update_damage(assembly, delta)
            dim = len(delta)
            # Traction follows the secant constitutive relation. The
            # consistent matrix is its derivative and is used only for the
            # Newton Jacobian.
            K_secant = self.local2global_K(self.get_secant_matrix(assembly))
            assembly.sv["InterfaceStress"] = [
                sum([delta[j] * K_secant[i][j] for j in range(dim)]) for i in range(dim)
            ]  # list of 3 objects
            K = self.get_K(assembly, delta, damage_gradient)

        assembly.sv["TangentMatrix"] = K

    # def GetInterfaceStress(self, Delta, time = None):
    #     #Delta is the relative displacement vector
    #     self.__UpdateDamageVariable(Delta)
    #     return Spring.GetInterfaceStress(self, Delta, time)


#    def SetLocalFrame(self, localFrame):
#        raise NameError("Not implemented: localFrame are not implemented in the context of cohesive laws")


# def __UpdateDamageVariable_old(self, delta):
#     #---------------------------------------------------------------------------------------------------------
#     ################# interface 90°/0° (Lower interface) ########################
#     #---------------------------------------------------------------------------------------------------------


#     alpha = 2 #for the power low
#     if np.isscalar(assembly.sv['DamageVariable']) and assembly.sv['DamageVariable'] == 0: assembly.sv['DamageVariable'] = 0*delta[0]
#     if np.isscalar(assembly.sv['DamageVariableOpening']) and assembly.sv['DamageVariableOpening'] == 0: assembly.sv['DamageVariableOpening']  = 0*delta[0]

#     # delta_n = delta.pop(self.parameters['axis'])
#     # delta_t = np.sqrt(delta[0]**2 + delta[1]**2)
#     delta_n = delta[self.parameters['axis']]
#     delta_t = [d for i,d in enumerate(delta) if i != self.parameters['axis'] ]
#     if len(delta_t) == 1:
#         delta_t = delta_t[0]
#     else:
#         delta_t = np.sqrt(delta_t[0]**2 + delta_t[1]**2)

#     # mode I
#     delta_0_I = self.parameters['SImax'] / self.parameters['KI']   # critical relative displacement (begining of the damage)
#     delta_m_I =  2*self.parameters['GIc'] / self.parameters['SImax']   # maximal relative displacement (total failure)

#     # mode II
#     SIImax = self.parameters['SIImax']
#     if SIImax == None: SIImax = self.parameters['SImax'] * np.sqrt(self.parameters['GIIc'] / self.parameters['GIc'])   #value by default used mainly to treat mode I dominant problems
#     delta_0_II = SIImax / self.parameters['KII']
#     delta_m_II =  2*self.parameters['GIIc'] / SIImax

#     for i in range (len(delta_n)):
#         if delta_n[i] > 0 :
#             beta= delta_t[i] / (delta_n[i]) # le rapport de mixité de mode

#             t0= (delta_0_II * delta_0_I) * (np.sqrt((1+ (beta**2)) / ((delta_0_II**2)+((beta*delta_0_I)**2)))) # Critical relative displacement in mixed mode

#             tm= (2*((1+ beta)**2)/t0) * ((((self.parameters['KI']/self.parameters['GIc'])**alpha) + \
#                     (((self.parameters['KII']*beta**2)/self.parameters['GIIc'])**alpha))**(-1/alpha)) #Maximal relative displacement in mixed mode (power low criterion)

#             dta= np.sqrt(delta_t[i]**2 + delta_n[i]**2)  # Actual relative displacement in mixed mode

#         else : #only mode II
#             t0= delta_0_II # Critical relatie displacement in mixed mode
#             print(delta_0_II)
#             tm= delta_m_II # Maximal relative displacement in mixed mode (power low criterion)

#             dta= delta_t[i] # Actual relative displacement in mixed mode

#         #---------------------------------------------------------------------------------------------------------------
#         # La variable d'endommagement "d"
#         #---------------------------------------------------------------------------------------------------------------
#         if dta <= t0:
#             di = 0
#         elif dta > t0  and dta < tm:
#             di = (tm / (tm - t0)) * (1 - (t0 / dta))
#         else:
#             di = 1

#         if (assembly.sv['DamageVariableIrreversible'] is 0) or (di >  assembly.sv['DamageVariableIrreversible'][i]):
#             assembly.sv['DamageVariable'][i] = di
#         else: assembly.sv['DamageVariable'][i] = assembly.sv['DamageVariableIrreversible'][i]

#         if delta_n[i] > 0 :
#             assembly.sv['DamageVariableOpening'] [i] = assembly.sv['DamageVariable'][i]
#         else :
#             assembly.sv['DamageVariableOpening'] [i] = 0

#     # verification : the damage variable should be between 0 and 1
#     if assembly.sv['DamageVariable'].min() < 0 or assembly.sv['DamageVariable'].max() > 1 :
#         print ("Warning : the value of damage variable is incorrect")


# if __name__=="__main__":
#     ModelingSpace("3D")
#     GIc = 0.3 ; SImax = 60
#     # delta_I_max = 2*GIc/SImax
#     delta_I_max = 0.04
#     nb_iter = 100
#     sig = []
#     delta_plot = []
#     law = CohesiveLaw(GIc=GIc, SImax = SImax, KI = 1e4, GIIc = 1, SIImax=60, KII=1e4, axis = 0)
#     for delta_z in np.arange(0,delta_I_max,delta_I_max/nb_iter):
#         delta = [np.array([0]), np.array([0]), np.array([delta_z])]
#         sig.append(law.GetInterfaceStress(delta)[2])
#         law.updateIrreversibleDamage()
#         delta_plot.append(delta_z)
#         # print(law.get_DamageVariable())

#     # for delta_z in np.arange(delta_I_max,-delta_I_max,-delta_I_max/nb_iter):
#     #     delta = [np.array([0]), np.array([0]), np.array([delta_z])]
#     #     sig.append(law.GetInterfaceStress(delta)[2])
#     #     law.updateIrreversibleDamage()
#     #     delta_plot.append(delta_z)

#     import matplotlib.pyplot as plt

#     plt.plot(delta_plot, sig)
