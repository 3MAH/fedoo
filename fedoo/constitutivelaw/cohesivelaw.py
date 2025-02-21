# derive de ConstitutiveLaw

from fedoo.constitutivelaw.spring import Spring
from fedoo.core.base import ConstitutiveLaw
from fedoo.core.base import AssemblyBase
import numpy as np
from numpy import linalg


class CohesiveLaw(Spring):
    """
    Bilinear cohesive Law based on the Crisfield model

    This constitutive Law should be associated with :mod:`fedoo.weakform.InterfaceForce`

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
        axis should be eiter 0,1 or 2 (default). It define the normal direction to the failure plane the is used for mode identification. The axis is defined in local coordinate system.
    name: str, optional
        The name of the constitutive law
    """

    # Use with WeakForm.InterfaceForce
    def __init__(
        self, GIc=0.3, SImax=60, KI=1e4, GIIc=1.6, SIImax=None, KII=5e4, axis=2, name=""
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
        self.parameters = {
            "GIc": GIc,
            "SImax": SImax,
            "KI": KI,
            "GIIc": GIIc,
            "SIImax": SIImax,
            "KII": KII,
            "axis": axis,
        }

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

    def get_tangent_matrix(self, assembly):
        Umd = 1 - assembly.sv["DamageVariable"]
        UmdI = 1 - assembly.sv["DamageVariableOpening"]

        axis = self.parameters["axis"]

        Kt = Umd * self.parameters["KII"]
        Kn = UmdI * self.parameters["KI"]
        Kdiag = [Kt if i != axis else Kn for i in range(3)]
        return [[Kdiag[0], 0, 0], [0, Kdiag[1], 0], [0, 0, Kdiag[2]]]

        #     return [[Kdiag[0], 0, 0], [0, Kdiag[1], 0], [0,0,Kdiag[2]]]
        # if get_Dimension() == "3D":        # tester si marche avec contrainte plane ou def plane
        #     Kdiag = [Umd*self.parameters['KII'] if i != axis else UmdI*self.parameters['KI'] for i in range(3)]
        #     return [[Kdiag[0], 0, 0], [0, Kdiag[1], 0], [0,0,Kdiag[2]]]
        # else:
        #     Kdiag = [Umd*self.parameters['KII'] if i != axis else UmdI*self.parameters['KI'] for i in range(2)]
        #     return [[Kdiag[0], 0], [0, Kdiag[1]]]

    def get_K(self, assembly):
        return self.local2global_K(self.get_tangent_matrix(assembly))

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

        op_delta = assembly.space.op_disp()
        if type_data == "Node":
            delta = [assembly.get_node_results(op, U) for op in op_delta]
        else:
            delta = [assembly.get_gp_results(op, U) for op in op_delta]

        self._update_damage(assembly, delta)

        if irreversible:
            assembly.sv["DamageVariableIrreversible"] = assembly.sv[
                "DamageVariable"
            ].copy()

    def _update_damage(self, assembly, delta):
        alpha = 2  # for the power low
        if (
            np.isscalar(assembly.sv["DamageVariable"])
            and assembly.sv["DamageVariable"] == 0
        ):
            assembly.sv["DamageVariable"] = 0 * delta[0]
        if (
            np.isscalar(assembly.sv["DamageVariableOpening"])
            and assembly.sv["DamageVariableOpening"] == 0
        ):
            assembly.sv["DamageVariableOpening"] = 0 * delta[0]

        delta_n = delta[self.parameters["axis"]]
        delta_t = [d for i, d in enumerate(delta) if i != self.parameters["axis"]]
        if len(delta_t) == 1:
            delta_t = delta_t[0]
        else:
            delta_t = np.sqrt(delta_t[0] ** 2 + delta_t[1] ** 2)

        # mode I
        delta_0_I = (
            self.parameters["SImax"] / self.parameters["KI"]
        )  # critical relative displacement (begining of the damage)
        delta_m_I = (
            2 * self.parameters["GIc"] / self.parameters["SImax"]
        )  # maximal relative displacement (total failure)

        # mode II
        SIImax = self.parameters["SIImax"]
        if SIImax == None:
            SIImax = self.parameters["SImax"] * np.sqrt(
                self.parameters["GIIc"] / self.parameters["GIc"]
            )  # value by default used mainly to treat mode I dominant problems
        delta_0_II = SIImax / self.parameters["KII"]
        delta_m_II = 2 * self.parameters["GIIc"] / SIImax

        t0 = 0.0 * delta_n
        tm = 0.0 * delta_n
        dta = 0.0 * delta_n

        test = delta_n > 0  # test if traction loading (opening mode)
        ind_traction = np.nonzero(test)[
            0
        ]  # indice of value where delta_n > 0 ie traction loading
        ind_compr = np.nonzero(test - 1)[0]

        #        beta = 0*delta_n
        beta = (
            delta_t[ind_traction] / delta_n[ind_traction]
        )  # le rapport de mixité de mode

        t0[ind_traction] = (delta_0_II * delta_0_I) * (
            np.sqrt((1 + (beta**2)) / ((delta_0_II**2) + ((beta * delta_0_I) ** 2)))
        )  # Critical relative displacement in mixed mode
        t0[ind_compr] = (
            delta_0_II  # Critical relative displacement in mixed mode (only mode II)
        )

        tm[ind_traction] = (2 * ((1 + beta) ** 2) / t0[ind_traction]) * (
            (
                ((self.parameters["KI"] / self.parameters["GIc"]) ** alpha)
                + (
                    ((self.parameters["KII"] * beta**2) / self.parameters["GIIc"])
                    ** alpha
                )
            )
            ** (-1 / alpha)
        )  # Maximal relative displacement in mixed mode (power low criterion)
        tm[ind_compr] = (
            delta_m_II  # Maximal relative displacement in mixed mode (power low criterion)
        )

        dta[ind_traction] = np.sqrt(
            delta_t[ind_traction] ** 2 + delta_n[ind_traction] ** 2
        )  # Actual relative displacement in mixed mode
        dta[ind_compr] = delta_t[
            ind_compr
        ]  # Actual relative displacement in mixed mode

        # ---------------------------------------------------------------------------------------------------------------
        # La variable d'endommagement "d"
        # ---------------------------------------------------------------------------------------------------------------
        d = (dta >= tm).astype(float)  # initialize d to 1 if dta>tm and else d=0
        test = np.nonzero((dta > t0) * (dta < tm))[
            0
        ]  # indices where dta>t0 and dta<tm ie d should be between 0 and 1

        d[test] = (tm[test] / (tm[test] - t0[test])) * (1 - (t0[test] / dta[test]))

        if (
            np.isscalar(assembly.sv["DamageVariableIrreversible"])
            and assembly.sv["DamageVariableIrreversible"] == 0
        ):
            assembly.sv["DamageVariable"] = (
                d  # I don't know why assembly.sv['DamageVariable'] = d end up in bads values
            )
        else:
            assembly.sv["DamageVariable"] = np.max(
                [assembly.sv["DamageVariableIrreversible"], d], axis=0
            )

        assembly.sv["DamageVariableOpening"] = (
            (delta_n > 0) * assembly.sv["DamageVariable"]
        )  # for opening the damage in considered to 0 when the relative displacement is negative (conctact)

        # verification : the damage variable should be between 0 and 1
        if (
            assembly.sv["DamageVariable"].min() < 0
            or assembly.sv["DamageVariable"].max() > 1
        ):
            print("Warning : the value of damage variable is incorrect")

    def reset(self):
        pass

    def set_start(self, assembly, pb):
        # Set Irreversible Damage
        self.update_irreversible_damage(assembly)

    # def to_start(self, assembly, pb):
    #     #Damage variable will be recompute. NPOthing to be done here (to be checked)
    #     pass

    def update(self, assembly, pb):
        displacement = pb.get_dof_solution()
        K = self.get_K()
        assembly.sv["TangentMatrix"] = K
        if np.isscalar(displacement) and displacement == 0:
            assembly.sv["InterfaceStress"] = assembly.sv["RelativeDisp"] = 0
        else:
            op_delta = (
                assembly.space.op_disp()
            )  # relative displacement = disp if used with cohesive element
            delta = [assembly.get_gp_results(op, displacement) for op in op_delta]
            assembly.sv["RelativeDisp"] = delta

            # Compute interface stress
            dim = len(delta)
            assembly.sv["InterfaceStress"] = [
                sum([delta[j] * K[i][j] for j in range(dim)]) for i in range(dim)
            ]  # list of 3 objects

    def update(self, assembly, pb):
        displacement = pb.get_dof_solution()

        if np.isscalar(displacement) and displacement == 0:
            assembly.sv["InterfaceStress"] = assembly.sv["RelativeDisp"] = 0
            K = self.get_K()
        else:
            op_delta = (
                assembly.space.op_disp()
            )  # relative displacement = disp if used with cohesive element
            delta = [assembly.get_gp_results(op, displacement) for op in op_delta]
            assembly.sv["RelativeDisp"] = delta

            self._update_damage(assembly, delta)
            dim = len(delta)
            K = self.get_K(assembly)
            assembly.sv["InterfaceStress"] = [
                sum([delta[j] * K[i][j] for j in range(dim)]) for i in range(dim)
            ]  # list of 3 objects

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
