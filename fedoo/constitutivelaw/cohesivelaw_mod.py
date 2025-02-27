# derive de ConstitutiveLaw
# Not working law

from fedoo.constitutivelaw.spring import Spring
from fedoo.core.base import ConstitutiveLaw
from fedoo.core.base import AssemblyBase
import numpy as np
from numpy import linalg


class CohesiveLaw_mod(Spring):
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

        ConstitutiveLaw.__init__(self, name)  # heritage
        self.__DamageVariable = 0  # damage variable
        self.__DamageVariableOpening = 0  # DamageVariableOpening is used for the opening mode (mode I). It is equal to DamageVariable in traction and equal to 0 in compression (soft contact law)
        self.__DamageVariableIrreversible = (
            0  # irreversible damage variable used for time evolution
        )
        self.__parameters = {
            "GIc": GIc,
            "SImax": SImax,
            "KI": KI,
            "GIIc": GIIc,
            "SIImax": SIImax,
            "KII": KII,
            "axis": axis,
        }
        self.__currentInterfaceStress = None

    def GetKelas(self):  # Get elastic rigidity in local coordinates
        Umd = 1 - self.__DamageVariable
        UmdI = 1 - self.__DamageVariableOpening

        axis = self.__parameters["axis"]

        Kt = Umd * self.__parameters["KII"]
        Kn = UmdI * self.__parameters["KI"]
        Kdiag = [Kt if i != axis else Kn for i in range(3)]
        return [[Kdiag[0], 0, 0], [0, Kdiag[1], 0], [0, 0, Kdiag[2]]]

        # if get_Dimension() == "3D":        # tester si marche avec contrainte plane ou def plane
        #     Kdiag = [Umd*self.__parameters['KII'] if i != axis else UmdI*self.__parameters['KI'] for i in range(3)]
        #     return [[Kdiag[0], 0, 0], [0, Kdiag[1], 0], [0,0,Kdiag[2]]]
        # else:
        #     Kdiag = [Umd*self.__parameters['KII'] if i != axis else UmdI*self.__parameters['KI'] for i in range(2)]
        #     return [[Kdiag[0], 0], [0, Kdiag[1]]]

    def get_tangent_matrix(self):  # Get tangent moduli
        if self.__currentInterfaceStress is None:
            return self.GetKelas()

        Umd = 1 - self.__DamageVariable
        UmdI = 1 - self.__DamageVariableOpening

        KIelas = UmdI * self.__parameters["KI"]
        KIIelas = Umd * self.__parameters["KII"]

        ### Cohesive Zones Data
        # mode I
        delta_0_I = (
            self.__parameters["SImax"] / self.__parameters["KI"]
        )  # critical relative displacement (begining of the damage)
        delta_m_I = (
            2 * self.__parameters["GIc"] / self.__parameters["SImax"]
        )  # maximal relative displacement (total failure)

        # mode II
        SIImax = self.__parameters["SIImax"]
        if SIImax == None:
            SIImax = self.__parameters["SImax"] * np.sqrt(
                self.__parameters["GIIc"] / self.__parameters["GIc"]
            )  # value by default used mainly to treat mode I dominant problems
        delta_0_II = SIImax / self.__parameters["KII"]
        delta_m_II = 2 * self.__parameters["GIIc"] / SIImax

        # KItangent = -self.__parameters['SImax']/(delta_m_I-delta_0_I)
        # KIItangent = -SIImax/(delta_m_II-delta_0_II)
        KItangent = 0
        KIItangent = 0

        # modeI :
        test = (self.__DamageVariableOpening - self.__DamageVariableIrreversible) > 0
        test2 = np.logical_not(test)
        test = np.logical_and(test, self.__DamageVariableOpening != 1)
        # test2 = (self.__DamageVariable == 1)
        KI = KIelas * test2 + KItangent * test

        # modeII :
        test = (self.__DamageVariable - self.__DamageVariableIrreversible) > 0
        test2 = np.logical_not(test)
        test = np.logical_and(test, self.__DamageVariable != 1)
        KII = KIIelas * test2 + KIItangent * test

        axis = self.__parameters["axis"]

        Kdiag = [KII if i != axis else KI for i in range(3)]
        return [[Kdiag[0], 0, 0], [0, Kdiag[1], 0], [0, 0, Kdiag[2]]]

        # if get_Dimension() == "3D":        # tester si marche avec contrainte plane ou def plane
        #     Kdiag = [KII if i != axis else KI for i in range(3)]
        #     return [[Kdiag[0], 0, 0], [0, Kdiag[1], 0], [0,0,Kdiag[2]]]
        # else:
        #     Kdiag = [KII if i != axis else KI for i in range(2)]
        #     return [[Kdiag[0], 0], [0, Kdiag[1]]]

    def set_DamageVariable(self, value):
        self.__DamageVariable = value

    def get_DamageVariable(self):
        return self.__DamageVariable

    def updateIrreversibleDamage(self):
        if np.isscalar(self.__DamageVariable) and self.__DamageVariable == 0:
            self.__DamageVariableIrreversible = 0
        else:
            self.__DamageVariableIrreversible = self.__DamageVariable.copy()

    #### Not working, need update
    def updateDamageVariable(
        self, CohesiveAssembly, U, Irreversible=False, typeData="PG"
    ):
        # Delta is the relative displacement
        # OperatorDelta  = assembly.space.op_disp() #relative displacement = disp if used with cohesive element
        # OperatorDelta, U_vir = get_DispOperator()
        if isinstance(CohesiveAssembly, str):
            CohesiveAssembly = AssemblyBase.get_all()[CohesiveAssembly]
        if typeData == "Node":
            delta = [CohesiveAssembly.get_node_results(op, U) for op in OperatorDelta]
        else:
            delta = [CohesiveAssembly.get_gp_results(op, U) for op in OperatorDelta]

        self.__UpdateDamageVariable(delta)

        if Irreversible == True:
            self.__DamageVariableIrreversible = self.__DamageVariable.copy()

    def __UpdateDamageVariable(self, delta):
        alpha = 2  # for the power low
        if np.isscalar(self.__DamageVariable) and self.__DamageVariable == 0:
            self.__DamageVariable = 0 * delta[0]
        if (
            np.isscalar(self.__DamageVariableOpening)
            and self.__DamageVariableOpening == 0
        ):
            self.__DamageVariableOpening = 0 * delta[0]

        # delta_n = delta.pop(self.__parameters['axis'])
        # if get_Dimension() == "3D":
        #     delta_t = np.sqrt(delta[0]**2 + delta[1]**2)
        # else: delta_t = delta[0]

        delta_n = delta[self.__parameters["axis"]]
        delta_t = [d for i, d in enumerate(delta) if i != self.__parameters["axis"]]

        if len(delta_t) == 1:
            delta_t = delta_t[0]
        else:
            delta_t = np.sqrt(delta_t[0] ** 2 + delta_t[1] ** 2)

        if get_Dimension() == "3D":
            delta_t = np.sqrt(delta_t[0] ** 2 + delta_t[1] ** 2)
        else:
            delta_t = delta_t[0]

        ### Cohesive Zones Data
        # mode I
        delta_0_I = (
            self.__parameters["SImax"] / self.__parameters["KI"]
        )  # critical relative displacement (begining of the damage)
        delta_m_I = (
            2 * self.__parameters["GIc"] / self.__parameters["SImax"]
        )  # maximal relative displacement (total failure)

        # mode II
        SIImax = self.__parameters["SIImax"]
        if SIImax == None:
            SIImax = self.__parameters["SImax"] * np.sqrt(
                self.__parameters["GIIc"] / self.__parameters["GIc"]
            )  # value by default used mainly to treat mode I dominant problems
        delta_0_II = SIImax / self.__parameters["KII"]
        delta_m_II = 2 * self.__parameters["GIIc"] / SIImax

        # Compute mixed mode relative displacement (actual values and limit values)

        test = delta_n > 0  # test if traction loading (opening mode)
        ind_traction = np.nonzero(test)[
            0
        ]  # indice of value where delta_n > 0 ie traction loading
        ind_compr = np.nonzero(test - 1)[0]

        t0 = 0.0 * delta_n
        tm = 0.0 * delta_n
        dta = 0.0 * delta_n

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
                ((self.__parameters["KI"] / self.__parameters["GIc"]) ** alpha)
                + (
                    ((self.__parameters["KII"] * beta**2) / self.__parameters["GIIc"])
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
            np.isscalar(self.__DamageVariableIrreversible)
            and self.__DamageVariableIrreversible == 0
        ):
            self.__DamageVariable = (
                d  # I don't know why self.__DamageVariable = d end up in bads values
            )
        else:
            self.__DamageVariable = np.max(
                [self.__DamageVariableIrreversible, d], axis=0
            )

        self.__DamageVariableOpening = (
            (delta_n > 0) * self.__DamageVariable
        )  # for opening the damage in considered to 0 when the relative displacement is negative (conctact)

        # verification : the damage variable should be between 0 and 1
        if self.__DamageVariable.min() < 0 or self.__DamageVariable.max() > 1:
            print("Warning : the value of damage variable is incorrect")

    def NewTimeIncrement(self):
        # Set Irreversible Damage
        self.updateIrreversibleDamage()
        self.__currentSigma = None

    def to_start(self):
        # Damage variable and currentInterfaceStress will be recomputed in the next call of GetInterfaceStress
        self.__currentInterfaceStress = None

    def reset(self):
        """
        reset the constitutive law (time history)
        """
        self.__DamageVariable = 0  # damage variable
        self.__DamageVariableOpening = 0  # DamageVariableOpening is used for the opening mode (mode I). It is equal to DamageVariable in traction and equal to 0 in compression (soft contact law)
        self.__DamageVariableIrreversible = (
            0  # irreversible damage variable used for time evolution
        )

    def GetInterfaceStress(self, Delta, time=None):
        # Delta is the relative displacement vector
        self.__UpdateDamageVariable(Delta)
        self.__currentInterfaceStress = Spring.GetInterfaceStress(self, Delta, time)
        return self.__currentInterfaceStress


#    def SetLocalFrame(self, localFrame):
#        raise NameError("Not implemented: localFrame are not implemented in the context of cohesive laws")


# def __UpdateDamageVariable_old(self, delta):
#     #---------------------------------------------------------------------------------------------------------
#     ################# interface 90°/0° (Lower interface) ########################
#     #---------------------------------------------------------------------------------------------------------


#     alpha = 2 #for the power low
#     if np.isscalar(self.__DamageVariable) and self.__DamageVariable == 0: self.__DamageVariable = 0*delta[0]
#     if np.isscalar(self.__DamageVariableOpening) and self.__DamageVariableOpening == 0: self.__DamageVariableOpening  = 0*delta[0]

#     # delta_n = delta.pop(self.__parameters['axis'])
#     # delta_t = np.sqrt(delta[0]**2 + delta[1]**2)
#     delta_n = delta[self.__parameters['axis']]
#     delta_t = [d for i,d in enumerate(delta) if i != self.__parameters['axis'] ]
#     if get_Dimension() == "3D":
#         delta_t = np.sqrt(delta_t[0]**2 + delta_t[1]**2)
#     else: delta_t = delta_t[0]

#     # mode I
#     delta_0_I = self.__parameters['SImax'] / self.__parameters['KI']   # critical relative displacement (begining of the damage)
#     delta_m_I =  2*self.__parameters['GIc'] / self.__parameters['SImax']   # maximal relative displacement (total failure)

#     # mode II
#     SIImax = self.__parameters['SIImax']
#     if SIImax == None: SIImax = self.__parameters['SImax'] * np.sqrt(self.__parameters['GIIc'] / self.__parameters['GIc'])   #value by default used mainly to treat mode I dominant problems
#     delta_0_II = SIImax / self.__parameters['KII']
#     delta_m_II =  2*self.__parameters['GIIc'] / SIImax

#     for i in range (len(delta_n)):
#         if delta_n[i] > 0 :
#             beta= delta_t[i] / (delta_n[i]) # le rapport de mixité de mode

#             t0= (delta_0_II * delta_0_I) * (np.sqrt((1+ (beta**2)) / ((delta_0_II**2)+((beta*delta_0_I)**2)))) # Critical relative displacement in mixed mode

#             tm= (2*((1+ beta)**2)/t0) * ((((self.__parameters['KI']/self.__parameters['GIc'])**alpha) + \
#                     (((self.__parameters['KII']*beta**2)/self.__parameters['GIIc'])**alpha))**(-1/alpha)) #Maximal relative displacement in mixed mode (power low criterion)

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

#         if (self.__DamageVariableIrreversible is 0) or (di >  self.__DamageVariableIrreversible[i]):
#             self.__DamageVariable[i] = di
#         else: self.__DamageVariable[i] = self.__DamageVariableIrreversible[i]

#         if delta_n[i] > 0 :
#             self.__DamageVariableOpening [i] = self.__DamageVariable[i]
#         else :
#             self.__DamageVariableOpening [i] = 0

#     # verification : the damage variable should be between 0 and 1
#     if self.__DamageVariable.min() < 0 or self.__DamageVariable.max() > 1 :
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
