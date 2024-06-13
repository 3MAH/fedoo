# derive de ConstitutiveLaw
####WARNING: not working constitutive law

from fedoo.core.mechanical3d import Mechanical3D
import scipy as sp


class ViscoElasticComposites(Mechanical3D):
    def __init__(
        self,
        EL,
        ET,
        GLT,
        GTT,
        nuLT,
        nuTT,
        CL=0,
        CT=0,
        CLT=0,
        RefStrainRate=1,
        SLc_T=None,
        SLc_C=None,
        SYc_T=None,
        SYc_C=None,
        SZc_T=None,
        SZc_C=None,
        SLYc=None,
        SLZc=None,
        name="",
    ):
        Mechanical3D.__init__(self, name)  # heritage

        self.__parameters = {
            "EL": EL,
            "ET": ET,
            "GLT": GLT,
            "GTT": GTT,
            "nuLT": nuLT,
            "nuTT": nuTT,
            "CL": CL,
            "CT": CT,
            "CLT": CLT,
            "RefStrainRate": RefStrainRate,
            "SLc_T": SLc_T,
            "SLc_C": SLc_C,
            "SYc_T": SYc_T,
            "SYc_C": SYc_C,
            "SZc_T": SZc_T,
            "SZc_C": SZc_C,
            "SLYc": SLYc,
            "SLZc": SLZc,
        }

        self.__DamageVariable = [0, 0, 0, 0, 0, 0]

    def SetStrainRate(self, StrainRate):
        self.__StrainRate = StrainRate

    def get_stress(self, localFrame=None):  # methode virtuel
        # tester si contrainte plane ou def plane
        # if get_Dimension() == "2Dstress":
        #     print('ViscoElasticComposites law for 2Dstress is not implemented')
        #     return NotImplemented

        for key in self.__parameters:
            exec(key + '= self.__parameters["' + key + '"]')
        StrainRate = self.__StrainRate

        if isinstance(EL, (float, int, np.number)):
            H = sp.empty((6, 6))
        elif isinstance(EL, (sp.ndarray, list)):
            H = sp.zeros((6, 6, len(EL)))
        else:
            H = sp.zeros((6, 6), dtype="object")

        d1, d2, d3, d4, d5, d6 = self.__DamageVariables

        StrainRateEffect = [
            (StrainRate[i] >= RefStrainRate) * np.log(StrainRate[i] / RefStrainRate)
            for i in range(6)
        ]

        nuTL = nuLT * ET / EL
        k = 1 - nuTT**2 - 2 * nuLT * nuTL - 2 * nuLT * nuTT * nuTL
        H[0, 0] = ((1 - d1) * EL * (1 + CL * StrainRateEffect[0])) * (1 - nuTT**2) / k
        H[1, 1] = (
            ((1 - d2) * ET * (1 + CT * StrainRateEffect[1])) * (1 - nuLT * nuTL) / k
        )
        H[2, 2] = (
            ((1 - d3) * ET * (1 + CT * StrainRateEffect[2])) * (1 - nuLT * nuTL) / k
        )
        H[0, 1] = H[1, 0] = H[0, 2] = H[2, 0] = (
            ((1 - d1) * EL * (1 + CL * StrainRateEffect[0])) * (nuTT * nuTL + nuTL) / k
        )
        H[1, 2] = H[2, 1] = (
            ((1 - d2) * ET * (1 + CT * StrainRateEffect[1])) * (nuLT * nuTL + nuTT) / k
        )
        H[3, 3] = (1 - d4) * GLT * (1 + CLT * StrainRateEffect[3])
        H[4, 4] = (1 - d5) * GLT * (1 + CLT * StrainRateEffect[4])
        H[5, 5] = (1 - d6) * GTT * (1 + CLT * StrainRateEffect[5])

        H = self._ConsitutiveLaw__ChangeBasisH(H)

        # eps, eps_vir = GetStrainOperator()
        sigma = [sum([eps[j] * H[i][j] for j in range(6)]) for i in range(6)]

        return sigma  # list de 6 objets de type DiffOp

    def updateDamage(self):
        for key in self.__parameters:
            exec(key + '= self.__parameters["' + key + '"]')
        return NotImplemented
