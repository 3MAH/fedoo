# derive de ConstitutiveLaw
# The elastoplastic law should be used with an InternalForce WeakForm

from fedoo.core.mechanical3d import Mechanical3D
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList

import numpy as np


class ElastoPlasticity(Mechanical3D):
    """
    Elasto-Plastic constitutive law.
    This law is based on the assumption of isotropic hardening with the Von-Mises plasticity criterion.
    After creating an ElastoPlasticity object, the hardening function must be set with the Method 'SetHardeningFunction'
    This constitutive Law should be associated with :mod:`fedoo.weakform.InternalForce`

    Parameters
    ----------
    YoungModulus: scalars or arrays of gauss point values
        Young modulus
    PoissonRatio: scalars or arrays of gauss point values
        Poisson's Ratio
    YieldStress: scalars or arrays of gauss point values
        Yield Stress Value
    name: str, optional
        The name of the constitutive law
    """

    def __init__(self, YoungModulus, PoissonRatio, YieldStress, name=""):
        # only scalar values of YoungModulus and PoissonRatio are possible
        Mechanical3D.__init__(self, name)  # heritage

        self.__YoungModulus = YoungModulus
        self.__PoissonRatio = PoissonRatio
        self.__YieldStress = YieldStress

        self.__P = None  # irrevesrible plasticity
        self.__currentP = None  # current iteration plasticity (reversible)
        self.__PlasticStrainTensor = None
        self.__currentPlasticStrainTensor = None
        self.__currentSigma = 0  # lissStressTensor object describing the last computed stress (GetStress method)
        self.__currentGradDisp = None
        self.__TangeantModuli = None

        self.__tol = 1e-6  # tolerance of Newton Raphson used to get the updated plasticity state (constutive law alogorithm)
        self.nlgeom = False  # will be updated with the Initialize function

    def GetYoungModulus(self):
        return self.__YoungModulus

    def GetPoissonRatio(self):
        return self.__PoissonRatio

    def GetYieldStress(self):
        return self.__YieldStress

    def SetNewtonRaphsonTolerance(self, tol):
        """
        Set the tolerance of the Newton Raphson algorithm used to get the updated plasticity state (constutive law alogorithm)
        """
        self.__tol = tol

    def GetHelas(self):
        H = np.zeros((6, 6), dtype="object")
        E = self.__YoungModulus
        nu = self.__PoissonRatio

        H[0, 0] = H[1, 1] = H[2, 2] = E * (
            1.0 / (1 + nu) + nu / ((1.0 + nu) * (1 - 2 * nu))
        )  # H1 = 2*mu+lamb
        H[0, 1] = H[0, 2] = H[1, 2] = E * (nu / ((1 + nu) * (1 - 2 * nu)))  # H2 = lamb
        H[3, 3] = H[4, 4] = H[5, 5] = 0.5 * E / (1 + nu)  # H3 = mu
        H[1, 0] = H[0, 1]
        H[2, 0] = H[0, 2]
        H[2, 1] = H[1, 2]  # symétrie

        return H

    def HardeningFunction(self, p):
        raise NameError(
            "Hardening function not defined. Use the method SetHardeningFunction"
        )

    def HardeningFunctionDerivative(self, p):
        raise NameError(
            "Hardening function not defined. Use the method SetHardeningFunction"
        )

    def SetHardeningFunction(self, FunctionType, **kargs):
        """
        Define the hardening function of the ElastoPlasticity law.
        FunctionType is the type of hardening function.

        For now, the only defined hardening function is a power law.
        * F = H*p^{beta} were p is the cumuled plasticity

        Other type of hardening function may be added in future versions.

        Parameters
        ----------
        FunctionType: str
            Type of hardening function.
            For now, the only possible value is 'power' for power law.
        H (keyword argument): scalar
        beta(keyword argument): scalar
        name: str, optional
            The name of the constitutive law

        """
        if FunctionType.lower() == "power":
            H = None
            beta = None
            for item in kargs:
                if item.lower() == "h":
                    H = kargs[item]
                if item.lower() == "beta":
                    beta = kargs[item]

            if H is None:
                raise NameError("Keyword arguments 'H' missing")
            if beta is None:
                raise NameError("Keyword arguments 'beta' missing")

            def HardeningFunction(p):
                return H * p**beta

            def HardeningFunctionDerivative(p):
                return np.nan_to_num(
                    beta * H * p ** (beta - 1), posinf=1
                )  # replace inf value by 1

        elif FunctionType.lower() == "user":
            HardeningFunction = None
            HardeningFunctionDerivative = None
            for item in kargs:
                if item.lower() == "hardeningfunction":
                    HardeningFunction = kargs[item]
                if item.lower() == "hardeningfunctionderivative":
                    HardeningFunctionDerivative = kargs[item]

            if HardeningFunction is None:
                raise NameError("Keyword arguments 'HardeningFunction' missing")
            if HardeningFunctionDerivative is None:
                raise NameError(
                    "Keyword arguments 'HardeningFunctionDerivative' missing"
                )

        self.HardeningFunction = HardeningFunction
        self.HardeningFunctionDerivative = HardeningFunctionDerivative

    def YieldFunction(self, Stress, p):
        return Stress.vonMises() - self.__YieldStress - self.HardeningFunction(p)

    def YieldFunctionDerivativeSigma(self, sigma):
        """
        Derivative of the Yield Function with respect to the stress tensor defined in sigma
        sigma should be a StressTensorList object
        """
        return StressTensorList(
            (3 / 2) * np.array(sigma.deviatoric()) / sigma.vonMises()
        ).toStrain()

    def GetPlasticity(self):
        return self.__currentP

    def get_strain(self, **kargs):
        return self.__currentPlasticStrainTensor

    def get_pk2(self):
        return self.__currentSigma

    def get_cauchy(self, **kargs):  # same as GetPKII
        # alias of GetPKII mainly use for small strain displacement problems
        return self.__currentSigma

    def get_stress(self, **kargs):  # same as GetPKII
        # alias of GetPKII mainly use for small strain displacement problems
        return self.__currentSigma

    def get_disp_grad(self):
        return self.__currentGradDisp

    def get_tangent_matrix(self):
        return self.__TangeantModuli

    # def GetStressOperator(self, localFrame=None):
    #     H = self.GetH()

    #     eps, eps_vir = GetStrainOperator(self.__currentGradDisp)
    #     sigma = [sum([0 if eps[j] is 0 else eps[j]*H[i][j] for j in range(6)]) for i in range(6)]

    #     return sigma # list de 6 objets de type DiffOp

    def NewTimeIncrement(self):
        # Set Irreversible Plasticity
        if self.__P is not None:
            self.__P = self.__currentP.copy()
            self.__PlasticStrainTensor = self.__currentPlasticStrainTensor.copy()
        self.__TangeantModuli = self.GetHelas()

    def to_start(self):
        if self.__P is None:
            self.__currentP = None
            self.__currentPlasticStrainTensor = None
        else:
            self.__currentP = self.__P.copy()
            self.__currentPlasticStrainTensor = self.__PlasticStrainTensor.copy()
        self.__TangeantModuli = self.GetHelas()

    def reset(self):
        """
        reset the constitutive law (time history)
        """
        self.__P = None  # irrevesrible plasticity
        self.__currentP = None  # current iteration plasticity (reversible)
        self.__PlasticStrainTensor = None
        self.__currentPlasticStrainTensor = None
        self.__currentSigma = 0  # lissStressTensor object describing the last computed stress (GetStress method)
        self.__TangeantModuli = self.GetHelas()

    def initialize(self, assembly, pb, t0=0.0, nlgeom=False):
        if self._dimension is None:
            self._dimension = assembly.space.get_dimension()
        self.NewTimeIncrement()
        self.nlgeom = nlgeom

    def update(self, assembly, pb, time):
        displacement = pb.get_dof_solution()

        if np.isscalar(displacement) and displacement == 0:
            self.__currentGradDisp = 0
            self.__currentSigma = 0
        else:
            self.__currentGradDisp = assembly.get_grad_disp(displacement, "GaussPoint")
            GradValues = self.__currentGradDisp
            if self.nlgeom == False:
                Strain = [GradValues[i][i] for i in range(3)]
                Strain += [
                    GradValues[0][1] + GradValues[1][0],
                    GradValues[0][2] + GradValues[2][0],
                    GradValues[1][2] + GradValues[2][1],
                ]
            else:
                Strain = [
                    GradValues[i][i]
                    + 0.5 * sum([GradValues[k][i] ** 2 for k in range(3)])
                    for i in range(3)
                ]
                Strain += [
                    GradValues[0][1]
                    + GradValues[1][0]
                    + sum([GradValues[k][0] * GradValues[k][1] for k in range(3)])
                ]
                Strain += [
                    GradValues[0][2]
                    + GradValues[2][0]
                    + sum([GradValues[k][0] * GradValues[k][2] for k in range(3)])
                ]
                Strain += [
                    GradValues[1][2]
                    + GradValues[2][1]
                    + sum([GradValues[k][1] * GradValues[k][2] for k in range(3)])
                ]

            TotalStrain = StrainTensorList(Strain)
            self.ComputeStress(
                TotalStrain, time
            )  # compute the total stress in self.__currentSigma

            # print(self.__currentP)

            dphi_dp = self.HardeningFunctionDerivative(self.__currentP)
            dphi_dsigma = self.YieldFunctionDerivativeSigma(self.__currentSigma)
            Lambda = dphi_dsigma  # for isotropic hardening only
            test = self.YieldFunction(self.__currentSigma, self.__P) > self.__tol

            Helas = self.GetHelas()
            ##### Compute new tangeant moduli
            B = sum(
                [
                    sum([dphi_dsigma[j] * Helas[i][j] for j in range(6)]) * Lambda[i]
                    for i in range(6)
                ]
            )
            Ap = B - dphi_dp
            CL = [
                sum([Lambda[j] * Helas[i][j] for j in range(6)]) for i in range(6)
            ]  # [C:Lambda]
            Peps = [
                sum([dphi_dsigma[i] * Helas[i][j] for i in range(6)]) / Ap
                for j in range(6)
            ]  # Peps
            #        TangeantModuli = [[Helas[i][j] - CL[i]*Peps[j] for j in range(6)] for i in range(6)]
            self.__TangeantModuli = [
                [Helas[i][j] - (CL[i] * Peps[j] * test) for j in range(6)]
                for i in range(6)
            ]
            ##### end Compute new tangeant moduli

    def ComputeStress(self, StrainTensor, time=None):
        # time not used here because this law require no time effect
        # initilialize values plasticity variables if required
        if self.__P is None:
            self.__P = np.zeros(len(StrainTensor[0]))
            self.__currentP = np.zeros(len(StrainTensor[0]))
        if self.__PlasticStrainTensor is None:
            self.__PlasticStrainTensor = StrainTensorList(
                np.zeros((6, len(StrainTensor[0])))
            )
            self.__currentPlasticStrainTensor = StrainTensorList(
                np.zeros((6, len(StrainTensor[0])))
            )

        H = (
            self.GetHelas()
        )  # no change of basis because only isotropic behavior are considered
        sigma = StressTensorList(
            [
                sum(
                    [
                        (StrainTensor[j] - self.__PlasticStrainTensor[j]) * H[i][j]
                        for j in range(6)
                    ]
                )
                for i in range(6)
            ]
        )
        test = self.YieldFunction(sigma, self.__P) > self.__tol
        #        print(sum(test)/len(test)*100)

        sigmaFull = np.array(sigma).T
        Ep = np.array(self.__PlasticStrainTensor).T
        #        Ep = np.array(self.__currentPlasticStrainTensor).T

        for pg in range(len(sigmaFull)):
            if test[pg] > 0:
                sigma = StressTensorList(sigmaFull[pg])
                p = self.__P[pg]
                iter = 0
                while abs(self.YieldFunction(sigma, p)) > self.__tol:
                    dphi_dp = self.HardeningFunctionDerivative(p)
                    dphi_dsigma = np.array(self.YieldFunctionDerivativeSigma(sigma))

                    Lambda = dphi_dsigma  # for associated plasticity
                    B = sum(
                        [
                            sum([dphi_dsigma[j] * H[i][j] for j in range(6)])
                            * Lambda[i]
                            for i in range(6)
                        ]
                    )

                    dp = self.YieldFunction(sigma, p) / (B - dphi_dp)
                    p += dp
                    Ep[pg] += Lambda * dp
                    sigma = StressTensorList(
                        [
                            sum(
                                [
                                    (StrainTensor[j][pg] - Ep[pg][j]) * H[i][j]
                                    for j in range(6)
                                ]
                            )
                            for i in range(6)
                        ]
                    )
                self.__currentP[pg] = p
                sigmaFull[pg] = sigma

        self.__currentPlasticStrainTensor = StrainTensorList(Ep.T)
        self.__currentSigma = StressTensorList(sigmaFull.T)  # list of 6 objets

        return self.__currentSigma
