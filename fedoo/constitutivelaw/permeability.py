"""Strain-dependent permeability models for poromechanics."""

import numpy as np


class HolmesMowPermeability:
    """Strain-dependent permeability of Holmes and Mow.

    ``k(J) = k0 * ((J - n0) / (1 - n0))^kappa * exp(M * (J^2 - 1) / 2)``

    Standard model for articular cartilage. Permeability drops in compression
    (J < 1) and increases in tension (J > 1).

    Reference: Holmes M.H. and Mow V.C., 1990, *J. Biomech.*

    Parameters
    ----------
    k0 : float
        Reference permeability at ``J = 1``.
    n0 : float
        Initial solid volume fraction ``(1 - phi_f0)``.
    kappa : float, default 2.0
        Porosity power-law exponent.
    M : float, default 4.638
        Exponential strain-stiffening coefficient.
    """

    def __init__(self, k0, n0, kappa=2.0, M=4.638):
        self.k0 = k0
        self.n0 = n0
        self.kappa = kappa
        self.M = M

    def __call__(self, J, sv=None):
        if J is None:
            return self.k0
        ratio = np.maximum((J - self.n0) / (1.0 - self.n0), 1e-12)
        return self.k0 * (ratio**self.kappa) * np.exp(0.5 * self.M * (J * J - 1.0))


class KozenyCarmanPermeability:
    """Kozeny-Carman permeability driven by current porosity ``phi(J)``.

    ``k(phi) = k0 * (phi^3 / (1 - phi)^2) / (phi0^3 / (1 - phi0)^2)``
    with ``phi(J) = 1 - (1 - phi0) / J`` from solid mass conservation.

    Parameters
    ----------
    k0 : float
        Reference permeability at ``J = 1``.
    phi0 : float
        Initial fluid volume fraction ``phi_f0`` in ``(0, 1)``.
    """

    def __init__(self, k0, phi0):
        self.k0 = k0
        self.phi0 = phi0
        self._norm = (phi0**3) / max((1.0 - phi0) ** 2, 1e-12)

    def __call__(self, J, sv=None):
        if J is None:
            return self.k0
        phi = 1.0 - (1.0 - self.phi0) / np.maximum(J, 1e-12)
        phi = np.clip(phi, 1e-12, 1.0 - 1e-12)
        return self.k0 * (phi**3) / ((1.0 - phi) ** 2) / self._norm
