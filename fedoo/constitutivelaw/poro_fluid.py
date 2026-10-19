"""Fluid-phase constitutive parameters for poromechanics."""

import numpy as np

from fedoo.core.base import ConstitutiveLaw


class PoroFluidProperties(ConstitutiveLaw):
    """Fluid-phase parameters for a saturated porous medium.

    Holds the Biot coefficient, the Biot modulus, the intrinsic permeability,
    the fluid viscosity, and the initial Lagrangian porosity. The skeleton
    constitutive law (hyperelastic, viscoelastic, ...) is provided separately
    and remains unaware of the pore pressure. The Biot/Terzaghi coupling is
    handled at the weak form level (see :py:class:`fedoo.weakform.PoroMechanics`).

    Parameters
    ----------
    permeability : float or callable
        Intrinsic permeability ``k``. If callable, signature is
        ``k(J, sv) -> float or ndarray`` for deformation-dependent permeability
        (e.g. :py:class:`HolmesMowPermeability`, :py:class:`KozenyCarmanPermeability`).
    fluid_viscosity : float, default 1.0
        Dynamic viscosity ``mu_f`` of the pore fluid. The Darcy mobility used
        in the mass balance is ``k / mu_f``. Use ``mu_f = 1`` if
        ``permeability`` already represents the mobility ``k / mu_f``.
    biot_coefficient : float, default 1.0
        Biot coefficient ``alpha``. The default ``1.0`` is the Terzaghi limit
        (incompressible skeleton constituent), appropriate for most soft
        biological tissues.
    biot_modulus : float or None, default None
        Biot modulus ``M``. The storage coefficient is ``1 / M``. If ``None``,
        the fluid is treated as fully incompressible (``1 / M = 0``).
    initial_porosity : float, default 0.8
        Initial Lagrangian fluid volume fraction ``phi_f0`` in ``[0, 1]``.
    fluid_density : float, default 1000.0
        Reference fluid density. Stored but not used yet: reserved for a
        future fluid gravity/body-force term.
    name : str, default ""
    """

    def __init__(
        self,
        permeability,
        fluid_viscosity=1.0,
        biot_coefficient=1.0,
        biot_modulus=None,
        initial_porosity=0.8,
        fluid_density=1000.0,
        name="",
    ):
        ConstitutiveLaw.__init__(self, name)
        self.permeability = permeability
        self.fluid_viscosity = fluid_viscosity
        self.biot_coefficient = biot_coefficient
        self.biot_modulus = biot_modulus
        self.initial_porosity = initial_porosity
        self.fluid_density = fluid_density

    def get_mobility(self, J=None, sv=None):
        """Return the isotropic mobility tensor ``k(J) / mu_f`` as a 3x3 list.

        Parameters
        ----------
        J : ndarray or None
            Jacobian ``det(F)`` at gauss points. Required when
            ``permeability`` is a callable. Ignored for constant permeability.
        sv : dict or None
            State-variable dict (forwarded to callable permeability models
            that may depend on other internal variables).

        Returns
        -------
        list of list
            Mobility tensor ``[[k/mu, 0, 0], [0, k/mu, 0], [0, 0, k/mu]]``.
        """
        if callable(self.permeability):
            k = self.permeability(J, sv)
        else:
            k = self.permeability
        K_mob = k / self.fluid_viscosity
        return [[K_mob, 0, 0], [0, K_mob, 0], [0, 0, K_mob]]

    def get_storage(self):
        """Return the storage coefficient ``1 / M`` (zero if fluid incompressible)."""
        if self.biot_modulus is None:
            return 0.0
        return 1.0 / self.biot_modulus
