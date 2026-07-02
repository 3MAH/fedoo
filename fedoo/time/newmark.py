from fedoo.time.generalized_alpha import (
    GeneralizedAlpha,
    GeneralizedAlphaDissipationTerm,
    GeneralizedAlphaStiffnessTerm,
    GeneralizedAlphaStorageTerm,
    GeneralizedAlphaWeakFormSum,
)


class Newmark(GeneralizedAlpha):
    """Newmark-beta time integrator.

    This is the ``alpha_m = 0`` and ``alpha_f = 0`` specialization of
    :class:`fedoo.time.GeneralizedAlpha`.
    """

    def __init__(self, beta=0.25, gamma=0.5):
        super().__init__(alpha_m=0.0, alpha_f=0.0, beta=beta, gamma=gamma)
