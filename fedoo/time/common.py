from dataclasses import dataclass


@dataclass(frozen=True)
class RayleighDamping:
    """Rayleigh damping descriptor: ``C = alpha*M + beta*K``."""

    alpha: float = 0.0
    beta: float = 0.0
