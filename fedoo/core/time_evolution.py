from dataclasses import dataclass


@dataclass(frozen=True)
class TimeEvolution:
    """Time-evolution category used to match weakforms and integrators."""

    kind: str


FIRST_ORDER = TimeEvolution("first_order")
SECOND_ORDER = TimeEvolution("second_order")


def normalize_time_evolution(evolution):
    """Return a canonical time-evolution category."""
    if isinstance(evolution, TimeEvolution):
        return evolution

    if isinstance(evolution, str):
        key = evolution.lower()
        if key in {"first_order", "firstorder", "rate", "temp", "thermal"}:
            return FIRST_ORDER
        if key in {
            "second_order",
            "secondorder",
            "dynamic",
            "dynamics",
            "disp",
            "mechanical",
        }:
            return SECOND_ORDER

    raise TypeError(
        "time evolution should be a TimeEvolution category "
        "(for instance fd.time.FIRST_ORDER or fd.time.SECOND_ORDER)"
    )
