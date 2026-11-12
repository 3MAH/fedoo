"""Capabilities of the installed simcoon, queried by feature and never by version.

simcoon is often an editable development build whose metadata version lags the
code, so fedoo asks simcoon what it serves. A simcoon that does not expose a
query is treated as not serving the feature, and fedoo keeps its fallback path.
"""

import simcoon as sim


def serves_total(umat_name):
    """Whether the law runs under ``kinematic_form="total"`` (ln U and T).

    False with simcoon 2.1, which has no total formulation.
    """
    serves = getattr(sim, "total_formulation_serves", None)
    return serves is not None and bool(serves(umat_name))


def has_total_formulation():
    """Whether simcoon provides the total formulation at all."""
    return hasattr(sim, "total_formulation_serves")


def serves_condensed(umat_name, ndi, nshr=None, kinematic_form=None):
    """Whether simcoon condenses the reduced stress state for this law.

    A served law returns the stress with zero condensed components and the
    tangent already condensed (zero rows and columns on the condensed slots).
    ``ndi = 3`` is always served. Without the ``condensation_serves`` query,
    reduced states rely on fedoo's own condensation of the 3D tangent, which
    is only valid for laws that take the strain, not the deformation gradient.
    """
    if ndi == 3:
        return True
    serves = getattr(sim, "condensation_serves", None)
    if serves is None:
        return False
    kwargs = {}
    if nshr is not None:
        kwargs["nshr"] = nshr
    if kinematic_form is not None:
        kwargs["kinematic_form"] = kinematic_form
    return bool(serves(umat_name, ndi, **kwargs))
