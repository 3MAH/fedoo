"""Internal helpers for 2Daxi weakforms.

Centralises the ``2*pi*r`` axisymmetric volume / surface integration
weight so that every mechanical, inertial, damping, contact and
interface weakform applies the factor consistently.
"""

import numpy as np


def axi_volume_weight(assembly):
    """Return ``2*pi*r`` evaluated at each Gauss point.

    Reuses ``assembly.sv["_R_gausspoints"]`` when it has already been
    populated by :class:`fedoo.weakform.StressEquilibrium` (which keeps
    it refreshed at the current configuration in UL). Otherwise builds
    the radial coordinate from the current mesh on the fly so that
    standalone uses (e.g. an ``Inertia`` weakform without a companion
    ``StressEquilibrium``) still get the correct weight.

    Parameters
    ----------
    assembly : fedoo.Assembly
        The assembly the weakform integrates over. Must be associated
        with a 2Daxi ``ModelingSpace``.

    Returns
    -------
    ndarray, shape (n_gauss_points,)
        ``2*pi*r`` at every Gauss point of the current mesh.
    """
    rr = assembly.sv.get("_R_gausspoints")
    if rr is None:
        mesh = assembly.current.mesh
        rr = mesh.convert_data(
            mesh.nodes[:, 0],
            "Node",
            "GaussPoint",
            n_elm_gp=assembly.n_elm_gp,
        )
    return (2 * np.pi) * rr
