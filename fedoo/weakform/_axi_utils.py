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
    return (2 * np.pi) * axi_radius(assembly)


def axi_radius(assembly):
    """Return the radius ``r`` at each Gauss point (see :func:`axi_volume_weight`)."""
    rr = assembly.sv.get("_R_gausspoints")
    if rr is None:
        mesh = assembly.current.mesh
        rr = mesh.convert_data(
            mesh.nodes[:, 0],
            "Node",
            "GaussPoint",
            n_elm_gp=assembly.n_elm_gp,
        )
    return rr


def axi_hoop_strain_op(space, rr):
    """Hoop strain operator ``u_r / r``, set to 0 on the axis (``r = 0``)."""
    return space.variable("DispX") * np.divide(
        1, rr, out=np.zeros_like(rr), where=rr != 0
    )
