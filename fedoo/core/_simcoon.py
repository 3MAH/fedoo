"""Simcoon batch calls shared by the constitutive laws and the weak forms.

Every call runs on ``fedoo.core.base.N_THREADS`` threads (``fedoo.get_config()["N_THREADS"]``,
0 = all the cores).
"""

import simcoon as sim

from fedoo.core import base


def rotate_voigt(values, DR, kind, active=True):
    """Rotate Voigt vectors ``(6, N)`` or 6x6 tensors ``(6, 6, N)`` by the rotation
    matrices ``DR`` ``(3, 3, N)``, or by one ``(3, 3)`` matrix for all of them.

    ``kind`` is the simcoon rotation rule: "stress", "strain", "stiffness", ...
    """
    # one parallel C++ loop, without building a scipy Rotation (from_matrix
    # is serial and costs about 1 us per point)
    return sim.rotate_voigt(values, DR, kind, active, base.N_THREADS)


def objective_rate(corate, F0, F1, dtime, return_de):
    """``simcoon.objective_rate`` on ``N_THREADS`` threads."""
    return sim.objective_rate(corate, F0, F1, dtime, return_de, base.N_THREADS)


def lt_convert(Lt, F, stress, converter_key):
    """``simcoon.Lt_convert`` on ``N_THREADS`` threads."""
    return sim.Lt_convert(Lt, F, stress, converter_key, base.N_THREADS)
