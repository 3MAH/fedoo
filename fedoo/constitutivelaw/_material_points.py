"""Material points integrated by structural section laws (shells, beams)."""

import copy

import numpy as np

from fedoo.util.voigt_tensors import StrainTensorList, StressTensorList


class _MaterialPointAssembly:
    """Minimal assembly interface used by section material points.

    ``dimension`` is the stress state seen by the material law: "2Dstress"
    for shell thickness points, "1Dstress" for beam fibers, "3D" otherwise.
    """

    class _MaterialSpace:
        def __init__(self, dimension):
            self._dimension = dimension

        def get_dimension(self):
            return self._dimension

        @property
        def ndim(self):
            return 2 if self._dimension == "2Dstress" else 3

        @staticmethod
        def list_variables():
            return ()

    def __init__(self, n_gauss_points, dimension="2Dstress"):
        self.n_gauss_points = n_gauss_points
        self.space = self._MaterialSpace(dimension)
        self.sv = {
            "Strain": StrainTensorList(np.zeros((6, n_gauss_points))),
            "Stress": StressTensorList(np.zeros((6, n_gauss_points))),
        }
        self.sv_start = dict(self.sv)
        self.sv_component = {}
        self._nlgeom = False

    def convert_data(self, data, *args, **kwargs):
        """Return pointwise material data without mesh interpolation."""
        return data


def _copy_state(state):
    """Deep-ish copy of a state-variable dict (shared by the section laws)."""
    copied = {}
    for key, value in state.items():
        if hasattr(value, "copy"):
            copied[key] = value.copy()
        else:
            copied[key] = copy.deepcopy(value)
    return copied


def _tangent_array(tangent, n_points):
    """Normalize scalar/pointwise tangent entries to a dense (6, 6, n) array."""
    dense = np.empty((6, 6, n_points), dtype=float)
    for row in range(6):
        for column in range(6):
            dense[row, column] = tangent[row][column]
    return dense
