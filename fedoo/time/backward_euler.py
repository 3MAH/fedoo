import numpy as np

from fedoo.core.time_evolution import FIRST_ORDER
from fedoo.core.weakform import WeakFormBase, WeakFormSum
from fedoo.time.base import TimeIntegratorBase


class BackwardEuler(TimeIntegratorBase):
    """Backward-Euler integrator for first-order storage weakforms."""

    evolution = FIRST_ORDER

    def _integrate_leaf(self, weakform):
        """Return a backward-Euler weakform sum when storage is declared."""
        if getattr(weakform, "dissipation", None) is not None:
            raise NotImplementedError(
                "BackwardEuler does not integrate dissipative terms yet. "
                "Remove set_dissipation() from this first-order weakform, or "
                "fold the dissipation into the storage term."
            )

        storage = getattr(weakform, "storage", None)
        if storage is None:
            # A first-order weakform without a storage term has no transient
            # contribution: keep it as a plain static term.
            return weakform
        if not isinstance(storage, WeakFormBase):
            raise TypeError(
                "BackwardEuler storage must be a weakform. Use set_storage() "
                "with a first-order storage weakform."
            )

        return WeakFormSum([weakform, BackwardEulerStorageTerm(storage)])


class BackwardEulerStorageTerm(WeakFormBase):
    """Backward-Euler discretization of a pure first-order storage weakform."""

    def __init__(self, storage, name=""):
        super().__init__(name, storage.space)
        self.storage = storage
        self.assembly_options = storage.assembly_options
        self._start_key = f"_BE_StorageStart_{id(self)}"

    def initialize(self, assembly, pb):
        self.storage.initialize(assembly, pb)
        assembly.sv[self._start_key] = self._current_storage_value(assembly, pb)

    def update(self, assembly, pb):
        self.storage.update(assembly, pb)

    def update_2(self, assembly, pb):
        self.storage.update_2(assembly, pb)

    def set_start(self, assembly, pb):
        self.storage.set_start(assembly, pb)
        assembly.sv[self._start_key] = self._current_storage_value(assembly, pb)

    def to_start(self, assembly, pb):
        self.storage.to_start(assembly, pb)

    def get_weak_equation(self, assembly, pb):
        dt = pb.dtime
        if dt == 0:
            return 0

        storage_wf = self.storage.get_weak_equation(assembly, pb)
        current_value = self._current_storage_value(assembly, pb)
        start_value = assembly.sv.get(self._start_key, 0)

        diff_op = (1.0 / dt) * storage_wf
        delta_value = current_value - start_value
        if np.any(delta_value):
            if hasattr(self.storage, "get_weak_equation_for_value"):
                diff_op += (1.0 / dt) * self.storage.get_weak_equation_for_value(
                    assembly, pb, delta_value
                )
            else:
                diff_op += (1.0 / dt) * assembly.operator_apply(
                    storage_wf, np.asarray(delta_value).ravel()
                )
        return diff_op

    def _current_storage_value(self, assembly, pb):
        if hasattr(self.storage, "get_storage_value"):
            value = self.storage.get_storage_value(assembly, pb)
        elif "Temp" in assembly.sv:
            value = assembly.sv["Temp"]
        else:
            raise NotImplementedError(
                "BackwardEulerStorageTerm needs the storage weakform to expose "
                "its current value."
            )
        return np.array(value, copy=True) if not np.isscalar(value) else value
