import numpy as np

from fedoo.core.assembly import Assembly
from fedoo.core.assembly_sum import AssemblySum
from fedoo.core.base import AssemblyBase
from fedoo.core.time_evolution import FIRST_ORDER, normalize_time_evolution
from fedoo.core.weakform import WeakFormBase, WeakFormSum


class BackwardEuler:
    """Backward-Euler integrator for first-order storage weakforms."""

    evolution = FIRST_ORDER

    def compile_assembly(self, assembly, evolution=None):
        """Compile compatible weakforms in an assembly tree in place."""
        evolution = normalize_time_evolution(evolution or self.evolution)
        if isinstance(assembly, AssemblySum):
            for child in assembly.list_assembly:
                self.compile_assembly(child, evolution)
            return assembly

        if isinstance(assembly, Assembly) and assembly.weakform is not None:
            assembly.weakform = self.compile_weakform(assembly.weakform, evolution)
        elif isinstance(assembly, AssemblyBase):
            self._compile_assembly_level_provider(assembly)
        return assembly

    def compile_weakform(self, weakform, evolution=None):
        """Return a backward-Euler weakform sum when storage is declared."""
        evolution = normalize_time_evolution(evolution or self.evolution)
        if getattr(weakform, "_fedoo_time_integrated", False):
            return weakform

        if isinstance(weakform, WeakFormSum):
            compiled = [
                self.compile_weakform(wf, evolution) for wf in weakform.list_weakform
            ]
            if all(old is new for old, new in zip(weakform.list_weakform, compiled)):
                return weakform
            return WeakFormSum(compiled, weakform.name)

        if getattr(weakform, "time_evolution", None) != evolution:
            return weakform

        storage = getattr(weakform, "storage", None)
        if storage is None:
            return weakform
        if not isinstance(storage, WeakFormBase):
            raise TypeError(
                "BackwardEuler storage must be a weakform. Use set_storage() "
                "with a first-order storage weakform."
            )

        integrated = WeakFormSum([weakform, BackwardEulerStorageTerm(storage)])
        integrated._fedoo_time_integrated = True
        return integrated

    def _compile_assembly_level_provider(self, assembly):
        if hasattr(assembly, "storage"):
            raise NotImplementedError(
                "Assembly-level storage providers are part of the architecture but "
                "do not have a concrete BackwardEuler adapter yet."
            )


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
        if not np.array_equal(delta_value, 0):
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
