"""Shared machinery for problem-level time integrators."""

from fedoo.core.assembly import Assembly
from fedoo.core.assembly_sum import AssemblySum
from fedoo.core.base import AssemblyBase
from fedoo.core.time_evolution import normalize_time_evolution
from fedoo.core.weakform import WeakFormSum


class TimeIntegratorBase:
    """Base class walking an assembly/weakform tree to inject transient terms.

    Subclasses set the class attribute ``evolution`` and implement
    :meth:`_integrate_leaf`, which turns a single static weakform (already
    matched to this integrator's evolution category) into its transient form.
    The tree traversal, the idempotency guard, and the ``WeakFormSum`` handling
    are shared so a traversal fix only has to be made once.
    """

    #: TimeEvolution category handled by this integrator (set by subclasses).
    evolution = None

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
        """Return a time-integrated version of ``weakform`` when applicable."""
        evolution = normalize_time_evolution(evolution or self.evolution)
        if getattr(weakform, "_fedoo_time_integrated", False):
            return weakform

        if isinstance(weakform, WeakFormSum):
            compiled = [
                self.compile_weakform(wf, evolution) for wf in weakform.list_weakform
            ]
            if all(old is new for old, new in zip(weakform.list_weakform, compiled)):
                return weakform
            # Build the compiled sum unnamed: a non-empty name would re-register
            # it in the WeakForm registry, clobbering the user's static sum.
            return WeakFormSum(compiled)

        if getattr(weakform, "time_evolution", None) != evolution:
            return weakform

        integrated = self._integrate_leaf(weakform)
        if integrated is weakform:
            return weakform
        integrated._fedoo_time_integrated = True
        return integrated

    def _integrate_leaf(self, weakform):
        """Turn a single matched static weakform into its transient form.

        Return ``weakform`` unchanged when there is nothing to integrate.
        """
        raise NotImplementedError

    def _compile_assembly_level_provider(self, assembly):
        if any(hasattr(assembly, attr) for attr in ("storage", "dissipation")):
            raise NotImplementedError(
                f"Assembly-level time providers are part of the architecture but "
                f"do not have a concrete {type(self).__name__} adapter yet."
            )
