"""Shared machinery for problem-level time integrators."""

import warnings

import numpy as np

from fedoo.core.assembly import Assembly
from fedoo.core.assembly_sum import AssemblySum
from fedoo.core.base import AssemblyBase
from fedoo.core.time_evolution import normalize_time_evolution
from fedoo.core.weakform import WeakFormSum
from fedoo.lib_elements.element_list import is_reduced_integration

_STAB_TOL = 1e-12


def warn_if_reduced_mass(assembly):
    """Warn when a consistent mass is assembled with a reduced integration.

    With a variant using centroid interpolation (``"tet4r"``, ``"hex8r"``,
    ...) or fewer Gauss points than the full rule of the element, the
    :math:`N_i N_j` element mass is rank deficient. Nothing is reported when
    a term of the assembly is lumped.

    By default the mass does not follow a reduced stiffness (see
    :py:func:`fedoo.time.common.storage_integration`): this only fires for a
    reduced integration requested on the storage weak form itself, or where
    stiffness and storage have to share one assembly.
    """
    if np.any(assembly.mat_lumping):
        return
    if is_reduced_integration(assembly.elm_type, assembly.n_elm_gp, assembly.mesh):
        warnings.warn(
            f"Consistent mass assembled with a reduced integration "
            f"(elm_type={assembly.elm_type!r}, n_elm_gp={assembly.n_elm_gp}): "
            "the element mass matrix is rank deficient. Use the full element "
            "(e.g. 'tet4', 'tet10', 'hex8') with its default number of Gauss "
            "points for the mass.",
            stacklevel=3,
        )


def _insert_before(root, leaf, new):
    """Insert ``new`` just before ``leaf`` in the sum holding it.

    Return the root of the tree: a new sum when ``root`` is ``leaf``.
    """
    if root is leaf:
        return AssemblySum([new, leaf])
    stack = [root]
    while stack:
        node = stack.pop()
        if not isinstance(node, AssemblySum):
            continue
        children = node.list_assembly
        for i, child in enumerate(children):
            if child is leaf:
                children.insert(i, new)
                new.associated_assembly_sum = node
                if node._reload != "all":
                    node._reload = [j + (j >= i) for j in node._reload] + [i]
                return root
        stack.extend(children)
    return root


def warn_if_conditionally_stable(beta, gamma, alpha_m=0.0, alpha_f=0.0, context=""):
    """Warn when a Newmark / generalized-alpha set is only conditionally stable.

    Unconditional (A-) stability of the generalized-alpha family
    (Chung-Hulbert convention) requires::

        alpha_m <= alpha_f <= 1/2
        gamma >= 1/2 - alpha_m + alpha_f
        beta >= gamma / 2

    (Newmark is the ``alpha_m = alpha_f = 0`` special case.) A violating
    set is CONDITIONALLY stable: with FE meshes (``omega_max*dt >> 1``) the
    high-frequency modes grow geometrically and the solve collapses after a
    few tens of increments regardless of dt or load level -- a failure that
    masquerades as Newton divergence / element inversion.
    """
    prefix = f"{context}: " if context else ""
    alpha_ordering_ok = alpha_m <= alpha_f + _STAB_TOL and alpha_f <= 0.5 + _STAB_TOL
    if not alpha_ordering_ok:
        warnings.warn(
            f"{prefix}(alpha_m={alpha_m}, alpha_f={alpha_f}) violates "
            "alpha_m <= alpha_f <= 1/2: the scheme is only CONDITIONALLY "
            "stable.",
            stacklevel=3,
        )
    gamma_min = 0.5 - alpha_m + alpha_f
    if gamma < gamma_min - _STAB_TOL or beta < 0.5 * gamma - _STAB_TOL:
        warnings.warn(
            f"{prefix}(beta={beta}, gamma={gamma}) is only "
            "CONDITIONALLY stable. Unconditional stability requires "
            f"gamma >= {gamma_min} and beta >= gamma/2 = "
            f"{0.5 * gamma:.4f} (recommended: beta = "
            f"(gamma + 1/2)^2 / 4 = {0.25 * (gamma + 0.5) ** 2:.4f}"
            " for oscillation-free high-frequency response). With "
            "typical FE meshes the high-frequency modes will grow and "
            "the solve will fail after a few tens of increments.",
            stacklevel=3,
        )


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
    is_explicit = False

    def compile_assembly(self, assembly, evolution=None):
        """Compile compatible weakforms in an assembly tree.

        The leaves are compiled in place. The returned root is a new
        ``AssemblySum`` when ``assembly`` is a leaf that needs a separate
        storage assembly (see :meth:`_split_storage`).
        """
        evolution = normalize_time_evolution(evolution or self.evolution)
        leaves = list(assembly.iter_leaf())
        for leaf in leaves:
            if not isinstance(leaf, Assembly) or leaf.weakform is None:
                if isinstance(leaf, AssemblyBase):
                    self._compile_assembly_level_provider(leaf)
                continue
            compiled = self.compile_weakform(leaf.weakform, evolution)
            if compiled is not leaf.weakform:
                storage = self._split_storage(leaf, compiled)
                if storage is None:
                    leaf.weakform = compiled
                else:
                    leaf._storage_assembly = storage
            # also re-attach the storage of a leaf compiled in another tree
            storage = getattr(leaf, "_storage_assembly", None)
            if storage is not None and all(storage is not a for a in leaves):
                assembly = _insert_before(assembly, leaf, storage)
        return assembly

    def _split_storage(self, leaf, compiled):
        """Return a separate storage assembly for ``leaf``, or None.

        Called when a leaf is compiled. An integrator whose storage needs
        another element or quadrature than ``leaf`` sets the remaining terms
        as ``leaf.weakform`` and returns the storage assembly, which is
        placed just before ``leaf``. None keeps ``compiled`` in ``leaf``.
        """
        return None

    def compile_weakform(self, weakform, evolution=None):
        """Return a time-integrated version of ``weakform`` when applicable."""
        evolution = normalize_time_evolution(evolution or self.evolution)
        if getattr(weakform, "_fedoo_time_integrated", False):
            return weakform

        if isinstance(weakform, WeakFormSum):
            leaves = tuple(weakform.iter_leaf())
            compiled = [self.compile_weakform(wf, evolution) for wf in leaves]
            if all(old is new for old, new in zip(leaves, compiled)):
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
        if getattr(assembly, "time_evolution", None) != self.evolution:
            return assembly
        if getattr(assembly, "_fedoo_time_integrated", False):
            return assembly
        if any(hasattr(assembly, attr) for attr in ("storage", "dissipation")):
            raise NotImplementedError(
                f"Assembly-level time providers are part of the architecture but "
                f"do not have a concrete {type(self).__name__} adapter yet."
            )
        return assembly
