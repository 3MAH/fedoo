from dataclasses import dataclass

import numpy as np
from scipy import sparse

from fedoo.core.assembly import Assembly
from fedoo.core.assembly_sum import AssemblySum
from fedoo.core.base import AssemblyBase
from fedoo.core.matrix import as_global_csr
from fedoo.core.time_evolution import normalize_time_evolution
from fedoo.core.weakform import WeakFormBase, WeakFormSum
from fedoo.weakform.inertia import Inertia


@dataclass(frozen=True)
class RayleighDamping:
    """Rayleigh damping descriptor: ``C = alpha*M + beta*K``."""

    alpha: float = 0.0
    beta: float = 0.0


@dataclass(frozen=True)
class StorageAssemblyGroup:
    """Storage and damping data associated with one leaf FE assembly."""

    source_assembly: Assembly
    storage_assembly: AssemblyBase
    weakforms: tuple
    rayleigh_damping: RayleighDamping


@dataclass(frozen=True)
class StorageAssemblyData:
    """Combined storage assembly and its per-part contributions."""

    assembly: AssemblyBase | None
    groups: tuple[StorageAssemblyGroup, ...]


def resolve_storage(weakform):
    """Resolve a weakform storage declaration to a storage weakform."""
    storage = weakform.get_storage()
    if storage is None or isinstance(storage, WeakFormBase):
        return storage
    return Inertia(storage, space=weakform.space)


def resolve_dissipation(weakform):
    """Return the dissipation explicitly attached to a weakform, if any."""
    if getattr(weakform, "dissipation", None) is not None:
        return weakform.dissipation
    return weakform.get_dissipation()


def build_storage_assembly(assembly, evolution):
    """Build storage assemblies while retaining per-part damping metadata.

    Different Rayleigh coefficients are supported across leaf assemblies. A
    single leaf assembly must use one coefficient pair because its assembled
    stiffness matrix cannot otherwise be separated into per-weakform parts.
    """
    evolution = normalize_time_evolution(evolution)
    groups = []
    for source in assembly.iter_leaf():
        if not isinstance(source, Assembly):
            continue

        weakforms = tuple(
            weakform
            for weakform in source.weakform.iter_leaf()
            if getattr(weakform, "time_evolution", None) == evolution
        )
        if not weakforms:
            continue

        storages = []
        damping_values = []
        for weakform in weakforms:
            storage = resolve_storage(weakform)
            if storage is None:
                raise ValueError(
                    f"Weakform {weakform.name!r} has {evolution.kind!r} evolution "
                    "but does not provide storage."
                )
            storages.append(storage)

            dissipation = resolve_dissipation(weakform)
            if dissipation is None:
                dissipation = RayleighDamping()
            elif not isinstance(dissipation, RayleighDamping):
                raise NotImplementedError(
                    "Cached dynamic problems currently support Rayleigh damping "
                    "descriptors, not custom dissipative weakforms."
                )
            damping_values.append(dissipation)

        damping = damping_values[0]
        if any(value != damping for value in damping_values[1:]):
            raise NotImplementedError(
                "Weakforms with different Rayleigh coefficients must belong to "
                "separate leaf assemblies. Combine those parts with AssemblySum."
            )

        storage = storages[0] if len(storages) == 1 else WeakFormSum(storages)
        storage_assembly = Assembly.create(storage, source.mesh, source.elm_type)
        groups.append(
            StorageAssemblyGroup(
                source_assembly=source,
                storage_assembly=storage_assembly,
                weakforms=weakforms,
                rayleigh_damping=damping,
            )
        )

    storage_assemblies = [group.storage_assembly for group in groups]
    if not storage_assemblies:
        combined = None
    elif len(storage_assemblies) == 1:
        combined = storage_assemblies[0]
    else:
        combined = AssemblySum(storage_assemblies)
    return StorageAssemblyData(combined, tuple(groups))


def assemble_rayleigh_damping_matrix(
    storage_data,
    size,
    mass_lumping=False,
):
    """Assemble ``sum(alpha_i*M_i + beta_i*K_i)`` for all model parts."""
    damping_matrix = sparse.csr_matrix((size, size))
    for group in storage_data.groups:
        alpha = group.rayleigh_damping.alpha
        beta = group.rayleigh_damping.beta
        if alpha:
            mass = as_global_csr(
                group.storage_assembly.get_global_matrix(), size, copy=False
            )
            if mass_lumping:
                mass = sparse.diags(np.asarray(mass.sum(axis=1)).ravel(), format="csr")
            damping_matrix = damping_matrix + alpha * mass
        if beta:
            stiffness = as_global_csr(
                group.source_assembly.current.get_global_matrix(),
                size,
                copy=False,
            )
            damping_matrix = damping_matrix + beta * stiffness
    return damping_matrix.tocsr()


def increment_solved(pb):
    """True when the problem's set_start closes an increment that was solved.

    On an empty increment the Newmark recurrence is not the identity but
    ``v <- v (1 - gamma/beta) + dt (1 - gamma/2beta) a``, i.e. exactly ``-v``
    for the standard parameters, so it must not run. Drivers that do not
    publish the flag are assumed to always close a solved increment.
    """
    return getattr(pb, "_increment_solved", True)


def newmark_acceleration_velocity(beta, gamma, dt, delta_disp, v_n, a_n):
    """Newmark update of acceleration and velocity from a displacement increment.

    Given the displacement increment ``delta_disp = u_{n+1} - u_n`` and the
    start-of-step velocity ``v_n`` and acceleration ``a_n``, return the
    end-of-step ``(acceleration, velocity)`` predicted by the Newmark relations::

        a_{n+1} = 1/(beta*dt**2) * (delta_disp - dt*v_n) + (1 - 0.5/beta) * a_n
        v_{n+1} = v_n + dt * ((1 - gamma) * a_n + gamma * a_{n+1})

    This single definition is shared by every generalized-alpha term (storage,
    stiffness, dissipation) so the recurrence constants live in exactly one
    place.
    """
    if delta_disp is None or (np.isscalar(delta_disp) and delta_disp == 0):
        delta_disp = np.zeros_like(v_n)
    else:
        delta_disp = np.asarray(delta_disp)
    if delta_disp.shape != np.shape(v_n):
        if delta_disp.size != np.size(v_n):
            raise ValueError(
                "delta_disp and velocity must contain the same number of DOFs."
            )
        # Assemblies can map the increment to the global flat DOF layout while
        # storing dynamic state variables in their node-shaped representation.
        delta_disp = delta_disp.reshape(np.shape(v_n))

    a0 = 1.0 / (beta * dt**2)
    acc = a0 * (delta_disp - dt * v_n) + (1.0 - 0.5 / beta) * a_n
    vel = v_n + dt * ((1.0 - gamma) * a_n + gamma * acc)
    return acc, vel
