"""Explicit dynamics with a consistent mass: the mass has its own solver
context, kept across the increments."""

import numpy as np
import pytest

import fedoo as fd
from fedoo.core.base import USE_MUMPS


def _count_factorizations(factor):
    calls = []
    ctx_factor = factor._ctx.factor

    def counted(*args, **kargs):
        calls.append(kargs.get("reuse_analysis"))
        return ctx_factor(*args, **kargs)

    factor._ctx.factor = counted
    return calls


def _explicit_consistent_mass(reuse):
    fd.ModelingSpace("2Dplane")
    mesh = fd.mesh.rectangle_mesh(nx=11, ny=4, x_max=4.0, elm_type="quad4")
    material = fd.constitutivelaw.ElasticIsotrop(200e3, 0.3)
    material.set_density(7.8e-9)
    pb = fd.problem.ExplicitDynamic(
        fd.Assembly.create(fd.weakform.StressEquilibrium(material), mesh),
        time_step=1e-8,
        mass_lumping=False,
    )
    pb.bc.add("Dirichlet", mesh.find_nodes("X", 0.0), "Disp", 0)
    pb.bc.add("Neumann", mesh.find_nodes("X", 4.0), "DispY", -1.0)
    if reuse:
        try:
            pb.set_reuse_factorization(True)
        except RuntimeError:
            pytest.skip("no factorization reuse backend")
    pb.initialize()
    pb.solve_history(20e-8)
    return pb


def test_explicit_consistent_mass_with_factorization_reuse():
    """The end-step mass solve has its own factorization: with factorization
    reuse, it must not use the one of the effective matrix."""
    ref = _explicit_consistent_mass(False)
    pb = _explicit_consistent_mass(True)
    for name in ("get_disp", "get_velocity", "get_acceleration"):
        a, b = getattr(ref, name)(), getattr(pb, name)()
        assert np.allclose(a, b, rtol=1e-9, atol=1e-9 * abs(a).max())


@pytest.mark.skipif(not USE_MUMPS, reason="python-mumps not installed")
def test_explicit_consistent_mass_factorizations_kept():
    """Mass and effective matrix keep their own factorization: with a
    constant time step, each one is factorized once."""
    pb = _explicit_consistent_mass(False)
    pb.solve_time_increment()  # nominal time step again (history shortened)
    calls = _count_factorizations(pb._kept_solver)
    mass_calls = _count_factorizations(pb._mass_solver[2])
    for _ in range(10):  # constant time step
        pb.solve_time_increment()
    assert calls == [] and mass_calls == []


def test_explicit_consistent_mass_balance_with_factorization_reuse():
    """M a = f_ext - K u on the free dof at the end of an increment (no
    damping): independent of the time step and of a reference run."""
    pb = _explicit_consistent_mass(True)
    residual = pb._MatCB.T @ (
        pb._mass_matrix @ pb.get_acceleration().ravel()
        - pb.get_B()
        + pb._stiffness_matrix @ pb.get_disp().ravel()
    )
    assert np.linalg.norm(residual) < 1e-10 * np.linalg.norm(pb.get_B())
