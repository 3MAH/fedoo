# Changelog

All notable changes to fedoo are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and fedoo aims to follow
semantic versioning.

## [Unreleased]

### Changed

- **`NonLinear.add_line_search` gained a `mode` argument** with three policies:
  `"natural"` (new default), `"minimize"` (the previous residual-descent
  behavior, with `method`) and `"safeguard"` (kinematic validity filter only,
  `det F > 0`). The default `"natural"` mode accepts a trial step when
  EITHER the classical Armijo test on `‖R‖` passes OR Deuflhard's
  affine-invariant test on the *simplified Newton correction*
  `K⁻¹ R(u + α dX)` does (one back-substitution per trial, the factorization
  of the current tangent being reused; accepted when this correction is
  smaller than the current one). The second merit, measured in displacement
  space, is invariant to the scaling of the equations and does not strangle
  the large legitimate steps of soft modes under force control (which
  inflate `‖R‖` without being bad steps); the first one remains the proven
  throttle against penalty-contact or plastic overshoot. `ls_mode` can also
  be set through `set_nr_criterion`.
  Factorization reuse (`set_reuse_factorization`) is enabled automatically
  for the default direct solver.
- A cached factorization (`set_reuse_factorization`) is now invalidated
  automatically when the constraint reduction matrix changes
  (`apply_boundary_conditions`), not only on `set_A`.

### Fixed

- `NonLinear`: the `elastic_initial_guess` / `force_elastic_matrix_next_iter`
  options re-assembled the elastic matrix but never handed it to the elastic
  prediction (a no-op beyond the first increment); they now refresh the
  tangent.
- `NonLinear.add_line_search` documents a contract that was only implicit: a
  custom step-size callback must return exactly 1 while a Dirichlet increment
  is pending. Returning `alpha < 1` defers the remaining `1 - alpha` of the
  prescribed displacement, and convergence is only declared once nothing is
  left to apply, so a callback that never returns 1 strands the increment and
  the time step collapses to `dt_min`.
- `NonLinear` with a `fd.time` integrator: the elastic prediction reused the
  tangent of the previous increment even after the time step changed. A
  transient tangent carries the `1/(beta dt^2)` inertia term, so that matrix
  is wrong by `(dt_prev/dt)^2` -- a factor 16 after the standard x0.25 cut,
  i.e. exactly when the solver is already struggling. The tangent is now
  refreshed when `dt` changed (`set_start`/`to_start` have just re-assembled
  it at the new step, so this only installs that matrix). Measured on a
  plastic dynamic bending case with repeated cuts: 12 increments completed
  instead of 5. Static problems and the legacy `ImplicitDynamic` weak form
  are unaffected.
- `NonLinear` with `adaptive_stiffness`: the iterate kept for the "redo the
  last iteration" rollback was saved once the error had already risen, so the
  rollback restored the bad iterate it was meant to undo. The last iterate
  that actually improved the error is now kept instead. Measured on
  `tube_compression`: 773 Newton iterations instead of 860 (and 779 instead
  of 815 with the residual-descent line search).
- `NonLinear` with `adaptive_stiffness`: the "safe" elastic matrix `KE` was
  read from the reference assembly, i.e. under `nlgeom="UL"` the matrix of
  the undeformed configuration (and, for an assembly sum, without the contact
  block) for the whole run. When the divergence guard restarted an increment
  with `set_A(KE)`, that matrix stayed installed across the time-step cuts:
  every retry then failed at its first iteration with an infinite error, down
  to `dt_min`. This made the `tube_compression` example abort at the contact
  folds (reproducibly single-threaded, run-dependent otherwise). `KE` is now
  read from the current assembly.
- **fedoo now requires `simcoon >= 2.0.0b1`** (previously `>= 1.14`). fedoo 1.0
  targets the simcoon 2.0 series, whose first release is the `2.0.0b1` beta.

### Migration — simcoon 2.0 `tangent_mode`

simcoon 2.0 **renumbered** the `umat()` tangent-operator enum. The mapping is:

| meaning                              | pre-2.0 | 2.0 |
| ------------------------------------ | :-----: | :-: |
| none (elastic operator)              |    –    |  0  |
| continuum tangent (default)          |    0    |  1  |
| Simo–Hughes algorithmic (consistent) |    1    |  2  |

fedoo's `Simcoon` law now defaults to `tangent_mode = 1` (continuum), preserving
the pre-2.0 numerical behavior and robustness. The algorithmic tangent stays
available as an explicit opt-in via `material.tangent_mode = 2`.

**Action required:** any code that passed **integer literals** for
`tangent_mode` must re-map them: **old `0` → `1`, old `1` → `2`**. Note that
`tangent_mode = 0` now selects *no* tangent (the elastic operator), which will
silently degrade convergence/accuracy if used unintentionally.
