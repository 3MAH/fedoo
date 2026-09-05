# Finite-strain consistency & nonlinear-solver robustness

Synthesis of the working-tree changes for the fedoo PR (built on top of
`fix/dynamics`). Companion of the simcoon PRs #101/#102 (exact spectral
tangent transport): **requires simcoon master ≥ the #102 merge** — the new
tests are gated by runtime capability probes and skip cleanly on older
simcoon builds.

## TL;DR

Three long-standing root causes were identified by finite-difference
verification of the assembled tangents and fixed; a robustness chain was
added so that a diverging Newton iterate can never crash the process again.
Benchmarks that were previously impossible now run:

| Benchmark (NEOHC cantilever, nu = 0.49) | before | after |
|---|---|---|
| Displacement control, static, UL | stalls at 0.82 L (dt collapse) | **1.0 L**, 40 inc, NR mean 3.6 |
| Force control 20 N, implicit dynamics | crash (`MUMPS -10` / `std::terminate`) at F ≈ 0.8 N | **45.6 mm = 0.912 L**, NR mean ≤ 2 |
| Force control, `fd.time` integrator + massive cap (`RigidBodyAssembly`) | erratic divergence at increment 4 | **45.6 mm**, full ramp |
| Test suite | 252 | **270 passed** (18 new tests) |

## Root causes fixed

### 1. 3D geometric stiffness was silently 2D-truncated
`_init_nl_strain_op_vir` compared `space.ndim` (an int) to the string
`"3D"` — always false — so every 3D finite-strain problem assembled the 2D
initial-stress operator: all terms involving a Z derivative (S33/S13/S23
slots and the k = Z rows) were missing. The tangent error was O(sigma),
grew with load, and destroyed Newton on soft bending modes.
*(converged solutions were never wrong — the residual was exact; only the
tangent, i.e. convergence, was affected. Same remark for #2 and #3.)*

### 2. UL tangent transport was not the Lie (Truesdell) tangent
The UL log-corate branch used the umat box tangent (÷J) directly, while the
tangent consistent with the Cauchy residual + standard initial-stress term
is the Lie one. `update_2` now converts `box → dS/dE → DSDE_2_Dsigma_LieDD`
for every simcoon law in UL (per-corate first-stage keys; native fedoo laws
keep their engineering tangent — gated by the new
`_corotational_box_tangent` attribute). `assume_sym` is now decided at
initialize: `True` except UL, where the Lie tangent is not major-symmetric
(a user-forced `True` warns and symmetrizes). Default corate is `log_r`
(exact polar frame increment — the corate whose transport is exact in
simcoon, including with rotated internal-variable history).

### 3. Conditionally-stable Newmark pairs failed undiagnosed
`(beta = 0.25, gamma = 0.6)` violates the A-stability condition
`beta ≥ gamma/2`: high-frequency modes grow geometrically and the solve
collapses after a few tens of increments **regardless of dt or load** — a
failure that masquerades as Newton divergence / element inversion.
`ImplicitDynamic` and `fd.time.GeneralizedAlpha` (hence `Newmark`) now warn
on any conditionally stable set (`alpha_m ≤ alpha_f ≤ 1/2`,
`gamma ≥ 1/2 − alpha_m + alpha_f`, `beta ≥ gamma/2`).

## Robustness chain (new)

- **`InvalidKinematicStateError`** (`core/base.py`): a trial state with
  `det F ≤ 0` at any Gauss point is rejected at the kinematics stage
  (`_comp_F` / `_comp_Fbar`) and treated by the NR loop as a failed
  increment (dt cut + state restore) — never assembled. Previously it
  produced NaN-filled matrices (reported as a spurious, non-deterministic
  `MUMPS -10`) or killed the interpreter through simcoon's polar
  decomposition.
- **Line search redesigned as a validity filter** (`problem/line_search.py`):
  the legitimate full Newton step of a soft mode can be large, and its
  quadratic remainder — amplified by the stiff nodal equations — grows the
  residual norm as step² (×10–×1000) while remaining an excellent step.
  Any residual-monotone rule then strangles Newton (measured: alpha
  collapsing to 3e-4, linear convergence, dt collapse). Default policy
  (`ls_mode="safeguard"`): accept alpha = 1 whenever the trial state is
  valid; geometric backtracking (×0.5, 0.8 safety) only on `det F ≤ 0`.
  The residual-descent methods remain available via `ls_mode="minimize"`.
  The line search is now also active on the elastic prediction under pure
  force control (gate on the actual `_Xbc` content instead of the
  `_boundary_is_0` proxy), and convergence acceptance no longer depends on
  the alpha-driven flag (`_xbc_is_applied`).
- **Duplicate-BC guard** (`core/boundary_conditions.py`): adding the same
  BC *object* twice (e.g. a `RigidTie` explicitly added by the user and
  auto-registered by `RigidBodyAssembly`) is now ignored. Duplicated MPCs
  silently corrupted the elimination (MatCB) and destroyed Newton — the
  cause of the `fd.time` + `RigidBodyAssembly` force-control failure.

## Changes by file

| File | Change |
|---|---|
| `fedoo/weakform/stress_equilibrium.py` | ndim fix (root cause 1); UL Lie conversion + per-corate first-stage keys; `corate="log_r"` default; assume_sym at initialize gated on `_corotational_box_tangent` (native UL laws keep the symmetric path); det F guard in `_comp_F`/`_comp_Fbar` incl. the element-center Jacobian (cofactor det, returned as J); tangent conversion skipped on line-search trials (`_line_search_update`, beam/plate pattern) |
| `fedoo/weakform/stress_equilibrium_mixed.py` | det F guard in the mixed `_comp_F` too (before `log(J)` turns an inverted state into silent NaN) |
| `fedoo/core/mechanical3d.py` | `_corotational_box_tangent = False` class default (native laws) |
| `fedoo/constitutivelaw/simcoon_umat.py` | `_corotational_box_tangent = True` (simcoon umats) |
| `fedoo/core/base.py` | `InvalidKinematicStateError` |
| `fedoo/problem/non_linear.py` | catch invalid iterates in the NR loop; `_xbc_is_applied` convergence gate; `add_line_search(mode=...)` API + `ls_mode`/`ls_method`/`ls_max_iter` accepted by `set_nr_criterion` |
| `fedoo/problem/line_search.py` | safeguard-first validity-filter line search; invalid trial → inf (sentinel never reaches the acceptance tests); minimize fallback bounded by the last valid alpha; `_xbc_is_applied` gate |
| `fedoo/weakform/implicit_dynamic.py` | Newmark stability warning (shared helper), on `ImplicitDynamic2` too |
| `fedoo/time/base.py` | `warn_if_conditionally_stable` — single stability guard used by `ImplicitDynamic` and `GeneralizedAlpha`/`Newmark` |
| `fedoo/time/generalized_alpha.py` | generalized-alpha stability warning (shared helper) |
| `fedoo/core/boundary_conditions.py` | identity guard in `ListBC.append` (the single insertion point; `extend` filters through it, `add` inherits it) |
| `tests/test_weakform_factory_types.py` | updated to the assume_sym-at-initialize contract |
| `examples/03-advanced/neohookean_cantilever_force.py` | NEW: force-driven companion example (massive rigid cap + `fd.time` Newmark) — exercises every fix of this PR; validated: 45.60 mm vs 45.5 expected |

## New tests (18)

- `tests/test_finite_strain_tangent_consistency.py` — FD consistency of the
  assembled global tangent: NEOHC/ELISO/EPICP × TL/UL × corate log/log_R
  (~1e-9 each), plus a committed-plastic-history case. Gated by runtime
  capability probes on the installed simcoon (skip, never fail, on older
  builds; TODO markers to replace the probes by a version pin once the
  simcoon release number is fixed).
- `tests/test_time_integrator_stability_warning.py` — the stability guards
  warn on every violating (beta, gamma, alpha) set and stay silent on valid
  ones.
- `tests/test_bc_duplicate_registration.py` — a `RigidTie` ends up exactly
  once in `pb.bc` in both registration orders.

## Behavior notes for reviewers

- Converged solutions are unchanged by 1–2 (tangent-only fixes); Newton
  behavior improves everywhere (e.g. EPICP UL shear benchmark: 185 → 81
  total NR iterations, flat 4/increment through gamma = 0.4).
- `corate` default moved `"log"` → `"log_r"`. Both compute the exact log
  strain; `log_r` uses the exact polar frame increment, for which the
  simcoon tangent transport is exact including rotated plastic history
  (with `"log"`/XBM a small, documented O(||EP||·dtheta) tangent residual
  remains — harmless in practice).
- `assume_sym` is now `False` in UL (unsymmetric solver path — slight cost,
  required for correctness of the Lie tangent). Non-UL problems keep
  `True`.
- The line-search default changed for `add_line_search` users (opt-in
  feature): safeguard mode. `add_line_search(mode="minimize")` restores the
  previous residual-descent behavior.
- Known cost (documented TODO in `line_search.py`): in safeguard mode the
  validity test is a full residual evaluation whose work is redone by the
  caller when the step is valid (~one vector assembly + umat sweep per NR
  iteration when the line search is active). A kinematics-only validity
  probe would remove it — deferred (touches the update machinery).
- Known limits, out of scope, documented: pure-static force control on
  quasi-rigid modes still requires continuation (Riks) or Tikhonov
  stabilization (not implemented); hypoelastic multi-increment transport is
  exact for `log_r` only.

## Do not commit

The root-level mission notes (`FORCE_CONTROL_RIGID_TIE.md`,
`FEDOO_EXACT_TANGENT_FOLLOWUP.md`, this file if desired), `MaillageCylindre/`
(~500 MB of meshes) and `fedoo.egg-info/` are untracked on purpose — do not
`git add -A`.
