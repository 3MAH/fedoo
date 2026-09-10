# Mission: force-controlled loading on free RigidTie DOFs

> **RESOLVED 2026-09-03.** Root cause: the benchmark scripts used Newmark
> (beta=0.25, gamma=0.6), which is only CONDITIONALLY stable (unconditional
> requires beta >= (gamma+1/2)^2/4 = 0.3025): geometric high-frequency
> growth killed every run after ~20-26 committed increments REGARDLESS of
> dt, ramp speed or load (the increment-count signature was the give-away).
> With beta=0.3025 the 20 N run completes: 200/200 increments,
> ux = 44.70 mm (0.894 L, vs 45.5 mm displacement-control cross-check,
> 1.8% = numerical damping), mean NR 1.95. A stability warning was added to
> ImplicitDynamic.__init__. The robustness chain built along the way
> (det F guard -> recoverable error, validity-only line search, LS on the
> elastic prediction, convergence-gate fix) turned crashes into clean
> failures and made the diagnosis possible -- keep it. Remaining side
> findings: RigidTie tangent EXONERATED by control-swap FD (missing block
> 4.4e-4 << k_rot); fd.time integrator + RigidBodyAssembly misbehaves on
> retries (erratic from increment 4) -- separate issue; cap inertia
> (PR#40) still structurally desirable.

> **For the agent working on this**: read entirely; §2 is the evidence — do
> not re-derive it. House rules: NEVER commit/push; leave user WIP alone.
> Env: `source ~/miniconda3/etc/profile.d/conda.sh && conda activate 3mah &&
> export KMP_DUPLICATE_LIB_OK=TRUE`. simcoon is the LOCAL git
> (~/Documents/GitHub/simcoon, nothing pushed); rebuild with
> `pip install --no-build-isolation .`.

Date: 2026-09-02. Context: the NEOHC cantilever benchmark (ArtiSynth
comparison). Displacement control (Dirichlet on RigidDispX) is fully healthy:
static reaches 1.0 L, curve saved. FORCE control (Neumann 20 N on
RigidDispX, all cap DOFs free) fails — this file scopes that.

## 1. Symptom

`Neumann("RigidDispX", 20 N)` on the RigidTie cap (implicit dynamics,
gamma=0.6, rayleigh [50,0], dt fixed 0.01, quadratic ramp) dies between
t ~ 0.15 and 0.21 (F ~ 0.5-0.9 N only!, cap rotY ~ 25-30 deg) with
`MUMPSError: Matrix is numerically singular` — at a WANDERING increment
(non-deterministic run to run).

## 2. Evidence (measured; do not redo)

- Displacement control, same everything: works to 1.0 L (static AND dynamic).
- All 6 cap DOFs blocked by Dirichlet: OK. ANY free cap DOF + the 20 N
  Neumann: fails. Tiny load (0.01 N, 2 increments): OK — so no structural
  zero mode at start.
- `pb.set_solver("direct_scipy")` (deterministic) survives further and
  exposes the true failure: **simcoon `RU_decomposition` aborts**
  (`sqrtmat_sympd(): given matrix is not symmetric / transformation failed`)
  = a Newton ITERATE produced garbage/inverted F (det F <= 0) at Gauss
  points. The MUMPS -10s are the same garbage states seen through the
  threaded factorization (hence the non-determinism).
- `pb.add_line_search(method="Quadratic")` + fixed dt: no more crashes
  (clean "NR has not converged" instead) — overshoot confirmed as the
  driver — but NR then stalls > 20 subiters at some increment.
- line search + update_dt=True + dt_max=0.01: MUMPS -10 came back (a retry
  path still assembles at a garbage state — the safeguard has holes).

Interpretation: the free cap DOFs are ultra soft (bending stiffness of a
soft slender rod: dF/du ~ 40 N/m at the origin; rotational ~ 2.5e-2 N.m/rad)
and carry NO inertia (RigidTie global DOFs are massless — known PR#40 gap),
so nothing bounds the first Newton iterate of an increment: it overshoots
into element inversion. ArtiSynth's mirror works in force control because
the cap is a true rigid body with mass + damping. A SECOND, unconfirmed
contributor: the RigidTie MPC linearization omits the constraint's
second-order term (tie forces x constraint curvature), which only enters
the free-free block when a tie DOF is loaded — possibly the reason line
search alone still stalls (> 20 iters).

## 3. The work

3.1 **fedoo NR safeguard (core, highest value)**: an NR iterate (or trial
    state in line search / after dt reduction) whose kinematics update
    yields det F <= 0 (or RU_decomposition failure) must be treated as a
    FAILED step: backtrack alpha (line search) or reduce dt — never
    assemble/factor at that state. Hook: the corate strain computation in
    `stress_equilibrium.py` (update_1 path) can check det F cheaply and
    raise a dedicated recoverable exception the NR loop catches like a
    convergence failure. Cover the retry paths (the dt_max experiment shows
    one is unguarded).

3.2 **simcoon robustness**: exceptions thrown inside the parallel umat loop
    (`simcoon_parallel_for` / GCD `dispatch_apply` in the python wrapper,
    and RU_decomposition callers) currently escape a worker thread ->
    `std::terminate` (libc++abi), killing Python. Capture per-worker (e.g.
    std::exception_ptr per point, rethrow after the region) so fedoo can
    catch and recover. This also matters beyond force control (any element
    inversion today kills the interpreter).

3.3 **RigidTie tangent completeness (verify, then fix or exonerate)**:
    FD-check the assembled tangent (pattern of
    tests/test_finite_strain_tangent_consistency.py) on a rotated,
    force-loaded free-tie-DOF state — reachable safely via displacement
    control to 0.4 L then RELEASING RigidDispX with the equilibrium force
    applied as Neumann (state unchanged, control swapped). If FD shows an
    O(F) inconsistency on the tie block, implement the constraint
    second-order term in `constraint/rigid_tie.py`; if FD-exact, exonerate
    and close.

3.4 Optional API gap (ties into PR#40 items): inertia/damping on RigidTie
    global DOFs (a cap with mass), which would make force-controlled
    dynamics as robust as ArtiSynth's.

## 4. Validation

- The §1 run (quadratic ramp, dt=0.01 fixed, 200 increments) completes and
  settles at ux ~ 45.5 mm = 0.911 L (cross-check from the
  displacement-control curve: ~/scratch/artisynth/fedoo_comparison/
  cantilever_curve_lin_0.050.csv), rotY ~ 60-70 deg at that point.
- No process-killing exception under any failure mode (kill 3.2).
- Full suite: `python -m pytest tests/ -q` -> 263 passed, 1 skipped today;
  nothing may regress. Add a regression test for 3.1 (a load step engineered
  to invert elements must retry, not crash).

## 5. Meanwhile (user-facing workaround)

The ArtiSynth comparison does NOT need fedoo force control: use
displacement control on the fedoo side (full curve, includes the F=20 N
point at 45.5 mm) and either protocol on the ArtiSynth side.
