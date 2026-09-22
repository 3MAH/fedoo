# Changelog

All notable changes to fedoo are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and fedoo aims to follow
semantic versioning.

## [Unreleased]

## [1.0.0] - 2026-09-22

### Added

- **`incompressibility` attribute of `StressEquilibrium`** (also a constructor
  argument) to select the treatment of volumetric locking: `None` (default,
  unchanged behavior), `"auto"`, `"mean_dilatation"`, `"sri"` or `"fbar"`.
- **Mean dilatation method** (`incompressibility="mean_dilatation"`): B-bar
  formulation based on a volume-weighted projection of the dilatation,
  equivalent to an element-wise discontinuous pressure eliminated at the
  element level. It supports small strain and updated Lagrangian formulations
  in 3D and plane strain, with a consistent tangent when geometric stiffness
  is enabled.
- **`Modal` problem** for linear free-vibration eigenvalue analysis of
  constrained and free-free systems. Computed modes are mass-normalized, and
  their mode indices, eigenvalues, angular frequencies, and frequencies can be
  written to result files.
- **`LinearBuckling` problem** for eigenvalue buckling analysis about a
  preloaded state, with critical load factors and buckling modes available in
  result files.
- **`Problem.clear_outputs`** for changing registered outputs between stages
  of chained analyses. FDH5 output also supports explicit `"overwrite"`,
  `"append"`, and `"error"` write policies.
- **`Problem.set_solver(..., symmetric=True)`** to request a
  symmetric-indefinite direct factorization: Pardiso `mtype=-2` and standalone
  MUMPS `sym=2`, with the appropriate stored triangle passed to each backend.
  SciPy/UMFPACK, PETSc and user-supplied solvers keep their general path, and
  the choice is propagated through factorization reuse. The caller is
  responsible for the symmetry of the reduced matrix.
- **`ignore_missing` output option** for `Problem.add_output` and
  `Problem.get_results`. Unavailable fields can now be omitted when a common
  output list is shared by constitutive laws with different state variables.
- An advanced gallery example comparing standard, mean-dilatation, F-bar and
  reduced-integration treatments of nearly incompressible plasticity.

### Changed

- Plane-strain incompressibility methods now retain the three-dimensional
  volumetric/deviatoric split: the assumed `zz` strain receives one third of
  the volumetric correction, and finite-strain mean dilatation uses the cube
  root scaling of the full assumed deformation gradient. Incompressibility
  options are ignored for plane-stress models.
- **Solid constitutive laws now follow a single documented finite-strain
  convention**: they return true Cauchy stress and an unnormalized
  corotational Kirchhoff ("box") tangent. `StressEquilibrium` converts that
  tangent centrally to `dS/dE` in TL or to the spatial Lie tangent in UL. The
  conversion can be bypassed with `convert_tangent=False` when a law already
  supplies the formulation tangent.
- **`StressEquilibrium.geometric_stiffness` now defaults to `None`**, meaning
  enabled for finite-strain tangent conversion and disabled in small strain.
  Explicit `True` and `False` values continue to override the default.
- **Native elastic laws in finite strain** now return Cauchy stress `tau / J`
  and use the same documented tangent convention as solid UMATs.
- **`ElastoPlasticity` in finite strain** now returns Cauchy stress `tau / J`.
- **`Heterogeneous`** forwards the finite-strain setting to every phase
  sub-assembly.
- **`Assembly.assume_sym` defaults to `None`** until the weak form resolves its
  formulation-dependent default during initialization. An explicit value set
  before initialization is preserved.
- Removed the dedicated `StressEquilibriumBbar` and `StressEquilibriumFbar`
  weak forms. Use `StressEquilibrium(..., incompressibility="sri")` and
  `StressEquilibrium(..., incompressibility="fbar")`, respectively.
- Removed the `StressEquilibrium.fbar` compatibility property. Select F-bar
  exclusively with `incompressibility="fbar"`.
- Updated-Lagrangian use of the legacy `incompressibility="sri"` method now
  emits a warning because its finite-strain formulation is not consistent.
- Incompressibility warnings are emitted independently of `Problem.print_info`
  and can be controlled with the standard Python warnings filters.
- Gauss-point-to-node conversion now uses a dedicated extrapolation matrix:
  full and over-integration use a pseudo-inverse, while reduced-integration
  elements use an independent reduced monomial basis. The mesh documentation
  clarifies the `"mean"` conversion method.
- `Modal.set_solver` and `LinearBuckling.set_solver` now configure their
  generalized eigensolver independently from linear-system solvers. The
  automatic, sparse `eigsh`, and dense `eigh` strategies are available, with
  persistent ARPACK options and per-solve mode counts and target shifts.
- Registering an output with `Problem.add_output` now requests automatic
  result writing from high-level solve methods. Modal and buckling analyses
  write one frame per mode.
- The documentation workflow no longer depends on the external `vtk-osmesa`
  package index or installs `microgen` for excluded gallery examples.

### Fixed

- The small-strain F-bar method now evaluates the volumetric strain at the
  element centroid.
- **F-bar tangent matrix.** The former implementation modified the stress but
  omitted the variation of the volume change at the element center. The
  consistent term is now included for `quad4` and `hex8`, uses the 1/3 factor
  associated with Fedoo's Lie tangent, and is no longer inadvertently
  symmetrized.
- `Mesh.get_volume` and `Mesh.get_element_volumes` now honor their `n_elm_gp`
  argument.
- Corrected the `Mechanical3D` symmetric-product indexing, which used
  `H[2][j]` where `H[j][2]` was required.
- Assembly caches now distinguish modeling spaces, preventing variable-rank
  mappings and change-of-basis matrices from being reused incorrectly between
  models, such as consecutive 3D and 2D beam examples.
- Bubble-enriched line and triangle elements now use their geometry element
  when constructing Gauss-point extrapolation matrices.
- Finite-strain weak forms now own and initialize the deformation-gradient
  field for the complete assembly before heterogeneous constitutive laws are
  initialized.
- `NonLinear.to_start` re-assembles the tangent matrix after restoring a
  failed increment instead of retaining the diverged iterate's matrix.
- The general MUMPS path remains compatible with releases whose `Context`
  constructor has no `sym` argument; requesting symmetric factorization on
  such a release raises `NotImplementedError`.
- The API documentation for `ElasticAnisotropic` and `ElasticIsotrop` is
  rendered again, and `Heterogeneous` is listed in the constitutive-law
  reference.
- Contact gallery examples now constrain initially detached indenters against
  rigid-body rotation, select only valid penalty-contact slave nodes, and use
  nonlinear-solver limits suitable for contact-pair changes.
- Periodic-composite, plasticity, thermal post-processing and fluid-mechanics
  examples were corrected and made self-contained where practical.

### Migration — `convert_tangent` in stress-equilibrium constructors

`convert_tangent` is inserted before `name` in `StressEquilibrium`,
`StressEquilibriumRI`, `StressEquilibriumMixed`, and the poromechanics momentum
weak forms. For `StressEquilibrium`, it follows `incompressibility`. Code that
passed a weak-form name positionally must use the keyword form:

```python
# before
wf = fd.weakform.StressEquilibrium(material, None, "my_weakform")
# after
wf = fd.weakform.StressEquilibrium(
    material, incompressibility=None, name="my_weakform"
)
```

A finite-strain law written against the previous behavior must return true
Cauchy stress. Divide by `J = det(F)` when the law integrates Kirchhoff stress.

## [1.0.0b2] - 2026-09-11

### Added

- **Material local frames for mechanical constitutive laws.** Uniform, nodal,
  elemental, and Gauss-point rotations can be supplied as matrices or SciPy /
  simcoon rotations. Anisotropic laws are transformed between material and
  global coordinates, including material-frame transport in finite strain.
- **`MechanicalUMAT`**, a generic corotational user-material base class that
  handles the Fedoo constitutive lifecycle, frame transformations, stress and
  tangent conversion, labeled properties and state variables, and initial
  state-variable values defined independently on each assembly with
  `set_initial_statev`.
- **Simcoon modular-material construction** through `Simcoon.from_modular`,
  with generated property and state-variable labels.
- **Assembly-level beam and shell frame control**, including per-node,
  per-element, and per-Gauss-point guides and projection onto the geometrical
  tangent or normal.
- **Local-frame mesh operations.** `extrude` can orient a profile from path
  frames, `thicken` creates solids from shell meshes using explicit or
  automatically oriented nodal normals, and cylindrical-frame generation is
  exposed from `fedoo.mesh`.
- **`Problem.set_dof`** for assigning scalar, vector, global, or complete DOF
  fields with problem-specific storage handling.
- A user-development guide covering custom weak forms, `MechanicalUMAT`, and
  advanced `Mechanical3D` constitutive laws.

### Changed

- Simcoon laws and the pedagogical `ElastoPlasticity` law now use the common
  `MechanicalUMAT` lifecycle. Isotropic laws skip unnecessary material-frame
  transformations.
- Constructor argument ordering is consistent across constitutive laws,
  constraints, and related helpers. Thermal weak forms remain unnamed unless
  a name is supplied explicitly instead of inheriting the material name.
- Material frames are owned by constitutive laws and assemblies rather than
  meshes, avoiding ambiguous nodal or element storage when integration rules
  vary.
- **Energy-based artificial damping now decreases more conservatively.** Its
  coefficient can fall by at most a factor of two after one converged
  increment, preventing a noisy incremental-energy estimate near an
  instability from abruptly removing most of the stabilization.
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
- **`NonLinear.add_line_search` gained an `apply_to_bc` argument**, defaulting
  to `True`. Pending prescribed-displacement increments are scaled only when
  required to maintain valid kinematics; residual and natural acceptance tests
  resume after the complete increment has been applied. Set `apply_to_bc=False`
  to apply it in one full step without this safeguard. Custom callbacks keep
  direct control of their returned alpha.
- The experimental `eigenvalue_shift` nonlinear-solver option and its
  `eigenvalue_shift_factor` / `eigenvalue_assume_sym` parameters were removed.
  The spectral estimate operated on the unreduced matrix, handled MPCs
  inconsistently, added substantial cost, and did not improve the measured
  nonlinear benchmarks.

### Fixed

- Movie export now replaces the previous plot actor at every frame instead of
  accumulating prior images.
- Finite-strain anisotropic updates now consistently transform kinematic,
  stress, rotation-increment, and tangent quantities between material and
  global frames.
- Line search no longer returns an untested minimum step when every trial
  produces invalid kinematics. It now searches down to a very small step and,
  if none is valid, reports a failed increment through the normal time-step
  reduction machinery. Natural line search falls back to its last valid trial
  when no acceptance test succeeds.
- Natural line search now releases factorization reuse when it is removed or
  replaced by another built-in mode. A factorization context subsequently
  installed by the user is preserved. Cached factorizations are also
  invalidated when `apply_boundary_conditions` changes the constraint
  reduction matrix, not only when `set_A` changes the system matrix.
- `NonLinear`: the `elastic_initial_guess` / `force_elastic_matrix_next_iter`
  options re-assembled the elastic matrix but never handed it to the elastic
  prediction (a no-op beyond the first increment); they now refresh the
  tangent.
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
