# Changelog

All notable changes to fedoo are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and fedoo aims to follow
semantic versioning.

## [Unreleased]

### Changed

- Prepare shell tangent fields once in the existing post-constitutive update
  and reuse them during residual/matrix assembly. State snapshots preserve the
  fields for rollback; the assembly and weak-form interfaces are unchanged.
- Differentiate FI/SRI/MITC interpolation in local in-plane coordinates before
  mapping to nodal increments, preserving all four weak-form corrections.
- Default beam/shell viewer components to global coordinates when local-frame
  data is available, with local coordinates as the fallback.
- Align shell viewer options with beam options: combine position/index/extrema
  in Value, move Coordinate system to the end, and add shell-only submesh
  selection with source preferences and fallback on other submeshes.
- Allow linear shell Stress/Strain-only files to reconstruct their thickness
  distribution from two distinct saved points without generalized outputs.
  Shell viewer stored selections use the nearest point within the requested
  layer; API interpolation remains the default. Reference point indices retain
  the selected side of a laminate interface.
- Beam viewer sampling shows the actual point count and snaps entered values
  to the nearest supported grid; fixed stored/custom counts are read-only.
  Rectangular grids select the closest odd square count, and count ties select
  the smaller grid.
- Beam viewer source choices require the selected source on the reference
  submesh and fall back between stored/recomputed data on other beam submeshes.
  Missing section fields remain absent instead of being treated as zero data.
- Select the nearest stored beam section point in normalized coordinates,
  independently for each submesh and saved location, with an optional exact-match
  policy. Section metadata supports `recovery="stored_only"` to disable analytical
  recomputation. The beam preview highlights the nearest stored point.
- Reduce optional shell tangent overhead by combining weak-form coefficients
  before operator expansion and contracting interpolation derivatives directly
  with local displacements/resultants, without storing the full derivative tensor.
- Vectorize shell tangent nodal increments, force rotations and coefficient
  selection; reuse invariant nodal operators without caching trial-state data.
- Reuse current shell interpolation operators and projected Jacobian inverses
  across tangent batches. Differentiate geometry from local nodal variations
  and compute the area gradient directly, reducing temporary array storage.

### Added

- Optional analytical UL corotational shell tangents with
  `PlateEquilibrium(consistent_tangent=True)` and `PlateEquilibriumFI`.
  Four explicit weak-form corrections include local finite rotations,
  changing surface geometry/area, FI/SRI/MITC interpolation and drilling.
  The default remains the historical material plus membrane-force geometric
  tangent. The optional tangent can be nonsymmetric and uses a fixed initial
  stiffness scale for the drilling penalty, including nonlinear section laws.
  Trial residuals use the trial frame; the existing force law and small local
  strain model are retained. Manually split shell weak forms are excluded.
- Beam output points configured by normalized coordinate pairs or an approximate
  `n_points` budget. `add_output(position=None)` saves all configured beam points
  on a separate section-point axis; scalar beam positions are rejected.
- Preserve ordinary continuum Stress/Strain blocks when evaluating section fields
  on meshes mixing beam, shell and solid elements. Stress-only files support
  sampled min/max and signed extrema without requesting generalized resultants.
- Shell output extraction at one or several normalized thickness coordinates.
  `position=None` saves configured output points; homogeneous linear shells
  default to bottom/middle/top, nonlinear shells to thickness integration points,
  and linear laminates to both sides of each interface. FDH5 retains numeric
  thickness metadata and local frames. Nonlinear/laminate outputs also save
  private thickness stresses for later layerwise interpolation.
- `shell_options` and a viewer **Options → Shell…** dialog for stored or recovered
  tensors, local/global components, individual thickness points and signed extrema.
  Generalized stress recovery is restricted to homogeneous linear shells;
  other shells interpolate saved material-point stresses, retaining the nearest
  value at faces and preserving laminate interface jumps.
- Shared beam result and viewer selection for stored or recomputed Stress/Strain.
  The default prefers stored tensors and uses signed `abs_max`; a reduction
  ignores point selection. Without reduction, normalized position or point index
  selects one point. Stored point coordinates are retained in FDH5; sampling
  resolution affects recomputed distributions only.
- Consistent UL corotational tangents for standard two-node `beam` elements
  in 2D and 3D, including length-dependent interpolation, frame rotation and
  local SO(3) rotation derivatives. `BeamEquilibrium(consistent_tangent=False)`
  retains the historical tangent. The correction differentiates the existing
  force residual and can be nonsymmetric; it does not introduce a new beam
  energy or finite-strain material model.
- Batch beam tangent derivatives across elements and columns, evaluate SO(3)
  Jacobians once per node, and bound dense temporary storage for large meshes.
- Assemble the consistent beam tangent entirely through strain, length and
  frame-spin weak operators with independent midpoint/chord interpolation.
  Remove the shared `get_matrix_correction` interface and sparse correction
  scatter; no extra nodal unknowns are introduced.
- Write the four consistent-tangent contributions explicitly in the beam
  weak form. Reuse cached strain interpolation and the current assembly frame,
  and differentiate local increments without rebuilding a dense nodal
  change-of-basis matrix.
- Beam section stress/strain recovery through `get_results` and `add_output`
  with physical `(y, z)` or normalized local-y section positions. Public
  `BeamStressList` and `BeamStrainList` support offline recovery, including
  optional rotation into global coordinates using the new `BeamLocalFrame`
  output. Section-specific shear recovery uses parabolic approximations for
  rectangles and solid disks, with average Q/A for other sections. Pointwise
  torsion supports circular/pipe sections and the rectangular Saint-Venant
  series. Rectangular torsional stiffness now uses the consistent series-based
  torsion constant instead of the previous polynomial approximation.
- A shared predefined component dictionary for datasets and the viewer,
  including named beam forces, moments, generalized strains, and frames.
- Numeric beam section descriptions saved with generalized Gauss-point fields
  in FDH5. Reloaded results expose lazy local and, when frames are available,
  global stress/strain fields. The viewer's section selector supports point
  recovery and sampled scalar extrema for circular, pipe and rectangular
  sections. Legacy files can attach properties with `set_beam_section`.

### Changed

- Respect Apply options to for beam viewer settings, validating every target
  pane before applying changes. Show indexed section-point physical coordinates
  at a selected beam Gauss point and highlight the point in the section preview.
- Replace the viewer's normalized local-y beam position with normalized (y,z)
  coordinates. Scale each signed direction using its centroid-relative
  bounding-box extent, supporting asymmetric and varying sections.
- Store beam section definitions once per submesh/frame as native HDF5
  metadata, replacing JSON for new files while reading legacy descriptions.
  Keep constant properties as scalars and varying properties in private
  Node/Element/GaussPoint fields. Section classes own their portable schema,
  normalized sampling and optional section meshing interfaces. Add custom
  output points, a viewer section preview, Show internal fields, and
  `add_output(..., private=True)`.
- Recover section Strain directly from BeamStrain using beam kinematics,
  without elastic constants, Poisson contraction or a shear correction factor.
  Native FDH5 recovery no longer stores or requires E, G and nu in section
  descriptions. Noncircular torsional strain still requires a warping model.
- Show one Stress and Strain field in the viewer for recovered beam tensors.
  Options → Beam… groups the section point/envelope and local/global coordinate
  controls. An optional Beam toolbar is hidden by default.
- Interpret beam envelope sampling resolution as an approximate total section
  point budget for every geometry. Circular and pipe sections refine angular
  and radial sampling together; label the control "Approximate sample points".
- Correct the `Iyy` and `Izz` axis assignments for non-square
  `BeamRectangular` sections (`a` along local y, `b` along local z).
- IPC contact thickness through an absolute `dmin` minimum separation on
  `IPCContact`, `IPCSelfContact`, `RigidBody.set_static_obstacle` and
  `RigidBody.set_static_plane`. Collision detection, barrier-stiffness
  tuning, friction and CCD use the same offset; zero remains the default.
  Nonzero offsets are rejected with OGC, whose trust region does not expose
  a minimum-separation parameter.
- `IPCContact` / `IPCSelfContact`: `use_area_weighting` option enabling the
  convergent (area-weighted) IPC formulation of ipctk.
- `RigidBody.set_static_plane`: IPC contact against an analytic static plane.
  Unlike a flat meshed obstacle, the contact force does not depend on which
  obstacle node, edge or face is the closest.
- Mesh import (`Mesh.read` for Abaqus `.inp` decks, `Mesh.from_meshio`) uses
  [meshlane](https://github.com/simvia-tech/meshlane), a maintained fork of
  meshio, when it is installed, and falls back to meshio otherwise. The `io`
  extra now installs `meshlane>=5.5.0` instead of `meshio`.
- Require Simcoon >= 2.1. UMAT callbacks receive `start` and a string `corate`
  explicitly. The Simcoon adapter handles `work_correction`, `tangent_output`,
  and conversion of the rate name to Simcoon's numeric code, without temporary
  call attributes or compatibility checks for older versions.
- Keep the state returned by UMAT initialization, with values assigned through
  `set_initial_statev()` taking precedence. Custom callbacks must accept the
  new keywords and handle `start=True` during initialization.
- The `tube_compression` example uses `IPCSelfContact` (ipctk >= 1.6) instead
  of the penalty self-contact.

### Fixed

- Rigid-body CCD bounds rotation-path curvature without relying on a single
  midpoint, preventing complete revolutions from bypassing collision checks.
  The line search never drops its curvature margin or physical offset.
- Analytic-plane CCD explicitly enforces the requested minimum distance,
  which IPC Toolkit 1.6 otherwise ignores for plane-vertex sweeps.
- IPC proximity safeguards compare the remaining linear gap above `dmin`
  with `dhat`, rather than comparing a squared distance with a length.
- IPC contact in `2Daxi`: the `2*pi*r` weight is now carried by the ipctk
  collisions, so the residual, tangent matrix, energy line search and
  automatic barrier stiffness are consistent (the tangent was previously
  weighted twice and the energy / barrier stiffness not at all).
- IPC friction produced no force: the friction potential was evaluated with
  absolute positions instead of the slip. Friction is now lagged over the
  time increment. With ipctk 1.6, the barrier stiffness was also passed as
  the static friction coefficient and missing from the normal force.
- `IPCContact(use_ogc=True)` raises in `2Daxi` (not supported).
- `RigidBody.set_static_obstacle` registers a CCD line search (`use_ccd=True`
  by default), so a time step that moves the body further than `dhat` no
  longer carries it through the obstacle and fails with a NaN barrier. The
  search follows the curved vertex paths of a rotating body (conservative
  piecewise linear CCD).

## [1.0.1] - 2026-10-01

### Added

- `CompositeUD` accepts `degrees=False` to specify fiber angles in radians;
  angles remain in degrees by default.
- The Simcoon UMAT adapter uses `corate` and direct `tangent_output` when the
  installed Simcoon supports them, while retaining Fedoo's box-tangent
  conversion for older versions. Simcoon step-cut requests now enter Fedoo's
  failed-increment handling.

### Fixed

- Nonlinear force recovery uses the evaluated residual, including inertia and
  damping, and refreshes it after vector updates and rollback. Linear implicit
  dynamics retain the completed step's force balance. Explicit dynamics recover
  the solved system's balance on demand, including diagonal mass operators,
  without storing force snapshots. Consistent
  mass central difference includes prescribed accelerations in the free equations.
- The Newton force criterion evaluates residuals even with a zero force reference.
- Rigid-body forces and torques are applied through constant Neumann conditions,
  keeping them separate from contact and inertia in the assembled residual.
  This preserves a nonzero current force reference in free-body dynamics,
  without retaining normalization from earlier Newton iterates.
- Finite-strain Simcoon laws that do not convect their own material axes now
  follow the rotating material frame, and tangent conversion uses the selected
  corotational rate in both TL and UL formulations.
- `CompositeUD` now rotates its stiffness matrix for a single nonzero angle
  as well as for arrays of Gauss point angles.
- The viewer's Plot Over Line now samples the currently displayed field and
  component after plot changes.
- Plot Over Line endpoints persist per view when its dialog is reopened or the
  active view changes. Duplicated views inherit the points independently.

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
