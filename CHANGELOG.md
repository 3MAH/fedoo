# Changelog

All notable changes to fedoo are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and fedoo aims to follow
semantic versioning.

## [Unreleased]

### Added

- `BeamFiberSection`: nonlinear multifiber beam section. A private copy of a
  3D material law (e.g. a Simcoon plasticity law) is evaluated under uniaxial
  stress at fiber points of a rectangular, circular or pipe section, and the
  normal force, bending moments and their full section tangent are integrated
  from the fibers. Transverse shear and torsion remain elastic. Works with
  `BeamEquilibrium` in 2D and 3D, small strain or corotational `nlgeom`,
  including `consistent_tangent=True`. Saved `Stress` fields are the fiber
  stresses (`recovery="stored_only"`).
- `"1Dstress"` (uniaxial stress) reduced state for mechanical laws, next to
  `"2Dstress"`: `Mechanical3D.get_H_condensed` statically condenses a 3D
  tangent onto any set of Voigt components; native elastic laws and UMAT laws
  (`ndi=1`) support it. Exact Simcoon stresses under `ndi=1` require a Simcoon
  version that condenses this state.
- `IPCContact` / `IPCSelfContact`: 3D `lin2` contact meshes, such as beam
  centerlines, are collision curves (edges only). `IPCSelfContact` uses a 3D
  `lin2` mesh directly instead of extracting its surface. With `dmin` equal
  to the fiber diameter, this models contact between equal-radius rods.
- Finite-strain Biot poromechanics. In `nlgeom="UL"` the pore-stress
  coupling now includes its geometric tangent. `nlgeom="TL"` is now the exact
  pull-back of UL and gives the same solution. The momentum coupling is
  -alpha p J C^{-1}:delta E, the storage and coupling terms of the mass
  balance carry the J of dv = J dV, and the mobility is pulled back to
  (k/mu) J C^{-1}. `PoroMechanicsSimple` now computes `lnJ`, so
  deformation-dependent permeabilities (Holmes-Mow, Kozeny-Carman) receive
  the current J.
- Poromechanics documentation page (`docs/poromechanics.rst`) and a
  Poromechanics section in the example gallery: Terzaghi consolidation
  compared with the analytical series, finite-strain consolidation of a
  Neo-Hookean skeleton, and the Taylor-Hood example.
  `PoroMechanics`, `PoroMechanicsSimple`, `PoroFluidProperties` and the
  permeability models are now listed in the API reference.

### Changed

- The material-point helpers of the nonlinear shells moved to
  `fedoo.constitutivelaw._material_points`, shared with the beam fibers.
- The poromechanics weak forms raise `NotImplementedError` in `2Daxi`: the
  Biot coupling lacked the axisymmetric measure and hoop strain.
- `PoroDarcy` warns and falls back to k(J=1) when a callable permeability is
  used in finite strain without `lnJ`.

### Fixed

- `PoroDarcy` in `nlgeom="UL"`: the pore-pressure gradient of the residual
  was computed on the reference mesh instead of the current one. The
  residual missed a factor of order J, while the tangent used the current
  mesh. Transient finite-strain consolidation was wrong by about the strain
  level (5 % at 5 % compression), but drained states were not affected.

## [1.1.0] - 2026-10-14

### Added

- Viewer frame-count refresh during navigation and an explicit Reload button
  for ongoing FDH5 simulations. Existing all-frame limits and history plots
  remain unchanged until requested again.
- Optional shell thickness display in the viewer's Shell options for
  `tri3`/`quad4` surfaces, using saved section thickness and deformed midsurface
  normals. Supports varying thickness and mixed meshes; unused shared nodes
  are excluded from display geometry.
- Total finite-strain formulation for `nlgeom="TL"`, following the strain-space
  framework of Miehe, Apel and Lambrecht (2002). Supported Simcoon laws use
  material logarithmic strain ln U and its dual stress T in the reference
  configuration. Strain is computed directly from F; constitutive integration
  can still depend on the time increment. This requires Simcoon's total-formulation
  API and a law it supports, including supported modular materials. With
  Simcoon 2.1, unsupported laws or plane stress, TL retains the rate update.
  `weakform.kinematic_form` reports the selected formulation.
- Result fields `LogStrainU`, `LogStrainV` and `LogStressU`, and
  `StressTensorList.cauchy_to_log_stress_u` / `log_stress_u_to_cauchy`, to compare a TL
  run and a UL run with `corate="log_r"` on the same measures.
- `IPCContact` / `IPCSelfContact`: `excluded_rings` blocks self-collision
  between primitives fewer than `excluded_rings` rings apart on the contact
  surface, so that `dmin` may exceed the edge length (shell thickness larger
  than the element size).
- `IPCContact` / `IPCSelfContact`: `psd_projection=False` assembles the exact
  contact Hessian instead of ipctk's per-collision positive-semidefinite
  projection. This can improve Newton convergence but may produce an indefinite
  tangent. Projection remains enabled by default; convergence gains depend on
  the problem and solver.
- `ListBC.check` and `Problem.check_boundary_conditions` report the boundary
  conditions that can't be applied together on a dof: Dirichlet bc with
  different values, Dirichlet bc on a dof eliminated by a mpc, dof eliminated
  by several mpc and nodal force ignored on a blocked dof. The check runs the
  first time the bc are applied after the list is modified, according to
  `pb.bc.check_mode` (`"warn"` by default, `"raise"` or `"ignore"`).
- Optional analytical corotational tangents for standard two-node beams in
  2D/3D and FI/SRI/MITC shells in updated Lagrangian geometry. Enable the
  keyword-only `consistent_tangent=True` constructor option on
  `BeamEquilibrium`, `PlateEquilibrium` or `PlateEquilibriumFI`. The
  historical tangent remains the default. Consistent tangents include changing
  geometry, interpolation and frame rotations, and may be nonsymmetric.
  The shell option excludes TL and manually split weak forms; alternative beam
  interpolations retain their historical tangent. Beam materials remain
  isotropic linear elastic, and the beam option is read-only after construction.
  Consistent tangents can substantially increase assembly cost, particularly
  for shells; their convergence benefit depends on the problem.
- Beam section stress/strain recovery at normalized `(y, z)` coordinates.
  `BeamStressList` and `BeamStrainList` support offline recovery; strain from
  `BeamStrain` uses beam kinematics without Poisson contraction. Standard
  sections provide transverse shear distributions and torsion recovery,
  including the rectangular Saint-Venant solution. Generic sections require
  suitable geometry/warping information for recovery.
- Configurable beam and shell `output_points` and `n_points`, with a separate
  section-point axis in saved tensors. Homogeneous linear shells sample
  bottom/middle/top by default, nonlinear shells use thickness integration
  points, and linear laminates include both sides of each interface.
- Dataset `beam_options` and `shell_options` for stored/recomputed fields,
  local/global coordinates, individual section points and reductions.
  `abs_max`, the beam default, returns the nonnegative largest magnitude.
  `signed_abs_max` retains the selected value's sign, choosing the first point
  for magnitudes tied within relative tolerance 1e-9. Both propagate NaNs.
  `max` and `min` select algebraic extrema and retain the sign. Full tensors
  are reduced componentwise; scalar invariants are evaluated at each point
  before reduction. A componentwise envelope is not a tensor at one physical
  point, and invariants computed from it are not guaranteed conservative.
- Automatic shell read selection: with no reduction or explicit point/index,
  use the sole saved thickness point, otherwise the top face. This also applies
  after reloading. Nonlinear stresses are interpolated within each layer and
  linearly extrapolated to its boundaries by default; set
  `shell_options.update(extrapolate=False)` for nearest-IP boundary values.
  Extrapolated stresses are estimates, and stored-point reductions use the
  actual saved values.
- Numeric section metadata and local-frame outputs for saved recovery. FDH5
  stores section descriptions and varying properties per submesh/frame;
  legacy NPZ/FDZ files preserve field metadata in an optional JSON entry.
  `set_beam_section` attaches section information to existing datasets.
  Linear shell distributions can be reconstructed from two saved points.
  Mixed beam/shell/solid results preserve ordinary continuum fields.
- Viewer Beam and Shell options dialogs with section previews, point/index/
  extrema selection, source preferences, submesh selection and local/global
  coordinates. Recovered tensors use global components when frames are
  available. Sampling controls respect fixed/custom point sets. Add
  `add_output(..., private=True)` and Show internal fields; private recovery
  arrays are excluded from PyVista mesh export.
- IPC minimum separation through `dmin` on `IPCContact`, `IPCSelfContact`,
  `RigidBody.set_static_obstacle` and `RigidBody.set_static_plane`. The default
  is zero; nonzero offsets are not supported with OGC.
- `IPCContact` / `IPCSelfContact` support `use_area_weighting` for the
  convergent area-weighted IPC formulation.
- `RigidBody.set_static_plane` provides IPC contact with an analytic plane,
  including minimum separation and conservative CCD, without dependence on a
  meshed obstacle's discretization.

### Changed

- Boundary conditions sharing some dof: the generalized forces of several
  Neumann bc are summed. A node repeated inside a single bc is still counted
  once. For Dirichlet bc, the last one in the list is still applied, and a
  Dirichlet bc prevails over a Neumann bc on the same dof.
- `nlgeom="TL"` and `nlgeom="UL"` now select the constitutive formulation as
  well as the configuration. Eligible TL models use the total formulation
  (ln U and its dual stress); UL uses the rate formulation with
  `weakform.corate`. Setting an explicit `corate` in TL raises at initialization
  only when the law runs the total formulation. TL models using the fallback
  rate formulation still accept `corate`. TL results can change for noncoaxial
  stress/stretch loading, such as plastic shear or off-axis anisotropy.
- `IPCContact` stores true minimum distances between contact primitives
  (ipctk returns squared distances). The adaptive barrier-stiffness update
  and the proximity safeguards act on the gap above `dmin`, compared with
  `dhat` as documented.
- `Problem.get_results` and `add_output` now default to `position=None`
  instead of 1, saving all configured beam/shell points. Shell Stress/Strain
  storage has shape `(6, n_thickness_points, n_locations)`. Mixed beam/shell
  tensor output requires `position=None`; use the respective dataset options
  for subsequent selection.
- Shell drilling penalties use a fixed initial membrane shear stiffness scale
  for both historical and consistent tangents, including nonlinear sections,
  instead of following the current stiffness.
- Rectangular beam torsional stiffness uses a Saint-Venant series rather than
  the previous polynomial approximation.
- Unavailable section recovery follows `ignore_missing`. Mixed assemblies keep
  fields provided by any submesh; a field unavailable everywhere raises
  `NameError` unless `ignore_missing=True`. Invalid positions remain errors.
- Mesh import uses [meshlane](https://github.com/simvia-tech/meshlane) when
  installed and falls back to meshio otherwise. The `io` extra installs
  `meshlane>=5.5.0` instead of meshio.
- Require Simcoon >= 2.1. UMAT callbacks receive `start` and the selected string
  `corate` explicitly, replacing the adapter's temporary call attributes and
  older-version signature detection. The adapter handles
  `work_correction`, `tangent_output` and conversion of the rate name to
  Simcoon's numeric code.
- The lattice beam example explicitly enables its consistent tangent for
  large-deformation convergence. The `tube_compression` example uses
  `IPCSelfContact` (ipctk >= 1.6) instead of penalty self-contact.
- Quick Start now uses current mesh/assembly APIs and runnable object names,
  retrieves stiffness after initialization, and recommends the Fedoo viewer
  with `fedoo[plot,gui]`. VTK export remains available for external viewers.

### Fixed

- Viewer node picking on thickened shells maps top/bottom display points to
  original node ids and reads the displayed field under the current section
  selection. Gauss-point values retain discontinuities; Element displays
  report adjacent element values at shared nodes.
- FDH5 output retries temporary file-lock conflicts with a live viewer instead
  of failing immediately. Eager mesh/frame reads use one short-lived handle;
  HDF5 locking remains enabled. Retries are bounded, and unrelated I/O errors
  still raise immediately.
- A distributed load added with `pb.bc.add` (`Pressure`, `SurfaceForce`,
  `DistributedForce`) no longer erases the Neumann bc defined before it, and
  two Neumann bc on the same dof no longer overwrite each other.
- In multi-step problems, the start value of a nodal Neumann bc no longer
  includes the forces applied by the other bc on the same dof.
- `MPC` docstring: the constraint equation is
  `sum(factor * dof) + constant = 0`.
- Analytic-plane CCD explicitly enforces the requested minimum distance,
  which IPC Toolkit 1.6 otherwise ignores for plane-vertex sweeps.
- IPC proximity safeguards compare the remaining linear gap above `dmin`
  with `dhat`, rather than comparing a squared distance with a length.
- Correct the `Iyy`/`Izz` assignments of non-square `BeamRectangular` sections:
  `a` is along local y and `b` along local z.
- Use the polar area moment `Iyy + Izz` for beam axial rotary inertia instead
  of the Saint-Venant torsion constant `Jx`. Torsional stiffness remains `G*Jx`.
- Beam assembly resolves its symmetry default during initialization and avoids
  the symmetric shortcut for coupled interpolations or consistent tangents,
  preserving explicitly configured symmetry settings.
- Correct the 3D beam frame update so rigid axial rotation produces no spurious
  local nodal rotation and the bending axes follow the section orientation.
- Correct the shell frame update with `true_drilling_rotation=False`: rigid
  rotation no longer produces spurious membrane strains or drilling reactions.
- Make shell thickness strain `epsilon0 - z*kappa` consistent with the existing
  curvature definition. Correct face stress signs, nonlinear material-point
  kinematics, laminate membrane/bending coupling (`-B`) and distribution plots.
- Correct finite-strain strain rotation for native anisotropic elastic laws.
- Committing a finite-strain increment no longer rotates an already current
  Cauchy stress a second time. History transport is handled by the constitutive
  update for the next increment.
- Tensor principal-value/direction routines support additional section-point
  axes instead of assuming a single point axis.
- Rigid-body CCD accounts for curved rotation paths, preventing complete
  revolutions from bypassing collision checks. The line search retains its
  curvature margin and physical offset.
- IPC contact in `2Daxi` consistently weights residual, tangent, energy and
  automatic barrier stiffness by `2*pi*r`. Unsupported OGC use raises an error.
- IPC friction uses incremental slip rather than absolute positions and is
  lagged over the time increment. Correct the ipctk 1.6 barrier-stiffness and
  friction-coefficient arguments.
- Static rigid obstacles register CCD by default, preventing steps from moving
  a body through an obstacle beyond the contact activation distance.

### Migration

- Shell output now saves all configured thickness points by default. Pass
  `position=1` to save only the top face. Raw stored tensors include a
  thickness-point axis; use `DataSet.get_data` and `shell_options` to select
  a `(6, n_locations)` tensor or a scalar envelope. Beam output positions are
  normalized `(y, z)` pairs, not scalar thickness coordinates.
- Corrected rectangular bending inertias and shell thickness signs can change
  existing results. Recompute affected beam, asymmetric-laminate or nonlinear
  shell analyses; nonlinear shell drilling penalties also use the fixed
  initial stiffness scale.
- Custom UMAT callbacks and `_call_umat` overrides must accept the new
  `start` and `corate` keyword arguments, and handle `start=True` during
  initialization. Install Simcoon >= 2.1.
- For a TL model that selects the total formulation, remove explicit `corate`
  assignments (or set `corate=None`). Choose UL when an objective rate is
  intended. `kinematic_form` is selected during problem initialization.
- Use an appropriate general solver for nonsymmetric consistent tangents.
  Disabling IPC's PSD projection can also require an indefinite solver;
  explicitly forcing positive-definite factorization is then unsuitable.

## [1.0.1] - 2026-10-01

### Added

- `CompositeUD` accepts `degrees=False` to specify fiber angles in radians;
  angles remain in degrees by default.
- The Simcoon UMAT adapter uses `corate` and direct `tangent_output` when the
  installed Simcoon supports them, while retaining Fedoo's box-tangent
  conversion for older versions. Simcoon step-cut requests now enter Fedoo's
  failed-increment handling.

### Fixed

- Preserve state returned by Simcoon UMAT initialization, with user-assigned
  `set_initial_statev()` components taking precedence. Initialization is
  distinguished from Newton updates at time zero. Assembly reset clears
  initial-state metadata; reapply initial values before reinitializing.
- Nonlinear force recovery uses the evaluated residual, including inertia and
  damping, and refreshes it after vector updates and rollback. Linear implicit
  dynamics retain the completed step's force balance. Explicit dynamics recover
  the solved system's balance on demand, including diagonal mass operators,
  without storing force snapshots. Consistent-mass central difference includes
  prescribed accelerations in the free equations.
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

- Require the stable Simcoon release >= 2.0.1, replacing the beta dependency
  >= 2.0.0b1 used in Fedoo 1.0.0b2.
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
