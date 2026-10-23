"""
=================================
Post-Processing
=================================

.. currentmodule:: fedoo

Get results from a problem
==========================

In fedoo, most of the standard results are easily exportable using the
:py:meth:`fedoo.Problem.get_results` method of the problem class.

The get_results method returns a :py:class:`fedoo.DataSet` object which comes
with several methods for plotting, saving and loading mesh-dependent results.
A dataset can be associated with either a :py:class:`fedoo.Mesh` or a
:py:class:`fedoo.MultiMesh`.


To avoid a redundent call of the get_results function, especially for time
dependent problems, one can simply add some required output with the
:py:meth:`fedoo.Problem.add_output` method. This create a
:py:class:`MultiFrameDataSet` object associated to the problem.
Once the required outputs are defined for a given problem, a call to the
:py:meth:`fedoo.Problem.save_results` method allow to save all the defined
fields on disk using the choosen file format, and associate the saved file to
an iteration of the :py:class:`MultiFrameDataSet`. For non linear problems
solved using :py:meth:`fedoo.Problem.nlsolve`, results are automatically saved at
some iterations dependending on the choosen parameters.

The :py:class:`MultiFrameDataSet` stores references to the saved iterations in
the MultiFrameDataSet.list_data attribute. The method
:py:meth:`MultiFrameDataSet.load` is called to read the data
of a given iteration.


Class DataSet
-------------

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   DataSet

Class ProblemDataSet
--------------------

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   ProblemDataSet

Class MultiFrameDataSet
-----------------------

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   MultiFrameDataSet

MultiMesh results
-----------------

A :py:class:`DataSet` or :py:class:`MultiFrameDataSet` associated with a
:py:class:`MultiMesh` can contain submeshes with different element types.
Node fields use the common global node numbering. Element and Gauss-point
fields returned by :py:meth:`DataSet.get_data` are stored in a
:py:class:`MultiMeshData` object, with one data block per submesh.

Global element ids use the concatenated submesh order: all elements of
submesh 0, then all elements of submesh 1, and so on. For example:

.. code-block:: python

    stress = results.get_data("Stress", "vm", "Element")

    # Access data by submesh or by global element id
    shell_stress = stress.submesh("tri3")
    value = stress.global_element_value(42)
    values = stress.global_element_values([3, 18, 42])

    # Build one NumPy array using the global element order
    global_stress = stress.to_global()

Normal NumPy-style access to a :py:class:`MultiMeshData` object refers to its
active submesh. Use :py:meth:`MultiMeshData.to_global` when a globally ordered
array is required. A global selection spanning several submeshes can be
plotted directly:

.. code-block:: python

    results.plot(
        "Stress",
        component="vm",
        data_type="Element",
        element_set=[3, 18, 42],
        global_element_set=True,
    )

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   MultiMeshData

Beam and shell section fields
-----------------------------

Requesting only ``Stress`` with ``position=None`` is sufficient for min/max and
largest-magnitude reductions over the saved section points. Generalized ``BeamStress`` and
``ShellStress`` are not required for these stored reductions. Mixed meshes retain
ordinary continuum stress fields alongside recovered beam/shell fields, including
von Mises and principal components.

These are extrema over the saved sampling, not guaranteed continuous section
extrema. Restricting the saved output to one point restricts stored reductions to
that point. Refining elastic beam sampling later requires the generalized
resultants. Homogeneous linear shell tensors at two distinct thickness points
determine their linear distribution and allow later resampling without saved
resultants; nonlinear/laminate shell files already retain the underlying thickness
stresses for interpolation.

Shell constitutive laws already recover ``Stress`` and ``Strain`` through
``get_results`` and ``add_output``. For shells, ``position`` is the normalized
through-thickness coordinate: -1 is the bottom face, 0 the midsurface, and 1
the top face. A scalar selects one point; a one-dimensional array selects
several points. ``position=None`` saves all ``shell.output_points``. Homogeneous
linear shells default to three equally spaced points (bottom, middle, top),
controlled by ``n_points``. Nonlinear shells default to their actual thickness
integration points. Linear laminates save both sides of every layer interface.

Shell tensors have shape ``(6, n_thickness_points, n_shell_gauss_points)``.
Their coordinates and layer identities are stored as native FDH5 metadata.
``ShellLocalFrame`` is included automatically for shell outputs and enables
global tensor components. Stored nodal tensors cannot be rotated unambiguously;
use element/Gauss-point tensors or recompute before converting to nodes.

Use ``res.shell_options`` to select the source and thickness value::

    res = pb.add_output("shell_results", shell_assembly,
                        ["ShellStress", "ShellStrain", "Stress", "Strain"])
    res.shell_options.update(reduction=None, position=0.25)
    stress = res.get_data("Stress", "XX")
    res.shell_options.update(reduction=None, point_index=0)
    res.shell_options.update(source="recomputed", reduction="abs_max", samples=11)
    von_mises = res.get_data("Stress_global", "vm")

The default source is ``"auto"``: prefer stored tensors, otherwise recover or
interpolate. ``"stored"`` and ``"recomputed"`` explicitly require that source.
With ``reduction=None`` and no position/index, the default reads the sole saved
thickness point, or the top face if multiple points or only generalized fields
are available. Explicit positions are respected. The
``"abs_max"`` reduction returns the largest absolute component value, without
its sign. ``"signed_abs_max"`` retains the selected sign, choosing the first
point for magnitudes tied within relative tolerance 1e-9. Both propagate NaNs;
``"max"`` and ``"min"`` select algebraic extrema. Invariants
and principal values are evaluated at each point before reduction. A reduction
ignores point selection; without a reduction, an explicit ``position`` or
``point_index`` overrides automatic selection. Updating either clears the other.
With a reduction, full-tensor reads reduce each component independently and return
``(6, n_locations)``. These components may come from different thickness points;
the envelope is not a tensor at one physical point. Invariants computed from
this envelope differ from pointwise invariant extrema and are not guaranteed
conservative.

Only homogeneous **linear** shell stresses can be recovered from ``ShellStress``
and the thickness/shear factor. Transverse shear uses the existing constant
distribution, ``Q/(k*h)``. ``ShellStrain`` recovers the Reissner--Mindlin strain
kinematics for all supported shell laws, including the existing zero normal
strain component.
If the respective generalized field is absent, two distinct saved thickness
points reconstruct the homogeneous linear stress distribution or linear strain
kinematics. The linear reconstruction can also evaluate positions beyond the
saved thickness interval, within [-1,1]. One saved point is insufficient.
Stored nodal tensors remain unsuitable for global rotation, including when
resampling their linear distribution.

Shell thickness recovery and material-point kinematics use
``epsilon(z) = epsilon0 - z*kappa``, with ``kappa_xx = -d(RotY)/dx``.
Positive thickness is along the local shell normal. Generalized bending
moments are ``M = -integral(z*sigma)``.
Previously computed asymmetric-laminate and nonlinear responses may need to
be recomputed with the corrected kinematics.

Nonlinear shells and laminates save private ``_ShellStress`` tensors at their
actual material points whenever ``Stress`` or ``ShellStress`` is requested.
Linear laminates instead save the exact stress limits at both faces of each
layer. This permits later piecewise linear interpolation independently within
each layer, preserving stress jumps. Interpolation also works for stored tensors
with multiple thickness points. At positions outside the layer's integration
points, nonlinear stresses are linearly extrapolated by default from the two
outermost points in the same layer to its boundary. Set
``shell_options.update(extrapolate=False)`` to retain the nearest-IP value.
Extrapolated plastic stresses are estimates and may exceed the yield surface;
material states and equilibrium are unchanged. Stored-point reductions still
use only the saved values; recomputed reductions include recovered boundary
values. Exact/nearest stored selection remains unchanged, and positions outside
[-1, 1] are rejected. One thickness integration point gives
a constant approximation within that layer. A coordinate exactly on an interface
selects the lower layer; point indices can select either saved side.

``samples`` sets the generated point count per layer for recomputed reductions
and index selection. Actual integration points are also retained to capture
extrema. Stored sampling is fixed, so ``samples`` is ignored for that source.
The viewer's **Options → Shell…** dialog combines normalized position, point
index and extrema in **Value**, with **Coordinate system** at the end. A shell
submesh selector lists only compatible submeshes. Stored/Recomputed is required
on the reference submesh and is a preference, with fallback, on other submeshes.
A reference point index supplies its normalized position to other submeshes;
on the reference it retains the exact index, including either side of a laminate
interface. Stored coordinates in the viewer use the nearest point within the
requested layer. The API keeps interpolation as the default stored selection;
``stored_position="nearest"`` or ``"exact"`` selects other policies. Source
fallback is opt-in through ``source_fallback=True`` and ``reference_submesh``.
The dialog respects **Apply options to**.
Beam and shell viewer components default to global coordinates when the local
frame is available; otherwise the viewer uses local coordinates. Section tensors
remain stored locally and the results API retains its existing local defaults.

Beam ``BeamStress`` and ``BeamStrain`` fields are generalized quantities in
the local element frame, not continuum tensors. Their component orders are::

    BeamStress: N, QY, QZ, MX, MY, MZ
    BeamStrain: epsX, gammaY, gammaZ, kappaX, kappaY, kappaZ

Beam sections also recover continuum ``Stress`` and ``Strain`` at normalized
``(y, z)`` coordinates. Scalar beam positions are rejected. Pass one pair or
an array with shape ``(n_section_points, 2)``. ``position=None`` (the default)
extracts every configured section output point::

    section = fd.constitutivelaw.BeamRectangular(
        material, a=2, b=4, output_points=[[-1, 0], [0, 0], [1, 0]]
    )
    res = pb.add_output(
        "beam_results", beam_assembly,
        ["BeamStress", "BeamStrain", "BeamLocalFrame", "Stress", "Strain"],
        position=None,
    )

Raw stored beam tensors have shape
``(6, n_section_points, n_beam_gauss_points)``. Section points remain a separate
axis from beam integration points. Their normalized coordinates are saved with
the field, including per-Gauss-point coordinates when a pipe's radius ratio
varies. Node and Element output retain the section-point axis.

The continuum fields obtained this way are expressed in the **local beam
frame**. ``BeamLocalFrame`` stores nine components at each Gauss point, in
row-major order; each row of its 3-by-3 matrix is a local basis vector expressed
in global coordinates. It follows the current element frame during a nonlinear
solve. Keep this output at Gauss points for later tensor recovery: averaging
frames at nodes shared by differently oriented beams loses the section axes.
Stored nodal tensors therefore support local extraction but reject global
rotation. Stored element tensors support local extraction and global rotation
when the Gauss-point frames agree within each element. Otherwise use stored
Gauss-point tensors, or recompute and rotate before nodal/element averaging.

Post-processing saved generalized fields requires the section properties used
in the analysis, but does not require the original problem or assembly::

    from fedoo.util import BeamStressList, BeamStrainList

    # Load a frame first if results is a MultiFrameDataSet.
    force = BeamStressList(results["BeamStress"])
    stress = force.get_stress(section, position=(y, z))
    strain = BeamStrainList(results["BeamStrain"]).get_strain(
        section, position=(y, z)
    )
    stress_global = force.get_stress(
        section, position=(y, z), local_frame=results["BeamLocalFrame"]
    )
    results.gausspoint_data["Stress"] = stress_global.asarray()
    # The standard viewer can now display XX, YY, XY, vm, etc.

Recovery assumes isotropic linear elasticity and a centroidal section with
principal y/z axes. The local axial stress is
``N/A + z*MY/Iyy - y*MZ/Izz``. Transverse shear uses parabolic Jourawski
approximations for rectangles and solid disks, with peak factors 3/2 and 4/3
relative to ``Q/A``. The disk expression is a chord-average approximation,
not the complete two-dimensional elasticity distribution. Pipes and generic
sections retain the section-average ``Q/A`` approximation.
Circular, pipe and rectangular sections include Saint-Venant torsion. The
rectangle uses 128 odd Fourier harmonics for both stress and strain recovery,
with a consistent series-based torsion constant replacing the previous
polynomial approximation. This describes free warping, not restrained-warping
axial stresses. Nonzero torsion in other section shapes raises
``NotImplementedError`` because a torsion constant
alone does not specify the pointwise stress distribution. Recovered strain
is recovered directly from ``BeamStrain`` using beam kinematics, without
``E``, ``G`` or Poisson's ratio. Its axial component is
``epsX + z*kappaY - y*kappaZ``. Engineering shear uses the generalized
``gammaY`` and ``gammaZ`` without multiplying by the shear correction factor.
Circular and pipe sections add ``-z*kappaX`` to ``gammaXY`` and
``y*kappaX`` to ``gammaXZ``; other sections require a warping model for
nonzero twist. ``epsYY``, ``epsZZ`` and ``gammaYZ`` are zero under these beam
kinematics. No isotropic Poisson contraction is inferred. These kinematic
strains are distinct from a full three-dimensional constitutive strain recovery.

The dataset and viewer share the predefined field/component dictionary
``fedoo.util.field_components.FIELD_COMPONENTS``. Beam components therefore
have physical labels rather than numerical indices, and named extraction such
as ``results["BeamStress", "MY"]`` uses the same labels. Unknown user fields
retain numerical component labels.

Automatic recovery from FDH5 beam results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Gauss-point outputs of ``BeamStress`` or ``BeamStrain`` automatically
include section metadata shared by both fields, associated with their submesh.
FDH5 writes typed HDF5 datasets under
``results/iter_N/section_metadata/submesh_I`` instead of JSON. Uniform properties
are scalars in this group; varying properties are private ``_Section_*`` fields
with an explicit Node, Element or GaussPoint association. Metadata references
these fields. Definitions are preserved per frame and per submesh.
A reloaded file therefore needs no live solver
or constitutive-law object to display section tensors::

    res = pb.add_output(
        "beam_results", beam_assembly,
        ["BeamStress", "BeamStrain", "BeamLocalFrame"],
    )
    # Later, after reading the FDH5 file and loading the desired frame:
    local_vm = results["Stress_local", "vm"]
    global_xx = results["Stress_global", "XX"]
    local_strain = results["Strain_local", "XX"]
    fd.viewer(results)

Native recovery follows ``BeamStress`` → ``Stress`` and ``BeamStrain`` →
``Strain``. Neither path needs elastic constants in the saved section
description. Older descriptions containing isotropic constants are still read,
but these constants do not influence kinematic strain recovery.

``Stress``, ``Stress_local`` and ``Stress_global`` share one selection policy
(likewise for strain). Global components require ``BeamLocalFrame``, which is
included automatically with beam stress or strain output, as for shells. The viewer
shows one Stress and one Strain entry. **Options → Beam…** selects the source,
coordinate system, section position/index and reduction. Its optional Beam
toolbar is hidden by default.

For mixed meshes, Beam options lists compatible beam submeshes as sampling
references. Choosing an index resolves its normalized coordinates on that
reference section; the same coordinate request applies to every beam submesh.
The preview highlights the nearest stored point when stored data is selected.
The viewer requires Stored/Recomputed on the selected reference submesh. On
other beam submeshes it uses that source when available and falls back to the
other source otherwise. Stored fallback uses the nearest normalized point;
recomputed fallback evaluates the requested normalized coordinates. These
preferences remain active when changing result frames.

``results.beam_options.update(...)`` validates updates atomically:

* ``source="auto"`` (default) prefers stored tensors when available, otherwise
  recomputes from the respective generalized field. Explicit ``"stored"`` or
  ``"recomputed"`` raises an error if the required data is unavailable.
  The viewer enables ``source_fallback=True`` for mixed meshes and sets
  ``reference_submesh`` to its sampling reference. The reference remains strict;
  other submeshes use the preferred source when available, otherwise the other
  source. API selections remain strict by default (``source_fallback=False``).
* ``reduction="abs_max"`` (default) returns the largest absolute magnitude as a
  nonnegative value. ``"signed_abs_max"`` retains the selected sign, choosing
  the first point for magnitudes tied within relative tolerance 1e-9. Both
  propagate NaNs. ``"max"`` and ``"min"`` use algebraic
  extrema and retain the sign (largest tension and largest compression for a
  normal stress).
* ``reduction=None`` selects one point using normalized ``position=(y,z)`` or
  zero-based ``point_index``. Supplying both non-None in one update is rejected.
  Selecting either clears the other. A reduction ignores both selectors.
* ``stored_position="nearest"`` (default) chooses the closest saved point in
  normalized (y,z), independently on each submesh and at each saved beam
  location. Equal-distance ties choose the first point. This selects a saved
  value without interpolation. ``stored_position="exact"`` requires a matching
  point at every location. Recomputed positions can be any coordinates supported by the
  recovery model, including coordinates beyond the bounding box.
* Section metadata declares ``recovery="linear"`` (the default for existing
  sections and legacy files) or ``"stored_only"``. Set ``section.recovery`` or
  override it in a section subclass. ``stored_only`` disables recovery from
  generalized fields even when those fields are saved; stored tensors remain
  available for point selection and reductions.
* ``samples`` changes generated recomputed sampling only. Stored points and
  explicitly supplied output points remain fixed.

For example::

    results.beam_options.update(source="stored", reduction=None, point_index=2)
    stress_at_point = results["Stress", "XX"]
    results.beam_options.update(position=(0.4, -0.2), stored_position="nearest")
    stress_at_nearest_saved_point = results["Stress", "XX"]
    results.beam_options.update(source="recomputed", position=(0.5, -0.25))
    stress_at_position = results["Stress", "XX"]
    results.beam_options.update(reduction="abs_max", samples=128)
    largest_magnitude = results["Stress", "XX"]
    maximum_vm = results["Stress", "vm"]

Zero is the section origin; -1 and +1 select the respective negative and
positive bounding faces. Asymmetric sections have separate scales on each
side. Bounds describe the geometry, not merely the extent of output points.
Generic sections without supplied bounds use ``(-1,1,-1,1)``, giving identity
mapping and thus physical coordinates. Bounding-box coordinates need not lie
in the material, for example inside a pipe hole.

Section reductions evaluate the requested scalar (including invariants) at
each point before selecting an extremum, and before nodal/element averaging.
With no scalar component requested, each tensor component is reduced
independently, returning ``(6, n_locations)``. Components may originate at
different section points; the result is an envelope, not a tensor at one
physical location. Computing Mises or principal values from that envelope is
not equivalent to reducing pointwise invariants and is not guaranteed
conservative. Use ``reduction=None`` with a position or index for a physical
point tensor. Extrema are sampled estimates; saving and viewer defaults are
unchanged.

Standard sections generate output points using ``n_points=32`` by default,
an approximate total point budget including boundaries and interiors. Pipes
exclude the hole. Increasing ``samples`` refines recomputed grids. A custom
section's fixed point set is not resampled. In the viewer, **Sample points number**
snaps to the closest supported count when editing finishes; fixed point counts
are displayed without allowing edits. For equal distances, the lower count wins.
An unsampled custom
section without supplied points or a sampling method requires a direct
position with ``reduction=None``; the viewer disables indices and reductions.
Recomputation still requires the recovery model for the requested stresses.

Section classes own storage and geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``BeamProperties.section_description()`` defines the portable section schema.
Subclasses declare a unique ``section_type`` and their ``section_dimensions``;
they are registered automatically. Implement ``sample_points(n_points=32)``,
``scale_section_points(points)`` and optionally ``section_mesh(n_points=32)``
to extend recovery without adding shape-specific code to FDH5 or the viewer.
The section class must be imported when recovering its saved results.

``section_mesh`` wraps existing mesh generators and accepts their optional
arguments. It returns a reference mesh in the centroid-relative local section
plane. ``n_points`` is an approximate total point budget for sampling and
meshing; their discretizations need not be identical. Rectangles sample mesh
nodes; circles and pipes retain analytical polar sampling.

Generic ``BeamProperties`` accepts optional ``section_points`` or
``section_mesh`` and ``section_scale=(scale_y, scale_z)``. Optional ``output_points`` are normalized pairs; ``n_points`` controls generated
output sampling. Supply true ``section_bounds=(ymin,ymax,zmin,zmax)`` for custom
geometry; bounds are never inferred from output points. Supplied section-mesh coordinates
must already be centred on the area centroid and aligned with principal axes.
Points are scaled about that origin, not the bounding-box centre. Only output
points and reference bounds are saved for custom meshes, not their connectivity.
Section stiffness properties must still be provided explicitly; a supplied
mesh does not compute a warping solution or enable noncircular torsion recovery.

Variable dimensions/properties default to GaussPoint storage after conversion
by the assembly. To retain another association, set, for example,
``section.section_property_associations = {"r": "Element", "A": "Element",
"Jx": "Element", "Iyy": "Element", "Izz": "Element"}`` before output.
Recovery converts these arrays to the saved beam Gauss-point locations.
Node-associated section arrays with element-subset output are currently rejected.

The viewer hides underscore-prefixed fields by default. **Options → Show
internal fields** exposes them. ``add_output(..., private=True)`` prefixes
requested mesh fields with ``_``. **Options → Beam… → Section geometry and
sampling…** displays section points and properties at a selected longitudinal
location. Select **Section point index** to evaluate Stress/Strain at a normalized
output point scaled separately at each beam location. In Python, use
``results.beam_options.update(reduction=None, point_index=0, samples=32)``;
updating ``position`` clears the previous index automatically.
Older JSON-based section descriptions remain readable.

Beam options respect **Options → Apply options to**. For **All windows**, each
beam pane evaluates the settings using its own section definition. Non-beam
panes are unaffected. If an index or coordinate-system choice is unsupported
in a target pane, no panes are changed and the viewer identifies the incompatible
pane. Use **Active window** to configure different sections independently.

In sample-index mode, the local y/z boxes display read-only physical coordinates
at the selected longitudinal Gauss point of the active submesh. Changing that
location updates the coordinates for varying sections; it does not restrict
the plotted field to that location. The index still follows the section mapping
throughout the mesh. The section preview highlights the selected sample.

Older FDH5 files cannot infer section geometry from resultants alone. If the
original section properties are available, attach them to a single-mesh dataset
before viewing or resaving::

    results.set_beam_section(section)

Nonuniform properties must follow the saved Gauss-point order. Files without
these descriptions continue to expose their original generalized fields.
Automatic recovery requires unaveraged Gauss-point generalized fields and
frames; requesting their output at nodes does not preserve this information.

Save data to disk
-----------------

Once a DataSet is created using for instance the
:py:meth:`fedoo.Problem.get_results` method, the data can easily be saved on
disk using for instance the :py:meth:`fedoo.DataSet.save` method.

FDH5 is the default and recommended format. If ``DataSet.save``,
``MultiFrameDataSet.save_all``, or ``read_data`` is called with a filename
without an extension, the ``.fdh5`` extension is assumed.

The available file types are:
    * 'fdh5': Fedoo's native HDF5 result format (default). See
      :ref:`fdh5_format` for its contents and write policies.
    * 'fdz': A legacy zipped archive containing the mesh using the 'vtk' format
      named '_mesh_.vtk', and data from several iterations named 'iter_x.npz'
      where x is the iteration number (x=0 for the 1st iteration).
    * 'vtk': The vtk format contains the mesh and the data in a single files.
      The gauss points data are not included in the file.
      This format is efficient for a linear problem when we need only one time
      iteration. In case of multiple saved iterations, a directory is created
      and one vtk file is saved per iteration. The mesh is included in every
      file which is not memory efficient.
    * 'msh': Format associated to gmsh. Have the same drawback as the vtk
      format for time depend results and missing gauss points data. The vtk
      format should be prefered.
    * 'npz': Save data in a numpy file npz which doesn't include the mesh. The
      mesh is generally saved beside in a raw vtk files without results.
    * 'npz_compressed': Same as npz with a compression of the zip archive.
    * 'csv': Save DataSet that contains only one type of data
      (ie Node, Element or Gauss point data) in a csv file (needs the library
      pandas installed).
      The mesh is not included and may be saved beside in a vtk file.
    * 'xlsx': Same as csv but with the excel format.

.. _fdh5_format:

FDH5 format
~~~~~~~~~~~

FDH5 stores the mesh, node and element sets, node/element/Gauss-point fields,
scalar outputs such as Time, and multiple saved frames in one file. It
preserves the complete MultiMesh structure and its per-submesh fields.
Beam/shell section descriptions and output-point coordinates are saved when
available, and local frames can be saved for recovery in global coordinates.
No conversion to nodal fields is required for storage.

.. note::

   Opening an FDH5 file in the viewer while its simulation is running is
   supported and should normally work: temporary read/write lock conflicts
   are retried. Persistent contention can still exceed the retry timeout and
   interrupt simulation output. Keep HDF5 file locking enabled; for critical
   runs, wait until the simulation finishes before opening the results.

The :doc:`FDH5 file layout <fdh5>` describes the HDF5 groups and array
conventions. That page is generated from the :mod:`fedoo.util.fdh5` module
docstring, keeping the detailed format description alongside its implementation.

The file structure below is illustrative: optional field and metadata groups
are present only when the corresponding outputs have been saved.

.. literalinclude:: /../fedoo/util/fdh5.py
   :language: text
   :start-after: .. code-block:: text
   :end-before: Field arrays retain their Fedoo shapes.
   :dedent: 4

FDH5 section tensors retain their point axis: stored Stress/Strain arrays
have shape ``(6, n_section_points, n_locations)`` when multiple section points
are saved. Use ``results.get_data`` with ``beam_options`` or ``shell_options``
to select one point or reduce across points. Field descriptions are saved
per frame and per submesh, so recovery can be performed after reopening the
file without the original material object. Global recovery additionally
requires the corresponding saved local-frame field.

The file stores the requested outputs and recovery metadata, rather than a
complete solver state for restarting an analysis. Dataset selection options
and viewer settings are not saved; read-time defaults apply after reopening.
Older files without section descriptions remain readable, but may only
provide their original fields.

For automatic simulation output, ``Problem.add_output`` accepts
``write_mode="overwrite"`` (default), ``"append"`` or ``"error"`` for FDH5.
Appending adds saved frames to an existing compatible mesh file; it does not
resume the solver. In contrast, ``DataSet.to_fdh5(..., overwrite=True)``
replaces the whole file, not only an existing iteration.

FDH5 compression
^^^^^^^^^^^^^^^^

FDH5 uses lossless gzip compression at level 4 by default. The viewer's
File > Save as dialog offers None, gzip (levels 1-9), and Zstandard
(levels 1-22, default 3). The last successful choice is reused by Save for
that dataset. Zstandard is optional: install ``pip install fedoo[zstd]``
or ``pip install hdf5plugin`` on both writers and readers. Fedoo registers
the filter automatically; other HDF5 readers also need a compatible filter.
Gzip remains the portable default and requires no extra dependency.

.. code-block:: python

    from fedoo.util.fdh5 import CompressionConfig

    data.to_fdh5("result.fdh5", overwrite=True, compressed="zstd")
    data.save("result.fdh5", compressed=CompressionConfig("zstd", 5))
    results.save_all("history.fdh5", compressed="zstd")
    data.save("uncompressed.fdh5", compressed=False)

Existing ``compressed=True`` and integer gzip levels retain their meaning.
Compression settings apply to newly written arrays; appending with a
different codec does not recompress earlier frames. Higher Zstandard
levels can need considerably more time and memory; start with level 3.


Read data from disk
-------------------

To read data saved on disk, use the function :py:func:`read_data`.
The data are imported as :py:class:`DataSet` or
:py:class:`MultiFrameDataSet` objects depending on the imported file(s).
FDH5 is used by default when the filename has no extension.

.. code-block:: python

    results = fd.read_data("simulation.fdh5")
    if isinstance(results, fd.MultiFrameDataSet):
        results.load(-1)  # load the last saved frame
    fd.viewer(results)

One saved frame returns a DataSet; several frames return a MultiFrameDataSet
whose frame fields are loaded on demand. ``load(i)`` selects the zero-based
index in the list of saved frames, not a Newton iteration number.
The viewer discovers new FDH5 frames during navigation; Reload also rereads
the current frame. Existing all-frame color limits and history plots are not
automatically recomputed. Reapply the limits or recreate the plot to include
new frames. If the simulation temporarily locks the file, retry after it
finishes writing the frame.
FDH5 writes retry temporary reader lock conflicts for up to 10 seconds, so a
brief viewer read does not abort output. Eager frame reads release their handle
before plotting. Keep HDF5 file locking enabled; read-only access still needs
protection against a writer changing the file. Explicit lazy-reader contexts
must be closed promptly to let the simulation write.


.. _very_simple_problem:

Example
-------

For example, defining and solving a very simple problem :

.. code-block:: python

    import fedoo as fd

    fd.ModelingSpace("2Dstress")
    mesh = fd.mesh.rectangle_mesh()
    material = fd.constitutivelaw.ElasticIsotrop(2e5, 0.3)
    wf = fd.weakform.StressEquilibrium(material)
    assembly = fd.Assembly.create(wf, mesh)

    # Define a new static problem
    pb = fd.problem.Linear(assembly)

    # Boundary conditions
    pb.bc.add('Dirichlet', 'left', 'Disp',    0 )
    pb.bc.add('Dirichlet', 'right', 'Disp',  [0.2,0] )

    # Solve problem
    pb.solve()


Then, we can catch the Stress, Displacement and Strain fields using:

.. code-block:: python

    results = pb.get_results(assembly, ["Stress", "Disp", "Strain"])
    # plot the sigma_xx averaged at nodes
    results.plot("Stress", component='XX', data_type='Node')


.. image:: ./_static/examples/example_plot_dataset.png

Alternatively, if we take the same problem, but accounting for geometric
non linearities (nlgeom = True), we can automatically save results at specified
time interval (here the results are saved on a file).


.. code-block:: python

    wf.nlgeom = True
    pb_nl = fd.problem.NonLinear(assembly)

    # Boundary conditions
    pb_nl.bc = pb.bc

    results_nl = pb_nl.add_output('nl_results', assembly, ["Stress", "Disp", "Strain"])

    pb_nl.nlsolve(dt = 0.1, tmax = 1, interval_output = 0.2)

    # plot the sigma_xx averaged at nodes at the last increment
    results_nl.plot("Stress", component='XX', data_type='Node')

.. image:: ./_static/examples/example_plot_dataset_nl.png

.. _fedoo_viewer:

Fedoo interactive viewer
========================

Fedoo includes a graphical viewer for result files, DataSet objects and live
Problems.
Install the plotting and GUI dependencies with ``pip install "fedoo[plot,gui]"``.

Then the viewer can either be launched as a standalone application from
command line:

.. code-block:: none

    $ python -m fedoo.viewer

or from Python:

.. code-block:: python

    import fedoo as fd
    result = fd.read_data('myfile.fdh5')  # load a DataSet from file

    fd.viewer()  # start the viewer with no file opened
    fd.viewer(result)  # start the viewer and open the result DataSet
    fd.viewer('myfile.fdh5')  # start the viewer with the data from a file
    fd.viewer(pb)  # inspect an existing Problem in the current Python session

The viewer includes the following tools and features:

* Manage independent views with Windows > Arrange Windows (tabbed,
  horizontal, vertical or automatic layout). Windows > Link windows links
  fields, iterations and/or cameras; Options > Apply options to separately
  selects whether plotting options affect the active view or all views.
  Unlinked views retain their own selected frame. Linked iteration navigation
  skips views that do not contain the requested frame. Windows > Copy Window
  creates another view of the current results.
* Field and iteration selectors for data exploration. The iteration spinbox
  decreases on the left and increases on the right.
* Automatic discovery of new FDH5 frames when navigating an ongoing simulation.
  The Reload button also rereads the current frame. A file temporarily locked
  by the writer can be retried on the next navigation or Reload.
  Existing all-frame color limits and history plots are kept: reapply
  All frames Min/Max or recreate the history plot to include new frames.
* A wide range of plotting options.
* Show saved shell thickness in Shell options (disabled by default).
  Each ``tri3``/``quad4`` shell submesh uses its saved section thickness,
  including varying thickness, along normals of the deformed midsurface.
  Beams and solids retain their own geometry. The selected field is repeated
  through the display thickness; this does not recover distinct face stresses
  or change the simulation mesh. Save shell stress/strain or generalized
  shell fields to include the section metadata needed for this display.
  Clicking a top/bottom display node reports its original node ID and the
  displayed field under the current section selection. Gauss-point displays
  retain element discontinuities; Element displays report adjacent element
  values at a shared node.
* Open Tools > Element Sets or Node Sets to manage selections. Element Sets
  can show or hide elements from predefined sets, rectangular selections,
  or arbitrary expressions; Node Sets manages nodes without visibility controls.
  Pick on screen displays a rectangle while dragging. New creates a named set
  from the current selection; Replace, Add and Remove load a set into the
  selection using the chosen operation.
* Manage node sets (selection, create, rename, remove) without visibility
  controls. Node Sets offers Create from element set; Element Sets offers
  Create from node set. Choose a source set in the dropdown and click Create.
  For element creation, a checkbox switches between requiring all element
  nodes or at least one node; its label shows the current rule.
  Node-to-element conversion can require all nodes or any node; element-to-node
  conversion collects unique nodes. MultiMesh element ids remain global.
* Grow an exterior surface selection in Node Sets. Click Select to pick a
  boundary node using the existing single-node picker, then select the patch
  with the chosen maximum normal angle. All viewer seeds are nodes;
  interior nodes are rejected.
  Corner seeds offer a starting-face choice. Boundary edges mode follows a
  line on a 2D mesh, including planar meshes stored with XYZ coordinates;
  Surface faces mode traverses shell faces. The result is a node selection
  that can be saved as a named set and reconstructed with ``extract_surface``.
  Exterior edges (3D) follows creases and open shell boundaries. Sharpness
  controls the minimum face-normal angle identifying a crease; Turn limit
  controls the allowed direction change along the edge. Branches stop
  traversal, and ambiguous corner seeds offer a starting-edge choice.
  Edge results are node IDs, including midside nodes, and can be saved as sets.
* Use field/component names such as ``Stress_XX`` in selection expressions.
  Element tests use element-center positions and element field values;
  node tests use nodal positions and values. Field conversions follow DataSet.
* Compute scalar expressions from fields, existing scalars and named sets.
  The help button beside Expression opens the full guidance and examples in
  a separate window, which can stay open while entering an expression.
  For example ``sum(Disp_X[nodes("Top")])``,
  ``mean(Stress_XX[elements("Wall")])`` with Element field values, or
  ``sum(scalar("Fext(RigidDisp)")[2])``. The result must be one finite number.
  Compute / update scalar evaluates the current frame immediately. Other
  frames are evaluated when loaded or requested by a history plot, rather than
  eagerly processing the whole file. File > Save or Save as evaluates and
  exports every frame of a multiframe dataset, including numeric scalar values,
  expression definitions and user-created sets. Definitions remain editable
  after reopening. A live Problem exports only its selected state or mode.
* Plot results along an interactively defined line.
* Tools > History plot accepts field/component names in quick search.
  For global DOFs and their associated forces, ID indexes the saved output
  array, not a node, element or Gauss point; changing the field association
  does not convert these global values.
* Optionally open Windows > Object inspector to browse the active dataset's
  mesh, named sets, stored fields, computed scalars and live Problem boundary
  conditions. Set and condition selections highlight their geometry; the
  read-only properties include counts, field shapes/ranges and expressions.
  Automatically generated submesh element sets are omitted here because the
  corresponding submesh already appears under Mesh. User-created sets remain
  visible, including sets whose names start with ``submesh_``.
  Manage delegates edits to the existing tools. The panel can float or
  dock on either side; fullscreen hides it and restores its prior visibility.
  A docked inspector returns to a compact width when opening another dataset.
  Right-click an element set, mesh, or submesh for Show only and a checkable
  Mask action. Unchecking Mask restores that object's elements without
  resetting other visibility choices. Visibility uses the same element
  filter as Element Sets, and the properties show partial masking.
* Optionally show the current ``Time`` scalar in the upper-right corner.
  It is hidden by default; enable Show time in Options > Plot.
  The existing active/all views target
  applies to this option too.
* Open a live Problem directly, or use a ``ProblemDataSet`` wrapper. The
  Problem > Open existing problem dialog also lists Problems from the current
  Python session's workspace variables, the named Fedoo registry and the
  active Problem. Refresh updates the list; reopening a Problem refreshes
  its existing viewer pane; different Problems open separate panes. Use
  Windows > Copy Window for another view when needed. This reads existing
  objects without running a solve. Reload reads its current state; Save as
  exports a single-state FDH5 snapshot.
  Problem > Boundary conditions inspects and highlights existing constraints,
  and adds/edits nodal Dirichlet or Neumann conditions using node sets or
  the nodes of element sets. Neumann values are per-node forces, not pressure.
  Custom constraints can be inspected. Edits affect the live Problem; solve
  from Python to update its solution. BC definitions are not exported to FDH5.
* File > Save as exports FDH5 with None, gzip or optional Zstandard
  compression. File > Save reuses that view's last successful compression
  choice. All compression options are lossless. User-created node and element
  sets are included; automatically generated All/submesh selection sets are
  not exported. Submesh geometry and names are preserved independently of
  these generated sets. Display visibility masks do not remove mesh elements
  from the saved result. See :ref:`fdh5_format` for compression dependencies.
* Clip the current mesh using an interactively defined plane.
* Save figures and create movies using the current visualization settings.


Basic operations
================

Live problem results
--------------------

``ProblemDataSet`` implements the DataSet interface for one current Problem
state. It extracts results using ``problem.get_results`` without running a
solve. Default fields include available DOFs, forces and stress/strain;
pass ``fields`` to request a specific set. Reading through ``get_data``
refreshes automatically; call ``refresh`` before reading the storage
dictionaries directly. Copies are detached DataSet snapshots.

Pass the Problem directly to ``fd.viewer(pb)``; creating the wrapper explicitly
is optional. In an already running viewer, **Problem > Open existing problem**
finds Problems in that same Python session. A standalone viewer cannot access
objects in a separate Python process. **Reload** refreshes the current results,
and **Problem > Boundary conditions** edits the live constraints; solving
remains an explicit operation in Python.

.. code-block:: python

    import fedoo as fd

    # pb is an existing Problem, for example after pb.solve() or pb.nlsolve().
    fd.viewer(pb)  # inspect its results and manage live boundary conditions

    # Optionally choose which fields the live wrapper exposes.
    current = fd.ProblemDataSet(pb, fields=["Disp", "Stress"])
    current.get_data("Disp", "X")
    current.save("current.fdh5")
    fd.viewer(current)

For modal and linear buckling Problems with computed modes, the viewer's
iteration slider, spinbox and Play button navigate the retained modes with
the same zero-based indexing as FDH5 frames. The ``Mode`` scalar is one-based,
as in saved mode output. Selection calls ``problem.set_mode`` and refreshes
the live fields and frequency/load-factor metadata without another eigensolve.
Reload discovers modes computed after opening the viewer. From Python use
``current.set_mode(index)``; ``current.n_modes`` gives the retained count.

Other Problems retain one current state. There is no accumulated time history
in a ProblemDataSet. Export saves the selected mode/current state's fields,
computed scalars and named sets; it does not create a solver restart file.

Shared mathematical expressions
-------------------------------

``fedoo.util.expressions.evaluate_expression`` provides the same evaluator
used by mesh selection and viewer tools. It accepts a mesh for positions
and an optional DataSet for fields and scalar outputs. ``X``, ``Y``, ``Z``
and ``ID`` refer to nodes or element centers, depending on ``data_type``.
Scalar computations use reference positions; viewer selection additionally
offers deformed positions. Logical ``and``/``or`` and ``&``/``|`` operate
elementwise. Named mathematical functions can also use the ``np.`` prefix.
Arbitrary Python statements and attribute access are not supported. The
expression language uses an explicit list of supported operations and functions,
not Python ``eval``. Expensive expressions can still consume substantial memory
or computation time; there are no strict resource limits. Stored definitions
are evaluated when frames load, so use trusted expression definitions.

.. code-block:: python

    from fedoo.util.expressions import evaluate_expression

    ids = mesh.find_nodes("Disp_X > 0 and X < 1", data=results)
    ids = mesh.find_elements("Stress_XX > 100", data=results)
    total = evaluate_expression('sum(Disp_X[nodes("Top")])', data=results)
    results.add_scalar_expression("TopDisplacement", 'sum(Disp_X[nodes("Top")])')
    results.save("computed.fdh5")  # one DataSet
    # For a MultiFrameDataSet, use results.save_all("computed.fdh5").

``field("Stress_XX", "Element")`` explicitly selects a field association.
``scalar("name")`` handles scalar names containing punctuation, and
``node_sets["Top"]`` / ``element_sets["Wall"]`` also provide set indices.

The principale methods/functions to extract, plot and manage
result data are listed in this section.

Extract data
------------

.. autosummary::
   :toctree: generated/

   DataSet.get_data
   MultiFrameDataSet.get_history


Plotting results
----------------

A few convenient methods are proposed to generate images or movies from
:py:class:`DataSet` and :py:class:`MultiFrameDataSet` objects.

.. autosummary::
   :toctree: generated/

   DataSet.plot
   MultiFrameDataSet.plot
   MultiFrameDataSet.plot_history
   MultiFrameDataSet.write_movie


Save results
------------

.. autosummary::
   :toctree: generated/

   DataSet.save
   DataSet.savez
   DataSet.savez_compressed
   DataSet.save_mesh
   DataSet.to_csv
   DataSet.to_excel
   DataSet.to_vtk
   DataSet.to_msh
   DataSet.to_fdh5
   MultiFrameDataSet.save_all


Read results
------------

.. autosummary::
   :toctree: generated/

   read_data
   DataSet.read
   DataSet.load
   MultiFrameDataSet.load


Advanced operations
===================

Write Movies
------------

A very simple way to write a movie from a :py:class:`MultiFrameDataSet` is
to call the embedded method :py:meth:`MultiFrameDataSet.write_movie`.

Though this method comes with lots of options, one may sometimes want to
fully control the movie rendering. This is easy to do by manualy writing the
movie using the pyvista library.

Here is an exemple to animate the linear results obtained in the
:ref:`problem <very_simple_problem>` defined above. The idea is to use a
scale_factor applied to the displacement (using the *scale* argument) and to
the stress field (modifiying the results data).

.. code-block:: python

    import pyvista as pv

    results = pb.get_results(assembly, ['Stress', 'Disp'], 'Node')
    stress = results.node_data['Stress']
    clim = [stress[3].min(), stress[3].max()]  # 3 -> xy in voigt notation

    pl = pv.Plotter(window_size = [600,400])
    pl.open_gif("my_movie.gif", fps=20)
    sargs = dict(height=0.10, position_x=0.2, position_y=0.05)
    for i in range(48):
        scale_factor = (i + 1) / 48
        results.node_data["Stress"] = scale_factor * stress
        results.plot(
            "Stress",
            "XY",
            plotter=pl,
            scale=scale_factor,
            clim=clim,
            title=f"Iter: {i}",
            title_size = 10,
            scalar_bar_args=sargs,
        )
        pl.hide_axes()
        pl.write_frame()

    pl.close()

.. image:: ./_static/examples/my_movie.gif


Multiplot feature
-----------------

It is possible to create the plotter before calling the plot function.
This allow for instance to use the pyvista multiplot capability.
For instance,
we can plot the stress results after the example
:ref:`example_plate_with_hole`:

.. code-block:: python

    import pyvista as pv

    pl = pv.Plotter(shape=(2,2))
    # or using the backgroundplotter:
    # from pyvistaqt import BackgroundPlotter
    # pl = BackgroundPlotter(shape = (2,2))
    results.plot('Stress', 'vm', 'Node', plotter=pl)
    pl.subplot(1,0)
    results.plot('Stress', 'XX', 'Node', plotter=pl)
    pl.subplot(0,1)
    results.plot('Stress', 'YY', 'Node', plotter=pl)
    pl.subplot(1,1)
    results.plot('Stress', 'XY', 'Node', plotter=pl)
    pl.show()

.. image:: ./_static/examples/multiplot.png
"""

from .axi_to_3d import (
    axi_to_3d,
)

__all__ = [
    "axi_to_3d",
]
