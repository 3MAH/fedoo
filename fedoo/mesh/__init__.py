"""
===========================================
Geometry and Mesh (:mod:`fedoo.mesh`)
===========================================

.. currentmodule:: fedoo

Mesh classes
====================

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   Mesh
   MultiMesh

.. _mesh_selection:

Node and element selection
==========================

Use the selection methods to obtain node or element IDs without changing
mesh geometry. Pass ``name`` to ``find_nodes`` or ``find_elements`` to also
store the result in ``mesh.node_sets`` or ``mesh.element_sets``. Node IDs
refer to the shared node table. For a MultiMesh, global element IDs follow
the concatenated submesh order.

.. autosummary::
   :toctree: generated/

   Mesh.find_nodes
   Mesh.find_elements
   Mesh.get_elements_from_nodes
   mesh.find_surface_nodes
   mesh.find_edge_nodes

Coordinates and field expressions
--------------------------------

``find_nodes`` selects by node coordinates. ``find_elements`` tests element
centers by default; use ``select_by="nodes"`` with ``all_nodes=True`` or
``False`` to require all nodes or at least one node to satisfy the criterion.
Both methods accept mathematical expressions, and ``data=results`` makes
result fields and components such as ``Stress_XX`` available. Fields are
converted to Node or Element values according to the selection method.

.. code-block:: python

   import fedoo as fd

   nodes = mesh.find_nodes("X", 0, name="Left")
   elements = mesh.find_elements("X > 0.5", name="Right")
   loaded_nodes = mesh.find_nodes("Disp_X > 0.01 and Y > 0", data=results)
   stressed_elements = mesh.find_elements("Stress_XX > 100", data=results)

Use ``mesh.get_elements_from_nodes(nodes)`` to select elements from node IDs.
See :ref:`fedoo_viewer` for the corresponding interactive node/element set
managers and conversions.

Connected exterior surface selection
------------------------------------

``mesh.find_surface_nodes(seed, seed_type="Node", angle=30)`` returns sorted
node IDs on the connected exterior patch containing a node or global element
seed. It crosses neighboring faces when their normal-angle difference is at
most ``angle`` degrees: 0 joins flat faces; 30 stops at a right-angle edge.
The test is local, so a smoothly curved patch can wrap around the geometry.

.. code-block:: python

   nodes = mesh.find_surface_nodes(top_node_id, angle=30)
   mesh.node_sets["selected_surface"] = nodes
   surface = fd.mesh.extract_surface(mesh, node_set=nodes)

The same function is available as ``fd.mesh.find_surface_nodes(mesh, ...)``.
Solid internal faces are removed, including interfaces between MultiMesh
submeshes. Midside nodes are included; normals use corner geometry. For a
three-dimensional shell mesh, existing tri/quad faces are traversed instead;
use ``mesh.extract_elements(mesh.get_elements_from_nodes(nodes))`` to extract
the selected shell patch. For two-dimensional meshes, the selection follows
exterior boundary edges. Non-manifold edges stop the selection.

A seed at a corner may belong to distinct patches. In this case
``AmbiguousSurfaceSeed`` exposes candidate ``face_ids`` and ``normals``;
repeat the call with ``seed_face`` set to the desired exterior face ID.
The viewer presents this choice in a dialog. Node Sets provides a seed ID,
a maximum normal angle, and **Select surface nodes**. **Select** activates
the single-node picker. All viewer surface/edge selection uses node seeds.
Interior nodes cannot seed a patch. For planar XYZ meshes use
**Boundary edges (2D mesh)**;
the equivalent Python option is ``boundary_edges=True``.
The resulting node selection can be saved as a named set and exported to FDH5.

Connected exterior edge selection
---------------------------------

For a 3D mesh, **Exterior edges (3D)** selects nodes along creases and open
shell boundaries. **Sharpness** is the minimum angle between adjacent exterior
face normals that identifies a crease. **Turn limit** is the maximum direction
change allowed while following that edge. Smooth surface tessellation lines
are excluded; branches and sharp changes of direction stop traversal.
The seed must lie on an eligible edge. A corner touching distinct edge chains
offers a starting-edge choice. Midside nodes are included.

.. code-block:: python

   nodes = mesh.find_edge_nodes(edge_node_id, angle=30, feature_angle=30)
   mesh.node_sets["selected_edge"] = nodes

The same function is available as ``fd.mesh.find_edge_nodes(mesh, ...)``.
``AmbiguousEdgeSeed`` exposes candidate ``edge_ids``, ``endpoints`` and
``directions``; repeat with ``seed_edge`` to choose one chain explicitly.

.. currentmodule:: fedoo.mesh

Mesh manipulation functions
==================================

.. autosummary::
   :toctree: generated/

   stack
   extrude
   thicken
   extract_surface
   extract_edges
   change_elm_type
   quad2tri

.. _importmesh:

Import and export meshes
==================================

Use :py:meth:`fedoo.Mesh.read` to import mesh geometry and connectivity:

VTK and other PyVista formats
----------------------------

VTK formats are read through PyVista: legacy ``.vtk`` meshes, XML
unstructured grids (``.vtu``) and XML polygonal surfaces (``.vtp``).
The imported geometry must contain cell types supported by Fedoo.

.. code-block:: python

   import fedoo as fd

   mesh = fd.Mesh.read("model.vtu", name="Domain")
   # Legacy VTK meshes and polygonal surfaces:
   # mesh = fd.Mesh.read("model.vtk")
   # mesh = fd.Mesh.read("surface.vtp")

:py:meth:`fedoo.Mesh.from_pyvista` also accepts existing PyVista
``UnstructuredGrid`` and ``PolyData`` objects, including mixed cell types.
It imports geometry and Fedoo's ``_nset_``/``_eset_`` field-data conventions,
but not ordinary point/cell result arrays. Element sets are not imported
through the PyVista path for mixed meshes.

Abaqus import
-------------

Use :py:meth:`fedoo.Mesh.read` to import the mesh from an Abaqus deck:

.. code-block:: python

   mesh = fd.Mesh.read("model.inp", name="Domain")

Abaqus ``.inp`` files use meshlane, with meshio as a fallback. Install the
recommended reader with ``pip install "fedoo[io]"``. Supported first- and
second-order element types are converted to Fedoo elements. ``*NSET`` and
``*ELSET`` definitions become ``mesh.node_sets`` and ``mesh.element_sets``.
Several element types produce a :py:class:`fedoo.MultiMesh`, with element
sets stored on the corresponding submeshes. Unsupported cell types are
skipped with a warning; isolated vertex cells are ignored.

Importing an Abaqus deck does not create its materials, loads or analysis
steps; define those in Fedoo after importing the mesh.

Importing with meshlane or meshio
--------------------------------

To explicitly use meshlane for another supported format, such as Gmsh,
pass its mesh object to :py:meth:`fedoo.Mesh.from_meshio`:

.. code-block:: python

   import meshlane

   mesh = fd.Mesh.from_meshio(meshlane.read("model.msh"))

:py:meth:`fedoo.Mesh.from_meshio` accepts both meshlane and meshio objects
and preserves their point/cell sets. Multiple supported cell types return
a :py:class:`fedoo.MultiMesh`.

Exporting mesh geometry
-----------------------

Export mesh geometry
with :py:meth:`fedoo.Mesh.save`, for example ``mesh.save("model.vtk")``.
Use :py:func:`fedoo.read_data` for saved simulation results, including FDH5,
rather than :py:meth:`fedoo.Mesh.read`.

Legacy readers
--------------

The following legacy readers handle specific ASCII VTK/Gmsh formats.
Prefer :py:meth:`fedoo.Mesh.read` or :py:meth:`fedoo.Mesh.from_meshio`
for new workflows.

.. autosummary::
   :toctree: generated/

   import_file
   import_msh
   import_vtk


.. _build_simple_mesh:

Mesh Creation Functions
===============================

.. autosummary::
   :toctree: generated/

   rectangle_mesh
   box_mesh
   disk_mesh
   hole_plate_mesh
   hollow_disk_mesh
   I_shape_mesh
   line_mesh_1D
   line_mesh
   line_mesh_cylindric
   circle_mesh
   grid_mesh_cylindric
   generate_cylindrical_local_frame
   structured_mesh_2D
   generate_nodes
"""


# import pkgutil

# for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
#     module = loader.find_module(module_name).load_module(module_name)
#     exec('from .'+module_name+' import *')

from .simple import (
    stack,
    rectangle_mesh,
    grid_mesh_cylindric,
    line_mesh_1D,
    line_mesh,
    line_mesh_cylindric,
    circle_mesh,
    box_mesh,
)

from .importmesh import import_file, import_vtk, import_msh
from .selection import (
    find_surface_nodes,
    find_edge_nodes,
    AmbiguousSurfaceSeed,
    AmbiguousEdgeSeed,
)

from .functions import (
    extrude,
    thicken,
    extract_surface,
    extract_edges,
    change_elm_type,
    quad2tri,
    axisymmetric_extrusion,
)

from .structured_mesh import (
    structured_mesh_2D,
    generate_nodes,
    hole_plate_mesh,
    disk_mesh,
    hollow_disk_mesh,
    I_shape_mesh,
)

from fedoo.util.localframe import generate_cylindrical_local_frame
