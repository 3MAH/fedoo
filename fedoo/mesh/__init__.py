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
