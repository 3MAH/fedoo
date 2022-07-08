"""
===========================================
Geometry and Mesh (:mod:`fedoo.mesh`)
===========================================

.. currentmodule:: fedoo.mesh.mesh

Class Mesh
====================

.. autosummary::   
   :toctree: generated/
   :template: custom-class-template.rst

   Mesh



.. currentmodule:: fedoo.mesh

Mesh manipulation functions
==================================

.. autosummary::
   :toctree: generated/
   
   get_all
   stack

Import/Export Fedoo Mesh object        
==================================

.. autosummary::
   :toctree: generated/

   import_file
   import_msh
   import_vtk


Mesh Creation Functions
===============================

.. autosummary::
   :toctree: generated/
   
   rectangle_mesh 
   grid_mesh_cylindric
   line_mesh_1D
   line_mesh_cylindric
   box_mesh
   structured_mesh_2D
   generate_nodes          
"""


# import pkgutil

# for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
#     module = loader.find_module(module_name).load_module(module_name)
#     exec('from .'+module_name+' import *')
  
from .simple import stack, rectangle_mesh, grid_mesh_cylindric, line_mesh_1D, \
                    line_mesh, line_mesh_cylindric, box_mesh, structured_mesh_2D, \
                    generate_nodes, hole_plate_mesh, quad2tri
    
from .importmesh import import_file, import_vtk, import_msh

from .base import get_all

from .mesh import Mesh