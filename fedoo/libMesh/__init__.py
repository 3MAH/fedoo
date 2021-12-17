"""
===========================================
Geometry and Mesh (:mod:`fedoo.libMesh`)
===========================================

.. currentmodule:: fedoo.libMesh.Mesh

Class Mesh
====================

.. autosummary::   
   :toctree: generated/
   :template: custom-class-template.rst

   Mesh



.. currentmodule:: fedoo.libMesh

Mesh manipulation functions
==================================

.. autosummary::
   :toctree: generated/
   
   GetAll
   Stack

Import/Export Fedoo Mesh object        
==================================

.. autosummary::
   :toctree: generated/

   ImportFromFile
   ImportFromMSH
   ImportFromVTK


Mesh Creation Functions
===============================

.. autosummary::
   :toctree: generated/
   
   RectangleMesh 
   GridMeshCylindric
   LineMesh1D
   LineMeshCylindric
   BoxMesh
   GridStructuredMesh2D
   GenerateNodes          
"""

# .. automodule:: fedoo.libMesh.MeshImport
#     :members:
#     :undoc-members:
    
# .. automodule:: fedoo.libMesh.MeshTools
#     :members:
#     :undoc-members:
# # 

import pkgutil

for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
    module = loader.find_module(module_name).load_module(module_name)
    exec('from .'+module_name+' import *')
    
    
