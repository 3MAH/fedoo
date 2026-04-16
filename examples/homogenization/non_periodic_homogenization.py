"""
Non-periodic homogenization with microgen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Homogenization of a non-periodic spinodoid microstructure
using microgen's BoxMesh to build the boundary interpolation
dictionary required by PeriodicBC with ``meshperio=False``.

The effective stiffness tensor is computed using the perturbation
method (``get_homogenized_stiffness`` / ``get_tangent_stiffness``).

This example requires `microgen <https://microgen.readthedocs.io>`_
(``pip install microgen``). Only the ``BoxMesh`` submodule is used,
which does **not** depend on cadquery.
"""

# sphinx_gallery_thumbnail_number = -1

import os

import fedoo as fd
import numpy as np
import pyvista as pv
from microgen.box_mesh import BoxMesh  # direct import avoids cadquery dependency

###############################################################################
# Dimension of the problem
# ------------------------------------------------------------------------------
fd.ModelingSpace("3D")

###############################################################################
# Load the non-periodic mesh and build microgen boundary data
# ------------------------------------------------------------------------------
dir_meshes = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../util/meshes/"
)

pvmesh = pv.read(os.path.join(dir_meshes, "spinodoid_pd.vtk"))
mesh = fd.Mesh.from_pyvista(pvmesh, name="Domain")

###############################################################################
# Build the boundary interpolation dictionary
# ------------------------------------------------------------------------------
# ``BoxMesh`` identifies boundary nodes and computes nearest-neighbour
# interpolation data for non-matching opposite faces.
box_mesh = BoxMesh.from_pyvista(pvmesh)

# Merge all boundary data into a single dictionary.
# ``closest_points_on_boundaries`` returns "+" side entries as
# ``(indices, distances)`` tuples for interpolation weights.
# ``faces`` / ``edges`` / ``corners`` provide the "-" side plain arrays.
# The "+" keys from ``closest_points`` overwrite the plain "+" arrays.
dic = {}
dic.update(box_mesh.faces)
dic.update(box_mesh.edges)
dic.update(box_mesh.corners)
dic.update(box_mesh.closest_points_on_boundaries(k_neighbours=3))
dic["d_rve"] = list(box_mesh.rve.dim)

###############################################################################
# Define the material and weak formulation
# ------------------------------------------------------------------------------
material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)
wf = fd.weakform.StressEquilibrium(material, nlgeom=False)

###############################################################################
# Assembly
# ------------------------------------------------------------------------------
assemb = fd.Assembly.create(wf, mesh, mesh.elm_type, name="Assembly")

###############################################################################
# Compute the effective stiffness tensor
# ------------------------------------------------------------------------------
L_eff = fd.homogen.get_homogenized_stiffness(
    assemb,
    meshperio=False,
    dic_closest_points_on_boundaries=dic,
)

np.set_printoptions(precision=2, suppress=True)
print("Effective stiffness tensor:")
print(L_eff)
