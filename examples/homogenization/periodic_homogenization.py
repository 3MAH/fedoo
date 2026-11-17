r"""
Homogenized elastic stiffness of periodic unit cells
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example computes the effective (homogenized) elastic stiffness matrix
:math:`\mathbf{L}^{\text{eff}}` of periodic lattice unit cells with
:func:`fedoo.homogen.get_homogenized_stiffness`. The function applies periodic
boundary conditions and solves six elementary load cases (one per component of
the mean strain).

Two unit cells are considered:

* an octet truss (small mesh read from a gmsh ``.msh`` file), for which the
  effective stiffness matrix is simply printed;
* a gyroid (periodic mesh generated with microgen), whose anisotropy is
  illustrated by the directional Young's modulus :math:`E(\mathbf{n})`.
"""

import fedoo as fd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors

dir_meshes = "../../util/meshes/"
np.set_printoptions(precision=3, suppress=True)

###############################################################################
# Octet truss unit cell
# ---------------------
# The ``.msh`` file contains several element types: only the ``tet4`` volume
# mesh is kept. ``meshperio=True`` states that the mesh is periodic (the nodes
# of opposite faces match), which is required for the periodic boundary
# conditions.

fd.ModelingSpace("3D")

mesh_octet = fd.mesh.import_file(dir_meshes + "octet_truss.msh")["tet4"]

material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)  # E (MPa), nu
wf = fd.weakform.StressEquilibrium(material)

assembly_octet = fd.Assembly.create(wf, mesh_octet)

L_eff_octet = fd.homogen.get_homogenized_stiffness(assembly_octet, meshperio=True)

###############################################################################
# The effective stiffness is given in the Voigt notation of fedoo, with the
# component order ``[11, 22, 33, 12, 13, 23]`` and engineering shear strains.
# The octet truss has a cubic symmetry, so only three independent constants
# appear (:math:`L_{11}`, :math:`L_{12}` and :math:`L_{44}`).
#
# For an isotropic cell, the equivalent Young's modulus and Poisson ratio could
# be extracted with simcoon:
# ``from simcoon import simmit as sim; sim.L_iso_props(L_eff)``.

print("L_eff (octet truss) =\n", L_eff_octet)

###############################################################################
# Gyroid unit cell
# ----------------
# Same procedure on a periodic gyroid mesh.

mesh_gyroid = fd.Mesh.read(dir_meshes + "gyroid_per_clean.vtk")

assembly_gyroid = fd.Assembly.create(wf, mesh_gyroid)

L_eff = fd.homogen.get_homogenized_stiffness(assembly_gyroid, meshperio=True)
print("L_eff (gyroid) =\n", L_eff)

###############################################################################
# Directional Young's modulus
# ---------------------------
# For a uniaxial stress along the unit direction :math:`\mathbf{n}`, the
# apparent Young's modulus is
#
# .. math::
#
#    \frac{1}{E(\mathbf{n})} = (\mathbf{n} \otimes \mathbf{n}) :
#    \mathbf{S}^{\text{eff}} : (\mathbf{n} \otimes \mathbf{n}),
#    \qquad \mathbf{S}^{\text{eff}} = \left(\mathbf{L}^{\text{eff}}\right)^{-1}.
#
# In fedoo's Voigt notation (engineering shear strains), the uniaxial stress
# :math:`\mathbf{n} \otimes \mathbf{n}` is the vector
# :math:`[n_1^2, n_2^2, n_3^2, n_1 n_2, n_1 n_3, n_2 n_3]`.
# The surface :math:`E(\mathbf{n})\,\mathbf{n}` is plotted below.

phi = np.linspace(0, 2 * np.pi, 128)  # azimuthal angle in the xy-plane
theta = np.linspace(0, np.pi, 128).reshape(128, 1)  # polar angle

n_1 = np.sin(theta) * np.cos(phi)
n_2 = np.sin(theta) * np.sin(phi)
n_3 = np.cos(theta) * np.ones(128)

n = (
    np.array([n_1 * n_1, n_2 * n_2, n_3 * n_3, n_1 * n_2, n_1 * n_3, n_2 * n_3])
    .transpose(1, 2, 0)
    .reshape(128, 128, 1, 6)
)

S_eff = np.linalg.inv(L_eff)
E = 1.0 / (n @ S_eff @ n.reshape(128, 128, 6, 1)).reshape(128, 128)

x = E * n_1
y = E * n_2
z = E * n_3

fig = plt.figure(figsize=plt.figaspect(1))
ax = fig.add_subplot(111, projection="3d")
ax.set_axis_off()

Emin, Eavg, Emax = np.min(E), np.average(E), np.max(E)
norm = colors.Normalize(vmin=Emin, vmax=Emax, clip=False)

ax.plot_surface(
    x,
    y,
    z,
    rstride=1,
    cstride=1,
    norm=norm,
    facecolors=cm.cividis(norm(E)),
    linewidth=0,
    antialiased=False,
    shade=False,
)

scalarmap = cm.ScalarMappable(cmap=plt.cm.cividis, norm=norm)
cbar = plt.colorbar(
    scalarmap,
    orientation="horizontal",
    fraction=0.06,
    pad=-0.1,
    ticks=[Emin, Eavg, Emax],
    ax=ax,
)
cbar.ax.tick_params(labelsize="large")
cbar.set_label(r"directional stiffness $E$ (MPa)", size=15, labelpad=20)

ax.azim = 30
ax.elev = 30

plt.show()
