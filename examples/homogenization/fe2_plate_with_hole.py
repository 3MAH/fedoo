r"""
FE2 multiscale analysis of a plate with a hole (octet-truss unit cell)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A quarter of a plate with a hole (2D plane strain, very coarse ``quad4`` mesh)
is stretched along :math:`x`. Its material is not given by a constitutive law
but computed by FE\ :sup:`2`: with :class:`fedoo.constitutivelaw.FE2`, each
macroscopic integration point carries a 3D finite element model of an
octet-truss unit cell (``tet4`` mesh) made of an elastoplastic steel (simcoon
``EPICP``). At every Newton iteration, the macroscopic strain is applied to
each cell with periodic boundary conditions, and the cells return the mean
stress and the homogenized tangent matrix.

Every macroscopic integration point solves a nonlinear 3D problem, so this
example is expensive although the macroscopic mesh is tiny: its page shows a
precomputed run.
"""

import os

import fedoo as fd
import numpy as np

###############################################################################
# Modeling spaces and meshes
# --------------------------
# Two modeling spaces are defined: 2D plane strain for the macroscopic problem
# and 3D for the microscopic unit cell. The last created space
# ("micro_space") becomes the active one, so the weak forms must be given
# their space explicitly.

fd.ModelingSpace("2Dplane", "macro_space")
fd.ModelingSpace("3D", "micro_space")

# Structured mesh of a quarter plate with a hole (sym=True).
mesh_macro = fd.mesh.hole_plate_mesh(
    nr=3,
    nt=3,
    length=100,
    height=100,
    radius=20,
    elm_type="quad4",
    sym=True,
    name="macro",
)

# The micro mesh file is read next to this script (independent of the cwd).
mesh_micro = fd.mesh.import_file(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "octet_surf.msh"),
    name="micro",
)["tet4"]

###############################################################################
# Microscopic model
# -----------------
# Elastoplastic material (simcoon ``EPICP`` law) for the micro cell:
# :math:`E = 200` GPa, :math:`\nu = 0.3` (steel), yield stress :math:`R_e`
# and power-law hardening :math:`\sigma = R_e + k p^m`. The micro assembly is
# wrapped in a :class:`fedoo.constitutivelaw.FE2` object, used below as the
# constitutive law of the macroscopic problem.

E = 200e3
nu = 0.3
Re = 300
k = 1000
alpha = 1e-5  # thermal expansion coefficient (no thermal load here)
m = 0.25

props = np.array([E, nu, alpha, Re, k, m])
Material = fd.constitutivelaw.Simcoon("EPICP", props, name="ConstitutiveLaw")

fd.weakform.StressEquilibrium("ConstitutiveLaw", name="micro_wf", space="micro_space")

micro_assembly = fd.Assembly.create("micro_wf", mesh_micro)

micro_cells = fd.constitutivelaw.FE2(micro_assembly, name="FEM")

###############################################################################
# Macroscopic problem
# -------------------
# The macroscopic weak form uses the ``FE2`` law. The results of one micro
# cell (here the cell of the 6th integration point) are also saved. They are
# written at each Newton iteration, so several frames correspond to the same
# time increment.

fd.weakform.StressEquilibrium("FEM", name="WeakForm", space="macro_space")

macro_assembly = fd.Assembly.create(
    "WeakForm", "macro", name="Assembly", mesh_change=True
)

pb = fd.problem.NonLinear("Assembly")

res = pb.add_output("results/FE2", "Assembly", ["Disp", "Stress", "Strain", "Wm"])

pb.initialize()
res_micro = micro_cells.list_problem[5].add_output(
    "results/micro_cell", micro_cells.list_assembly[5], ["Disp", "Stress", "Strain"]
)

###############################################################################
# Boundary conditions and solve
# -----------------------------
# Symmetry conditions on the left and bottom edges, and a displacement of
# 0.1 mm on the right edge.

crd = mesh_macro.nodes
left = np.where(crd[:, 0] == np.min(crd[:, 0]))[0]
right = np.where(crd[:, 0] == np.max(crd[:, 0]))[0]
bottom = np.where(crd[:, 1] == np.min(crd[:, 1]))[0]

pb.bc.add("Dirichlet", left, "DispX", 0)
pb.bc.add("Dirichlet", bottom, "DispY", 0)
pb.bc.add("Dirichlet", right, "DispX", 0.1)

pb.nlsolve()
