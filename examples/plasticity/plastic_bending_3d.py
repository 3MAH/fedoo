r"""
3D elastoplastic three-point bending
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A plate (length 16 mm, thickness 2 mm, width 10 mm, hex8 elements) lies on two
line supports and an 8 mm displacement is imposed along the line at the middle
of its top face. The material follows the J2 elastoplastic law with isotropic
power-law hardening of Simcoon (``EPICP``). The problem is solved in finite
strain with the updated Lagrangian formulation (``nlgeom="UL"``) and the F-bar
method (``incompressibility="fbar"``) to avoid the volumetric locking of the
nearly incompressible plastic flow.

The shear stress :math:`\sigma_{xy}` at the end of the loading is plotted.
"""

import os

import fedoo as fd
import numpy as np

###############################################################################
# Geometry, mesh and material
# ---------------------------
# Units: N, mm, MPa. ``mat = 2`` uses the native
# :class:`fedoo.constitutivelaw.ElastoPlasticity` law instead of Simcoon, and
# ``mat = 0`` the linear elastic Simcoon law ``ELISO``.

space = fd.ModelingSpace("3D")

NLGEOM = "UL"
typeBending = "3nodes"  # '3nodes' or '4nodes'
h = 2
w = 10
L = 16
E = 200e3
nu = 0.3
alpha = 1e-5  # thermal expansion coefficient (required by Simcoon, isothermal here)
meshname = "Domain"
uimp = -8

mesh = fd.mesh.box_mesh(
    nx=21,
    ny=7,
    nz=7,
    x_min=0,
    x_max=L,
    y_min=0,
    y_max=h,
    z_min=0,
    z_max=w,
    elm_type="hex8",
    name=meshname,
)

mat = 1
if mat == 0:
    props = np.array([E, nu, alpha])
    material = fd.constitutivelaw.Simcoon("ELISO", props, name="constitutivelaw")
elif mat == 1 or mat == 2:
    Re = 300
    k = 1000
    m = 0.3
    if mat == 1:
        props = np.array([E, nu, alpha, Re, k, m])
        material = fd.constitutivelaw.Simcoon("EPICP", props, name="constitutivelaw")
    elif mat == 2:
        material = fd.constitutivelaw.ElastoPlasticity(
            E, nu, Re, name="constitutivelaw"
        )
        material.set_hardening_function("power", h=k, beta=m)
else:
    material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="constitutivelaw")

###############################################################################
# Weak form, assembly and problem
# -------------------------------
# :class:`fedoo.weakform.StressEquilibriumRI` (reduced integration with
# hourglass stiffness) is an alternative to the F-bar method.

wf = fd.weakform.StressEquilibrium(
    "constitutivelaw", nlgeom=NLGEOM, incompressibility="fbar"
)

# node sets for boundary conditions
nodes_bottomLeft = mesh.find_nodes("XY", (0, 0))
nodes_bottomRight = mesh.find_nodes("XY", (L, 0))

if typeBending == "3nodes":
    nodes_topCenter = mesh.find_nodes("XY", (L / 2, h))
else:
    nodes_top1 = mesh.find_nodes("XY", (L / 4, h))
    nodes_top2 = mesh.find_nodes("XY", (3 * L / 4, h))
    nodes_topCenter = np.hstack((nodes_top1, nodes_top2))

assemb = fd.Assembly.create(wf, meshname, name="Assembling")

pb = fd.problem.NonLinear("Assembling")
pb.set_nr_criterion("Force", tol=5e-3, max_subiter=16)

# create a 'results' folder and set the desired outputs
if not (os.path.isdir("results")):
    os.mkdir("results")
res = pb.add_output(
    "results/bendingPlastic3D",
    "Assembling",
    ["Disp", "Stress", "Strain"],
    file_format="fdz",
)

###############################################################################
# Bending
# -------

pb.bc.add("Dirichlet", nodes_bottomLeft, "Disp", 0)
pb.bc.add("Dirichlet", nodes_bottomRight, "DispY", 0)
bc = pb.bc.add("Dirichlet", nodes_topCenter, "DispY", uimp)

pb.nlsolve(dt=0.025, tmax=1, update_dt=True, print_info=1, interval_output=0.05)

###############################################################################
# Shear stress at the end of the loading
# --------------------------------------

res.plot("Stress", component="XY", data_type="Node")
