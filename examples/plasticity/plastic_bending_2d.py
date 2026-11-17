"""
2D elastoplastic three-point bending (Simcoon EPICP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A plane strain beam (length 16 mm, height 2 mm, quad4 elements) is bent by a
5 mm displacement imposed at the middle of its top edge, then released to
show the residual stresses. The material follows the J2 elastoplastic law with
isotropic power-law hardening of Simcoon (``EPICP``, through
:class:`fedoo.constitutivelaw.Simcoon`), and the problem is solved in finite
strain (``nlgeom=True``) with :class:`fedoo.problem.NonLinear`.

The von Mises stress after the release is plotted at the end.
"""

import os

import fedoo as fd
import numpy as np

###############################################################################
# Geometry, mesh and material
# ---------------------------
# Units: N, mm, MPa. ``2Dplane`` is the plane strain assumption. Set
# ``mat = 0`` for a linear elastic material.

fd.ModelingSpace("2Dplane")

NLGEOM = True
typeBending = "3nodes"  # '3nodes' or '4nodes'
h = 2
L = 16
E = 200e3
nu = 0.3
alpha = 1e-5  # thermal expansion coefficient (required by EPICP, isothermal here)
uimp = -5

mesh = fd.mesh.rectangle_mesh(
    nx=41,
    ny=21,
    x_min=0,
    x_max=L,
    y_min=0,
    y_max=h,
    elm_type="quad4",
    name="Domain",
)

mat = 1
if mat == 0:  # linear
    material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="ConstitutiveLaw")
elif mat == 1:
    # J2 plasticity with isotropic power-law hardening:
    # yield stress Re + k * p**m, p = accumulated plastic strain
    Re = 300
    k = 1000
    m = 0.25
    props = np.array([E, nu, alpha, Re, k, m])
    material = fd.constitutivelaw.Simcoon("EPICP", props, name="ConstitutiveLaw")

###############################################################################
# Weak form, assembly and problem
# -------------------------------
# :class:`fedoo.weakform.StressEquilibriumRI` (quad4 with one Gauss point and
# hourglass control) is an alternative to the fully integrated element.

wf = fd.weakform.StressEquilibrium("ConstitutiveLaw", nlgeom=NLGEOM)

# node sets for boundary conditions
bottom_left = mesh.nearest_node([0, 0])
bottom_right = mesh.nearest_node([L, 0])

if typeBending == "3nodes":
    top_center = mesh.nearest_node([L / 2, h])
else:
    nodes_top1 = mesh.find_nodes(f"X=={L / 4} and Y=={h}")
    nodes_top2 = mesh.find_nodes(f"X=={3 * L / 4} and Y=={h}")
    top_center = np.hstack((nodes_top1, nodes_top2))

assemb = fd.Assembly.create(
    wf,
    "Domain",
    name="Assembling",
)

pb = fd.problem.NonLinear("Assembling")

# create a 'results' folder and set the desired outputs
if not (os.path.isdir("results")):
    os.mkdir("results")
if mat == 0:
    res = pb.add_output(
        "results/bendingPlastic",
        "Assembling",
        ["Disp", "Stress", "Strain"],
        output_type="Node",
        file_format="vtk",
    )
elif mat == 1:
    res = pb.add_output(
        "results/bendingPlastic",
        "Assembling",
        ["Disp", "Stress", "Strain", "Statev", "Wm"],
        output_type="Node",
        file_format="fdz",
        compressed=True,
    )

###############################################################################
# Step 1: bending
# ---------------
# The beam lies on two supports and the displacement of the top center node is
# prescribed.

pb.bc.add(
    "Dirichlet",
    bottom_left,
    "Disp",
    0,
)
pb.bc.add("Dirichlet", bottom_right, "DispY", 0)
pb.bc.add("Dirichlet", top_center, "DispY", uimp, name="disp")

pb.nlsolve(dt=0.05, tmax=1, update_dt=True, tol_nr=0.05, interval_output=0.05)

###############################################################################
# Step 2: release and residual stresses
# -------------------------------------
# The prescribed displacement is replaced by a force that starts from the
# reaction force and decreases to zero.

pb.bc.remove("disp")
F = pb.get_ext_forces("DispY")[top_center]

pb.bc.add(
    "Neumann", top_center, "DispY", 0, start_value=F
)  # no force applied = relaxation

# The force criterion is normalized by the current external forces, which
# vanish as the load is released: use the displacement criterion instead.
pb.set_nr_criterion("Displacement", adaptive_stiffness=True)

pb.add_line_search()
pb.nlsolve(dt=0.25, t0=1, tmax=2, update_dt=True, tol_nr=0.05)

res.plot("Stress", "vm")
