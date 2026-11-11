r"""
Elastoplastic simple shear of a cube (finite strain, UL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A unit cube (hex8 elements) is clamped on its bottom face while its top face
is moved by one side length in the :math:`x` direction (shear
:math:`\gamma = 1`), with no vertical motion. The material follows the J2
elastoplastic law with isotropic power-law hardening of Simcoon (``EPICP``).
The problem is solved with the updated Lagrangian formulation
(``nlgeom="UL"``), the F-bar method and the logarithmic objective rate
(``wf.corate = "log"``).

The von Mises stress at the end of the loading is plotted.
"""

import os

import fedoo as fd
import numpy as np

###############################################################################
# Geometry, mesh and material
# ---------------------------
# Units: N, mm, MPa. ``mat`` selects the law: 0 elastic (Simcoon ``ELISO``),
# 1 Simcoon ``EPICP``, 2 native
# :class:`fedoo.constitutivelaw.ElastoPlasticity`, 3 Simcoon Neo-Hookean
# ``NEOHC``.

fd.ModelingSpace("3D")

NLGEOM = "UL"
h = 1
w = 1
L = 1
E = 200e3
nu = 0.3
alpha = 1e-5  # thermal expansion coefficient (required by Simcoon, isothermal here)
meshname = "Domain"
uimp = 1

filename = "sheartest_ref"
res_dir = "results/"

fd.mesh.box_mesh(
    nx=11,
    ny=11,
    nz=11,
    x_min=0,
    x_max=L,
    y_min=0,
    y_max=h,
    z_min=0,
    z_max=w,
    elm_type="hex8",
    name=meshname,
)
mesh = fd.Mesh[meshname]

mat = 1
if mat == 0:
    props = np.array([E, nu, alpha])
    material = fd.constitutivelaw.Simcoon("ELISO", props, name="ConstitutiveLaw")
elif mat == 1 or mat == 2:
    Re = 300
    k = 1000
    m = 0.3
    if mat == 1:
        props = np.array([E, nu, alpha, Re, k, m])
        material = fd.constitutivelaw.Simcoon("EPICP", props, name="ConstitutiveLaw")
    elif mat == 2:
        material = fd.constitutivelaw.ElastoPlasticity(
            E, nu, Re, name="ConstitutiveLaw"
        )
        material.set_hardening_function("power", h=k, beta=m)
elif mat == 3:
    props = np.array([3.0, 0.5e2])
    material = fd.constitutivelaw.Simcoon("NEOHC", props, name="ConstitutiveLaw")
else:
    material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="ConstitutiveLaw")

###############################################################################
# Weak form, assembly and problem
# -------------------------------

wf = fd.weakform.StressEquilibrium(
    "ConstitutiveLaw", nlgeom=NLGEOM, incompressibility="fbar"
)
wf.corate = "log"  # objective rate of the UL formulation (default "log_r")

fd.Assembly.create(wf, meshname, name="Assembling")

pb = fd.problem.NonLinear("Assembling")
pb.set_nr_criterion("Force", err0=None, tol=5e-3, max_subiter=10)

# create a 'results' folder and set the desired outputs
if not (os.path.isdir("results")):
    os.mkdir("results")

results = pb.add_output(
    res_dir + filename, "Assembling", ["Disp", "Stress", "Strain", "P", "EP", "Wm"]
)

###############################################################################
# Simple shear
# ------------

# node sets for boundary conditions
nodes_bottom = mesh.find_nodes("Y", 0)
nodes_top = mesh.find_nodes("Y", 1)

pb.bc.add("Dirichlet", nodes_bottom, "Disp", 0)
pb.bc.add("Dirichlet", nodes_top, ["DispY", "DispZ"], 0)
pb.bc.add("Dirichlet", nodes_top, "DispX", uimp)

pb.nlsolve(dt=0.01, tmax=1, update_dt=True, print_info=1, interval_output=0.05)

###############################################################################
# Von Mises stress at the end of the loading
# ------------------------------------------
# The history of a component at a given point can also be plotted with
# :py:meth:`fedoo.MultiFrameDataSet.plot_history`, and an animation written
# with :py:meth:`fedoo.MultiFrameDataSet.write_movie`.

results.plot("Stress", component="vm", show=True, data_type="Node")
