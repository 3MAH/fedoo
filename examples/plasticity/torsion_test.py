"""
Large torsion of a cube (finite strain, UL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A unit cube (hex8 elements) is clamped on its left face and its right face is
twisted by 180 degrees about the :math:`x` axis. The right face is driven as a
rigid surface with :class:`fedoo.constraint.RigidTie`, whose rotation dof
``RigidRotX`` is prescribed. The problem is solved with the updated Lagrangian
formulation (``nlgeom="UL"``).

By default (``mat = 0``) the material is linear elastic (Simcoon ``ELISO``);
``mat = 1`` or ``mat = 2`` switch to J2 elastoplasticity. The von Mises stress
at the end of the loading is plotted, and an animation of the twist is written
in ``results/torsion_test.mp4``.
"""

import os

import fedoo as fd
import numpy as np

###############################################################################
# Geometry, mesh and material
# ---------------------------
# Units: N, mm, MPa. ``mat`` selects the law: 0 elastic (Simcoon ``ELISO``),
# 1 Simcoon ``EPICP``, 2 native
# :class:`fedoo.constitutivelaw.ElastoPlasticity`.

fd.ModelingSpace("3D")

NLGEOM = "UL"
h = 1
w = 1
L = 1
E = 200e3
nu = 0.3
alpha = 1e-5  # thermal expansion coefficient (required by Simcoon, isothermal here)

filename = "torsion_test"
res_dir = "results/"

mesh = fd.mesh.box_mesh(
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
    name="Domain",
)

mat = 0
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
else:
    material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="ConstitutiveLaw")

###############################################################################
# Weak form, assembly and problem
# -------------------------------

wf = fd.weakform.StressEquilibrium("ConstitutiveLaw", nlgeom=NLGEOM, name="wf")
wf.incompressibility = "auto"

assemb = fd.Assembly.create("wf", mesh, name="Assembling")

# node set for boundary conditions
left = mesh.find_nodes("X", 0)
right = mesh.find_nodes("X", 1)

pb = fd.problem.NonLinear("Assembling")

# create a 'results' folder and set the desired outputs
if not (os.path.isdir("results")):
    os.mkdir("results")

results = pb.add_output(
    res_dir + filename,
    "Assembling",
    ["Disp", "Stress", "Strain"],
)

###############################################################################
# Rigid tie on the right face and twist
# -------------------------------------

pb.bc.add(fd.constraint.RigidTie(right))

pb.bc.add("Dirichlet", left, "Disp", 0)
pb.bc.add("Dirichlet", "RigidRotX", np.pi)  # 180 deg twist of the right end

pb.nlsolve(dt=0.05, tmax=1, update_dt=True, print_info=1, interval_output=0.025)

###############################################################################
# Von Mises stress and animation
# ------------------------------
# ``write_movie`` also accepts camera options, for instance
# ``rot_azimuth=-1.5`` to turn the camera around the scene.

results.plot("Stress", component="vm", data_type="Node", show=True)

results.write_movie(res_dir + filename, "Stress", "vm", framerate=12, quality=5)
