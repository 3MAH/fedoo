r"""
Laminated plate cantilever (ShellLaminate)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A rectangular plate (100 x 20, plate elements) is clamped on one end and loaded
by a transverse force at the center of the other end. The section is a
symmetric sandwich defined with :class:`fedoo.constitutivelaw.ShellLaminate`:
two stiff skins (thickness 0.2) on a core ten times softer (thickness 1), with
transverse shear (Reissner-Mindlin plate). The problem is linear
(:class:`fedoo.weakform.PlateEquilibrium`, :class:`fedoo.problem.Linear`).

The through-thickness distribution of :math:`\sigma_{xx}` at one Gauss point
shows the stress jump at the skin/core interfaces; the von Mises stress on the
lower face is then plotted.
"""

import os

import fedoo as fd
from matplotlib import pylab as plt

###############################################################################
# Materials, laminated section and mesh
# -------------------------------------

fd.ModelingSpace("3D")
E = 1e5
nu = 0.3

L = 100
h = 20
F = -100

geom_elm_type = "quad4"  # choose among 'tri3', 'tri6', 'quad4', 'quad8', 'quad9'
# True, False or 'auto'. If True, use selective reduced integration (SRI) for
# linear elements. If 'auto', use the PlateEquilibrium defaults.
reduced_integration = "auto"
save_results = True

mat1 = fd.constitutivelaw.ElasticIsotrop(E, nu, name="Mat1")
mat2 = fd.constitutivelaw.ElasticIsotrop(E / 10, nu, name="Mat2")

# total thickness 1.4; by default k=1 i.e. with transverse shear
fd.constitutivelaw.ShellLaminate(
    ["Mat1", "Mat2", "Mat1"], [0.2, 1, 0.2], name="PlateSection"
)

mesh = fd.mesh.rectangle_mesh(
    201, 21, 0, L, -h / 2, h / 2, geom_elm_type, name="plate", ndim=3
)

nodes_left = mesh.node_sets["left"]
nodes_right = mesh.node_sets["right"]

node_right_center = nodes_right[(mesh.nodes[nodes_right, 1] ** 2).argmin()]

###############################################################################
# Plate weak form, assembly and problem
# -------------------------------------

fd.weakform.PlateEquilibrium("PlateSection", name="WFplate")

if reduced_integration == "auto":
    plate_elm_type = None
elif reduced_integration:
    if geom_elm_type not in ["tri3", "quad4"]:
        raise ValueError("SRI plate elements are only available for tri3 and quad4")
    plate_elm_type = f"p{geom_elm_type}sri"
else:
    plate_elm_type = f"p{geom_elm_type}"

if plate_elm_type is None:
    fd.Assembly.create("WFplate", "plate", name="plate")
else:
    fd.Assembly.create("WFplate", "plate", plate_elm_type, name="plate")

pb = fd.problem.Linear("plate")

# create a 'results' folder and set the desired outputs
if not (os.path.isdir("results")):
    os.mkdir("results")
res = pb.add_output(
    "results/laminate",
    "plate",
    ["Disp", "Rot", "Stress"],
    file_format="npz",
    position=-1,
)  # position = -1 for the lower face

###############################################################################
# Boundary conditions and solve
# -----------------------------
# Clamped left end (displacements and rotations), transverse force at the
# center of the right end.

pb.bc.add("Dirichlet", nodes_left, "Disp", 0)
pb.bc.add("Dirichlet", nodes_left, "Rot", 0)

pb.bc.add("Neumann", node_right_center, "DispZ", F)

pb.apply_boundary_conditions()
pb.solve()

if save_results:
    pb.save_results()  # save in npz

###############################################################################
# Through-thickness stress and von Mises stress
# ---------------------------------------------

# stress distribution through the thickness at Gauss point 200
z, StressDistribution = fd.ConstitutiveLaw["PlateSection"].get_stress_distribution(
    fd.Assembly["plate"], 200
)
plt.plot(StressDistribution[0], z)
plt.xlabel("SXX")
plt.ylabel("z")

# von Mises stress on the lower face
res.plot("Stress", component="vm")
