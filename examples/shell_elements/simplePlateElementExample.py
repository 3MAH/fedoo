#
# Plate element to model the canteleaver beam using different kind of plate elements
#

import fedoo as fd
import numpy as np
from matplotlib import pylab as plt
import os

fd.ModelingSpace("3D")
E = 1e5
nu = 0.3

L = 100
h = 20
thickness = 1
F = -10

geomElementType = "quad4"  # choose among 'tri3', 'tri6', 'quad4', 'quad8', 'quad9'
reduced_integration = "auto"  # choose among True, False and 'auto'. if True, use SRI for linear elements. if 'auto', use PlateEquilibrium defaults
save_results = True

material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="Material")
fd.constitutivelaw.ShellHomogeneous("Material", thickness, name="PlateSection")

mesh = fd.mesh.rectangle_mesh(
    201, 21, 0, L, -h / 2, h / 2, geomElementType, name="plate", ndim=3
)

nodes_left = mesh.node_sets["left"]
nodes_right = mesh.node_sets["right"]

node_right_center = nodes_right[(mesh.nodes[nodes_right, 1] ** 2).argmin()]

fd.weakform.PlateEquilibrium("PlateSection", name="WFplate")

if reduced_integration == "auto":
    plate_elm_type = None
elif reduced_integration:
    if geomElementType not in ["tri3", "quad4"]:
        raise ValueError("SRI plate elements are only available for tri3 and quad4")
    plate_elm_type = f"p{geomElementType}sri"
else:
    plate_elm_type = f"p{geomElementType}"

if plate_elm_type is None:
    assemb = fd.Assembly.create("WFplate", "plate", name="plate")
else:
    assemb = fd.Assembly.create("WFplate", "plate", plate_elm_type, name="plate")

pb = fd.problem.Linear(assemb)

# create a 'result' folder and set the desired ouputs
if not (os.path.isdir("results")):
    os.mkdir("results")
res = pb.add_output(
    "results/simple_plate",
    assemb,
    ["Disp", "Rot", "Stress", "Strain"],
    output_type="Node",
    file_format="vtk",
    position=+1,
)

pb.bc.add("Dirichlet", nodes_left, "Disp", 0)
pb.bc.add("Dirichlet", nodes_left, "Rot", 0)

pb.bc.add("Neumann", node_right_center, "DispZ", F)

pb.apply_boundary_conditions()
pb.solve()

I = h * thickness**3 / 12
print("Beam analitical deflection: ", F * L**3 / (3 * E * I))
print(
    "Numerical deflection with shell model: ", pb.get_disp("DispZ")[node_right_center]
)

if save_results:
    pb.save_results()  # save in vtk

z, StressDistribution = fd.ConstitutiveLaw.get_all()[
    "PlateSection"
].get_stress_distribution(assemb, 20)
plt.plot(StressDistribution[0], z)

# plot the von mises stress on the upper face (position = +1)
res.plot("Stress", component="XX")
