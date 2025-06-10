#
# Plate element to model the canteleaver beam using different kind of plate elements
#

import fedoo as fd
from matplotlib import pylab as plt
import os

fd.ModelingSpace("3D")
E = 1e5
nu = 0.3

L = 100
h = 20
thickness = 1
F = -100

geom_elm_type = "quad4"  # choose among 'tri3', 'tri6', 'quad4', 'quad8', 'quad9'
reduced_integration = "auto"  # choose among True, False and 'auto'. if True, use reduce integration for shear. if 'auto', depend on the order of the element
save_results = True

mat1 = fd.constitutivelaw.ElasticIsotrop(E, nu, name="Mat1")
mat2 = fd.constitutivelaw.ElasticIsotrop(E / 10, nu, name="Mat2")

# ConstitutiveLaw.ShellHomogeneous('Material', thickness, name = 'PlateSection')
fd.constitutivelaw.ShellLaminate(
    ["Mat1", "Mat2", "Mat1"], [0.2, 1, 0.2], name="PlateSection"
)  # by default k=1 i.e. with shear effect

mesh = fd.mesh.rectangle_mesh(
    201, 21, 0, L, -h / 2, h / 2, geom_elm_type, name="plate", ndim=3
)

nodes_left = mesh.node_sets["left"]
nodes_right = mesh.node_sets["right"]

node_right_center = nodes_right[(mesh.nodes[nodes_right, 1] ** 2).argmin()]


if reduced_integration == "auto":
    fd.weakform.PlateEquilibrium("PlateSection", name="WFplate")
elif reduced_integration:
    # selective integration: reduced integration for shear terms and full integration for flexural terms
    fd.weakform.PlateEquilibriumSI("PlateSection", name="WFplate")
else:
    # full integration
    fd.weakform.PlateEquilibriumFI("PlateSection", name="WFplate")

fd.Assembly.create("WFplate", "plate", name="plate")

pb = fd.problem.Linear("plate")

# create a 'result' folder and set the desired ouputs
if not (os.path.isdir("results")):
    os.mkdir("results")
# res = pb.add_output('results/simplePlate', 'plate', ['Disp','Rot','Stress', 'Stress_vm'], output_type='Node', file_format ='vtk', position = -1)
res = pb.add_output(
    "results/laminate",
    "plate",
    ["Disp", "Rot", "Stress"],
    file_format="npz",
    position=-1,
)  # position = -1 for the lower face

pb.bc.add("Dirichlet", nodes_left, "Disp", 0)
pb.bc.add("Dirichlet", nodes_left, "Rot", 0)

pb.bc.add("Neumann", node_right_center, "DispZ", F)

pb.apply_boundary_conditions()
pb.solve()

if save_results:
    pb.save_results()  # save in vtk

# plot the stress distribution
z, StressDistribution = fd.ConstitutiveLaw["PlateSection"].GetStressDistribution(
    fd.Assembly["plate"], 200
)
plt.plot(StressDistribution[0], z)

# plot the von mises stress on the lower face
res.plot("Stress", component="vm")
