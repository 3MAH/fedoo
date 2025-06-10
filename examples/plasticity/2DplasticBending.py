import fedoo as fd
import numpy as np
from time import time
import os

# import pylab as plt
from numpy import linalg

start = time()
# --------------- Pre-Treatment --------------------------------------------------------

fd.ModelingSpace("2Dplane")

NLGEOM = True
typeBending = "3nodes"  #'3nodes' or '4nodes'
# Units: N, mm, MPa
h = 2
w = 10
L = 16
E = 200e3
nu = 0.3
alpha = 1e-5  # ???
uimp = -5

mesh = fd.mesh.rectangle_mesh(
    nx=41, ny=21, x_min=0, x_max=L, y_min=0, y_max=h, elm_type="quad4", name="Domain"
)

crd = mesh.nodes

mat = 1
if mat == 0:  # linear
    material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="ConstitutiveLaw")
elif mat == 1:
    # isotropic plasticity with power law hardening sigma = k*eps_p**m
    Re = 300
    k = 1000
    m = 0.25
    props = np.array([E, nu, alpha, Re, k, m])
    material = fd.constitutivelaw.Simcoon("EPICP", props, name="ConstitutiveLaw")


wf = fd.weakform.StressEquilibrium("ConstitutiveLaw", nlgeom=NLGEOM)
wf.fbar = True

# alternative using element 'quad4' with reduced integration
# ie n_elm_gp = 1 combined with hourglass control
# wf = fd.weakform.StressEquilibriumRI("ConstitutiveLaw", 0.005, nlgeom=NLGEOM)


# note set for boundary conditions
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

# Problem.set_solver('cg', precond = True)

pb.set_nr_criterion("Displacement")
# Problem.set_nr_criterion("Work")
# Problem.set_nr_criterion("Force")

# create a 'result' folder and set the desired ouputs
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
    # elm_set = mesh.get_elements_from_nodes(mesh.find_nodes('X<8'))
    # res = pb.add_output('results/bendingPlastic', 'Assembling', ['Disp', 'Stress', 'Strain', 'Statev', 'Wm'], element_set = elm_set )

################### step 1 ################################
tmax = 1
pb.bc.add(
    "Dirichlet",
    bottom_left,
    "Disp",
    0,
)
pb.bc.add("Dirichlet", bottom_right, "DispY", 0)
pb.bc.add("Dirichlet", top_center, "DispY", uimp, name="disp")

pb.nlsolve(dt=0.05, tmax=1, update_dt=True, tol_nr=0.05, interval_output=0.05)

################### step 2 ################################
# compute residual stresses
pb.bc.remove("disp")

pb.bc.add("Neumann", top_center, "DispY", 0)  # no force applied = relaxation

pb.nlsolve(dt=1.0, update_dt=True, tol_nr=0.05)

print(time() - start)

res.plot("Stress", "vm")
res.write_movie("test", "Stress", "vm")
