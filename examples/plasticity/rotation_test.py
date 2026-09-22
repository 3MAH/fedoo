import fedoo as fd
import numpy as np
import os

# --------------- Pre-Treatment --------------------------------------------------------

fd.ModelingSpace("3D")

NLGEOM = True
# Units: N, mm, MPa
h = 1
w = 1
L = 1
E = 200e3
nu = 0.3
alpha = 1e-5  # ???
meshname = "Domain"
uimp = 2

fd.mesh.box_mesh(
    nx=5,
    ny=5,
    nz=5,
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

crd = mesh.nodes

mat = 1
if mat == 0:
    props = np.array([E, nu, alpha])
    material = fd.constitutivelaw.Simcoon("ELISO", props, name="ConstitutiveLaw")
    material.corate = "log_r"
elif mat == 1 or mat == 2:
    Re = 300
    k = 1000  # 1500
    m = 0.3  # 0.25
    if mat == 1:
        props = np.array([E, nu, alpha, Re, k, m])
        material = fd.constitutivelaw.Simcoon("EPICP", props, name="ConstitutiveLaw")
        # material.corate = 'log'

    elif mat == 2:
        material = fd.constitutivelaw.ElastoPlasticity(
            E, nu, Re, name="ConstitutiveLaw"
        )
        material.set_hardening_function("power", h=k, beta=m)
else:
    material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="ConstitutiveLaw")

wf = fd.weakform.StressEquilibrium("ConstitutiveLaw", nlgeom=NLGEOM)


# note set for boundary conditions
nodes_bottom = mesh.find_nodes("Y", 0)
nodes_top = mesh.find_nodes("Y", 1)

node_center = mesh.nearest_node([0.5, 0.5, 0.5])

assemb = fd.Assembly.create(wf, meshname, "hex8", name="Assembling", n_elm_gp=8)

pb = fd.problem.NonLinear("Assembling")


# create a 'result' folder and set the desired ouputs
if not (os.path.isdir("results")):
    os.mkdir("results")
res = pb.add_output(
    "results/rot_test",
    ["Disp", "Stress", "Strain", "P", "EP", "Wm"],
    ignore_missing=True,
)

# Add periodic BC
bc_periodic = fd.constraint.PeriodicBC("finite_strain", dim=3)
pb.bc.add(bc_periodic)

tmax = 1

theta = np.pi / 2

pb.bc.add("Dirichlet", node_center, "Disp", 0)

# Prescribe F(t) = R(theta * t) by defining time_func, instead of the default
# linear interpolation of grad(u) = R(theta) - I. Linear component interpolation
# would give F(t) = I + t * (R(theta) - I), which contains an artificial stretch
# between the initial and final rotations.
pb.bc.add(
    "Dirichlet",
    "DU_xx",
    1,
    start_value=0,
    time_func=lambda t: np.cos(theta * t) - 1,
)
pb.bc.add(
    "Dirichlet",
    "DU_xy",
    1,
    start_value=0,
    time_func=lambda t: -np.sin(theta * t),
)
pb.bc.add("Dirichlet", "DU_xz", 0)
pb.bc.add(
    "Dirichlet",
    "DU_yx",
    1,
    start_value=0,
    time_func=lambda t: np.sin(theta * t),
)
pb.bc.add(
    "Dirichlet",
    "DU_yy",
    1,
    start_value=0,
    time_func=lambda t: np.cos(theta * t) - 1,
)
pb.bc.add("Dirichlet", "DU_yz", 0)
pb.bc.add("Dirichlet", "DU_zx", 0)
pb.bc.add("Dirichlet", "DU_zy", 0)
pb.bc.add("Dirichlet", "DU_zz", 0)


pb.nlsolve(dt=0.05, tmax=1, update_dt=False, print_info=1, interval_output=0.05)

# res.plot('Stress', 'XX')
# res.write_movie('rigid_rot', 'Stress', 'XX')


# Show the saved iterations with the Fedoo viewer. The viewer provides iteration
# and animation controls as well as field and component selection.
fd.viewer(res)
