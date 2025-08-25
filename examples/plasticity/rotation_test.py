import fedoo as fd
import numpy as np
import os
import pylab as plt
from numpy import linalg
import pyvista as pv
from pyvistaqt import BackgroundPlotter

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
    props = np.array([[E, nu, alpha]])
    material = fd.constitutivelaw.Simcoon("ELISO", props, 1, name="ConstitutiveLaw")
    material.corate = "log"
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
        material.SetHardeningFunction("power", H=k, beta=m)
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
res = pb.add_output("results/rot_test", ["Disp", "Stress", "Strain", "Statev", "Wm"])

# Add periodic BC
bc_periodic = fd.constraint.PeriodicBC("finite_strain", dim=3)
pb.bc.add(bc_periodic)

tmax = 1

theta = np.pi / 2
grad_u = np.array(
    [
        [np.cos(theta) - 1, -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta) - 1, 0],
        [0, 0, 0],
    ]
)
# or using simcoon
# from simcoon import simmit as simcoon
# rot_mat = np.array(
#     [
#         [np.cos(theta), -np.sin(theta), 0],
#         [np.sin(theta), np.cos(theta), 0],
#         [0, 0, 1],
#     ]
# )
# F = simcoon.eR_to_F(simcoon.v2t_strain([0,0,0,0,0,0]), rot_mat)
# grad_u = F - np.eye(3)

pb.bc.add("Dirichlet", node_center, "Disp", 0)
pb.bc.add("Dirichlet", "DU_xx", grad_u[0, 0])
pb.bc.add("Dirichlet", "DU_xy", grad_u[0, 1])
pb.bc.add("Dirichlet", "DU_xz", grad_u[0, 2])
pb.bc.add("Dirichlet", "DU_yx", grad_u[1, 0])
pb.bc.add("Dirichlet", "DU_yy", grad_u[1, 1])
pb.bc.add("Dirichlet", "DU_yz", grad_u[1, 2])
pb.bc.add("Dirichlet", "DU_zx", grad_u[2, 0])
pb.bc.add("Dirichlet", "DU_zy", grad_u[2, 1])
pb.bc.add("Dirichlet", "DU_zz", grad_u[2, 2])


pb.nlsolve(dt=0.05, tmax=1, update_dt=False, print_info=1, interval_output=0.05)

# res.plot('Stress', 'XX')
# res.write_movie('rigid_rot', 'Stress', 'XX')


### Show results with slider
pl = pv.Plotter()


def change_iter(value):
    res.load(int(value))
    pl.clear_actors()
    res.plot(
        "Stress",
        "XX",
        plotter=pl,
        title="",
        # title_size=10,
        # azimuth=0,
        # elevation=-70,
        show_scalar_bar=False,
        show_edges=True,
    )
    pl.hide_axes()
    pl.write_frame()


slider = pl.add_text_slider_widget(
    change_iter,
    [str(i) for i in range(20)],
    style="modern",
)
pl.show()
