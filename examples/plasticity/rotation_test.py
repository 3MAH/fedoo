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

StrainNodes = mesh.add_nodes(crd[node_center], 3)  # add virtual nodes for macro strain

# Assembly.create("ConstitutiveLaw", meshname, 'hex8', name="Assembling", MeshChange = False, n_elm_gp = 27)     #uses MeshChange=True when the mesh change during the time
assemb = fd.Assembly.create(
    wf, meshname, "hex8", name="Assembling", MeshChange=False, n_elm_gp=8
)  # uses MeshChange=True when the mesh change during the time

pb = fd.problem.NonLinear("Assembling")
# pb.set_solver('cg', precond = True)
pb.set_nr_criterion("Displacement", err0=1, tol=5e-3, max_subiter=5)

# pb.set_nr_criterion("Displacement")
# pb.set_nr_criterion("Work")
# pb.set_nr_criterion("Force")

# create a 'result' folder and set the desired ouputs
if not (os.path.isdir("results")):
    os.mkdir("results")
res = pb.add_output(
    "results/rot_test", "Assembling", ["Disp", "Stress", "Strain", "Statev", "Wm"]
)


# Add periodic BC
list_strain_nodes = [
    [StrainNodes[0], StrainNodes[0], StrainNodes[0]],
    [StrainNodes[1], StrainNodes[1], StrainNodes[1]],
    [StrainNodes[2], StrainNodes[2], StrainNodes[2]],
]
list_strain_var = [["DispX", "DispY", "DispZ"] for i in range(3)]

bc_periodic = fd.homogen.PeriodicBC(list_strain_nodes, list_strain_var, dim=3)
pb.bc.add(bc_periodic)

################### step 1 ################################

tmax = 1

theta = np.pi / 2
grad_u = np.array(
    [
        [np.cos(theta) - 1, -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta) - 1, 0],
        [0, 0, 0],
    ]
)

pb.bc.add("Dirichlet", node_center, "Disp", 0)
pb.bc.add("Dirichlet", [StrainNodes[0]], "DispX", grad_u[0, 0])  # EpsXX
pb.bc.add("Dirichlet", [StrainNodes[0]], "DispY", grad_u[0, 1])  # EpsYY
pb.bc.add("Dirichlet", [StrainNodes[0]], "DispZ", grad_u[0, 2])  # EpsZZ
pb.bc.add("Dirichlet", [StrainNodes[1]], "DispX", grad_u[1, 0])  # EpsXX
pb.bc.add("Dirichlet", [StrainNodes[1]], "DispY", grad_u[1, 1])  # EpsYY
pb.bc.add("Dirichlet", [StrainNodes[1]], "DispZ", grad_u[1, 2])  # EpsZZ
pb.bc.add("Dirichlet", [StrainNodes[2]], "DispX", grad_u[2, 0])  # EpsXX
pb.bc.add("Dirichlet", [StrainNodes[2]], "DispY", grad_u[2, 1])  # EpsYY
pb.bc.add("Dirichlet", [StrainNodes[2]], "DispZ", grad_u[2, 2])  # EpsZZ

# pb.apply_boundary_conditions()

pb.nlsolve(dt=0.05, tmax=1, update_dt=False, print_info=1, interval_output=0.05)

# pb.solve()
# pb.save_results()


E = np.array(
    fd.Assembly["Assembling"].get_strain(pb.get_dof_solution(), "GaussPoint", False)
).T

# ################### step 2 ################################
# bc.Remove()
# #We set initial condition to the applied force to relax the load
# F_app = pb.get_ext_forces('DispY')[nodes_topCenter]
# bc = pb.bc.add('Neumann','DispY', 0, nodes_topCenter, initialValue=F_app)#face_center)

# pb.nlsolve(dt = 1., update_dt = True, ToleranceNR = 0.01)


# ### plot with pyvista and slider


# meshplot = mesh.to_pyvista()
# # meshplot = pv.read('results/bendingPlastic3D_15.vtk')
# # meshplot = pv.read('results/rot_test.vtk')
# # meshplot.point_data['svm'] = np.c_[meshplot.point_data['Cauchy_Mises']]

# # pl = pv.Plotter()
# pl = BackgroundPlotter()

# pl.set_background('White')

# sargs = dict(
#     interactive=True,
#     title_font_size=20,
#     label_font_size=16,
#     color='Black',
#     # n_colors= 10
# )

# res.load(0)
# for item in res.node_data:
#     meshplot.point_data[item] = res.node_data[item].T
# for item in res.gausspoint_data:
#     meshplot.point_data[item] = res.get_data(item, data_type = 'Node').T
# for item in res.element_data:
#     meshplot.cell_data[item] = res.element_data[item].T

# # # cpos = [(-2.69293081283409, 0.4520024822911473, 2.322209100082263),
# # #         (0.4698685969042552, 0.46863550630755524, 0.42428354242422084),
# # #         (0.5129241539116808, 0.07216479580221505, 0.8553952621921701)]
# # # pl.camera_position = cpos

# # global actor
# # actor = pl.add_axes(color='Black', interactive = True)


# # # pl.add_mesh(meshplot.warp_by_vector(factor = 5), scalars = 'Stress', component = 2, clim = [0,10000], show_edges = True, cmap="bwr")
# actor = pl.add_mesh(meshplot.warp_by_vector('Disp',factor = 1), scalars = 'Strain', component = 0, show_edges = True, scalar_bar_args=sargs, cmap="jet")
# # pl.add_mesh(meshplot.warp_by_vector(factor = 1), scalars = 'svm', component = 0, show_edges = True, scalar_bar_args=sargs, cmap="jet")


# def change_iter(value):
#     global actor
#     print(int(value))
#     pl.remove_actor(actor)

#     res = np.load(int(value))
#     for item in res.node_data:
#         meshplot.point_data[item] = res.node_data[item].T
#     for item in res.gausspoint_data:
#         meshplot.point_data[item] = res.get_data(item, data_type = 'Node').T
#     for item in res.element_data:
#         meshplot.cell_data[item] = res.element_data[item].T

#     # pl.update_scalars(scalars, mesh=None, render=True)
#     # pl.update_scalars(res['Strain_Node'][0])
#     # pl.update()

#     actor = pl.add_mesh(meshplot.warp_by_vector('Disp',factor = 1), scalars = 'Strain', component = 0, show_edges = True, scalar_bar_args=sargs, cmap="jet")
#     # pl.update()


# # slider = pl.add_slider_widget(

# #     change_iter,

# #     [0, 20],

# #     title="Iter",

# #     title_opacity=0.5,

# #     title_color="red",

# #     fmt="%0.9f",

# #     title_height=0.08,

# #     style = 'modern',

# # )

# slider = pl.add_text_slider_widget(

#     change_iter,

#     [str(i) for i in range(20)],

#     style = 'modern',

# )

# pl.show()
# # cpos = pl.show(return_cpos = True)
# # pl.save_graphic('test.pdf', title='PyVista Export', raster=True, painter=True)
