import os

n_threads = 4
os.environ["OMP_NUM_THREADS"] = f"{n_threads}"
os.environ["OPENBLAS_NUM_THREADS"] = f"{n_threads}"
os.environ["MKL_NUM_THREADS"] = f"{n_threads}"
os.environ["VECLIB_MAXIMUM_THREADS"] = f"{n_threads}"
os.environ["NUMEXPR_NUM_THREADS"] = f"{n_threads}"

import fedoo as fd
import numpy as np
from time import time
import pylab as plt
from numpy import linalg

start = time()
# --------------- Pre-Treatment --------------------------------------------------------

space = fd.ModelingSpace("3D")

NLGEOM = "UL"
typeBending = "3nodes"  #'3nodes' or '4nodes'
# Units: N, mm, MPa
h = 2
w = 10
L = 16
E = 200e3
nu = 0.3
alpha = 1e-5  # ???
meshname = "Domain"
uimp = -8

mesh = fd.mesh.box_mesh(
    nx=21,
    ny=7,
    nz=7,
    x_min=0,
    x_max=L,
    y_min=0,
    y_max=h,
    z_min=0,
    z_max=w,
    elm_type="hex8",
    name=meshname,
)

crd = mesh.nodes

mat = 1
if mat == 0:
    props = np.array([E, nu, alpha])
    material = fd.constitutivelaw.Simcoon("ELISO", props, name="constitutivelaw")
elif mat == 1 or mat == 2:
    Re = 300
    k = 1000  # 1500
    m = 0.3  # 0.25
    if mat == 1:
        props = np.array([E, nu, alpha, Re, k, m])
        material = fd.constitutivelaw.Simcoon("EPICP", props, name="constitutivelaw")
    elif mat == 2:
        material = fd.constitutivelaw.ElastoPlasticity(
            E, nu, Re, name="constitutivelaw"
        )
        material.SetHardeningFunction("power", H=k, beta=m)
else:
    material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="constitutivelaw")

wf = fd.weakform.StressEquilibrium("constitutivelaw", nlgeom=NLGEOM)
wf.corate = "green_naghdi"
wf.fbar = True

# or alternatively with reduced integration + hourglass stiffness :
# wf = fd.weakform.StressEquilibriumRI("constitutivelaw", nlgeom = True)

# note set for boundary conditions
nodes_bottomLeft = mesh.find_nodes("XY", (0, 0))
nodes_bottomRight = mesh.find_nodes("XY", (L, 0))

if typeBending == "3nodes":
    nodes_topCenter = mesh.find_nodes("XY", (L / 2, h))
    # nodes_topCenter = np.where((crd[:,0]==L/2) * (crd[:,1]==h))[0]
else:
    nodes_top1 = mesh.find_nodes("XY", (L / 4, h))
    nodes_top2 = mesh.find_nodes("XY", (3 * L / 4, h))
    nodes_topCenter = np.hstack((nodes_top1, nodes_top2))

# Assembly.create("constitutivelaw", meshname, 'hex8', name="Assembling", MeshChange = False, n_elm_gp = 27)     #uses MeshChange=True when the mesh change during the time
# assemb = fd.Assembly.create("constitutivelaw", meshname, 'hex8', name="Assembling", MeshChange = False, n_elm_gp = 8)     #uses MeshChange=True when the mesh change during the time
assemb = fd.Assembly.create(wf, meshname, name="Assembling")


pb = fd.problem.NonLinear("Assembling")
# pb.set_solver('cg', precond = True)
# pb.set_solver('petsc', solver_type='preonly', pc_type='lu', pc_factor_mat_solver_type='mumps')
pb.set_nr_criterion("Displacement", err0=None, tol=5e-3, max_subiter=5)

# Problem.set_nr_criterion("Displacement")
# pb.set_nr_criterion("Work")
# Problem.set_nr_criterion("Force")

# create a 'result' folder and set the desired ouputs
if not (os.path.isdir("results")):
    os.mkdir("results")
# res = pb.add_output('results/bendingPlastic3D', 'Assembling', ['Disp', 'Cauchy', 'Strain', 'Cauchy_vm', 'Statev', 'Wm'], output_type='Node', file_format ='vtk')
# Problem.add_output('results/bendingPlastic3D', 'Assembling', ['Cauchy', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')
# res = pb.add_output('results/bendingPlastic3D', 'Assembling', ['Disp', 'Cauchy', 'Strain', 'Cauchy_vm', 'Statev', 'Wm'])
res = pb.add_output(
    "results/bendingPlastic3D",
    "Assembling",
    ["Disp", "Cauchy", "Strain", "Cauchy_vm"],
    file_format="fdz",
)


################### step 1 ################################
tmax = 1
pb.bc.add("Dirichlet", nodes_bottomLeft, "Disp", 0)
pb.bc.add("Dirichlet", nodes_bottomRight, "DispY", 0)
bc = pb.bc.add("Dirichlet", nodes_topCenter, "DispY", uimp)

pb.nlsolve(dt=0.025, tmax=1, update_dt=False, print_info=1, interval_output=0.05)

E = assemb.sv["Strain"]
# E = np.array(fd.Assembly.get_all()['Assembling'].get_strain(pb.get_dof_solution(), "GaussPoint", False)).T
#
# ################### step 2 ################################
# bc.Remove()
# #We set initial condition to the applied force to relax the load
# F_app = Problem.get_ext_forces('DispY')[nodes_topCenter]
# bc = Problem.bc.add('Neumann','DispY', 0, nodes_topCenter, initialValue=F_app)#face_center)

# Problem.nlsolve(dt = 1., update_dt = True, ToleranceNR = 0.01)

print(time() - start)


### plot with pyvista

# print(assemb.sv['TangentMatrix'])

# from pyvistaqt import BackgroundPlotter
# plotter = BackgroundPlotter()

res.plot("Stress", component="XY", data_type="Node")

# res.plot('Statev', 'Node', component = 1)
# res.write_movie('test', 'Stress', component=0)

# import pyvista as pv

# meshplot = pv.read('results/bendingPlastic3D_15.vtk')
# # meshplot.point_data['svm'] = np.c_[meshplot.point_data['Cauchy_Mises']]

# pl = pv.Plotter()
# pl.set_background('White')

# sargs = dict(
#     interactive=True,
#     title_font_size=20,
#     label_font_size=16,
#     color='Black',
#     # n_colors= 10
# )

# # cpos = [(-2.69293081283409, 0.4520024822911473, 2.322209100082263),
# #         (0.4698685969042552, 0.46863550630755524, 0.42428354242422084),
# #         (0.5129241539116808, 0.07216479580221505, 0.8553952621921701)]
# # pl.camera_position = cpos

# # pl.add_mesh(meshplot.warp_by_vector(factor = 5), scalars = 'Stress', component = 2, clim = [0,10000], show_edges = True, cmap="bwr")
# pl.add_mesh(meshplot.warp_by_vector(factor = 1), scalars = 'Disp', component = 1, show_edges = True, scalar_bar_args=sargs, cmap="jet")
# # pl.add_mesh(meshplot.warp_by_vector(factor = 1), scalars = 'svm', component = 0, show_edges = True, scalar_bar_args=sargs, cmap="jet")

# cpos = pl.show(return_cpos = True)
# # pl.save_graphic('test.pdf', title='PyVista Export', raster=True, painter=True)
