import fedoo as fd
import numpy as np
from time import time
import os
import pylab as plt
from numpy import linalg
import pyvista as pv


start = time()
# --------------- Pre-Treatment --------------------------------------------------------

fd.ModelingSpace("3D")

NLGEOM = "UL"
# Units: N, mm, MPa
h = 1
w = 1
L = 1
E = 200e3
nu = 0.3
alpha = 1e-5  # ???
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

crd = mesh.nodes

mat = 1
if mat == 0:
    props = np.array([E, nu, alpha])
    material = fd.constitutivelaw.Simcoon("ELISO", props, name="ConstitutiveLaw")
    material.corate = 2
elif mat == 1 or mat == 2:
    Re = 300
    k = 1000  # 1500
    m = 0.3  # 0.25
    if mat == 1:
        props = np.array([E, nu, alpha, Re, k, m])
        material = fd.constitutivelaw.Simcoon("EPICP", props, name="ConstitutiveLaw")
    elif mat == 2:
        material = fd.constitutivelaw.ElastoPlasticity(
            E, nu, Re, name="ConstitutiveLaw"
        )
        material.SetHardeningFunction("power", H=k, beta=m)
elif mat == 3:
    props = np.array([3.0, 0.5e2])
    material = fd.constitutivelaw.Simcoon("NEOHC", props, name="ConstitutiveLaw")
else:
    material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="ConstitutiveLaw")

wf = fd.weakform.StressEquilibriumFbar("ConstitutiveLaw", nlgeom=NLGEOM)
wf.fbar = True
wf.corate = "jaumann"

# fd.Assembly.create("ConstitutiveLaw", meshname, 'hex8', name="Assembling", MeshChange = False, n_elm_gp = 27)     #uses MeshChange=True when the mesh change during the time
fd.Assembly.create(
    wf, meshname, name="Assembling"
)  # uses MeshChange=True when the mesh change during the time

pb = fd.problem.NonLinear("Assembling")
# Problem.set_solver('cg', precond = True)
pb.set_nr_criterion("Displacement", err0=None, tol=5e-4, max_subiter=20)

# Problem.set_nr_criterion("Displacement")
# Problem.set_nr_criterion("Work")
# Problem.set_nr_criterion("Force")

# create a 'result' folder and set the desired ouputs
if not (os.path.isdir("results")):
    os.mkdir("results")
# results = pb.add_output(res_dir+filename, 'Assembling', ['Disp'], output_type='Node', file_format ='npz')
# results = pb.add_output(res_dir+filename, 'Assembling', ['Cauchy', 'PKII', 'Strain', 'Cauchy_vm', 'Statev', 'Wm'], output_type='GaussPoint', file_format ='npz')

results = pb.add_output(
    res_dir + filename, "Assembling", ["Disp", "Stress", "Strain", "Statev", "Wm"]
)

################### step 1 ################################
# node sets for boundary conditions
nodes_bottom = mesh.find_nodes("Y", 0)
nodes_top = mesh.find_nodes("Y", 1)


# pb.bc.add('Dirichlet','Disp',0,nodes_bottom)
# pb.bc.add('Dirichlet','DispY', 0,nodes_top)
# pb.bc.add('Dirichlet','DispZ', 0,nodes_top)
# pb.bc.add('Dirichlet','DispX', uimp,nodes_top)

pb.bc.add("Dirichlet", nodes_bottom, "Disp", 0)
pb.bc.add("Dirichlet", nodes_top, ["DispY", "DispZ"], 0)
pb.bc.add("Dirichlet", nodes_top, "DispX", uimp)


pb.nlsolve(dt=0.01, tmax=1, update_dt=True, print_info=1, interval_output=0.05)

# E = np.array(
#     fd.Assembly.get_all()["Assembling"].get_strain(
#         pb.get_dof_solution(), "GaussPoint", False
#     )
# ).T

# ################### step 2 ################################
# bc.Remove()
# #We set initial condition to the applied force to relax the load
# F_app = Problem.get_ext_forces('DispY')[nodes_topCenter]
# bc = Problem.bc.add('Neumann','DispY', 0, nodes_topCenter, initialValue=F_app)#face_center)

# Problem.nlsolve(dt = 1., update_dt = True, ToleranceNR = 0.01)

print(time() - start)


# =============================================================
# Example of plots with pyvista - uncomment the desired plot
# =============================================================

# ------------------------------------
# Simple plot with default options
# ------------------------------------
results.plot("Stress", component="vm", show=True, data_type="Node")
# results.plot("Stress", component=0, data_type='Node', show=True)

# ------------------------------------
# Simple plot with default options and save to png
# ------------------------------------
# pl = results.plot('Cauchy_vm', component = 0, show = False)
# pl.show(screenshot = "test.png")

# ------------------------------------
# Write movie with default options
# ------------------------------------
# results.write_movie(res_dir+filename, 'Stress', 'vm', framerate = 5, quality = 5)

# ------------------------------------
# Save pdf plot
# ------------------------------------
# pl = results.plot(scalars = 'Cauchy_vm', show = False)
# pl.save_graphic('test.pdf', title='PyVista Export', raster=True, painter=True)

# ------------------------------------
# Plot the automatically saved mesh
# ------------------------------------
# pv.read(res_dir+filename+'/'+filename+'.vtk').plot()

# ------------------------------------
# Write movie with moving camera
# ------------------------------------
# results.write_movie(res_dir+filename, 'Stress_vm', framerate = 5, quality = 5, rot_azimuth = 2, rot_elevation = 0.5)

# ------------------------------------
# Plot time history
# ------------------------------------
# from matplotlib import pylab
# t, sigma = results.get_history(('Time','Cauchy'), (0,12), component = 3)
# pylab.plot(t,sigma)
# #or results.plot_history('Cauchy', 12)


# filename = 'results/sheartest_ref'

# filename_27 = 'results/sheartest'


# res = np.load(filename+'_{}.npz'.format(19))
# res_27 = np.load(filename_27+'_{}.npz'.format(19))
# meshplot = mesh.to_pyvista()

# # for item in res:
# #     if item[-4:] == 'Node':
# #         if len(res[item]) == len(crd):
# #             meshplot.point_data[item[:-5]] = res[item]
# #         else:
# #             meshplot.point_data[item[:-5]] = res[item].T
# #     else:
# #         meshplot.cell_data[item] = res[item].T

# meshplot.point_data['Disp'] = res['Disp_Node'].T

# pl = pv.Plotter(shape=(1, 2))
# # pl = pv.Plotter()
# pl.set_background('White')

# sargs = dict(
#     interactive=False,
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
# pl.subplot(0,0)
# meshplot.point_data['Disp'] = res['Disp_Node'].T
# pl.add_mesh(meshplot.warp_by_vector('Disp', factor = 1), scalars = res['Cauchy_vm_Node'], component = 0, show_edges = True, scalar_bar_args=sargs, cmap="jet")
# # pl.add_mesh(meshplot.warp_by_vector(factor = 1), scalars = 'svm', component = 0, show_edges = True, scalar_bar_args=sargs, cmap="jet")
# pl.add_axes()

# pl.subplot(0,1)
# meshplot.point_data['Disp'] = res_27['Disp_Node'].T
# pl.add_mesh(meshplot.warp_by_vector('Disp', factor = 1), scalars = res_27['Cauchy_vm_Node'], component = 0, show_edges = True, scalar_bar_args=sargs, cmap="jet")
# pl.add_axes()
# pl.link_views()
# cpos = pl.show(return_cpos = True)
# # pl.save_graphic('test.pdf', title='PyVista Export', raster=True, painter=True)


# meshplot = pv.read('results/bendingPlastic3D_19.vtk')
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


# a = res_dir+filename
# test = fd.core.dataset.read_data(a)
