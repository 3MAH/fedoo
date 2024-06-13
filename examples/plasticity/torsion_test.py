import fedoo as fd
import numpy as np
import os

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
uimp = 1

filename = "torsion_test"
res_dir = "results/"

# mesh = fd.mesh.box_mesh(nx=21, ny=21, nz=21, x_min=0, x_max=L, y_min=0, y_max=h, z_min = 0, z_max = w, elm_type = 'hex8', name = 'Domain')
# mesh = fd.mesh.import_file('../../util/meshes/octet_truss.msh', name = "Domain")['tet4']
mesh = fd.mesh.import_file("../../util/meshes/octet_truss_2.msh", name="Domain")["tet4"]

crd = mesh.nodes

mat = 1
if mat == 0:
    props = np.array([E, nu, alpha])
    material = fd.constitutivelaw.Simcoon("ELISO", props, name="ConstitutiveLaw")
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
else:
    material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="ConstitutiveLaw")

wf = fd.weakform.StressEquilibrium("ConstitutiveLaw", nlgeom=NLGEOM, name="wf")

# fd.Assembly.create("ConstitutiveLaw", mesh, 'hex8', name="Assembling", MeshChange = False, n_elm_gp = 27)     #uses MeshChange=True when the mesh change during the time
assemb = fd.Assembly.create(
    "wf", mesh, name="Assembling"
)  # uses MeshChange=True when the mesh change during the time

# node set for boundary conditions
left = mesh.find_nodes("X", 0)
right = mesh.find_nodes("X", 1)

# add CD nodes
# ref_node = mesh.add_nodes(2) #reference node for rigid body motion
ref_node = mesh.add_virtual_nodes(2)  # reference node for rigid body motion
node_cd = [ref_node[0] for i in range(3)] + [ref_node[1] for i in range(3)]
var_cd = ["DispX", "DispY", "DispZ", "DispX", "DispY", "DispZ"]


pb = fd.problem.NonLinear("Assembling")
# Problem.set_solver('cg', precond = True)
pb.set_nr_criterion("Displacement", err0=1, tol=1e-3, max_subiter=5)

# Problem.set_nr_criterion("Displacement")
# Problem.set_nr_criterion("Work")
# Problem.set_nr_criterion("Force")

# create a 'result' folder and set the desired ouputs
if not (os.path.isdir("results")):
    os.mkdir("results")
# results = pb.add_output(res_dir+filename, 'Assembling', ['Disp'], output_type='Node', file_format ='npz')
# results = pb.add_output(res_dir+filename, 'Assembling', ['Cauchy', 'PKII', 'Strain', 'Cauchy_vm', 'Statev', 'Wm'], output_type='GaussPoint', file_format ='npz')

results = pb.add_output(
    res_dir + filename,
    "Assembling",
    ["Disp", "Cauchy", "Strain", "Cauchy_vm", "Statev", "Wm", "Fext"],
)
# results = pb.add_output(res_dir+filename, 'Assembling', ['Disp', 'Cauchy', 'Strain', 'Fext'])

# Problem.add_output(res_dir+filename, 'Assembling', ['cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')


pb.bc.add(fd.constraint.RigidTie(right, node_cd, var_cd))

# pb.bc.add('Dirichlet','Disp',0,nodes_bottom)
# pb.bc.add('Dirichlet','DispY', 0,nodes_top)
# pb.bc.add('Dirichlet','DispZ', 0,nodes_top)
# pb.bc.add('Dirichlet','DispX', uimp,nodes_top)

pb.bc.add("Dirichlet", left, "Disp", 0)
# pb.bc.add('Dirichlet',ref_node[0], 'Disp', 0) #Displacement of the right end
# pb.bc.add('Dirichlet',ref_node[1], ['DispX','DispY','DispZ'], [np.pi,0,0.]) #Rigid rotation of the right end
pb.bc.add(
    "Dirichlet", ref_node[1], "DispX", 2 * np.pi / 2
)  # Rigid rotation of the right end

# pb.bc.add('Dirichlet',ref_node[0], 'DispX', 0.5) #Rigid displacement of the right end
# pb.bc.add('Neumann',ref_node[0], 'DispX', 69384) #Rigid displacement of the right end


# pb.bc.add('Neumann',ref_node[1], 'DispX', 300) #Rigid rotation of the right end


pb.nlsolve(dt=0.05, tmax=1, update_dt=True, print_info=1, interval_output=0.025)

E = np.array(
    fd.Assembly.get_all()["Assembling"].get_strain(
        pb.get_dof_solution(), "GaussPoint", False
    )
).T


# =============================================================
# Example of plots with pyvista - uncomment the desired plot
# =============================================================

# ------------------------------------
# Simple plot with default options
# ------------------------------------
results.plot("Stress", component="vm", data_type="Node", show=True)

# ------------------------------------
# Write movie with default options
# ------------------------------------
results.write_movie(res_dir + filename, "Stress_vm", framerate=12, quality=5)

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
# results.write_movie(res_dir+filename, 'Stress', component = 0, framerate = 12, quality = 5, rot_azimuth = -1.5, rot_elevation = 0)


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

# from matplotlib import pylab

# Fhist = []
# for it in range(results.n_iter):
#     results.load(it)
#     # Fhist.append((pb._MFext.T @ pb.get_A() @ pb.get_X() - pb._MFext.T @ pb.get_D()).reshape(3,-1)[0,-2])
#     Fhist.append(sum(results['Fext'][0,right]))

# pylab.plot(Fhist)
