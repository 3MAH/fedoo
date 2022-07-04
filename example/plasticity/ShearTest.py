from fedoo import *
import numpy as np
from time import time
import os
import pylab as plt
from numpy import linalg
import pyvista as pv


start = time()
#--------------- Pre-Treatment --------------------------------------------------------

Util.ProblemDimension("3D")

NLGEOM = 2
#Units: N, mm, MPa
h = 1
w = 1
L = 1
E = 200e3
nu=0.3
alpha = 1e-5 #???
meshname = "Domain"
uimp = 1

filename = 'sheartest_ref'
res_dir = 'results/'

Mesh.box_mesh(Nx=11, Ny=11, Nz=11, x_min=0, x_max=L, y_min=0, y_max=h, z_min = 0, z_max = w, ElementShape = 'hex8', name = meshname)
mesh = Mesh.get_all()[meshname]

crd = mesh.nodes 

mat =1
if mat == 0:
    props = np.array([[E, nu, alpha]])
    Material = ConstitutiveLaw.Simcoon("ELISO", props, 1, name='ConstitutiveLaw')
    Material.corate = 2
    # Material.SetMaskH([[] for i in range(6)])
    # mask = [[3,4,5] for i in range(3)]
    # mask+= [[0,1,2,4,5], [0,1,2,3,5], [0,1,2,3,4]]
    # Material.SetMaskH(mask)
elif mat == 1 or mat == 2:
    Re = 300
    k=1000 #1500
    m=0.3 #0.25
    if mat == 1:
        props = np.array([[E, nu, alpha, Re,k,m]])
        Material = ConstitutiveLaw.Simcoon("EPICP", props, 8, name='ConstitutiveLaw')
        Material.corate = 2
        # Material.SetMaskH([[] for i in range(6)])
    
        # mask = [[3,4,5] for i in range(3)]
        # mask+= [[0,1,2,4,5], [0,1,2,3,5], [0,1,2,3,4]]
        # Material.SetMaskH(mask)

    elif mat == 2:
        Material = ConstitutiveLaw.ElastoPlasticity(E,nu,Re, name='ConstitutiveLaw')
        Material.SetHardeningFunction('power', H=k, beta=m)
else:
    Material = ConstitutiveLaw.ElasticIsotrop(E, nu, name='ConstitutiveLaw')




#### trouver pourquoi les deux fonctions suivantes ne donnent pas la même chose !!!!
WeakForm.InternalForce("ConstitutiveLaw", nlgeom = NLGEOM)
# WeakForm.InternalForceUL("ConstitutiveLaw")



#note set for boundary conditions
nodes_bottom = mesh.find_nodes('Y',0)
nodes_top = mesh.find_nodes('Y',1)

# Assembly.Create("ConstitutiveLaw", meshname, 'hex8', name="Assembling", MeshChange = False, nb_gp = 27)     #uses MeshChange=True when the mesh change during the time
Assembly.Create("ConstitutiveLaw", meshname, name="Assembling")     #uses MeshChange=True when the mesh change during the time

Problem.NonLinearStatic("Assembling")
# Problem.SetSolver('cg', precond = True)
Problem.SetNewtonRaphsonErrorCriterion("Displacement", err0 = 1, tol = 5e-4, max_subiter = 5)

# Problem.SetNewtonRaphsonErrorCriterion("Displacement")
# Problem.SetNewtonRaphsonErrorCriterion("Work")
# Problem.SetNewtonRaphsonErrorCriterion("Force")

#create a 'result' folder and set the desired ouputs
if not(os.path.isdir('results')): os.mkdir('results')
Problem.AddOutput(res_dir+filename, 'Assembling', ['Disp', 'Cauchy', 'PKII', 'Strain', 'Cauchy_vm', 'Statev', 'Wm'], output_type='Node', file_format ='npz')    
# Problem.AddOutput(res_dir+filename, 'Assembling', ['cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')    


################### step 1 ################################
tmax = 1
Problem.BoundaryCondition('Dirichlet','Disp',0,nodes_bottom)
Problem.BoundaryCondition('Dirichlet','DispY', 0,nodes_top)
Problem.BoundaryCondition('Dirichlet','DispZ', 0,nodes_top)
Problem.BoundaryCondition('Dirichlet','DispX', uimp,nodes_top)

Problem.NLSolve(dt = 0.05, tmax = 1, update_dt = False, print_info = 1, intervalOutput = 0.05)


E = np.array(Assembly.get_all()['Assembling'].get_strain(Problem.GetDoFSolution(), "GaussPoint", False)).T

# ################### step 2 ################################
# bc.Remove()
# #We set initial condition to the applied force to relax the load
# F_app = Problem.get_ext_forces('DispY')[nodes_topCenter]
# bc = Problem.BoundaryCondition('Neumann','DispY', 0, nodes_topCenter, initialValue=F_app)#face_center)

# Problem.NLSolve(dt = 1., update_dt = True, ToleranceNR = 0.01)

print(time()-start)

### plot with pyvista



res_name = res_dir+filename

res = np.load(res_name+'_{}.npz'.format(18))

meshplot = mesh.to_pyvista()

# for item in res:
#     if item[-4:] == 'Node':
#         if len(res[item]) == len(crd):
#             meshplot.point_data[item[:-5]] = res[item]
#         else:
#             meshplot.point_data[item[:-5]] = res[item].T
#     else:
#         meshplot.cell_data[item] = res[item].T

meshplot.point_data['Disp'] = res['Disp_Node'].T

pl = pv.Plotter()
# pl = pv.Plotter()
pl.set_background('White')

sargs = dict(
    interactive=True,
    title_font_size=20,
    label_font_size=16,
    color='Black',
    # n_colors= 10
)

# cpos = [(-2.69293081283409, 0.4520024822911473, 2.322209100082263),
#         (0.4698685969042552, 0.46863550630755524, 0.42428354242422084),
#         (0.5129241539116808, 0.07216479580221505, 0.8553952621921701)]
# pl.camera_position = cpos

# pl.add_mesh(meshplot.warp_by_vector(factor = 5), scalars = 'Stress', component = 2, clim = [0,10000], show_edges = True, cmap="bwr")
pl.subplot(0,0)
meshplot.point_data['Disp'] = res['Disp_Node'].T
pl.add_mesh(meshplot.warp_by_vector('Disp', factor = 1), scalars = res['Cauchy_vm_Node'].T, component = 3, show_edges = True, scalar_bar_args=sargs, cmap="jet")
# pl.add_mesh(meshplot.warp_by_vector(factor = 1), scalars = 'svm', component = 0, show_edges = True, scalar_bar_args=sargs, cmap="jet")
pl.add_axes(color='Black', interactive = True)

cpos = pl.show(return_cpos = True)         
# cpos = pl.show(interactive = False, auto_close=False, return_cpos = True)
# pl.save_graphic('test.pdf', title='PyVista Export', raster=True, painter=True)



















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





   

    
    
    
    






