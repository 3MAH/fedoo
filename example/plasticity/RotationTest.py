from fedoo import *
import numpy as np
from time import time
import os
import pylab as plt
from numpy import linalg
import pyvista as pv
from pyvistaqt import BackgroundPlotter

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
uimp = 2

mesh.box_mesh(Nx=5, Ny=5, Nz=5, x_min=0, x_max=L, y_min=0, y_max=h, z_min = 0, z_max = w, ElementShape = 'hex8', name = meshname)
mesh = mesh.get_all()[meshname]

crd = mesh.nodes 

mat =1
if mat == 0:
    props = np.array([[E, nu, alpha]])
    Material = ConstitutiveLaw.Simcoon("ELISO", props, 1, name='ConstitutiveLaw')
    Material.corate = 2
    # Material.SetMaskH([[] for i in range(6)])
    mask = [[3,4,5] for i in range(3)]
    mask+= [[0,1,2,4,5], [0,1,2,3,5], [0,1,2,3,4]]
    Material.SetMaskH(mask)
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

node_center = mesh.nearest_node([0.5,0.5,0.5])

StrainNodes = mesh.add_nodes(crd[node_center],3) #add virtual nodes for macro strain

# Assembly.create("ConstitutiveLaw", meshname, 'hex8', name="Assembling", MeshChange = False, n_elm_gp = 27)     #uses MeshChange=True when the mesh change during the time
assemb = Assembly.create("ConstitutiveLaw", meshname, 'hex8', name="Assembling", MeshChange = False, n_elm_gp = 8)     #uses MeshChange=True when the mesh change during the time

Problem.NonLinearStatic("Assembling")
# Problem.SetSolver('cg', precond = True)
Problem.SetNewtonRaphsonErrorCriterion("Displacement", err0 = 1, tol = 5e-3, max_subiter = 5)

# Problem.SetNewtonRaphsonErrorCriterion("Displacement")
# Problem.SetNewtonRaphsonErrorCriterion("Work")
# Problem.SetNewtonRaphsonErrorCriterion("Force")

#create a 'result' folder and set the desired ouputs
if not(os.path.isdir('results')): os.mkdir('results')
Problem.AddOutput('results/rot_test', 'Assembling', ['Disp', 'Cauchy', 'PKII', 'Strain', 'Cauchy_vm', 'Statev', 'Wm'], output_type='Node', file_format ='npz')    
# Problem.AddOutput('results/bendingPlastic3D', 'Assembling', ['cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')    


Homogen.DefinePeriodicBoundaryConditionGrad(meshname,
        [[StrainNodes[0], StrainNodes[0], StrainNodes[0]], [StrainNodes[1], StrainNodes[1], StrainNodes[1]], [StrainNodes[2], StrainNodes[2], StrainNodes[2]]],
        [['DispX',        'DispY',        'DispZ'] for i in range(3)], dim='3D', tol=1e-4)

################### step 1 ################################

tmax = 1

theta = np.pi/2
grad_u = np.array([[np.cos(theta)-1,-np.sin(theta),0], [np.sin(theta),np.cos(theta)-1,0], [0,0,0]])

Problem.BoundaryCondition('Dirichlet','Disp',0,node_center)
Problem.BoundaryCondition('Dirichlet','DispX', grad_u[0,0], [StrainNodes[0]]) #EpsXX
Problem.BoundaryCondition('Dirichlet','DispY', grad_u[0,1], [StrainNodes[0]]) #EpsYY
Problem.BoundaryCondition('Dirichlet','DispZ', grad_u[0,2], [StrainNodes[0]]) #EpsZZ
Problem.BoundaryCondition('Dirichlet','DispX', grad_u[1,0], [StrainNodes[1]]) #EpsXX
Problem.BoundaryCondition('Dirichlet','DispY', grad_u[1,1], [StrainNodes[1]]) #EpsYY
Problem.BoundaryCondition('Dirichlet','DispZ', grad_u[1,2], [StrainNodes[1]]) #EpsZZ
Problem.BoundaryCondition('Dirichlet','DispX', grad_u[2,0], [StrainNodes[2]]) #EpsXX
Problem.BoundaryCondition('Dirichlet','DispY', grad_u[2,1], [StrainNodes[2]]) #EpsYY
Problem.BoundaryCondition('Dirichlet','DispZ', grad_u[2,2], [StrainNodes[2]]) #EpsZZ

# Problem.ApplyBoundaryCondition()

Problem.NLSolve(dt = 0.05, tmax = 1, update_dt = False, print_info = 1, intervalOutput = 0.05)

# Problem.Solve()
# Problem.SaveResults()


E = np.array(Assembly.get_all()['Assembling'].get_strain(Problem.GetDoFSolution(), "GaussPoint", False)).T

# ################### step 2 ################################
# bc.Remove()
# #We set initial condition to the applied force to relax the load
# F_app = Problem.get_ext_forces('DispY')[nodes_topCenter]
# bc = Problem.BoundaryCondition('Neumann','DispY', 0, nodes_topCenter, initialValue=F_app)#face_center)

# Problem.NLSolve(dt = 1., update_dt = True, ToleranceNR = 0.01)

print(time()-start)


### plot with pyvista



meshplot = mesh.to_pyvista()
# meshplot = pv.read('results/bendingPlastic3D_15.vtk')
# meshplot = pv.read('results/rot_test.vtk')
# meshplot.point_data['svm'] = np.c_[meshplot.point_data['Cauchy_Mises']]

# pl = pv.Plotter()
pl = BackgroundPlotter()

pl.set_background('White')

sargs = dict(
    interactive=True,
    title_font_size=20,
    label_font_size=16,
    color='Black',
    # n_colors= 10
)

res = np.load('results/rot_test'+'_{}.npz'.format(0))
for item in res:
    if item[-4:] == 'Node':
        if len(res[item]) == meshplot.n_points:
            meshplot.point_data[item[:-5]] = res[item]
        else:
            meshplot.point_data[item[:-5]] = res[item].T
    else:
        meshplot.cell_data[item] = res[item].T

# # cpos = [(-2.69293081283409, 0.4520024822911473, 2.322209100082263),
# #         (0.4698685969042552, 0.46863550630755524, 0.42428354242422084),
# #         (0.5129241539116808, 0.07216479580221505, 0.8553952621921701)]
# # pl.camera_position = cpos

# global actor 
# actor = pl.add_axes(color='Black', interactive = True)


# # pl.add_mesh(meshplot.warp_by_vector(factor = 5), scalars = 'Stress', component = 2, clim = [0,10000], show_edges = True, cmap="bwr")
actor = pl.add_mesh(meshplot.warp_by_vector('Disp',factor = 1), scalars = 'Strain', component = 0, show_edges = True, scalar_bar_args=sargs, cmap="jet")
# pl.add_mesh(meshplot.warp_by_vector(factor = 1), scalars = 'svm', component = 0, show_edges = True, scalar_bar_args=sargs, cmap="jet")


def change_iter(value):
    global actor 
    
    pl.remove_actor(actor)

    # print(int(value))    
    res = np.load('results/rot_test'+'_{}.npz'.format(int(value)))
    for item in res:
        if item[-4:] == 'Node':
            if len(res[item]) == meshplot.n_points:
                meshplot.point_data[item[:-5]] = res[item]
            else:
                meshplot.point_data[item[:-5]] = res[item].T
        else:
            meshplot.cell_data[item] = res[item].T
    
    # pl.update_scalars(scalars, mesh=None, render=True)
    # pl.update_scalars(res['Strain_Node'][0])
    # pl.update()
    
    actor = pl.add_mesh(meshplot.warp_by_vector('Disp',factor = 1), scalars = 'Strain', component = 0, show_edges = True, scalar_bar_args=sargs, cmap="jet")
            
# slider = pl.add_slider_widget(

#     change_iter,

#     [0, 20],

#     title="Iter",

#     title_opacity=0.5,

#     title_color="red",

#     fmt="%0.9f",

#     title_height=0.08,
    
#     style = 'modern',

# )

slider = pl.add_text_slider_widget(

    change_iter,

    [str(i) for i in range(20)],
    
    style = 'modern',

)

pl.show()
# cpos = pl.show(return_cpos = True)
# pl.save_graphic('test.pdf', title='PyVista Export', raster=True, painter=True)




#ANIMATE

   
    
    
    
    






