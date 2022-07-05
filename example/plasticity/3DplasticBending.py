from fedoo import *
import numpy as np
from time import time
import os
import pylab as plt
from numpy import linalg

start = time()
#--------------- Pre-Treatment --------------------------------------------------------

Util.ProblemDimension("3D")

NLGEOM = 2
typeBending = '3nodes' #'3nodes' or '4nodes'
#Units: N, mm, MPa
h = 2
w = 10
L = 16
E = 200e3
nu=0.3
alpha = 1e-5 #???
meshname = "Domain"
uimp = -8

Mesh.box_mesh(Nx=21, Ny=7, Nz=7, x_min=0, x_max=L, y_min=0, y_max=h, z_min = 0, z_max = w, ElementShape = 'hex8', name = meshname)
mesh = Mesh.get_all()[meshname]

crd = mesh.nodes 

mat =0
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
nodes_bottomLeft = mesh.find_nodes('XY',(0,0))
nodes_bottomRight = mesh.find_nodes('XY',(L,0))

if typeBending == '3nodes':
    nodes_topCenter = mesh.find_nodes('XY',(L/2,h))
    # nodes_topCenter = np.where((crd[:,0]==L/2) * (crd[:,1]==h))[0]
else: 
    nodes_top1 = mesh.find_nodes('XY',(L/4,h))
    nodes_top2 = mesh.find_nodes('XY',(3*L/4,h))
    nodes_topCenter = np.hstack((nodes_top1, nodes_top2))

# Assembly.Create("ConstitutiveLaw", meshname, 'hex8', name="Assembling", MeshChange = False, n_elm_gp = 27)     #uses MeshChange=True when the mesh change during the time
Assembly.Create("ConstitutiveLaw", meshname, 'hex8', name="Assembling", MeshChange = False, n_elm_gp = 8)     #uses MeshChange=True when the mesh change during the time

Problem.NonLinearStatic("Assembling")
# Problem.SetSolver('cg', precond = True)
Problem.SetNewtonRaphsonErrorCriterion("Displacement", err0 = 1, tol = 5e-3, max_subiter = 5)

# Problem.SetNewtonRaphsonErrorCriterion("Displacement")
# Problem.SetNewtonRaphsonErrorCriterion("Work")
# Problem.SetNewtonRaphsonErrorCriterion("Force")

#create a 'result' folder and set the desired ouputs
if not(os.path.isdir('results')): os.mkdir('results')
Problem.AddOutput('results/bendingPlastic3D', 'Assembling', ['Disp', 'Cauchy', 'PKII', 'Strain', 'Cauchy_vm', 'Statev', 'Wm'], output_type='Node', file_format ='vtk')    
# Problem.AddOutput('results/bendingPlastic3D', 'Assembling', ['cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')    


################### step 1 ################################
tmax = 1
Problem.BoundaryCondition('Dirichlet','DispX',0,nodes_bottomLeft)
Problem.BoundaryCondition('Dirichlet','DispY', 0,nodes_bottomLeft)
Problem.BoundaryCondition('Dirichlet','DispZ', 0,nodes_bottomLeft)
Problem.BoundaryCondition('Dirichlet','DispY',0,nodes_bottomRight)
bc = Problem.BoundaryCondition('Dirichlet','DispY', uimp, nodes_topCenter)

Problem.NLSolve(dt = 0.025, tmax = 1, update_dt = True, print_info = 1, intervalOutput = 0.05)


E = np.array(Assembly.get_all()['Assembling'].get_strain(Problem.GetDoFSolution(), "GaussPoint", False)).T

# ################### step 2 ################################
# bc.Remove()
# #We set initial condition to the applied force to relax the load
# F_app = Problem.get_ext_forces('DispY')[nodes_topCenter]
# bc = Problem.BoundaryCondition('Neumann','DispY', 0, nodes_topCenter, initialValue=F_app)#face_center)

# Problem.NLSolve(dt = 1., update_dt = True, ToleranceNR = 0.01)

print(time()-start)


### plot with pyvista



import pyvista as pv

meshplot = pv.read('results/bendingPlastic3D_15.vtk')
# meshplot.point_data['svm'] = np.c_[meshplot.point_data['Cauchy_Mises']]

pl = pv.Plotter()
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
pl.add_mesh(meshplot.warp_by_vector(factor = 1), scalars = 'Disp', component = 1, show_edges = True, scalar_bar_args=sargs, cmap="jet")
# pl.add_mesh(meshplot.warp_by_vector(factor = 1), scalars = 'svm', component = 0, show_edges = True, scalar_bar_args=sargs, cmap="jet")

cpos = pl.show(return_cpos = True)
# pl.save_graphic('test.pdf', title='PyVista Export', raster=True, painter=True)





   
    
    
    
    






