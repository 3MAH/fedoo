from fedoo import *
import numpy as np
from time import time
import os
import pylab as plt
from numpy import linalg

start = time()
#--------------- Pre-Treatment --------------------------------------------------------

Util.ProblemDimension("3D")

meshname = "Domain"
nb_iter = 100

# Mesh.box_mesh(Nx=3, Ny=3, Nz=3, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1, ElementShape = 'hex8', name = meshname) 
# Mesh.import_file('octet_surf.msh', meshname = "Domain")
# Mesh.import_file('data/octet_1.msh', meshname = "Domain")
Mesh.import_file('data/gyroid.msh', meshname = "Domain")

mesh = Mesh.get_all()[meshname]

crd = mesh.nodes 

K = 500 # K = 18 #W/K/m
c = 0.500 #J/kg/K
rho = 7800 #kg/m2
Material = ConstitutiveLaw.ThermalProperties(K, c, rho, name='ThermalLaw')
wf = WeakForm.HeatEquation("ThermalLaw")
assemb = Assembly.Create("ThermalLaw", meshname, name="Assembling")    

#note set for boundary conditions
Xmin, Xmax = mesh.bounding_box()
bottom = mesh.find_nodes('Z', Xmin[2])
top = mesh.find_nodes('Z', Xmax[2])
left = mesh.find_nodes('X', Xmin[2])
right = mesh.find_nodes('X', Xmax[2])

Problem.NonLinearStatic("Assembling")

# Problem.SetSolver('cg', precond = True)

Problem.SetNewtonRaphsonErrorCriterion("Displacement", tol = 5e-2, max_subiter=5, err0 = 100)

#create a 'result' folder and set the desired ouputs
if not(os.path.isdir('results')): os.mkdir('results')
Problem.AddOutput('results/thermal3D', 'Assembling', ['temp'], output_type='Node', file_format ='npz')    
# Problem.AddOutput('results/bendingPlastic', 'Assembling', ['cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')    

tmax = 10
# Problem.BoundaryCondition('Dirichlet','Temp',0,bottom)
def timeEvolution(timeFactor): 
    if timeFactor == 0: return 0
    else: return 1

# Problem.BoundaryCondition('Dirichlet','Temp',100,left, timeEvolution=timeEvolution)
Problem.BoundaryCondition('Dirichlet','Temp',100,right, timeEvolution=timeEvolution)
# Problem.BoundaryCondition('Dirichlet','Temp',100,top, timeEvolution=timeEvolution)


# Problem.BoundaryCondition('Dirichlet','DispY', 0,nodes_bottomLeft)
# Problem.BoundaryCondition('Dirichlet','DispY',0,nodes_bottomRight)
# bc = Problem.BoundaryCondition('Dirichlet','DispY', uimp, nodes_topCenter)

Problem.NLSolve(dt = tmax/nb_iter, tmax = tmax, update_dt = True)



#Generate video using pyvista
import pyvista as pv

# meshplot = pv.read('results/thermal3D_43.vtk')
meshplot = mesh.to_pyvista()
# meshplot["nd_label"] = [str(i) for i in range(meshplot.n_points)]

pl = pv.Plotter()
pl.set_background('White')
# pl.add_point_labels(meshplot, "nd_label", point_size=10, font_size=10)

sargs = dict(
    interactive=True,
    title_font_size=20,
    label_font_size=16,
    color='Black',
    # n_colors= 10
)

# cpos = [(-2.05821994783786, 1.967185181335808, 1.731622321366397),
#         (0.7843951189600558, 0.21392551951633995, 0.16595366983397655),
#         (0.4236410761781089, -0.11545074454760883, 0.8984427439509189)]

cpos = [(-2.090457552750125, 1.7582929402632352, 1.707926514944027),
        (0.20739316009534275, -0.2296587829717462, -0.38339561081860574),
        (0.42357673667356105, -0.37693638734293083, 0.8237121512068624)]

pl.camera_position = cpos

res = np.load('results/thermal3D_43.npz')
meshplot.point_data["data"] = res['Temp_Node']
# meshplot = meshplot.clip('X', (0.8,0,0))
# pl.add_mesh(meshplot.warp_by_vector(factor = 5), scalars = 'Stress', component = 2, clim = [0,10000], show_edges = True, cmap="bwr")
pl.add_mesh(meshplot, scalars = "data", show_edges = True, scalar_bar_args=sargs, cmap="bwr")
# pl.add_mesh(meshplot, show_edges = True, scalar_bar_args=sargs, cmap="bwr")


# pl.save_graphic('test.pdf', title='PyVista Export', raster=True, painter=True)
cpos = pl.show(return_cpos = True, screenshot='gyroid.png')  

# assert 0