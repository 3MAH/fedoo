from fedoo import *
import numpy as np
from time import time
import os
import pylab as plt
from numpy import linalg

start = time()
#--------------- Pre-Treatment --------------------------------------------------------


# -------------------- MESH ------------------------------
# Mesh.box_mesh(Nx=3, Ny=3, Nz=3, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1, ElementShape = 'hex8', name = meshname) 
# Mesh.import_file('octet_surf.msh', meshname = "Domain")
# Mesh.import_file('data/octet_1.msh', meshname = "Domain")
Mesh.import_file('data/gyroid.msh', meshname = "Domain")
meshname = "Domain"
filename_res = 'results/thermo_meca_nl'

mesh = Mesh.get_all()[meshname]

crd = mesh.nodes 

#note set for boundary conditions
Xmin, Xmax = mesh.bounding_box()
left = mesh.find_nodes('X', Xmin[0])
right = mesh.find_nodes('X', Xmax[0])
bottom = mesh.find_nodes('Y', Xmin[1])
top = mesh.find_nodes('Y', Xmax[1])
back = mesh.find_nodes('Z', Xmin[2])
front = mesh.find_nodes('Z', Xmax[2])



boundary = np.unique(np.hstack((bottom,top,left,right, back, front)))

# -------------------- Thermal Problem ------------------------------
thermal_space = Util.ProblemDimension("3D")


K = 500 # K = 18 #W/K/m
c = 0.500 #J/kg/K
rho = 7800 #kg/m2
thermal_law = ConstitutiveLaw.ThermalProperties(K, c, rho, name='ThermalLaw')
wf_th = WeakForm.HeatEquation("ThermalLaw", space = thermal_space)
assemb = Assembly.Create("ThermalLaw", meshname, name="Assembling_T")    

pb_th = Problem.NonLinearStatic("Assembling_T")

# Problem.SetSolver('cg', precond = True)
pb_th.SetNewtonRaphsonErrorCriterion("Displacement", tol = 5e-2, max_subiter=5, err0 = 100)

# -------------------- Mechanical Problem ------------------------------
mech_space = Util.ProblemDimension("3D")

E = 200e3
nu=0.3
alpha = 1e-3 # alpha = 1e-5 #???
NLGEOM = False
mat =0
if mat == 0:
    props = np.array([[E, nu, alpha]])
    mechancial_law = ConstitutiveLaw.Simcoon("ELISO", props, 1, name='MechanicalLaw')
    mechancial_law.corate = 2
    # Material.SetMaskH([[] for i in range(6)])
elif mat == 1 or mat == 2:
    Re = 300
    k=1000 #1500
    m=1#0.3 #0.25
    if mat == 1:
        props = np.array([[E, nu, alpha, Re,k,m]])
        mechancial_law = ConstitutiveLaw.Simcoon("EPICP", props, 8, name='MechanicalLaw')
        mechancial_law.corate = 2
    elif mat == 2:
        mechancial_law = ConstitutiveLaw.ElastoPlasticity(E,nu,Re, name='MechanicalLaw')
        mechancial_law.SetHardeningFunction('power', H=k, beta=m)
else:
    mechancial_law = ConstitutiveLaw.ElasticIsotrop(E, nu, name='MechanicalLaw')

WeakForm.InternalForce("MechanicalLaw", nlgeom = NLGEOM)

Assembly.Create("MechanicalLaw", meshname, name="Assembling_M")     #uses MeshChange=True when the mesh change during the time

pb_m = Problem.NonLinearStatic("Assembling_M")
# pb_m.SetSolver('cg', precond = True)
pb_m.SetNewtonRaphsonErrorCriterion("Displacement", tol = 1e-2, err0 = 1)

# -------------------- Set output ------------------------------
#create a 'result' folder and set the desired ouputs
pb_m.AddOutput(filename_res+'_me', 'Assembling_M', ['disp', 'cauchy', 'strain', 'cauchy_vm', 'statev', 'wm'], output_type='Node', file_format ='npz')    

pb_th.AddOutput(filename_res+'_th', 'Assembling_T', ['temp'], output_type='Node', file_format ='npz')    
# # Problem.AddOutput('results/bendingPlastic', 'Assembling', ['cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')    

# -------------------- Boundary Conditions ------------------------------
# def timeEvolution(timeFactor): 
#     if timeFactor == 0: return 0
#     else: return 1
def timeEvolution(timeFactor): 
    if timeFactor < 0.5: return 2*timeFactor
    else: return 1


pb_th.BoundaryCondition('Dirichlet','Temp',100,right, timeEvolution=timeEvolution, name='temp')
# pb_th.BoundaryCondition('Dirichlet','Temp',100,right, name='temp')

pb_m.BoundaryCondition('Dirichlet','Disp',0,boundary)

# -------------------- Solve  ------------------------------

nb_iter = 100
tmax = 30
dt = tmax/nb_iter
   
pb_th.Initialize()
pb_m.Initialize()

pb_th.tmax = tmax
pb_m.tmax = tmax

for i in range(nb_iter):    
    time = i*dt
    convergence, nbNRiter, normRes = pb_th.SolveTimeIncrement(time, dt)
    # assert convergence, 'thermal problem has not converged'
    if not(convergence): print('WARNING: iteration', i, 'has not converged: err =', normRes)
    pb_th.SaveResults(i)
    
    print('Iter {} - Therm - Time: {:.5f} - NR iter: {} - Err: {:.5f}'.format(i, time, nbNRiter, normRes))
    temp = assemb.ConvertData(pb_th.GetTemp(), convertFrom='Node', convertTo='GaussPoint')
    mechancial_law.set_T(temp)
    
    convergence, nbNRiter, normRes = pb_m.SolveTimeIncrement(time, dt)
    assert convergence, 'mechanical problem has not converged'
    pb_m.SaveResults(i)
    
    print('Iter {} - Mech - Time: {:.5f} - NR iter: {} - Err: {:.5f}'.format(i, time, nbNRiter, normRes))


# pb_th.RemoveBC('temp')
# pb_th.BoundaryCondition('Dirichlet','Temp',0,right, timeEvolution=timeEvolution, initialValue = 100, name='temp')
pb_th.GetBC('temp')[0].ChangeValue(0, initialValue = 'Current')
# pb_th.ApplyBoundaryCondition()

pb_th.t0 = tmax
pb_th.tmax = 2*tmax
pb_m.t0 = tmax
pb_m.tmax = 2*tmax

for i in range(nb_iter, 2*nb_iter):    
    time = i*dt
    convergence, nbNRiter, normRes = pb_th.SolveTimeIncrement(time, dt)
    # assert convergence, 'thermal problem has not converged'
    if not(convergence): print('WARNING: iteration', i, 'has not converged: err =', normRes)
    pb_th.SaveResults(i)
    
    print('Iter {} - Therm - Time: {:.5f} - NR iter: {} - Err: {:.5f}'.format(i, time, nbNRiter, normRes))
    temp = assemb.ConvertData(pb_th.GetTemp(), convertFrom='Node', convertTo='GaussPoint')
    mechancial_law.set_T(temp)
    
    convergence, nbNRiter, normRes = pb_m.SolveTimeIncrement(time, dt)
    assert convergence, 'mechanical problem has not converged'
    pb_m.SaveResults(i)
    
    print('Iter {} - Mech - Time: {:.5f} - NR iter: {} - Err: {:.5f}'.format(i, time, nbNRiter, normRes))

nb_iter = 2*nb_iter

# #Generate video using pyvista
# import pyvista as pv

# # meshplot = pv.read('results/thermal3D_43.vtk')
# meshplot = mesh.to_pyvista()
# # meshplot["nd_label"] = [str(i) for i in range(meshplot.n_points)]

# pl = pv.Plotter()
# pl.set_background('White')
# # pl.add_point_labels(meshplot, "nd_label", point_size=10, font_size=10)

# sargs = dict(
#     interactive=True,
#     title_font_size=20,
#     label_font_size=16,
#     color='Black',
#     # n_colors= 10
# )

# # cpos = [(-2.05821994783786, 1.967185181335808, 1.731622321366397),
# #         (0.7843951189600558, 0.21392551951633995, 0.16595366983397655),
# #         (0.4236410761781089, -0.11545074454760883, 0.8984427439509189)]

# cpos = [(-2.090457552750125, 1.7582929402632352, 1.707926514944027),
#         (0.20739316009534275, -0.2296587829717462, -0.38339561081860574),
#         (0.42357673667356105, -0.37693638734293083, 0.8237121512068624)]

# pl.camera_position = cpos

# res = np.load(filename_res+'_th_{}.npz'.format(nb_iter-1))
# meshplot.point_data["data"] = res['Temp_Node']

# res = np.load(filename_res+'_me_{}.npz'.format(nb_iter-1))
# meshplot.point_data["disp"] = res['Disp_Node'].T
# meshplot.point_data["stress_vm"] = res['Cauchy_Mises_Node']
# # meshplot.set_active_vectors("disp")

# # meshplot = meshplot.clip('X', (0.8,0,0))
# # pl.add_mesh(meshplot.warp_by_vector(factor = 5), scalars = 'Stress', component = 2, clim = [0,10000], show_edges = True, cmap="bwr")
# pl.add_mesh(meshplot.warp_by_vector('disp',factor = 0.5), scalars = "data", show_edges = True, scalar_bar_args=sargs, cmap="jet")
# # pl.add_mesh(meshplot, show_edges = True, scalar_bar_args=sargs, cmap="bwr")


# # pl.save_graphic('test.pdf', title='PyVista Export', raster=True, painter=True)
# cpos = pl.show(return_cpos = True)  



