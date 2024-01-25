# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:25:38 2022

@author: Etienne
"""
import fedoo as fd
from simcoon import simmit as sim
import numpy as np
import os 

#--------------- Pre-Treatment --------------------------------------------------------
space = fd.ModelingSpace("2Dstress")

mesh = fd.mesh.hole_plate_mesh(elm_type='quad4',sym=False)
crd = mesh.nodes
elm = mesh.elements
bounds = mesh.bounding_box
right = mesh.find_nodes('X', bounds.xmax)
left = mesh.find_nodes('X', bounds.xmin)


type_el = mesh.elm_type
# type_el = 'hex20'
xmax = np.max(crd[:,0]) ; xmin = np.min(crd[:,0])
ymax = np.max(crd[:,1]) ; ymin = np.min(crd[:,1])
center = mesh.nearest_node(mesh.bounding_box.center)
    
strain_nodes = mesh.add_virtual_nodes(2)

# BC_perturb = np.eye(6)
# BC_perturb[3:6,3:6] *= 2 #2xEXY

young_modulus = 1e5
poisson_ratio = 0.3
material = fd.constitutivelaw.ElasticIsotrop(young_modulus,poisson_ratio)
   
#Assembly
wf = fd.weakform.StressEquilibrium(material)

assemb = fd.Assembly.create(wf, mesh, type_el)

#Type of problem
pb = fd.problem.Linear(assemb)

#Shall add other conditions later on
bc_periodic = fd.constraint.PeriodicBC([strain_nodes[0], strain_nodes[1], strain_nodes[0]],
                                       ['DispX',        'DispY',        'DispY'], dim=2)
pb.bc.add(bc_periodic)

pb.bc.add('Dirichlet', strain_nodes[1], 'DispX', 0)
pb.bc.add('Dirichlet', center, 'Disp', 0, name = 'center')


#create a 'result' folder and set the desired ouputs
if not(os.path.isdir('results')): os.mkdir('results')
# res = pb.add_output('results/test', assemb, ['Disp', 'Stress', 'Strain'])    


pb.apply_boundary_conditions()

eps_xx = 0.1
eps_yy = 0
gamma_xy = 0.5

pb.bc.remove("_Strain")
pb.bc.add('Dirichlet', [strain_nodes[0]], 'DispX',
      eps_xx, start_value=0, name = '_Strain')  # EpsXX
pb.bc.add('Dirichlet', [strain_nodes[1]], 'DispY',
      eps_yy, start_value=0, name = '_Strain')  # EpsYY        
pb.bc.add('Dirichlet', [strain_nodes[0]], 'DispY',
      gamma_xy, start_value=0, name = '_Strain')  # 2EpsXY



pb.apply_boundary_conditions()

pb.solve()
i = 0
assemb.sv

res = pb.get_results(assemb, ['Disp', 'Stress', 'Strain'], 'Node')
# pb.save_results(i)

# res = fd.core.dataset.read_data('results/test')

#mean_stress
# X = pb.get_X()  
# DofFree = pb._Problem__dof_free
# MatCB = pb._Problem__MatCB
# F = MatCB.T @ pb.get_A() @ MatCB @ X[DofFree]

# F = F.reshape(2, -1)
# stress = [F[0, -2], F[1, -2], 0, F[0, -1], 0, 0]

volume = mesh.bounding_box.volume
stress = [mesh.integrate_field(res['Stress',i], type_field = 'GaussPoint')/volume for i in [0,1,3]]

assert 0


# X = pb.get_X()  

#     DStrain.append(np.array([pb._get_vect_component(X, 'DispX')[strain_nodes[0]], pb._get_vect_component(X, 'DispY')[strain_nodes[0]], 0,
#                              pb._get_vect_component(X, 'DispX')[strain_nodes[1]], 0, 0]))        

#     F = MatCB.T @ pb.get_A() @ MatCB @ X[DofFree]

#     F = F.reshape(2, -1)
#     stress = [F[0, -2], F[1, -2], 0, F[0, -1], 0, 0]

#     DStress.append(stress)


# if typeBC == "Neumann":
#     L_eff = np.linalg.inv(np.array(DStrain).T)
# else:
#     L_eff = np.array(DStress).T




# # np.set_printoptions(precision=3, suppress=True)
# print('L_eff = ', L_eff)

# props_test_eff = sim.L_iso_props(L_eff)
# print('props', props_test_eff)












# import matplotlib.pyplot as plt

# from matplotlib import cm, colors

# plt.rcParams['text.usetex'] = True

# plt.rcParams["figure.figsize"] = (20,8)



# phi = np.linspace(0,2*np.pi, 128) # the angle of the projection in the xy-plane

# theta = np.linspace(0, np.pi, 128).reshape(128,1) # the angle from the polar axis, ie the polar angle



# n_1 = np.sin(theta)*np.cos(phi)

# n_2 = np.sin(theta)*np.sin(phi)

# n_3 = np.cos(theta)*np.ones(128)



# n = np.array([n_1*n_1, n_2*n_2, n_3*n_3, n_1*n_2, n_1*n_3, n_2*n_3]).transpose(1,2,0).reshape(128,128,1,6)



# M = np.linalg.inv(L_eff)



# S = (n@M@n.reshape(128,128,6,1)).reshape(128,128)



# E = (1./S)

# x = E*n_1

# y = E*n_2

# z = E*n_3



# #E = E/E.max()



# fig = plt.figure(figsize=plt.figaspect(1))  # Square figure

# ax = fig.add_subplot(111, projection='3d')



# # make the panes transparent

# ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# # make the grid lines transparent

# ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)

# ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)

# ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

# ax.set_axis_off()



# #ax.plot_surface(x, y, z, cmap='hot',c=E)



# #norm = colors.Normalize(vmin = 0., vmax = 10000, clip = False)

# Emin = np.min(E)

# Eavg = np.average(E)

# Emax = np.max(E)

# norm = colors.Normalize(vmin = Emin, vmax = Emax, clip = False)

# surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, norm=norm, facecolors=cm.cividis(norm(E)),linewidth=0, antialiased=False, shade=False)



# #ax.set_xlim(0,20000)

# #ax.set_ylim(0,20000)

# #ax.set_zlim(0,20000)

# #ax.set_xlabel(r'$E_x$ (MPa)')

# #ax.set_ylabel(r'$E_y$ (MPa)')

# #ax.set_zlabel(r'$E_z$ (MPa)')



# scalarmap = cm.ScalarMappable(cmap=plt.cm.cividis, norm=norm)

# scalarmap.set_clim(np.min(E),np.max(E))

# #m.set_array([])

# cbar = plt.colorbar(scalarmap, orientation="horizontal", fraction=0.06, pad=-0.1, ticks=[Emin, Eavg, Emax])

# cbar.ax.tick_params(labelsize='large')

# cbar.set_label(r'directional stiffness $E$ (MPa)', size=15, labelpad=20)



# #ax.figure.axes[0].tick_params(axis="both", labelsize=5)

# ax.figure.axes[1].tick_params(axis="x", labelsize=20)



# ax.azim = 30

# ax.elev = 30



# #Volume_mesh = Assembly.get_all()['Assembling'].integrate_field(np.ones_like(TensorStress[0]))



# plt.savefig("directional.png", transparent=True)

# plt.show()





