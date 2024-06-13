import fedoo as fd
import numpy as np
from time import time
import os
import pylab as plt
from numpy import linalg

import pyvista as pv


start = time()
# --------------- Pre-Treatment --------------------------------------------------------

fd.ModelingSpace("2D")

NLGEOM = "UL"
# Units: N, mm, MPa
h = 100
w = 1
L = 100
E = 200e3
nu = 0.3
alpha = 1e-5  # ???
meshname = "Domain"
uimp = 1


filename = "self_contact"
res_dir = "results/"


# mesh = fd.mesh.box_mesh(nx=11, ny=11, nz=11, x_min=0, x_max=L, y_min=0, y_max=h, z_min = 0, z_max = w, elm_type = 'hex8', name = meshname)

mesh = fd.mesh.hole_plate_mesh(nr=15, nt=15, length=100, height=100, radius=45)
surf = fd.mesh.extract_surface(mesh)

# surf.plot() #to plot the mesh
# surf.plot_normals(5) #to check normals orientation

# contact = fd.constraint.contact.SelfContact(surf, 'biliear')
contact = fd.constraint.contact.SelfContact(surf, "linear", search_algorithm="bucket")
contact.contact_search_once = True
contact.eps_n = 1e6
# contact.eps_a = 1e4
# contact.limit_soft_contact = 0.01

# node sets for boundary conditions
nodes_top = mesh.find_nodes("Y", mesh.bounding_box.ymax)
nodes_bottom = mesh.find_nodes("Y", mesh.bounding_box.ymin)

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


#### trouver pourquoi les deux fonctions suivantes ne donnent pas la même chose !!!!
wf = fd.weakform.StressEquilibrium("ConstitutiveLaw", nlgeom=NLGEOM)

solid_assembly = fd.Assembly.create(
    wf, mesh, name="Assembling"
)  # uses MeshChange=True when the mesh change during the time

assembly = fd.Assembly.sum(solid_assembly, contact)
# assembly = solid_assembly

pb = fd.problem.NonLinear(assembly)

# create a 'result' folder and set the desired ouputs
if not (os.path.isdir("results")):
    os.mkdir("results")
# results = pb.add_output(res_dir+filename, 'Assembling', ['Disp'], output_type='Node', file_format ='npz')
# results = pb.add_output(res_dir+filename, 'Assembling', ['Cauchy', 'PKII', 'Strain', 'Cauchy_vm', 'Statev', 'Wm'], output_type='GaussPoint', file_format ='npz')

# results = pb.add_output(res_dir+filename, solid_assembly, ['Disp', 'Cauchy', 'PKII', 'Strain', 'Cauchy_vm', 'Statev', 'Wm'])
results = pb.add_output(
    res_dir + filename, solid_assembly, ["Disp", "Stress", "Strain", "Fext"]
)

# Problem.add_output(res_dir+filename, 'Assembling', ['cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')


################### step 1 ################################
pb.bc.add("Dirichlet", nodes_bottom, "Disp", 0)
pb.bc.add("Dirichlet", nodes_top, "Disp", [0, -70])

# Problem.set_solver('cg', precond = True)
# pb.set_nr_criterion("Force", err0 = None, tol = 5e-3, max_subiter = 10)
pb.set_nr_criterion("Displacement", err0=None, tol=5e-3, max_subiter=5)

# pb.nlsolve(dt = 0.001, tmax = 1, update_dt = False, print_info = 2, interval_output = 0.001)
pb.nlsolve(dt=0.01, tmax=1, update_dt=True, print_info=1, interval_output=0.01)

E = np.array(
    fd.Assembly.get_all()["Assembling"].get_strain(
        pb.get_dof_solution(), "GaussPoint", False
    )
).T

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
results.plot("Stress", "vm", "Node", show=True, scale=1)
# results.plot('Stress', 'XX', 'Node', show = True, scale = 1, show_nodes=True)
# results.plot('Stress', 'XX', 'Node', show = True, scale = 1, show_nodes=True,  node_labels =True)
# results.plot('Fext', 'X', 'Node', show = True, scale = 1, show_nodes=True)

# ------------------------------------
# Simple plot with default options and save to png
# ------------------------------------
# pl = results.plot('Stress', 0, show = False)
# pl.show(screenshot = "test.png")

# ------------------------------------
# Write movie with default options
# ------------------------------------
# results.write_movie(res_dir+filename, 'Stress', 'vm', framerate = 5, quality = 5)

results.write_movie(
    res_dir + filename,
    "Stress",
    "vm",
    "Node",
    framerate=24,
    quality=10,
    clim=[0, 1.5e3],
)

# ------------------------------------
# Save pdf plot
# ------------------------------------
# pl = results.plot('Stress', 'vm', show = False)
# pl.save_graphic('test.pdf', title='PyVista Export', raster=True, painter=True)

# ------------------------------------
# Plot time history
# ------------------------------------
# from matplotlib import pylab
# t, sigma = results.get_history(('Time','Cauchy'), (0,12), component = 3)
# pylab.plot(t,sigma)
# #or results.plot_history('Cauchy', 12)
