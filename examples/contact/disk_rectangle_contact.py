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
h = 1
L = 1
E = 200e3
nu = 0.3
alpha = 1e-5

filename = "disk_rectangle_contact"
res_dir = "results/"

mesh_rect = fd.mesh.rectangle_mesh(
    nx=11, ny=21, x_min=0, x_max=L, y_min=0, y_max=h, elm_type="quad4", name="Domain"
)
mesh_rect.element_sets["rect"] = np.arange(0, mesh_rect.n_elements)
mesh_disk = fd.mesh.disk_mesh(radius=L / 2, nr=6, nt=6, elm_type="quad4")
# surf = fd.constraint.contact.extract_surface_mesh(fd.Mesh.from_pyvista(pv.Circle(radius=0.5, resolution = 30).triangulate()))
mesh_disk.nodes += np.array([1.5, 0.48])
mesh_disk.element_sets["disk"] = np.arange(0, mesh_disk.n_elements)

mesh = fd.Mesh.stack(mesh_rect, mesh_disk)

# node sets for boundary conditions
nodes_left = mesh.find_nodes("X", 0)
nodes_right = mesh.find_nodes("X", L)

# nodes_top = mesh.find_nodes('Y',1)
nodes_bc = mesh.find_nodes("X>1.5")
nodes_bc = list(set(nodes_bc).intersection(mesh.node_sets["boundary"]))

# if slave surface = disk
# nodes_contact = mesh.node_sets['boundary']
# surf = fd.mesh.extract_surface(mesh.extract_elements('rect')) #extract the surface of the rectangle
# surf = surf.extract_elements(surf.get_elements_from_nodes(nodes_right))

# if slave surface = rectangle
nodes_contact = nodes_right
surf = fd.mesh.extract_surface(
    mesh.extract_elements("disk")
)  # extract the surface of the disk

contact = fd.constraint.Contact(nodes_contact, surf)
contact.contact_search_once = True
contact.eps_n = 5e5
contact.max_dist = 1

mat = 1
if mat == 0:
    props = np.array([E, nu, alpha])
    material_rect = fd.constitutivelaw.Simcoon("ELISO", props, name="ConstitutiveLaw")
elif mat == 1 or mat == 2:
    Re = 300
    k = 1000  # 1500
    m = 0.3  # 0.25
    if mat == 1:
        props = np.array([E, nu, alpha, Re, k, m])
        material_rect = fd.constitutivelaw.Simcoon(
            "EPICP", props, name="ConstitutiveLaw"
        )
    elif mat == 2:
        material_rect = fd.constitutivelaw.ElastoPlasticity(
            E, nu, Re, name="ConstitutiveLaw"
        )
        material_rect.SetHardeningFunction("power", H=k, beta=m)
else:
    material_rect = fd.constitutivelaw.ElasticIsotrop(E, nu, name="ConstitutiveLaw")

material_disk = fd.constitutivelaw.ElasticIsotrop(50e3, nu, name="ConstitutiveLaw")

material = fd.constitutivelaw.Heterogeneous(
    (material_rect, material_disk), ("rect", "disk")
)

wf = fd.weakform.StressEquilibrium(material, nlgeom=NLGEOM)
solid_assembly = fd.Assembly.create(wf, mesh)

# assembly = fd.Assembly.sum(solid_assembly1, solid_assembly2, contact)
assembly = fd.Assembly.sum(solid_assembly, contact)

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

pb.bc.add("Dirichlet", nodes_left, "Disp", 0)
pb.bc.add("Dirichlet", nodes_bc, "Disp", [-0.4, 0.2])

# Problem.set_solver('cg', precond = True)
# pb.set_nr_criterion("Force", err0 = None, tol = 5e-3, max_subiter = 10)
pb.set_nr_criterion("Displacement", err0=None, tol=5e-3, max_subiter=5)

# pb.nlsolve(dt = 0.001, tmax = 1, update_dt = False, print_info = 2, interval_output = 0.001)
pb.nlsolve(dt=0.005, tmax=1, update_dt=True, print_info=1, interval_output=0.01)


pb.bc.remove(-1)  # remove last boundary contidion
pb.bc.add("Dirichlet", nodes_bc, "Disp", [0, 0])

pb.nlsolve(dt=0.005, tmax=1, update_dt=True, print_info=1, interval_output=0.01)

print(time() - start)


# =============================================================
# Example of plots with pyvista - uncomment the desired plot
# =============================================================

# ------------------------------------
# Simple plot with default options
# ------------------------------------
results.plot("Stress", "vm", "Node", show=True, scale=1, show_nodes=True)
results.plot("Stress", "XX", "Node", show=True, scale=1, show_nodes=True)
# results.plot('Fext',  'X', 'Node', show = True, scale = 1, show_nodes=True)

results.plot("Disp", 0, "Node", show=True, scale=1, show_nodes=True)

# ------------------------------------
# Write movie with default options
# ------------------------------------
# results.write_movie(res_dir+filename, 'Stress', 'vm', framerate = 5, quality = 5)

results.write_movie(
    res_dir + filename,
    "Stress",
    "XX",
    "Node",
    framerate=24,
    quality=5,
    clim=[-3e3, 3e3],
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
# t, sigma = results.get_history(('Time','Stress'), (0,12), component = 3)
# pylab.plot(t,sigma)
# #or results.plot_history('Stress', 12)
