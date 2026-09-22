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

mesh = fd.mesh.box_mesh(
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
    name="Domain",
)
# mesh = fd.mesh.import_file('../../util/meshes/octet_truss.msh', name = "Domain")['tet4']
# mesh = fd.mesh.import_file("../../util/meshes/octet_truss_2.msh", name="Domain")["tet4"]

crd = mesh.nodes

mat = 0
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
        material.set_hardening_function("power", h=k, beta=m)
else:
    material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="ConstitutiveLaw")

wf = fd.weakform.StressEquilibrium("ConstitutiveLaw", nlgeom=NLGEOM, name="wf")
wf.incompressibility = "auto"

assemb = fd.Assembly.create("wf", mesh, name="Assembling")

# node set for boundary conditions
left = mesh.find_nodes("X", 0)
right = mesh.find_nodes("X", 1)

pb = fd.problem.NonLinear("Assembling")

# create a 'result' folder and set the desired ouputs
if not (os.path.isdir("results")):
    os.mkdir("results")

results = pb.add_output(
    res_dir + filename,
    "Assembling",
    ["Disp", "Stress", "Strain"],  # "P", "EP", "Wm", "Fext", "RigidDisp", "RigidRot"],
)

pb.bc.add(fd.constraint.RigidTie(right))

pb.bc.add("Dirichlet", left, "Disp", 0)
pb.bc.add("Dirichlet", "RigidRotX", 2 * np.pi / 2)  # Rigid rotation of the right end

pb.nlsolve(dt=0.05, tmax=1, update_dt=True, print_info=1, interval_output=0.025)

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
results.write_movie(res_dir + filename, "Stress", "vm", framerate=12, quality=5)

# ------------------------------------
# Save pdf plot
# ------------------------------------
# pl = results.plot('Stress', 'vm', show = False)
# pl.save_graphic('test.pdf', title='PyVista Export', raster=True, painter=True)

# ------------------------------------
# Plot the automatically saved mesh
# ------------------------------------
# fd.read_data(res_dir+filename).plot()

# ------------------------------------
# Write movie with moving camera
# ------------------------------------
# results.write_movie(res_dir+filename, 'Stress', component = 'XX', framerate = 12, quality = 5, rot_azimuth = -1.5, rot_elevation = 0)
