import fedoo as fd
import numpy as np
from time import time
import os
import pylab as plt
from numpy import linalg

start = time()
# --------------- Pre-Treatment --------------------------------------------------------


# -------------------- MESH ------------------------------
# fd.mesh.box_mesh(Nx=3, Ny=3, Nz=3, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1, ElementShape = 'hex8', name = meshname)
# fd.mesh.import_file('octet_surf.msh', meshname = "Domain")
# fd.mesh.import_file('data/octet_1.msh', meshname = "Domain")
fd.mesh.import_file("../../util/meshes/gyroid.msh", name="Domain")
meshname = "Domain"
filename_res = "results/thermo_meca_nl"

mesh = fd.Mesh[meshname]

crd = mesh.nodes

# note set for boundary conditions
Xmin, Xmax = mesh.bounding_box
left = mesh.find_nodes("X", Xmin[0])
right = mesh.find_nodes("X", Xmax[0])
bottom = mesh.find_nodes("Y", Xmin[1])
top = mesh.find_nodes("Y", Xmax[1])
back = mesh.find_nodes("Z", Xmin[2])
front = mesh.find_nodes("Z", Xmax[2])


boundary = np.unique(np.hstack((bottom, top, left, right, back, front)))

# -------------------- Thermal Problem ------------------------------
thermal_space = fd.ModelingSpace("3D")


K = 500  # K = 18 #W/K/m
c = 0.500  # J/kg/K
rho = 7800  # kg/m2
thermal_law = fd.constitutivelaw.ThermalProperties(K, c, rho, name="ThermalLaw")
wf_th = fd.weakform.HeatEquation("ThermalLaw", space=thermal_space)
assemb = fd.Assembly.create(wf_th, meshname, name="Assembling_T")

pb_th = fd.problem.NonLinear("Assembling_T")

# Problem.set_solver('cg', precond = True)
pb_th.set_nr_criterion("Displacement", tol=5e-2, max_subiter=5, err0=100)

# -------------------- Mechanical Problem ------------------------------
mech_space = fd.ModelingSpace("3D")

E = 200e3
nu = 0.3
alpha = 1e-3  # alpha = 1e-5 #???
NLGEOM = False
mat = 0
if mat == 0:
    props = np.array([E, nu, alpha])
    mechancial_law = fd.constitutivelaw.Simcoon("ELISO", props, name="MechanicalLaw")
elif mat == 1 or mat == 2:
    Re = 300
    k = 1000  # 1500
    m = 1  # 0.3 #0.25
    if mat == 1:
        props = np.array([E, nu, alpha, Re, k, m])
        mechancial_law = fd.constitutivelaw.Simcoon(
            "EPICP", props, name="MechanicalLaw"
        )
    elif mat == 2:
        mechancial_law = fd.constitutivelaw.ElastoPlasticity(
            E, nu, Re, name="MechanicalLaw"
        )
        mechancial_law.SetHardeningFunction("power", H=k, beta=m)
else:
    mechancial_law = fd.constitutivelaw.ElasticIsotrop(E, nu, name="MechanicalLaw")

wf_mech = fd.weakform.StressEquilibrium("MechanicalLaw", nlgeom=NLGEOM)

fd.Assembly.create(
    wf_mech, meshname, name="Assembling_M"
)  # uses MeshChange=True when the mesh change during the time

pb_m = fd.problem.NonLinear("Assembling_M")
# pb_m.set_solver('cg', precond = True)
pb_m.set_nr_criterion("Displacement", tol=1e-2, err0=1)

# -------------------- Set output ------------------------------
# create a 'result' folder and set the desired ouputs
res_m = pb_m.add_output(
    filename_res + "_me",
    "Assembling_M",
    ["Disp", "Stress", "Strain", "Stress_vm", "Statev", "Wm"],
)

res_th = pb_th.add_output(filename_res + "_th", "Assembling_T", ["Temp"])
# # Problem.add_output('results/bendingPlastic', 'Assembling', ['cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')


# -------------------- Boundary Conditions ------------------------------
# def timeEvolution(timeFactor):
#     if timeFactor == 0: return 0
#     else: return 1
def timeEvolution(timeFactor):
    if timeFactor < 0.5:
        return 2 * timeFactor
    else:
        return 1


pb_th.bc.add("Dirichlet", right, "Temp", 100, time_func=timeEvolution, name="temp")
# pb_th.bc.add('Dirichlet',right,'Temp',100, name='temp')

pb_m.bc.add("Dirichlet", boundary, "Disp", 0)

# -------------------- Solve  ------------------------------

nb_iter = 100
tmax = 30
dt = tmax / nb_iter

pb_th.initialize()
pb_m.initialize()

pb_th.tmax = tmax
pb_m.tmax = tmax
pb_th.dtime = pb_m.dtime = dt

pb_th.set_start()
pb_m.set_start()

for i in range(nb_iter):
    pb_th.time = pb_m.time = time = i * dt
    convergence, nbNRiter, normRes = pb_th.solve_time_increment()
    # assert convergence, 'thermal problem has not converged'
    if not (convergence):
        print("WARNING: iteration", i, "has not converged: err =", normRes)
    pb_th.set_start()
    pb_th.save_results(i)

    print(
        "Iter {} - Therm - Time: {:.5f} - NR iter: {} - Err: {:.5f}".format(
            i, time, nbNRiter, normRes
        )
    )
    pb_m.set_start(dt)
    # temp = assemb.convert_data(pb_th.get_temp(), convert_from='Node', convert_to='GaussPoint')
    pb_m.assembly.sv["Temp"] = pb_th.assembly.sv["Temp"]

    convergence, nbNRiter, normRes = pb_m.solve_time_increment()
    assert convergence, "mechanical problem has not converged"
    pb_m.set_start()
    pb_m.save_results(i)

    print(
        "Iter {} - Mech - Time: {:.5f} - NR iter: {} - Err: {:.5f}".format(
            i, time, nbNRiter, normRes
        )
    )


# pb_th.RemoveBC('temp')
# pb_th.bc.add('Dirichlet','Temp',0,right, timeEvolution=timeEvolution, initialValue = 100, name='temp')
# pb_th.GetBC('temp')[0].ChangeValue(0, initialValue = 'Current')
pb_th.bc[0].value = 0
# pb_th.apply_boundary_conditions()

pb_th.t0 = tmax
pb_th.tmax = 2 * tmax
pb_m.t0 = tmax
pb_m.tmax = 2 * tmax

for i in range(nb_iter, 2 * nb_iter):
    time = i * dt
    convergence, nbNRiter, normRes = pb_th.solve_time_increment()
    # assert convergence, 'thermal problem has not converged'
    if not (convergence):
        print("WARNING: iteration", i, "has not converged: err =", normRes)
    pb_th.save_results(i)

    print(
        "Iter {} - Therm - Time: {:.5f} - NR iter: {} - Err: {:.5f}".format(
            i, time, nbNRiter, normRes
        )
    )
    # temp = pb_th.assembly.convert_data(pb_th.get_temp(), convert_from='Node', convert_to='GaussPoint')
    pb_m.assembly.sv["Temp"] = pb_th.assembly.sv["Temp"]

    convergence, nbNRiter, normRes = pb_m.solve_time_increment()
    assert convergence, "mechanical problem has not converged"
    pb_m.save_results(i)

    print(
        "Iter {} - Mech - Time: {:.5f} - NR iter: {} - Err: {:.5f}".format(
            i, time, nbNRiter, normRes
        )
    )

nb_iter = 2 * nb_iter


### Simple movie output
# res_m.write_movie('stress_field.mp4', 'Stress_vm')
# res_th.write_movie('temp_field.mp4', 'Temp')
