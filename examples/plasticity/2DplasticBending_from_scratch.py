import fedoo as fd
import numpy as np
from time import time
import os
import pylab as plt
from numpy import linalg

start = time()
# --------------- Pre-Treatment --------------------------------------------------------

fd.ModelingSpace("2Dplane")

typeBending = "3nodes"  #'3nodes' or '4nodes'
# Units: N, mm, MPa
h = 2
w = 10
L = 16
E = 200e3
nu = 0.3
alpha = 1e-5  # ???
uimp = -5

mesh = fd.mesh.rectangle_mesh(
    nx=41, ny=21, x_min=0, x_max=L, y_min=0, y_max=h, elm_type="quad4", name="Domain"
)

crd = mesh.nodes

mat = 1
if mat == 0:  # linear
    material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="ConstitutiveLaw")
elif mat == 1:
    # isotropic plasticity with power law hardening sigma = k*eps_p**m
    Re = 300
    k = 1000
    m = 0.25
    props = np.array([E, nu, alpha, Re, k, m])
    material = fd.constitutivelaw.Simcoon("EPICP", props, name="ConstitutiveLaw")

wf = fd.weakform.StressEquilibrium("ConstitutiveLaw", nlgeom=False)


# note set for boundary conditions
bottom_left = mesh.nearest_node([0, 0])
bottom_right = mesh.nearest_node([L, 0])

if typeBending == "3nodes":
    top_center = mesh.nearest_node([L / 2, h])
else:
    nodes_top1 = mesh.find_nodes(f"X=={L / 4} and Y=={h}")
    nodes_top2 = mesh.find_nodes(f"X=={3 * L / 4} and Y=={h}")
    top_center = np.hstack((nodes_top1, nodes_top2))

fd.Assembly.create(
    wf, "Domain", "quad4", name="Assembling", MeshChange=False
)  # uses MeshChange=True when the mesh change during the time

pb = fd.problem.NonLinear("Assembling")  # incremental non linear problems

# create a 'result' folder and set the desired ouputs
if not (os.path.isdir("results")):
    os.mkdir("results")

if mat == 0:
    res = pb.add_output(
        "results/bendingPlastic",
        "Assembling",
        ["Disp", "Stress", "Strain"],
        output_type="Node",
        file_format="vtk",
    )
elif mat == 1:
    res = pb.add_output(
        "results/bendingPlastic",
        "Assembling",
        ["Disp", "Stress", "Strain", "Statev", "Wm"],
        output_type="Node",
        file_format="vtk",
    )

################### step 1 ################################
# bending with non linear behavior
tmax = 1
pb.bc.add(
    "Dirichlet",
    bottom_left,
    "Disp",
    0,
)
pb.bc.add("Dirichlet", bottom_right, "DispY", 0)
pb.bc.add("Dirichlet", top_center, "DispY", uimp, name="disp")

# pb.nlsolve(dt = 0.05, tmax = 1, update_dt = False, tol_nr = 0.05, interval_output = 0.05)

# parameters
n_iter = 20
tol_nr = 0.05  # newton-raphson tolerance
max_subiter = 5  # newton-raphson of iterations per time step
err_num = 1e-8  # numerical error to avoid very small iterations
tmax = 1

pb.dtime = tmax / n_iter  # time step

pb.init_bc_start_value()  # set the initial displacement and forces

pb.time = 0  # time at the begining of the iteration

pb.initialize()  # compute the initial stress and elastic matrix
ii = 0  # iterations

# Newton raphson loop
while pb.time < tmax - err_num:
    pb.assembly.set_start(
        pb
    )  # set the tangent matrix to the elastic and make the global matrices assembly
    if ii != 0:
        pb.save_results(ii)  # Save results

    # Begin with an elastic prediction using the elastic tangent matrix

    # apply progressively the bounding condition using a time factor
    # as we solve an incremental problem, the load variation is considered
    t_fact_old = pb.time / tmax
    t_fact = (pb.time + pb.dtime) / (tmax)  # adimensional time

    pb.apply_boundary_conditions(t_fact, t_fact_old)

    # build and solve the linearized system with elastic rigidity matrix
    pb.set_A(pb.assembly.get_global_matrix())
    pb.set_D(pb.assembly.get_global_vector())

    # pb.updateD(start = True) #not modified in principle if dt is not modified, except the very first iteration. May be optimized by testing the change of dt
    pb.solve()

    pb._Xbc *= 0  # boundary conditions increment are now set to 0 because the current prediction already satisfy the bc

    # update displacement increment
    pb._dU += pb.get_X()

    err0 = np.max(np.abs(pb._U + pb._dU))
    convergence = False

    for subiter in range(max_subiter):  # newton-raphson iterations
        # Check convergence (displacement criterion. Force and work criteria are also possible)
        norm_res = np.max(np.abs(pb.get_X())) / err0

        # print('     Subiter {} - Time: {:.5f} - Err: {:.5f}'.format(subiter, pb.time+pb.dtime, norm_res))

        if norm_res < tol_nr:  # convergence of the NR algorithm
            convergence = True
            break

        # --------------- Solve --------------------------------------------------------
        pb.assembly.update(
            pb
        )  # update the initial stress and tangent matrix from the constitutive law and make the globel matrix and vector assemblies

        pb.set_A(pb.assembly.get_global_matrix())
        pb.set_D(pb.assembly.get_global_vector())
        pb.solve()
        pb._dU += pb.get_X()

    if not (convergence):
        raise NameError(
            "Newton Raphson iteration has not converged (err: {:.5f})- Reduce the time step.".format(
                norm_res
            )
        )

    pb.time = pb.time + pb.dtime  # update time value
    ii += 1

    pb.assembly.update(
        pb
    )  # update the stress and tangent matrix from the constitutive law and make the globel matrix and vector assemblies

    print(
        "Iter {} - Time: {:.5f} - NR iter: {} - Err: {:.5f}".format(
            ii, pb.time, subiter, norm_res
        )
    )


# pb.assembly.set_start(pb)
pb.save_results(ii)


################### step 2 ################################
# compute residual stresses
# assume only elastic return, so no need newton raphson here -> only linear problem

# change boundary conditions
pb.bc.remove("disp")
pb.bc.add("Neumann", top_center, "DispY", 0)  # no force applied = relaxation


pb.assembly.set_start(
    pb
)  # set the tangent matrix to the elastic and make the global matrices assembly

pb.apply_boundary_conditions(t_fact=1, t_fact_old=0)

# build and solve the linearized system with elastic rigidity matrix
pb.set_A(pb.assembly.get_global_matrix())
pb.set_D(pb.assembly.get_global_vector())

# pb.updateD(start = True) #not modified in principle if dt is not modified, except the very first iteration. May be optimized by testing the change of dt
pb.solve()

# update displacement
pb._U += pb.get_X()

pb.assembly.update(
    pb
)  # update the stress and tangent matrix from the constitutive law and make the globel matrix and vector assemblies

pb.save_results(ii + 1)


print(time() - start)

res.plot("Stress", "vm")
res.write_movie("test", "Stress", "vm")
