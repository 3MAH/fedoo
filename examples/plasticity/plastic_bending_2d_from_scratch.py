"""
2D elastoplastic three-point bending written from low-level operators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Same plane strain bending test as the 2D elastoplastic three-point bending
example (Simcoon ``EPICP`` law, 5 mm central displacement, then release), but
in small strain and without :py:meth:`fedoo.problem.NonLinear.nlsolve`: the
time loop and the Newton-Raphson iterations are written by hand with the
low-level methods of the problem and of the assembly (``set_start``,
``update``, ``apply_boundary_conditions``, ``set_A``, ``set_D``, ``solve``).
It shows what ``nlsolve`` does internally.
"""

import os

import fedoo as fd
import numpy as np

###############################################################################
# Geometry, mesh and material
# ---------------------------
# Units: N, mm, MPa. Set ``mat = 0`` for a linear elastic material.

fd.ModelingSpace("2Dplane")

typeBending = "3nodes"  # '3nodes' or '4nodes'
h = 2
L = 16
E = 200e3
nu = 0.3
alpha = 1e-5  # thermal expansion coefficient (required by EPICP, isothermal here)
uimp = -5

mesh = fd.mesh.rectangle_mesh(
    nx=41, ny=21, x_min=0, x_max=L, y_min=0, y_max=h, elm_type="quad4", name="Domain"
)

mat = 1
if mat == 0:  # linear
    material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="ConstitutiveLaw")
elif mat == 1:
    # J2 plasticity with isotropic power-law hardening:
    # yield stress Re + k * p**m, p = accumulated plastic strain
    Re = 300
    k = 1000
    m = 0.25
    props = np.array([E, nu, alpha, Re, k, m])
    material = fd.constitutivelaw.Simcoon("EPICP", props, name="ConstitutiveLaw")

wf = fd.weakform.StressEquilibrium("ConstitutiveLaw", nlgeom=False)

###############################################################################
# Boundary node sets, assembly and problem
# ----------------------------------------

bottom_left = mesh.nearest_node([0, 0])
bottom_right = mesh.nearest_node([L, 0])

if typeBending == "3nodes":
    top_center = mesh.nearest_node([L / 2, h])
else:
    nodes_top1 = mesh.find_nodes(f"X=={L / 4} and Y=={h}")
    nodes_top2 = mesh.find_nodes(f"X=={3 * L / 4} and Y=={h}")
    top_center = np.hstack((nodes_top1, nodes_top2))

fd.Assembly.create(
    wf, "Domain", "quad4", name="Assembling", mesh_change=False
)  # use mesh_change=True when the mesh changes during the time

pb = fd.problem.NonLinear("Assembling")  # incremental nonlinear problem

# create a 'results' folder and set the desired outputs
if not (os.path.isdir("results")):
    os.mkdir("results")

if mat == 0:
    res = pb.add_output(
        "results/bendingPlastic_from_scratch",
        "Assembling",
        ["Disp", "Stress", "Strain"],
        output_type="Node",
        file_format="vtk",
    )
elif mat == 1:
    res = pb.add_output(
        "results/bendingPlastic_from_scratch",
        "Assembling",
        ["Disp", "Stress", "Strain", "Statev", "Wm"],
        output_type="Node",
        file_format="vtk",
    )

###############################################################################
# Step 1: bending with a hand-written Newton-Raphson loop
# -------------------------------------------------------
# Each increment starts with an elastic prediction that applies the increment
# of the boundary conditions, followed by Newton-Raphson corrections with the
# tangent matrix of the constitutive law. Convergence is checked on the
# displacement correction.

pb.bc.add(
    "Dirichlet",
    bottom_left,
    "Disp",
    0,
)
pb.bc.add("Dirichlet", bottom_right, "DispY", 0)
pb.bc.add("Dirichlet", top_center, "DispY", uimp, name="disp")

# parameters
n_iter = 20
tol_nr = 0.05  # newton-raphson tolerance
max_subiter = 5  # maximum number of newton-raphson iterations per time step
err_num = 1e-8  # numerical error to avoid very small iterations
tmax = 1

pb.dtime = tmax / n_iter  # time step

pb.init_bc_start_value()  # set the initial displacement and forces

pb.time = 0  # time at the beginning of the iteration

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

    # apply progressively the boundary conditions using a time factor
    # as we solve an incremental problem, the load variation is considered
    t_fact_old = pb.time / tmax
    t_fact = (pb.time + pb.dtime) / (tmax)  # adimensional time

    pb.apply_boundary_conditions(t_fact, t_fact_old)

    # build and solve the linearized system with elastic rigidity matrix
    pb.set_A(pb.assembly.get_global_matrix())
    pb.set_D(pb.assembly.get_global_vector())

    pb.solve()

    pb._Xbc *= 0  # boundary conditions increment are now set to 0 because the current prediction already satisfy the bc

    # update displacement increment
    pb._dU += pb.get_X()

    err0 = np.max(np.abs(pb._U + pb._dU))
    convergence = False

    for subiter in range(max_subiter):  # newton-raphson iterations
        # Check convergence (displacement criterion. Force and work criteria are also possible)
        norm_res = np.max(np.abs(pb.get_X())) / err0

        if norm_res < tol_nr:  # convergence of the NR algorithm
            convergence = True
            break

        # update the stress and tangent matrix from the constitutive law and
        # make the global matrix and vector assemblies
        pb.assembly.update(pb)

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
    )  # update the stress and tangent matrix from the constitutive law and make the global matrix and vector assemblies

    print(
        "Iter {} - Time: {:.5f} - NR iter: {} - Err: {:.5f}".format(
            ii, pb.time, subiter, norm_res
        )
    )


pb.save_results(ii)

###############################################################################
# Step 2: elastic release and residual stresses
# ---------------------------------------------
# The prescribed displacement is removed. The unloading is assumed to be
# elastic, so a single linear solve with the elastic matrix is enough.

pb.bc.remove("disp")

pb.assembly.set_start(
    pb
)  # set the tangent matrix to the elastic and make the global matrices assembly

pb.apply_boundary_conditions(t_fact=1, t_fact_old=0)

# build and solve the linearized system with elastic rigidity matrix
pb.set_A(pb.assembly.get_global_matrix())
pb.set_D(pb.assembly.get_global_vector())

pb.solve()

# update displacement
pb._U += pb.get_X()

pb.assembly.update(
    pb
)  # update the stress and tangent matrix from the constitutive law and make the global matrix and vector assemblies

pb.save_results(ii + 1)

# residual stresses after the elastic release
res.plot("Stress", "vm")
