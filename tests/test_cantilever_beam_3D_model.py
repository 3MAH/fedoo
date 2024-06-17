import time

import numpy as np

import fedoo as fd


def test_cantilever_beam_3D_model():
    # --------------- Pre-Treatment --------------------------------------------------------

    fd.ModelingSpace("3D")

    # Units: N, mm, MPa
    # Mesh.box_mesh(Nx=101, Ny=21, Nz=21, x_min=0, x_max=1000, y_min=0, y_max=100, z_min=0, z_max=100, ElementShape = 'hex8', name = 'Domain')
    mesh = fd.mesh.box_mesh(
        nx=11,
        ny=5,
        nz=5,
        x_min=0,
        x_max=1000,
        y_min=0,
        y_max=100,
        z_min=0,
        z_max=100,
        elm_type="hex8",
        name="Domain",
    )

    # Material definition
    fd.constitutivelaw.ElasticIsotrop(200e3, 0.3, name="ElasticLaw")
    fd.weakform.StressEquilibrium("ElasticLaw", name="weakform")

    # Assembly (print the time required for assembling)
    fd.Assembly.create("weakform", "Domain", "hex8", name="Assembling")

    # Type of problem
    pb = fd.problem.Linear("Assembling")

    # Boundary conditions
    nodes_left = mesh.node_sets["left"]
    nodes_right = mesh.node_sets["right"]
    nodes_top = mesh.node_sets["top"]
    nodes_bottom = mesh.node_sets["bottom"]

    pb.bc.add("Dirichlet", nodes_left, "DispX", 0)
    pb.bc.add("Dirichlet", nodes_left, "DispY", 0)
    pb.bc.add("Dirichlet", nodes_left, "DispZ", 0)

    pb.bc.add("Dirichlet", nodes_right, "DispY", -10)

    pb.apply_boundary_conditions()

    # --------------- Solve --------------------------------------------------------
    t0 = time.time()
    # Problem.set_solver('cg') #uncomment for conjugate gradient solver
    print("Solving...")
    pb.solve()
    print("Done in " + str(time.time() - t0) + " seconds")

    # --------------- Post-Treatment -----------------------------------------------
    # Get the displacement vector on nodes for export to vtk
    U = np.reshape(pb.get_dof_solution("all"), (3, -1)).T

    # Get the stress tensor (nodal values) - Old way to get strain and stress -> prefer the use of get_results or direct access to assembly.sv field
    TensorStrain = fd.Assembly["Assembling"].get_strain(
        pb.get_dof_solution(), "Node", nlgeom=False
    )
    TensorStress = fd.ConstitutiveLaw["ElasticLaw"].get_stress_from_strain(
        fd.Assembly["Assembling"], TensorStrain
    )

    assert np.abs(TensorStress[5][-1] + 0.9007983467254552) < 1e-10
