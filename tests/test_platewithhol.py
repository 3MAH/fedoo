import numpy as np

import fedoo as fd


def test_plate_with_hole():
    # Define the Modeling Space - Here 2D problem with plane stress assumption.
    fd.ModelingSpace("2Dstress")

    # Generate a simple structured mesh "Domain" (plate with a hole).
    mesh = fd.mesh.hole_plate_mesh(
        nr=11, nt=11, length=100, height=100, radius=20, elm_type="quad4", name="Domain"
    )

    # Define an elastic isotropic material with E = 2e5MPa et nu = 0.3 (steel)
    fd.constitutivelaw.ElasticIsotrop(2e5, 0.3, name="ElasticLaw")

    # Create the weak formulation of the mechanical equilibrium equation
    fd.weakform.StressEquilibrium("ElasticLaw", name="WeakForm")

    # Create a global assembly
    fd.Assembly.create("WeakForm", "Domain", name="Assembly", MeshChange=True)

    # Define a new static problem
    pb = fd.problem.Linear("Assembly")

    # Definition of the set of nodes for boundary conditions
    left = mesh.find_nodes("X", mesh.bounding_box.xmin)
    right = mesh.find_nodes("X", mesh.bounding_box.xmax)
    bottom = mesh.find_nodes("Y", mesh.bounding_box.ymin)

    # Boundary conditions
    # symetry condition on left (ux = 0)
    pb.bc.add("Dirichlet", left, "DispX", 0)
    # symetry condition on bottom edge (ux = 0)
    pb.bc.add("Dirichlet", bottom, "DispY", 0)
    # displacement on right (ux=0.1mm)
    pb.bc.add("Dirichlet", right, "DispX", 0.1)

    pb.apply_boundary_conditions()

    # Solve problem
    pb.solve()

    # ---------- Post-Treatment ----------
    # Get the stress tensor, strain tensor, and displacement (nodal values)
    res = pb.get_results("Assembly", ["Stress_vm", "Strain"], "Node")

    U = pb.get_disp()

    assert U[0, 40] == 0.1
    assert np.abs(U[1, 40] + 0.01962855744173) < 1e-10
    assert np.abs(res.node_data["Stress_vm"][282] - 175.50126302014) < 1e-10

    # res.plot('Stress_vm')
    # fd.util.field_plot_2d("Assembly", disp = pb.get_disp(), dataname = 'stress', component='vm', data_min=None, data_max = None, scale_factor = 6, plot_edge = True, nb_level = 10, type_plot = "smooth")
    # fd.util.field_plot_2d("Assembly", disp = pb.get_disp(), dataname = 'disp', component=0, scale_factor = 6, plot_edge = True, nb_level = 6, type_plot = "smooth")
