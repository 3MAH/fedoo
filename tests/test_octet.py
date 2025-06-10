from pathlib import Path

import numpy as np

import fedoo as fd


def test_octet():
    # --------------- Pre-Treatment --------------------------------------------------------

    fd.Assembly.delete_memory()

    # Define the Modeling Space - Here 3D problem
    fd.ModelingSpace("3D")

    # Import the mesh generated with Microgen
    octet_truss = Path(__file__).resolve().parent / "octet_truss.msh"
    fd.mesh.import_file(str(octet_truss), name="Domain")

    # Get the imported mesh
    mesh = fd.Mesh["Domain2"]

    # Get the bounding box (corners coordinates and center)
    bounds = mesh.bounding_box
    crd_center = bounds.center
    # Nearest node to the center of the bounding box for boundary conditions
    center = mesh.nearest_node(crd_center)

    # Add 2 virtual nodes for macro strain
    StrainNodes = fd.Mesh["Domain2"].add_virtual_nodes(crd_center, 2)

    # Material definition and simcoon elasto-plastic constitutive law
    Re = 300
    k = 1000
    m = 0.25
    alpha = 1e-5
    props = np.array([1e5, 0.3, alpha, Re, k, m])
    material = fd.constitutivelaw.Simcoon("EPICP", props, name="ConstitutiveLaw")

    # Create the weak formulation of the mechanical equilibrium equation
    wf = fd.weakform.StressEquilibrium("ConstitutiveLaw", name="WeakForm", nlgeom=False)

    # Assembly
    assemb = fd.Assembly.create("WeakForm", "Domain2", "tet4", name="Assembly")

    # Type of problem
    pb = fd.problem.NonLinear("Assembly")
    pb.set_nr_criterion(criterion="Work")

    # Set the desired ouputs at each time step
    # pb.add_output(
    #     "results",
    #     "Assembly",
    #     ["Disp", "Stress", "Strain", "Statev"],
    #     output_type="Node",
    #     file_format="vtk",
    # )

    # Boundary conditions for the linearized strain tensor
    E = [0, 0, 0, 0.1, 0, 0]  # [EXX, EYY, EZZ, EXY, EXZ, EYZ]

    list_strain_nodes = [
        StrainNodes[0],
        StrainNodes[0],
        StrainNodes[0],
        StrainNodes[1],
        StrainNodes[1],
        StrainNodes[1],
    ]
    list_strain_var = ["DispX", "DispY", "DispZ", "DispX", "DispY", "DispZ"]
    # or equivalent:
    # list_strain_nodes = [[StrainNodes[0], StrainNodes[1], StrainNodes[1]],
    #                      [StrainNodes[1], StrainNodes[0], StrainNodes[1]],
    #                      [StrainNodes[1], StrainNodes[1], StrainNodes[0]]]
    # list_strain_var = [['DispX', 'DispX', 'DispY'],
    #                    ['DispX', 'DispY', 'DispZ'],
    #                    ['DispY', 'DispZ', 'DispZ']]

    bc_periodic = fd.constraint.PeriodicBC(list_strain_nodes, list_strain_var, dim=3)
    pb.bc.add(bc_periodic)

    # fixed point on the center to avoid rigid body motion
    pb.bc.add("Dirichlet", center, "Disp", 0)

    # Enforced mean strain
    pb.bc.add(
        "Dirichlet", [StrainNodes[0]], "Disp", [E[0], E[1], E[2]]
    )  # EpsXX, EpsYY, EpsZZ
    pb.bc.add(
        "Dirichlet", [StrainNodes[1]], "Disp", [E[3], E[4], E[5]]
    )  # EpsXY, EpsXZ, EpsYZ

    # ---------------  Non linear solver---------------------------------------
    # pb.set_solver('CG') #conjugate gradient solver
    pb.nlsolve(dt=0.2, tmax=1, update_dt=False, tol_nr=0.1)

    # --------------- Post-Treatment ------------------------------------------
    # Get the stress and strain tensor (PG values)
    res = pb.get_results("Assembly", ["Strain", "Stress"], "GaussPoint")
    TensorStrain = res.gausspoint_data["Strain"]
    TensorStress = res.gausspoint_data["Stress"]

    assert np.abs(TensorStress[4][222] - 72.3765265291865) < 1e-3
    assert np.abs(TensorStrain[2][876] - 0.03046909551762696) < 1e-6
