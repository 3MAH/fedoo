import os
from pathlib import Path

import numpy as np

import fedoo as fd


def test_thermal3D():
    # --------------- Pre-Treatment --------------------------------------------------------

    fd.ModelingSpace("3D")

    meshname = "Domain"
    nb_iter = 3

    # Mesh.box_mesh(Nx=3, Ny=3, Nz=3, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1, ElementShape = 'hex8', name = meshname)
    # Mesh.import_file('octet_surf.msh', meshname = "Domain")
    # Mesh.import_file('data/octet_1.msh', meshname = "Domain")
    # gyroid = Path(__file__).resolve().parent / "../util/meshes/gyroid.msh"
    gyroid = Path(__file__).resolve().parent / "gyroid.msh"
    fd.mesh.import_file(str(gyroid), name="Domain")

    mesh = fd.Mesh[meshname]

    crd = mesh.nodes

    K = 500  # K = 18 #W/K/m
    c = 0.500  # J/kg/K
    rho = 7800  # kg/m2
    material = fd.constitutivelaw.ThermalProperties(K, c, rho, name="ThermalLaw")
    wf = fd.weakform.HeatEquation("ThermalLaw")
    assemb = fd.Assembly.create("ThermalLaw", meshname, name="Assembling")

    # note set for boundary conditions
    Xmin, Xmax = mesh.bounding_box
    bottom = mesh.find_nodes("Z", Xmin[2])
    top = mesh.find_nodes("Z", Xmax[2])
    left = mesh.find_nodes("X", Xmin[2])
    right = mesh.find_nodes("X", Xmax[2])

    pb = fd.problem.NonLinear("Assembling")
    pb.set_nr_criterion(norm_type=np.inf, tol=5e-3)
    # Problem.set_solver('cg', precond = True)

    pb.set_nr_criterion("Displacement", tol=5e-2, max_subiter=5, err0=100)
    if not (os.path.isdir("results")):
        os.mkdir("results")
    results = pb.add_output(
        "results/thermal3D",
        "Assembling",
        ["Temp"],
        output_type="Node",
        file_format="npz",
    )
    # Problem.add_output('results/bendingPlastic', 'Assembling', ['cauchy', 'PKII', 'strain', 'cauchy_vm', 'statev'], output_type='Element', file_format ='vtk')

    tmax = 10

    # Problem.bc.add('Dirichlet','Temp',0,bottom)
    def time_func(t_fact):
        if t_fact == 0:
            return 0
        else:
            return 1

    # Problem.bc.add('Dirichlet','Temp',100,left, timeEvolution=timeEvolution)
    pb.bc.add("Dirichlet", right, "Temp", 3, time_func=time_func)
    # Problem.bc.add('Dirichlet','Temp',100,top, timeEvolution=timeEvolution)

    # Problem.bc.add('Dirichlet','DispY', 0,nodes_bottomLeft)
    # Problem.bc.add('Dirichlet','DispY',0,nodes_bottomRight)
    # bc = Problem.bc.add('Dirichlet','DispY', uimp, nodes_topCenter)

    pb.nlsolve(dt=tmax / nb_iter, tmax=tmax, update_dt=True)

    results.load()
    assert np.abs(results.node_data["Temp"][8712] - 2.610859332847924) < 1e-8
