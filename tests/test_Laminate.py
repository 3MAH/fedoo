#
# Plate element to model the canteleaver beam using different kind of plate elements
#

import numpy as np

import fedoo as fd

# from matplotlib import pylab as plt
# import os


def test_laminate():
    fd.Assembly.delete_memory()

    fd.ModelingSpace("3D")
    E = 1e5
    nu = 0.3

    L = 51
    h = 11
    thickness = 1
    F = -100

    geomElementType = "quad4"  # choose among 'tri3', 'tri6', 'quad4', 'quad9'
    plateElementType = (
        "p" + geomElementType
    )  # plate interpolation. Same as geom interpolation in local element coordinate (change of basis)

    mat1 = fd.constitutivelaw.ElasticIsotrop(E, nu, name="Mat1")
    mat2 = fd.constitutivelaw.ElasticIsotrop(E / 10, nu, name="Mat2")

    # ConstitutiveLaw.ShellHomogeneous('Material', thickness, name = 'PlateSection')
    fd.constitutivelaw.ShellLaminate(
        ["Mat1", "Mat2", "Mat1"], [0.2, 1, 0.2], name="PlateSection"
    )

    mesh = fd.mesh.rectangle_mesh(
        21, 5, 0, L, -h / 2, h / 2, geomElementType, ndim=3, name="plate"
    )

    nodes_left = mesh.node_sets["left"]
    nodes_right = mesh.node_sets["right"]

    node_right_center = nodes_right[(mesh.nodes[nodes_right, 1] ** 2).argmin()]

    # reduced_integration = True #if true, use reduce integration for shear
    # if reduced_integration == False:
    #     fd.weakform.PlateFI("PlateSection", name = "WFplate") #by default k=0 i.e. no shear effect
    #     fd.Assembly.create("WFplate", "plate", plateElementType, name="plate")
    #     post_tt_assembly = 'plate'
    # else:
    #     fd.weakform.PlateShear("PlateSection", name = "WFplate_RI") #by default k=0 i.e. no shear effect
    #     fd.Assembly.create("WFplate_RI", "plate", plateElementType, name="plate_RI", n_elm_gp = 1)

    #     fd.weakform.PlateKirchhoffLove("PlateSection", name = "WFplate_FI") #by default k=0 i.e. no shear effect
    #     fd.Assembly.create("WFplate_FI", "plate", plateElementType, name="plate_FI")

    #     fd.Assembly.sum("plate_RI", "plate_FI", name = "plate")
    #     post_tt_assembly = 'plate_FI'

    fd.weakform.PlateEquilibrium(
        "PlateSection", name="WFplate"
    )  # by default k=0 i.e. no shear effect
    fd.Assembly.create("WFplate", "plate", plateElementType, name="plate")

    pb = fd.problem.Linear("plate")

    # create a 'result' folder and set the desired ouputs
    # if not(os.path.isdir('results')): os.mkdir('results')
    # Problem.add_output('results/simplePlate', post_tt_assembly, ['disp','rot', 'stress', 'strain'], output_type='Node', file_format ='vtk', position = -1)

    pb.bc.add("Dirichlet", nodes_left, "DispX", 0)
    pb.bc.add("Dirichlet", nodes_left, "DispY", 0)
    pb.bc.add("Dirichlet", nodes_left, "DispZ", 0)
    pb.bc.add("Dirichlet", nodes_left, "RotX", 0)
    pb.bc.add("Dirichlet", nodes_left, "RotY", 0)
    pb.bc.add("Dirichlet", nodes_left, "RotZ", 0)

    pb.bc.add("Neumann", node_right_center, "DispZ", F)

    pb.apply_boundary_conditions()
    pb.solve()

    assert np.abs(pb.get_disp("DispZ")[node_right_center] + 25.7688952234) < 1e-7

    # plot the stress distribution
    # z, StressDistribution = fd.ConstitutiveLaw['PlateSection'].GetStressDistribution(fd.Assembly['plate'],200)
    # plt.plot(StressDistribution[0], z)
