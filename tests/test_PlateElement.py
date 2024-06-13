#
# Plate element to model the canteleaver beam using different kind of plate elements
#

import numpy as np

import fedoo as fd

# from matplotlib import pylab as plt
# import os


def test_plate_element():
    fd.Assembly.delete_memory()

    fd.ModelingSpace("3D")
    E = 1e5
    nu = 0.3

    L = 100
    h = 20
    thickness = 1
    F = -10

    geomElementType = "quad4"  # choose among 'tri3', 'tri6', 'quad4', 'quad9'
    plateElementType = (
        "p" + geomElementType
    )  # plate interpolation. Same as geom interpolation in local element coordinate (change of basis)

    material = fd.constitutivelaw.ElasticIsotrop(E, nu, name="Material")
    fd.constitutivelaw.ShellHomogeneous("Material", thickness, k=1, name="PlateSection")

    mesh = fd.mesh.rectangle_mesh(
        51, 11, 0, L, -h / 2, h / 2, geomElementType, ndim=3, name="plate"
    )

    nodes_left = mesh.node_sets["left"]
    nodes_right = mesh.node_sets["right"]

    node_right_center = nodes_right[(mesh.nodes[nodes_right, 1] ** 2).argmin()]

    fd.weakform.PlateEquilibrium("PlateSection", name="WFplate")
    fd.Assembly.create("WFplate", "plate", plateElementType, name="plate")

    # or by hand
    # reduced_integration = True #if true, use reduce integration for shear
    # if reduced_integration == False:
    #     fd.weakform.PlateEquilibriumFI("PlateSection", name = "WFplate") #by default k=0 i.e. no shear effect
    #     fd.Assembly.create("WFplate", "plate", plateElementType, name="plate")
    #     post_tt_assembly = 'plate'
    # else:
    #     fd.weakform.PlateShearEquilibrium("PlateSection", name = "WFplate_RI") #by default k=0 i.e. no shear effect
    #     fd.Assembly.create("WFplate_RI", "plate", plateElementType, name="plate_RI", n_elm_gp = 1)

    #     fd.weakform.PlateKirchhoffLoveEquilibrium("PlateSection", name = "WFplate_FI") #by default k=0 i.e. no shear effect
    #     fd.Assembly.create("WFplate_FI", "plate", plateElementType, name="plate_FI")

    #     fd.Assembly.sum("plate_RI", "plate_FI", name = "plate")
    #     post_tt_assembly = 'plate_FI'

    pb = fd.problem.Linear("plate")

    pb.bc.add("Dirichlet", nodes_left, "DispX", 0)
    pb.bc.add("Dirichlet", nodes_left, "DispY", 0)
    pb.bc.add("Dirichlet", nodes_left, "DispZ", 0)
    pb.bc.add("Dirichlet", nodes_left, "RotX", 0)
    pb.bc.add("Dirichlet", nodes_left, "RotY", 0)
    pb.bc.add("Dirichlet", nodes_left, "RotZ", 0)

    pb.bc.add("Neumann", node_right_center, "DispZ", F)

    pb.apply_boundary_conditions()
    pb.solve()

    # I = h*thickness**3/12
    # # print('Beam analitical deflection: ', F*L**3/(3*E*I))
    # # print('Numerical deflection: ', pb.get_disp('DispZ')[node_right_center])

    assert np.abs(pb.get_disp("DispZ")[node_right_center] + 19.62990873) < 1e-7

    # z, StressDistribution = fd.ConstitutiveLaw['PlateSection'].GetStressDistribution(fd.Assembly['plate'],20)
    # plt.plot(StressDistribution[0], z)
