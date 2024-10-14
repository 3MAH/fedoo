# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 16:25:38 2022

@author: Etienne
"""

import fedoo as fd
from simcoon import simmit as sim
import numpy as np
import os

# --------------- Pre-Treatment --------------------------------------------------------
dim = 3
meshperio = True
method = 2
dir_meshes = "../../util/meshes/"

if dim == 2:
    fd.ModelingSpace("2Dplane")
else:
    fd.ModelingSpace("3D")

if dim == 3:
    # Mesh.import_file('./meshes/octet_surf.msh', meshname = "Domain")
    # Mesh.import_file('./meshes/gyroid.msh', meshname = "Domain2") meshperio = False
    # Mesh.import_file('./meshes/MeshPeriodic2_quad.msh', meshname = "Domain")
    # mesh = fd.mesh.box_mesh(10,10,10, elm_type = 'hex8', name = "Domain")

    mesh = fd.Mesh.read(dir_meshes + "gyroid_per.vtk", name="Domain")
else:
    mesh = fd.Mesh.rectangle_mesh(10, 10, ElementShape="quad8", name="Domain2")

crd = mesh.nodes
elm = mesh.elements

bounds = mesh.bounding_box
right = mesh.find_nodes("X", bounds.xmax)
left = mesh.find_nodes("X", bounds.xmin)

umat_name = "ELISO"
props = np.array([[1e5, 0.3, 1]])
nstatev = 1

L = sim.L_iso(np.array([1e5, 0.3]), "Enu")
props_test = sim.L_iso_props(L)
print("props", props_test)

if method == 0:
    material = fd.constitutivelaw.ElasticAnisotropic(L, name="ElasticLaw")
    wf = fd.weakform.StressEquilibrium(material, name="WeakForm", nlgeom=False)

    # Assembly
    assemb = fd.Assembly.create("WeakForm", mesh, mesh.elm_type, name="Assembly")

    # if '_perturbation' in fd.Problem.get_all():
    #     del fd.Problem.get_all()['_perturbation']
    L_eff = fd.homogen.get_homogenized_stiffness(assemb, meshperio)
else:
    type_el = mesh.elm_type
    # type_el = 'hex20'
    xmax = np.max(crd[:, 0])
    xmin = np.min(crd[:, 0])
    ymax = np.max(crd[:, 1])
    ymin = np.min(crd[:, 1])
    if dim == 3:
        zmax = np.max(crd[:, 2])
        zmin = np.min(crd[:, 2])
        crd_center = (np.array([xmin, ymin, zmin]) + np.array([xmax, ymax, zmax])) / 2
    else:
        crd_center = (np.array([xmin, ymin]) + np.array([xmax, ymax])) / 2

    center = [np.linalg.norm(crd - crd_center, axis=1).argmin()]

    StrainNodes = mesh.add_nodes(crd_center, 2)  # add virtual nodes for macro strain

    BC_perturb = np.eye(6)
    # BC_perturb[3:6,3:6] *= 2 #2xEXY

    DStrain = []
    DStress = []

    material = fd.constitutivelaw.ElasticAnisotropic(L, name="ElasticLaw")

    # Assembly
    wf = fd.weakform.StressEquilibrium("ElasticLaw")
    assemb = fd.Assembly.create(wf, mesh, type_el, name="Assembling")

    # Type of problem
    pb = fd.problem.Linear("Assembling")

    # Shall add other conditions later on
    bc_periodic = fd.constraint.PeriodicBC(
        [
            StrainNodes[0],
            StrainNodes[0],
            StrainNodes[0],
            StrainNodes[1],
            StrainNodes[1],
            StrainNodes[1],
        ],
        ["DispX", "DispY", "DispZ", "DispX", "DispY", "DispZ"],
        dim=dim,
        meshperio=meshperio,
    )
    pb.bc.add(bc_periodic)

    # if dim == 3:
    #     if meshperio:
    #         Homogen.DefinePeriodicBoundaryCondition(mesh,
    #         [StrainNodes[0], StrainNodes[0], StrainNodes[0], StrainNodes[1], StrainNodes[1], StrainNodes[1]],
    #         ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D', Problemname = Problemname)
    #     else:
    #         Homogen.DefinePeriodicBoundaryConditionNonPerioMesh(mesh,
    #         [StrainNodes[0], StrainNodes[0], StrainNodes[0], StrainNodes[1], StrainNodes[1], StrainNodes[1]],
    #         ['DispX',        'DispY',        'DispZ',       'DispX',         'DispY',        'DispZ'], dim='3D', Problemname = Problemname)
    # elif dim == 2:
    #     if meshperio:
    #         Homogen.DefinePeriodicBoundaryCondition(mesh,
    #         [StrainNodes[0], StrainNodes[0], StrainNodes[1]],
    #         ['DispX',        'DispY',        'DispX'], dim='2D', Problemname = Problemname)
    #     else:
    #         assert 0, 'NotImplemented'

    # create a 'result' folder and set the desired ouputs
    if not (os.path.isdir("results")):
        os.mkdir("results")
    pb.add_output("results/test", "Assembling", ["Disp", "Stress", "Strain"])
    # pb.add_output('results/test', 'Assembling', ['Disp', 'Stress', 'Strain'], output_type='Node', file_format ='vtk')
    # pb.add_output('results/test', 'Assembling', ['Stress', 'Strain'], output_type='Element', file_format ='vtk')

    pb.bc.add("Dirichlet", center, "Disp", 0, name="center")

    # typeBC = 'Dirichlet'
    typeBC = "Neumann"
    for i in range(6):
        pb.bc.remove("_Strain")
        pb.bc.add(
            typeBC,
            [StrainNodes[0]],
            "DispX",
            BC_perturb[i][0],
            start_value=0,
            name="_Strain",
        )  # EpsXX
        pb.bc.add(
            typeBC,
            [StrainNodes[0]],
            "DispY",
            BC_perturb[i][1],
            start_value=0,
            name="_Strain",
        )  # EpsYY
        pb.bc.add(
            typeBC,
            [StrainNodes[1]],
            "DispX",
            BC_perturb[i][3],
            start_value=0,
            name="_Strain",
        )  # EpsXY

        if dim == 3:
            pb.bc.add(
                typeBC,
                [StrainNodes[0]],
                "DispZ",
                BC_perturb[i][2],
                start_value=0,
                name="_Strain",
            )  # EpsZZ
            pb.bc.add(
                typeBC,
                [StrainNodes[1]],
                "DispY",
                BC_perturb[i][4],
                start_value=0,
                name="_Strain",
            )  # EpsXZ
            pb.bc.add(
                typeBC,
                [StrainNodes[1]],
                "DispZ",
                BC_perturb[i][5],
                start_value=0,
                name="_Strain",
            )  # EpsYZ
        else:
            pb.bc.add("Dirichlet", StrainNodes[1], "DispY", 0, name="_Strain")

        pb.apply_boundary_conditions()

        pb.solve()
        pb.save_results(i)

        X = pb.get_X()
        if dim == 3:
            DStrain.append(
                np.array(
                    [
                        pb._get_vect_component(X, "DispX")[StrainNodes[0]],
                        pb._get_vect_component(X, "DispY")[StrainNodes[0]],
                        pb._get_vect_component(X, "DispZ")[StrainNodes[0]],
                        pb._get_vect_component(X, "DispX")[StrainNodes[1]],
                        pb._get_vect_component(X, "DispY")[StrainNodes[1]],
                        pb._get_vect_component(X, "DispZ")[StrainNodes[1]],
                    ]
                )
            )

            Fext = pb.get_ext_forces()
            Fext = Fext.reshape(3, -1)
            stress = [
                Fext[0, -2],
                Fext[1, -2],
                Fext[2, -2],
                Fext[0, -1],
                Fext[1, -1],
                Fext[2, -1],
            ]

            DStress.append(stress)

            Fext2 = pb.get_ext_forces(include_mpc=False)
            Fext2 = Fext2.reshape(3, -1)

        elif dim == 2:
            DStrain.append(
                np.array(
                    [
                        pb._get_vect_component(X, "DispX")[StrainNodes[0]],
                        pb._get_vect_component(X, "DispY")[StrainNodes[0]],
                        0,
                        pb._get_vect_component(X, "DispX")[StrainNodes[1]],
                        0,
                        0,
                    ]
                )
            )

            Fext = pb.get_ext_forces()
            Fext = Fext.reshape(2, -1)
            stress = [Fext[0, -2], Fext[1, -2], 0, Fext[0, -1], 0, 0]

            DStress.append(stress)

    if typeBC == "Neumann":
        L_eff = np.linalg.inv(np.array(DStrain).T)
    else:
        L_eff = np.array(DStress).T


# np.set_printoptions(precision=3, suppress=True)
print("L_eff = ", L_eff)

props_test_eff = sim.L_iso_props(L_eff)
print("props", props_test_eff)


import matplotlib.pyplot as plt

from matplotlib import cm, colors

plt.rcParams["text.usetex"] = False

plt.rcParams["figure.figsize"] = (20, 8)


phi = np.linspace(0, 2 * np.pi, 128)  # the angle of the projection in the xy-plane

theta = np.linspace(0, np.pi, 128).reshape(
    128, 1
)  # the angle from the polar axis, ie the polar angle


n_1 = np.sin(theta) * np.cos(phi)

n_2 = np.sin(theta) * np.sin(phi)

n_3 = np.cos(theta) * np.ones(128)


n = (
    np.array([n_1 * n_1, n_2 * n_2, n_3 * n_3, n_1 * n_2, n_1 * n_3, n_2 * n_3])
    .transpose(1, 2, 0)
    .reshape(128, 128, 1, 6)
)


M = np.linalg.inv(L_eff)


S = (n @ M @ n.reshape(128, 128, 6, 1)).reshape(128, 128)


E = 1.0 / S

x = E * n_1

y = E * n_2

z = E * n_3


# E = E/E.max()


fig = plt.figure(figsize=plt.figaspect(1))  # Square figure

ax = fig.add_subplot(111, projection="3d")


# make the panes transparent

ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# make the grid lines transparent

ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)

ax.set_axis_off()


# ax.plot_surface(x, y, z, cmap='hot',c=E)


# norm = colors.Normalize(vmin = 0., vmax = 10000, clip = False)

Emin = np.min(E)

Eavg = np.average(E)

Emax = np.max(E)

norm = colors.Normalize(vmin=Emin, vmax=Emax, clip=False)

surf = ax.plot_surface(
    x,
    y,
    z,
    rstride=1,
    cstride=1,
    norm=norm,
    facecolors=cm.cividis(norm(E)),
    linewidth=0,
    antialiased=False,
    shade=False,
)


# ax.set_xlim(0,20000)

# ax.set_ylim(0,20000)

# ax.set_zlim(0,20000)

# ax.set_xlabel(r'$E_x$ (MPa)')

# ax.set_ylabel(r'$E_y$ (MPa)')

# ax.set_zlabel(r'$E_z$ (MPa)')


scalarmap = cm.ScalarMappable(cmap=plt.cm.cividis, norm=norm)

scalarmap.set_clim(np.min(E), np.max(E))

# m.set_array([])

cbar = plt.colorbar(
    scalarmap,
    orientation="horizontal",
    fraction=0.06,
    pad=-0.1,
    ticks=[Emin, Eavg, Emax],
    ax=ax,
)

cbar.ax.tick_params(labelsize="large")

cbar.set_label(r"directional stiffness $E$ (MPa)", size=15, labelpad=20)


# ax.figure.axes[0].tick_params(axis="both", labelsize=5)

ax.figure.axes[1].tick_params(axis="x", labelsize=20)


ax.azim = 30

ax.elev = 30


# Volume_mesh = Assembly.get_all()['Assembling'].integrate_field(np.ones_like(TensorStress[0]))


plt.savefig("directional.png", transparent=True)

plt.show()
