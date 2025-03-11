import fedoo as fd
import numpy as np
import os

# --------------- Pre-Treatment --------------------------------------------------------
dim = 3
meshperio = True
method = 0
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
    mesh = fd.mesh.rectangle_mesh(10, 10, elm_type="quad8", name="Domain2")

crd = mesh.nodes
elm = mesh.elements

bounds = mesh.bounding_box
right = mesh.find_nodes("X", bounds.xmax)
left = mesh.find_nodes("X", bounds.xmin)

umat_name = "ELISO"
props = np.array([[1e5, 0.3, 1]])
nstatev = 1

material = fd.constitutivelaw.ElasticIsotrop(1e5, 0.3)  # E, nu
wf = fd.weakform.StressEquilibrium(material, nlgeom=False)

# Assembly
assemb = fd.Assembly.create(wf, mesh, mesh.elm_type, name="Assembly")

# if '_perturbation' in fd.Problem.get_all():
#     del fd.Problem.get_all()['_perturbation']
L_eff = fd.homogen.get_homogenized_stiffness(assemb, meshperio)

# print(L_eff)
# # from simcoon import simmit as sim
# # print(sim.L_iso_props(L_eff)) #to check material properties

if dim == 3:
    import matplotlib.pyplot as plt

    from matplotlib import cm, colors

    plt.rcParams["text.usetex"] = True

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
