"""
Compression of a tube using 2D axisymmetric model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This model uses self-contact, elasto-plastic material law with finite strain
assumption in a 2D axisymetric modeling space.
The full 3D result is ploted during the post processing phase.
"""

import fedoo as fd
import numpy as np
import pyvista as pv
import os

###############################################################################
# The tube in the 2D axisymmetric space is modeled by a rectangle.

fd.ModelingSpace("2Daxi")  # 2D axisymmetric space
mesh = fd.mesh.rectangle_mesh(5, 240, 23, 25, 0, 180)  # tube geometry

###############################################################################
# The elasto-plastic constitutive law "EPICP" from the Simcoon library is used.
# This law assume an isotropic hardening modeled with a power-law:
#
# .. math::
#    \sigma = \sigma_y + k p^m
#
# where
#   - :math:`\sigma` is the equivalent stress defining the yield surface,
#   - :math:`p` is the equivalent plastic strain,
#   - :math:`\sigma_y` is the initial yield stress,
#   - :math:`k` is the strain hardening constant,
#   - :math:`m` is the strain hardening exponent.

sigma_y = 300  # Yield stress
k = 1000
m = 0.3
E = 200e3  # Elasticity modulus (for steel)
nu = 0.3  # Poisson ratio
props = np.array([E, nu, 1e-5, sigma_y, k, m])
material = fd.constitutivelaw.Simcoon("EPICP", props)

###############################################################################
# We build two assemblies for:
#   - the mechanical static equilibrium
#   - the self contact

wf = fd.weakform.StressEquilibrium(material)
assembly = fd.Assembly.create(wf, mesh)

# Add self contact....
surf = fd.mesh.extract_surface(mesh)
contact = fd.constraint.contact.SelfContact(surf)

# contact parameters
contact.contact_search_once = True
contact.eps_n = 1e6  # contact penalty
contact.max_dist = 1.5  # max distance for the contact search

###############################################################################
# We define a non linear problem including geometrical non linearities with the
# updated lagrangian method (NLGEOM = 'UL') which is the default method in
# fedoo (equivalent to NLGEOM = True).
# Some parameters of the newton-raphson algorithm are changed and the output
# files to save at each iteration are defined.

NLGEOM = "UL"
pb = fd.problem.NonLinear(assembly + contact, nlgeom=NLGEOM)
pb.set_nr_criterion("Displacement", err0=None, tol=1e-2, max_subiter=5)

# create a 'result' folder and set the desired ouputs
if not (os.path.isdir("results")):
    os.mkdir("results")
res = pb.add_output(
    "results/tube_compressoin", assembly, ["Disp", "Stress", "Strain", "P"]
)


# Node sets for boundary conditions
bottom = mesh.node_sets["bottom"]
top = mesh.node_sets["top"]

pb.bc.add("Dirichlet", bottom, "Disp", 0)
pb.bc.add("Dirichlet", top, "Disp", [0, -150])
pb.nlsolve(dt=0.01, tmax=1, update_dt=True, print_info=0)


###############################################################################
# Plot with pyvista
# ~~~~~~~~~~~~~~~~~
# A simple 3D plot of the :math:`\sigma_{zz}` field which coorespond to the
# :math:`\sigma_{\theta \theta}` component since the axisymmetric model work in
# cylindrical coordinates.

res.plot("Stress", component="YY", data_type="Node")

###############################################################################
# Write an animated gif of the equivalent plasticity :math:`p`, with full 3D
# reconstruction using :func:`fedoo.post_processing.axi_to_3d`.
#
# .. note::
#   The 3D reconstruction convert all fields to node_data. The fields are kept
#   in the axisymmetric cylindrical coordinate system except for the 'Disp'
#   field (displacement) that is converted to the 3D global coordinate system.
#   For instance, the 'XX' component of 'Stress' is the radial stress whereas
#   the 'X' component of displacement is the true 3d displacement along x.

clim = res.get_all_frame_lim("P")[2]
pl = pv.Plotter(window_size=[400, 600], off_screen=True)
pl.open_gif("tube_compression.gif", fps=20)
for i in range(res.n_iter):
    res.load(i)
    pl.clear_actors()
    fd.post_processing.axi_to_3d(res, 41).plot(
        "P",
        plotter=pl,
        clim=clim,
        title=f"Iter: {i}",
        title_size=10,
        azimuth=0,
        elevation=-70,
        show_scalar_bar=False,
        show_edges=True,
    )
    pl.hide_axes()
    pl.write_frame()

pl.close()

# We can also write a mp4 movie with:
# data_3d = fd.post_processing.axi_to_3d_multi('full_3d_data', res)
# data_3d.write_movie('tube_compression', 'P')

###############################################################################
# An example of how to do a realistic plot using the vtk physical based
# renderic availbale through pyvista.

pl = pv.Plotter(window_size=[608, 800])
res.load(62)
fd.post_processing.axi_to_3d(res, 41).plot(
    'Disp',
    'Z',
    show_edges=False,
    pbr=True,
    metallic=1,
    roughness=0.5,
    diffuse=1.,
    azimuth=0,
    elevation=-70,
    show_scalar_bar=False,
    plotter=pl,
)
pl.show()

###############################################################################
# This example generate a relastic mp4 movie of the 3d deformation of the
# cylinder. It is commented because it is not possible to render the mp4 movie
# with sphinx-gallery.

# pl = pv.Plotter(window_size=[608, 800], off_screen=True)

# cubemap = pv.examples.download_sky_box_cube_map()
# pl.add_actor(cubemap.to_skybox())
# pl.set_environment_texture(cubemap)
# pl.open_movie("tube_compression.mp4", quality=6)

# for i in range(res.n_iter):
#     res.load(i)
#     fd.post_processing.axi_to_3d(res, 41).plot(
#         show_edges=False,
#         pbr=True,
#         metallic=0.9,
#         roughness=0.4,
#         diffuse=0.8,
#         azimuth=0,
#         elevation=-70,
#         color="orange",
#         # clim=clim,
#         show_scalar_bar=False,
#         plotter=pl,
#         name="mymesh",
#     )

#     pl.hide_axes()
#     pl.write_frame()
#     pl.remove_actor("mymesh")

# pl.close()
