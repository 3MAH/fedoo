"""
Compression of a tube using 2D axisymmetric model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This model uses IPC self-contact (requires ``ipctk >= 1.6``), elasto-plastic
material law with finite strain assumption in a 2D axisymetric modeling space.
The full 3D result is ploted during the post processing phase.
"""

# sphinx_gallery_thumbnail_number = 3
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
#
# The self contact is treated with the IPC (Incremental Potential Contact)
# method from the ipctk library: a barrier potential is activated when two
# parts of the surface get closer than ``dhat`` (here relative to the size of
# the model), and a continuous collision detection (CCD) line search
# guarantees that the folds never interpenetrate. The barrier stiffness is
# tuned automatically.
#
# ipctk works on the planar (r, z) outline of the tube. In the "2Daxi" space,
# fedoo weights each collision by the circumference :math:`2 \pi r` of the
# ring it represents. The option ``use_area_weighting=True`` (convergent
# formulation) can be added to weight each collision by the area of the ring,
# so that the contact response does not depend on the mesh density, at the
# price of a few more increments.

wf = fd.weakform.StressEquilibrium(material)
solid_assembly = fd.Assembly.create(wf, mesh)

contact = fd.constraint.IPCSelfContact(
    mesh,
    dhat=1e-3,  # barrier activation distance / bounding box diagonal
)
assembly = fd.Assembly.sum(solid_assembly, contact)

###############################################################################
# We define a non-linear problem including geometric nonlinearities using the
# Updated Lagrangian method (NLGEOM = 'UL'), which is the default in fedoo
# (equivalent to NLGEOM = True).
#
# The Line Search method is activated to improve the convergence of the
# Newton-Raphson algorithm. This necessitates increasing the maximum
# sub-iterations (max_subiter from 10 to 20) to allow for step-length
# optimization. Additionally, 'adaptive_stiffness' is enabled to stabilize
# convergence during sharp elastic-to-plastic transitions.

NLGEOM = "UL"
pb = fd.problem.NonLinear(assembly, nlgeom=NLGEOM)
pb.set_nr_criterion(
    "Displacement",
    tol=1e-2,
    max_subiter=20,
    adaptive_stiffness=True,
)

# create a 'result' folder and set the desired ouputs
if not (os.path.isdir("results")):
    os.mkdir("results")
res = pb.add_output(
    "results/tube_compression", solid_assembly, ["Disp", "Stress", "Strain", "P"]
)


# Node sets for boundary conditions
bottom = mesh.node_sets["bottom"]
top = mesh.node_sets["top"]

pb.bc.add("Dirichlet", bottom, "Disp", 0)
pb.bc.add("Dirichlet", top, "Disp", [0, -150])
pb.add_line_search()
pb.nlsolve(dt=0.01, tmax=1, update_dt=True, print_info=0, dt_min=1e-8)


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

clim = res.get_all_frame_lim("P")[2]  # extract the min/max values
data_3d = fd.post_processing.axi_to_3d(res)  # recontructed 3d data
pl = pv.Plotter(window_size=[400, 600], off_screen=True)
pl.open_gif("tube_compression.gif", fps=20)
for i in range(data_3d.n_iter):
    data_3d.load(i)
    pl.clear_actors()
    data_3d.plot(
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
# data_3d.write_movie('tube_compression', 'P')

###############################################################################
# An example of how to do a realistic plot using the vtk physical based
# renderic availbale through pyvista.

pl = pv.Plotter(window_size=[608, 800])
data_3d.load(int(0.6 * data_3d.n_iter))  # load an intermediate iteration
data_3d.plot(
    "Disp",
    "Z",
    show_edges=False,
    pbr=True,
    metallic=1,
    roughness=0.5,
    diffuse=1.0,
    azimuth=0,
    elevation=-70,
    show_scalar_bar=False,
    plotter=pl,
)
pl.show()

###############################################################################
# This example generate a realistic mp4 movie of the 3d deformation of the
# cylinder. It is commented because it is not possible to render the mp4 movie
# with sphinx-gallery.

# pl = pv.Plotter(window_size=[608, 800], off_screen=True)

# cubemap = pv.examples.download_sky_box_cube_map()
# pl.add_actor(cubemap.to_skybox())
# pl.set_environment_texture(cubemap)
# pl.open_movie("tube_compression.mp4", quality=6)

# for i in range(data_3d.n_iter):
#     data_3d.load(i)
#     data_3d.plot(
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
