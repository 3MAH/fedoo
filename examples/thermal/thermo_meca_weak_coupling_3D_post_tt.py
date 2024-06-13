import os
from time import time

import numpy as np
import pylab as plt
import pyvista as pv
from numpy import linalg

from fedoo import Mesh

start = time()
# --------------- Pre-Treatment --------------------------------------------------------


# -------------------- MESH ------------------------------
# Mesh.box_mesh(Nx=3, Ny=3, Nz=3, x_min=0, x_max=1, y_min=0, y_max=1, z_min=0, z_max=1, ElementShape = 'hex8', name = meshname)
# Mesh.import_file('octet_surf.msh', meshname = "Domain")
# Mesh.import_file('data/octet_1.msh', meshname = "Domain")
Mesh.import_file("data/gyroid.msh", meshname="Domain")
meshname = "Domain"

mesh = Mesh.get_all()[meshname]
nb_iter = 200
filename = "results/thermo_meca_nl"

# field_name = 'State_Variables'
field_name = "Temp"
component = 1
# clim = None
clim = [0, 100]

crd = mesh.nodes

# #note set for boundary conditions
Xmin, Xmax, center = mesh.bounding_box.center


#### save a video (need imageio-ffmpeg - conda install imageio-ffmpeg -c conda-forge
pl = pv.Plotter(window_size=[1024, 768])
pl.set_background("White")
sargs = dict(
    interactive=True,
    title_font_size=20,
    label_font_size=16,
    color="Black",
    # n_colors= 10
)

factor = 5

pl.open_movie(filename + ".mp4", framerate=24, quality=4)

meshplot = mesh.to_pyvista()

# pl.show(auto_close=False)  # only necessary for an off-screen movie
pl.camera.SetFocalPoint(center)
pl.camera.position = (-2.090457552750125, 1.7582929402632352, 1.707926514944027)

for i in range(0, nb_iter):
    # meshplot = pv.read('results/thermal3D_' +str(i)+'.vtk')
    res_th = np.load(filename + "_th_{}.npz".format(i))
    res_me = np.load(filename + "_me_{}.npz".format(i))

    for res in [res_th, res_me]:
        for item in res:
            if item[-4:] == "Node":
                if len(res[item]) == len(crd):
                    meshplot.point_data[item[:-5]] = res[item]
                else:
                    meshplot.point_data[item[:-5]] = res[item].T
            else:
                meshplot.cell_data[item] = res[item].T

    # actor = pl.add_mesh(meshplot, scalars = 'data', show_edges = True, scalar_bar_args=sargs, cmap="bwr", clim = [0,100])
    meshplot.points = crd + factor * meshplot.point_data["Disp"]

    if i == 0:
        pl.add_mesh(
            meshplot,
            scalars=field_name,
            component=component,
            show_edges=True,
            scalar_bar_args=sargs,
            cmap="jet",
            clim=clim,
        )

    if clim is None:
        pl.update_scalar_bar_range(
            [
                meshplot.point_data[field_name].min(),
                meshplot.point_data[field_name].max(),
            ]
        )

    # pl.camera.Azimuth(2*i/360*np.pi)
    pl.camera.Azimuth(360 / nb_iter)
    # Run through each frame
    # pl.add_text(f"Iteration: {i}", name='time-label', color='Black')
    pl.write_frame()  # write initial data
    # pl.remove_actor(actor)

pl.close()
