"""
=================================
Post-Processing
=================================

.. currentmodule:: fedoo

Get results from a problem
==========================

In fedoo, most of the standard results are easily exportable using the
:py:meth:`fedoo.Problem.get_results` method of the problem class.

The get_results method returns a :py:class:`fedoo.DataSet` object which come
with several methods for plotting, saving and loading mesh dependent results.


To avoid a redundent call of the get_results function, especially for time
dependent problems, one can simply add some required output with the
:py:meth:`fedoo.Problem.add_output` method. This create a
:py:class:`MultiFrameDataSet` object associated to the problem.
Once the required outputs are defined for a given problem, a call to the
:py:meth:`fedoo.Problem.save_results` method allow to save all the defined
fields on disk using the choosen file format, and associate the saved file to
an iteration of the :py:class:`MultiFrameDataSet`. For non linear problems
solved using :py:meth:`Problem.nlsolve`, results are automatically saved at
some iterations dependending on the choosen parameters.

The :py:class:`MultiFrameDataSet` store the path of the saved files for each
iteration in the MultiFrameDataSet.list_data attribute. The method
:py:meth:`MultiFrameDataSet.load` is called to read the data
of a given iteration.


Class DataSet
-------------

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   DataSet

Class MultiFrameDataSet
-----------------------

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   MultiFrameDataSet

Save data to disk
-----------------

Once a DataSet is created using for instance the
:py:meth:`fedoo.Problem.get_results` method, the data can easily be saved on
disk using for instance the :py:meth:`fedoo.DataSet.save` method.

The available file types are:
    * 'fdz': A zipped archive containing the mesh using the 'vtk' format
      named '_mesh_.vtk', and data from several iterations named 'iter_x.npz'
      where x is the iteration number (x=0 for the 1st iteration).
    * 'vtk': The vtk format contains the mesh and the data in a single files.
      The gauss points data are not included in the file.
      This format is efficient for a linear problem when we need only one time
      iteration. In case of multiple saved iterations, a directory is created
      and one vtk file is saved per iteration. The mesh is included in every
      file which is not memory efficient.
    * 'msh': Format associated to gmsh. Have the same drawback as the vtk
      format for time depend results and missing gauss points data. The vtk
      format should be prefered.
    * 'npz': Save data in a numpy file npz which doesn't include the mesh. The
      mesh is generally saved beside in a raw vtk files without results.
    * 'npz_compressed': Same as npz with a compression of the zip archive.
    * 'csv': Save DataSet that contains only one type of data
      (ie Node, Element or Gauss point data) in a csv file (needs the library
      pandas installed).
      The mesh is not included and may be saved beside in a vtk file.
    * 'xlsx': Same as csv but with the excel format.


Read data from disk
-------------------

To read data saved on disk, use the function :py:func:`read_data`.


.. _very_simple_problem:

Example
-------

For example, defining and solving a very simple problem :

.. code-block:: python

    import fedoo as fd

    fd.ModelingSpace("2Dstress")
    mesh = fd.mesh.rectangle_mesh()
    material = fd.constitutivelaw.ElasticIsotrop(2e5, 0.3)
    wf = fd.weakform.StressEquilibrium(material)
    assembly = fd.Assembly.create(wf, mesh)

    # Define a new static problem
    pb = fd.problem.Linear(assembly)

    # Boundary conditions
    pb.bc.add('Dirichlet', 'left', 'Disp',    0 )
    pb.bc.add('Dirichlet', 'right', 'Disp',  [0.2,0] )

    # Solve problem
    pb.solve()


Then, we can catch the Stress, Displacement and Strain fields using:

.. code-block:: python

    results = pb.get_results(assembly, ["Stress", "Disp", "Strain"])
    # plot the sigma_xx averaged at nodes
    results.plot("Stress", "Node", component='XX')


.. image:: ./_static/examples/example_plot_dataset.png

Alternatively, if we take the same problem, but accounting for geometric
non linearities (nlgeom = True), we can automatically save results at specified
time interval (here the results are saved on a file).


.. code-block:: python

    wf.nlgeom = True
    pb_nl = fd.problem.NonLinear(assembly)

    # Boundary conditions
    pb_nl.bc = pb.bc

    results_nl = pb_nl.add_output('nl_results', assembly, ["Stress", "Disp", "Strain"])

    pb_nl.nlsolve(dt = 0.1, tmax = 1, interval_output = 0.2)

    # plot the sigma_xx averaged at nodes at the last increment
    results_nl.plot("Stress", "Node", component='XX')

.. image:: ./_static/examples/example_plot_dataset_nl.png

Basic operations
================

Plotting results
----------------

A few convinient methods are proposed to generate image of movie from
:py:class:`DataSet` and :py:class:`MultiFrameDataSet` objects.

.. autosummary::
   :toctree: generated/

   DataSet.plot
   MultiFrameDataSet.plot
   MultiFrameDataSet.plot_history
   MultiFrameDataSet.write_movie


Save results
------------

.. autosummary::
   :toctree: generated/

   DataSet.save
   DataSet.savez
   DataSet.savez_compressed
   DataSet.save_mesh
   DataSet.to_csv
   DataSet.to_excel
   DataSet.to_vtk
   DataSet.to_msh
   MultiFrameDataSet.save_all


Read results
------------

.. autosummary::
   :toctree: generated/

   read_data
   DataSet.read
   DataSet.load
   MultiFrameDataSet.load


Advanced operations
===================

Write Movies
------------

A very simple way to write a movie from a :py:class:`MultiFrameDataSet` is
to call the embedded method :py:meth:`MultiFrameDataSet.write_movie`.

Though this method comes with lots of options, one may sometimes want to
fully control the movie rendering. This is easy to do by manualy writing the
movie using the pyvista library.

Here is an exemple to animate the linear results obtained in the
:ref:`problem <very_simple_problem>` defined above. The idea is to use a
scale_factor applied to the displacement (using the *scale* argument) and to
the stress field (modifiying the results data).

.. code-block:: python

    results = pb.get_results(assembly, ['Stress', 'Disp'], 'Node')
    stress = results.node_data['Stress']
    clim = [stress[3].min(), stress[3].max()]  # 3 -> xy in voigt notation

    pl = pv.Plotter(window_size = [600,400])
    pl.open_gif("my_movie.gif", fps=20)
    sargs = dict(height=0.10, position_x=0.2, position_y=0.05)
    for i in range(48):
        scale_factor = (i + 1) / 48
        results.node_data["Stress"] = scale_factor * stress
        results.plot(
            "Stress",
            "XY",
            plotter=pl,
            scale=scale_factor,
            clim=clim,
            title=f"Iter: {i}",
            title_size = 10,
            scalar_bar_args=sargs,
        )
        pl.hide_axes()
        pl.write_frame()

    pl.close()

.. image:: ./_static/examples/my_movie.gif


Multiplot feature
-----------------

It is possible to create the plotter before calling the plot function.
This allow for instance to use the pyvista multiplot capability.
For instance,
we can plot the stress results after the example
:ref:`example_plate_with_hole`:

.. code-block:: python

    import pyvista as pv
    pl = pv.Plotter(shape=(2,2))
    # or using the backgroundplotter:
    # from pyvistaqt import BackgroundPlotter
    # pl = BackgroundPlotter(shape = (2,2))
    results.plot('Stress', 'vm', 'Node', plotter=pl)
    pl.subplot(1,0)
    results.plot('Stress', 'XX', 'Node', plotter=pl)
    pl.subplot(0,1)
    results.plot('Stress', 'YY', 'Node', plotter=pl)
    pl.subplot(1,1)
    results.plot('Stress', 'XY', 'Node', plotter=pl)
    pl.show()

.. image:: ./_static/examples/multiplot.png
"""

from .axi_to_3d import (
    axi_to_3d,
    axi_to_3d_multi,
)

__all__ = [
    "axi_to_3d",
    "axi_to_3d_multi",
]
