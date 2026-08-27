Overview
========

About
-----

.. image:: _static/fedoo_logos.png

Fedoo is an open-source finite element library written in Python. It is focused
on solid and structural mechanics, with additional support for thermal
problems. Its script-based interface gives direct access to meshes, elements,
constitutive laws, weak forms, assemblies, and solution procedures, making it
suitable for research, teaching, and the development of custom numerical
models.

Fedoo has a particular focus on heterogeneous materials and computational
homogenization. `Simcoon <https://github.com/3MAH/simcoon>`_ is integrated into
Fedoo as a dependency and provides high-performance nonlinear constitutive
models. Fedoo can also be used alongside `Microgen
<https://github.com/3MAH/microgen>`_, another project from the 3MAH ecosystem,
for geometry and mesh generation.


Main features
-------------

* **Static and transient analyses:** linear and nonlinear solution procedures,
  implicit and explicit time integration, and thermal analyses.
* **Mechanical models:** small and finite strain formulations, elasticity,
  plasticity, hyperelasticity, cohesive laws, beams, shells, solids, and rigid
  bodies.
* **Meshes and assemblies:** 2D, axisymmetric, and 3D elements, mixed element
  families, reduced integration, and multi-mesh models.
* **Constraints and interactions:** periodic boundary conditions, multi-point
  constraints, mean-value and mean-motion constraints, and 2D/3D contact.
* **Homogenization:** tools for periodic and non-periodic boundary conditions
  and extraction of homogenized quantities.
* **Input and output:** many standard mesh formats can be imported and exported
  through MeshIO. Results can be stored in Fedoo's FDH5 format, which is based
  on HDF5 and designed for fast reading and writing from Fedoo.
* **Graphical results viewer:** an integrated graphical interface based on
  PyVista provides interactive visualization and inspection of computed
  fields.
* **Extensibility:** users can implement their own weak equations, as well as
  custom constitutive laws, elements, and solution workflows, through a
  readable Python architecture.

Fedoo favors an open and inspectable implementation while still paying
attention to computational cost. Performance-critical constitutive
calculations are handled by Simcoon where applicable, while additional sparse
linear solvers can be provided by optional compiled dependencies.
