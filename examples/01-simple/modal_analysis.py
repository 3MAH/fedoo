r"""
Free vibration of a cantilever
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example computes the natural frequencies and mode shapes of an elastic
cantilever. A modal analysis solves the generalized eigenvalue problem

.. math::

    \mathbf{K}\boldsymbol{\phi} = \omega^2
    \mathbf{M}\boldsymbol{\phi},

where :math:`\mathbf{K}` and :math:`\mathbf{M}` are the stiffness and mass
matrices, :math:`\omega` is an angular frequency, and
:math:`\boldsymbol{\phi}` is a mode shape. Because no external load is needed,
this analysis is also called a *free-vibration* analysis.
r"""

import numpy as np

import fedoo as fd


###############################################################################
# Geometry and mesh
# ~~~~~~~~~~~~~~~~~
# A slender rectangular cantilever is represented with plane-strain
# quadrilateral elements. The dimensions are expressed in metres.

fd.ModelingSpace("2Dplane")

length = 1.0
height = 0.1
mesh = fd.mesh.rectangle_mesh(
    nx=21,
    ny=5,
    x_min=0.0,
    x_max=length,
    y_min=-height / 2,
    y_max=height / 2,
    elm_type="quad4",
)

###############################################################################
# Material and assembly
# ~~~~~~~~~~~~~~~~~~~~~
# Both elastic properties and mass density are required. The density is used
# by Fedoo to assemble the consistent mass matrix of the model.

steel = fd.constitutivelaw.ElasticIsotrop(210e9, 0.3)
steel.set_density(7800.0)
weakform = fd.weakform.StressEquilibrium(steel)
assembly = fd.Assembly.create(weakform, mesh)

###############################################################################
# Modal problem
# ~~~~~~~~~~~~~
# The left edge is clamped. ``Modal.solve`` then extracts the six modes with
# the lowest eigenvalues. ``sigma=0`` explicitly asks the sparse eigensolver
# to search around zero frequency.
#
# Registering an output writes one frame per mode to the ``fdh5`` file. Each
# frame contains the requested fields as well as the mode number, eigenvalue,
# angular frequency, and frequency as scalar metadata.

modal = fd.problem.Modal(assembly)
clamped = mesh.find_nodes("X", mesh.bounding_box.xmin)
modal.bc.add("Dirichlet", clamped, "Disp", 0.0)

results = modal.add_output(
    "cantilever_modes.fdh5", assembly, ["Disp", "Strain", "Stress"]
)
modal.solve(n_modes=6, sigma=0.0)

print("Natural frequencies [Hz]:")
for mode, frequency in enumerate(modal.frequencies, start=1):
    print(f"  mode {mode}: {frequency:.6g}")

###############################################################################
# Plot a mode shape
# ~~~~~~~~~~~~~~~~~
# Modal amplitudes have no physical magnitude: only their relative values and
# signs define the shape. The deformation scale is therefore chosen so that
# the largest displayed displacement is 15% of the beam length. The first
# frame is plotted and coloured by displacement magnitude using the standard
# :class:`fedoo.DataSet` plotting interface.

results.load(0)
max_displacement = np.max(np.linalg.norm(results["Disp"], axis=0))
display_scale = 0.15 * length / max_displacement
results.plot(
    "Disp",
    component="norm",
    iteration=0,
    scale=display_scale,
    show_edges=True,
    title=f"First bending mode - {modal.frequencies[0]:.2f} Hz",
)
