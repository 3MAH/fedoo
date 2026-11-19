"""Element library.

=============================================
Element library (:mod:`fedoo.lib_elements`)
=============================================

An element type is selected by its name, given as ``elm_type`` when a mesh
is built or an assembly is created (``"hex8"``, ``"hex20"``, ``"tet4"``,
``"tet10"``, ``"quad4"``, ``"quad8"``, ``"tri3"``, ``"tri6"``, ``"lin2"``,
...). The names known to fedoo are the keys of
:py:func:`~fedoo.lib_elements.get_all`.

Each element has a default number of Gauss points, which the ``n_elm_gp``
argument of :py:meth:`fedoo.Assembly.create` overrides: 4 for ``"tet4"``,
15 for ``"tet10"``, 8 for ``"hex8"``, 27 for ``"hex20"``. The reduced
variants are selected with ``elm_type`` at the assembly level, the mesh
keeping its base element:

* ``"tri3r"``, ``"quad4r"``, ``"hex8r"`` and ``"tet4r"`` evaluate the
  interpolation at the element centre whatever ``n_elm_gp`` and default to
  1 point. Another ``n_elm_gp`` raises a warning: selective integration
  goes through the combined elements (``"quad4sri"``, ``"hex8sri"``).
* ``"hex20r"`` (14 points) and ``"tet10r"`` (4 points) keep the full
  interpolation and only change the default rule: with another ``n_elm_gp``
  they are ``"hex20"`` and ``"tet10"`` with that rule, and a warning is
  raised.

In dynamics the mass does not follow a reduced stiffness: it is assembled on
the element of the mesh with its full rule (4 points on a ``"tet4"`` mesh
even with ``elm_type="tet4r"``), on the reference configuration. A reduced
mass must be requested on the storage weak form (``assembly_options`` of the
:py:class:`~fedoo.weakform.Inertia` given to ``set_inertia``) and raises a
warning.

By default every variable of the modeling space is interpolated with the
element of the mesh. A :py:class:`~fedoo.lib_elements.CombinedElement`
defines a new element type in which some variables use another
interpolation, for instance quadratic displacement and linear pressure
(Taylor-Hood pair) for mixed formulations and poromechanics::

    import fedoo as fd

    elm = fd.lib_elements.CombinedElement("hex20lbb", "hex20")
    elm.set_variable_interpolation("PorePressure", "hex8")
    # assembly = fd.Assembly.create(wf, mesh_hex20, elm_type="hex20lbb")

.. autosummary::
   :toctree: generated/
   :template: custom-class-template.rst

   CombinedElement

.. autosummary::
   :toctree: generated/

   get_all
   get_element
   get_full_integration
   is_reduced_integration
"""

from .beam import (
    Beam,
    Beam_dispY,
    Beam_dispZ,
    Beam_rotY,
    Beam_rotZ,
    BeamFCQ,
    BeamFCQ_disp,
    BeamFCQ_lin2,
    BeamFCQ_rot,
    BernoulliBeam,
    BernoulliBeam_disp,
    BernoulliBeam_rot,
)
from .cohesive import (
    Node2,
    Node2Jump,
    Lin2Interface,
    Lin2InterfaceJump,
    Lin2MeanPlane,
    Quad4Interface,
    Quad4InterfaceJump,
    Quad4MeanPlane,
)
from .element_base import Element, Element1D, Element1DGeom2, Element2D
from .element_list import (
    CombinedElement,
    add_elements,
    get_all,
    get_default_n_gp,
    get_element,
    get_full_integration,
    get_node_elm_coordinates,
    is_reduced_integration,
)
from .finite_difference_1d import (
    BackwardFiniteDifference,
    BackwardFiniteDifferenceOrder2,
    FiniteDifference1D,
    ForwardFiniteDifference,
    ForwardFiniteDifferenceOrder2,
    Node,
    Parameter,
)
from .hexahedron import ElementHexahedron, Hex8, Hex20, Hex20r, Hex8r
from .line import Lin2, Lin2Bubble, Lin3, Lin3Bubble
from .plate import pquad4, pquad8, pquad9, ptri3, ptri6
from .quadrangle import ElementQuadrangle, Quad4, Quad8, Quad9
from .tetrahedron import ElementTetrahedron, Tet4, Tet4r, Tet10, Tet10r
from .triangle import ElementTriangle, Tri3, Tri3Bubble, Tri6
from .wedge import ElementWedge, Wed6, Wed15, Wed18
from .incompressible import hex8sri

__all__ = [
    "Beam",
    "Beam_dispY",
    "Beam_dispZ",
    "Beam_rotY",
    "Beam_rotZ",
    "BeamFCQ",
    "BeamFCQ_disp",
    "BeamFCQ_lin2",
    "BeamFCQ_rot",
    "BernoulliBeam",
    "BernoulliBeam_disp",
    "BernoulliBeam_rot",
    "Cohesive1D",
    "Lin2Interface",
    "Lin2InterfaceJump",
    "Lin2MeanPlane",
    "Quad4Interface",
    "Quad4InterfaceJump",
    "Quad4MeanPlane",
    "Element",
    "Element1D",
    "Element1DGeom2",
    "Element2D",
    "CombinedElement",
    "add_elements",
    "get_all",
    "get_default_n_gp",
    "get_element",
    "get_node_elm_coordinates",
    "BackwardFiniteDifference",
    "BackwardFiniteDifferenceOrder2",
    "FiniteDifference1D",
    "ForwardFiniteDifference",
    "ForwardFiniteDifferenceOrder2",
    "Node",
    "Parameter",
    "ElementHexahedron",
    "Hex8",
    "Hex8r",
    "Hex20",
    "Hex20r",
    "Lin2",
    "Lin2Bubble",
    "Lin3",
    "Lin3Bubble",
    "pquad4",
    "pquad8",
    "pquad9",
    "ptri3",
    "ptri6",
    "ElementQuadrangle",
    "Quad4",
    "Quad8",
    "Quad9",
    "ElementTetrahedron",
    "Tet4",
    "Tet4r",
    "Tet10",
    "Tet10r",
    "ElementTriangle",
    "Tri3",
    "Tri3Bubble",
    "Tri6",
    "ElementWedge",
    "Wed6",
    "Wed15",
    "Wed18",
]
