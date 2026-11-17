"""Element library.

=============================================
Element library (:mod:`fedoo.lib_elements`)
=============================================

An element type is selected by its name, given as ``elm_type`` when a mesh
is built or an assembly is created (``"hex8"``, ``"hex20"``, ``"tet4"``,
``"tet10"``, ``"quad4"``, ``"quad8"``, ``"tri3"``, ``"tri6"``, ``"lin2"``,
...). The names known to fedoo are the keys of
:py:func:`~fedoo.lib_elements.get_all`.

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
    get_node_elm_coordinates,
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
from .tetrahedron import ElementTetrahedron, Tet4, Tet10, Tet10r
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
