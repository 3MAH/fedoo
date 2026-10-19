import numpy as np
from fedoo.lib_elements.finite_difference_1d import Node, Parameter

# from fedoo.lib_elements.beam import *
# from fedoo.lib_elements.cohesive import Cohesive1D
from fedoo.lib_elements.hexahedron import Hex8, Hex20, Hex20r, Hex8r, Hex8Hourglass
from fedoo.lib_elements.line import Lin2, Lin2Bubble, Lin3, Lin3Bubble
from fedoo.lib_elements.quadrangle import (
    Quad4,
    Quad8,
    Quad9,
    Quad4r,
    Quad4Hourglass,
)

# from fedoo.lib_elements.plate import *
from fedoo.lib_elements.tetrahedron import Tet4, Tet10, Tet10r
from fedoo.lib_elements.triangle import Tri3, Tri3r, Tri3Bubble, Tri6
from fedoo.lib_elements.wedge import Wed6, Wed15, Wed18

_dict_elements = {
    "lin2": Lin2,
    "lin3": Lin3,
    "lin2bubble": Lin2Bubble,
    "lin3bubble": Lin3Bubble,
    # 'cohesive1d':Cohesive1D, 'lin2interface':Lin2Interface, 'quad4interface':Quad4Interface,
    "tri3": Tri3,
    "tri3r": Tri3r,
    "tri6": Tri6,
    "tri3bubble": Tri3Bubble,
    "quad4": Quad4,
    "quad4r": Quad4r,
    "quad4hourglass": Quad4Hourglass,
    "quad8": Quad8,
    "quad9": Quad9,
    "tet4": Tet4,
    "tet10": Tet10,
    "tet10r": Tet10r,
    "hex8": Hex8,
    "hex8r": Hex8r,
    "hex8hourglass": Hex8Hourglass,
    "hex20": Hex20,
    "hex20r": Hex20r,
    "wed6": Wed6,
    "wed15": Wed15,
    "wed18": Wed18,
    "parameter": Parameter,
    "node": Node,
}

_dict_default_n_gp = {
    "lin2": 2,
    "lin3": 3,
    "lin2bubble": 2,
    "lin3bubble": 3,
    "lin4": 4,
    "cohesive1d": 1,
    "lin2interface": 2,
    "quad4interface": 4,
    "tri3": 3,
    "tri6": 7,
    "tri3bubble": 3,
    "quad4": 4,
    "quad4hourglass": 1,
    "quad8": 9,
    "quad9": 9,
    "tet4": 4,
    "tet10": 15,
    "tet10r": 4,
    "hex8": 8,
    "hex8r": 1,
    "hex20": 27,
    "hex20r": 14,
    "wed6": 6,
    "wed15": 21,
    "wed18": 21,
    "beam": 4,
    "beamfcq": 4,
    "bernoullibeam": 4,
    "parameter": 0,
    "node": 0,
    "pquad4": 4,
    "ptri3": 3,
    "pquad8": 9,
    "ptri6": 7,
    "pquad9": 9,
}


def add_elements(*args):
    for elm in args:
        _dict_elements[elm.name] = elm
        _dict_default_n_gp[elm.name] = elm.default_n_gp


def get_node_elm_coordinates(element, nNd_elm=None):
    if nNd_elm is None:
        nNd_elm = get_element(element).n_nodes
    # return xi_nd ie the position of nodes in the element local coordinate
    if element in ["lin2", "lin3", "lin2bubble", "lin3bubble", "cohesive2d"]:
        if nNd_elm == 2:
            return np.c_[[0.0, 1.0]]
        elif nNd_elm == 3:
            return np.c_[[0.0, 1.0, 0.5]]
    elif element == "lin2interface":
        if nNd_elm == 4:
            return np.c_[[0.0, 1.0, 0.0, 1.0]]
    elif element in ["tri3", "tri6", "tri3bubble"]:
        if nNd_elm == 3:
            return np.c_[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        elif nNd_elm == 4:
            return np.c_[[0.0, 1.0, 0.0, 1 / 3], [0.0, 0.0, 1.0, 1 / 3]]
        elif nNd_elm == 6:
            return np.c_[[0.0, 1.0, 0.0, 0.5, 0.5, 0.0], [0.0, 0.0, 1.0, 0.0, 0.5, 0.5]]
    elif element in ["quad4", "quad8", "quad9"]:
        if nNd_elm == 4:
            return np.c_[[-1.0, 1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 1.0]]
        elif nNd_elm == 8:
            return np.c_[
                [-1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0],
            ]
        elif nNd_elm == 9:
            return np.c_[
                [-1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                [-1.0, -1.0, 1.0, 1.0, -1.0, 0.0, 1.0, 0.0, 0.0],
            ]
    elif element in ["quad4interface"]:
        if nNd_elm == 8:
            return np.c_[
                [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
            ]
    elif element in ["tet4", "tet10", "tet10r"]:
        if nNd_elm == 4:
            return np.c_[
                [0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]
            ]
        elif nNd_elm == 10:
            return np.c_[
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5],
                [1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0],
            ]
    elif element in ["hex8", "hex20", "hex8r", "hex20r"]:
        if nNd_elm == 8:
            return np.c_[
                [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        elif nNd_elm == 20:
            return np.array(
                [
                    [-1.0, -1.0, -1.0],
                    [1.0, -1.0, -1.0],
                    [1.0, 1.0, -1.0],
                    [-1.0, 1.0, -1.0],
                    [-1.0, -1.0, 1.0],
                    [1.0, -1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [-1.0, 1.0, 1.0],
                    [0.0, -1.0, -1.0],
                    [1.0, 0.0, -1.0],
                    [0.0, 1.0, -1.0],
                    [-1.0, 0.0, -1.0],
                    [0.0, -1.0, 1.0],
                    [1.0, 0.0, 1.0],
                    [0.0, 1.0, 1.0],
                    [-1.0, 0.0, 1.0],
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                ]
            )
    elif element in ["wed6", "wed15", "wed18"]:
        if nNd_elm == 6:
            return np.c_[
                [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            ]
        elif nNd_elm == 15:
            return np.c_[
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.5,
                    0.5,
                    0.0,
                    0.5,
                    1.0,
                    0.0,
                    0.0,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.5,
                    0.5,
                    0.0,
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ],
            ]
        elif nNd_elm == 18:
            return np.c_[
                [
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    1.0,
                    -1.0,
                    -1.0,
                    -1.0,
                    1.0,
                    1.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.5,
                    0.5,
                    0.0,
                    0.5,
                    1.0,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.5,
                ],
                [
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.5,
                    0.5,
                    0.0,
                    0.5,
                    0.5,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.5,
                    0.5,
                    0.0,
                ],
            ]

    elif element in ["spring"]:
        # The values are arbitrary, only the size is important
        return np.c_[[0.0, 0.0]]


def get_all():
    """Return the registry of element types, a dict ``{name: element}``.

    Its keys are the names accepted as ``elm_type`` (``"hex8"``, ``"tet10"``,
    ``"quad8"``, ...), including the combined elements created by the user.
    """
    return _dict_elements


def get_default_n_gp(element, mesh=None, raise_error=True):
    n_elm_gp = _dict_default_n_gp.get(element)
    if n_elm_gp is None:
        if mesh is not None:
            n_elm_gp = _dict_default_n_gp.get(mesh.elm_type)
        if n_elm_gp is None and raise_error:
            raise NameError("Element unknown: no default number of integration points")

    return n_elm_gp


def get_element(element_str):
    """Return the element registered under the name ``element_str``.

    Raises ``KeyError`` for an unknown name (see :py:func:`get_all`).
    """
    # if not isinstance(element_str, str): return element_str
    # return _dict_elements[element_str.lower()]
    return _dict_elements[element_str]


class CombinedElement:  # element that use several interpolation depending on the variable
    """Element type whose variables may use different interpolations.

    The base element is used for the geometry and, by default, for every
    variable. :py:meth:`set_variable_interpolation` assigns another element
    to one variable, typically a lower-order element built on the vertex
    nodes of the base element. The classical use is a Taylor-Hood pair,
    quadratic displacement and linear pressure, which satisfies the inf-sup
    (LBB) condition of mixed and undrained poromechanical formulations.

    The new element type is registered globally under ``name`` when the
    object is created: create it before the assembly and pass ``name`` as
    ``elm_type`` to :py:meth:`fedoo.Assembly.create`. The mesh keeps the
    element type of the base element. Creating another combined element with
    the same name replaces the previous one.

    Parameters
    ----------
    name : str
        Name of the new element type.
    base_elm : str or element class
        Element used for the geometry and for the variables without a
        specific interpolation, e.g. ``"hex20"``.
    **kargs
        Advanced options: ``default_n_gp`` (number of integration points,
        default that of ``base_elm``), ``n_nodes``, ``local_csys``,
        ``dict_elm_type`` and ``associated_variables``.

    Notes
    -----
    The DOFs of a variable on nodes that its interpolation does not use
    (mid-side nodes for a linear pressure on a quadratic mesh) receive no
    contribution: they are eliminated by the solver and stay at zero.

    Examples
    --------
    Quadratic displacement with a linear pore pressure:

    >>> import fedoo as fd
    >>> fd.ModelingSpace("3D")  # doctest: +SKIP
    >>> elm = fd.lib_elements.CombinedElement("hex20lbb", "hex20")
    >>> elm.set_variable_interpolation("PorePressure", "hex8")
    >>> # assembly = fd.Assembly.create(wf, mesh_hex20, elm_type="hex20lbb")
    """

    def __init__(self, name, base_elm, **kargs):
        assert isinstance(name, str), "Element name should be a str."
        if isinstance(base_elm, str):
            base_elm = _dict_elements[base_elm]
        self.base_elm = (
            base_elm  # element for variables not defined in associated_variables
        )
        self.geometry_elm = base_elm  # element used for geometrical interpolation (default = same as base_elm)
        self.name = name
        self.default_n_gp = kargs.get("default_n_gp", base_elm.default_n_gp)
        self.n_nodes = kargs.get("n_nodes", base_elm.n_nodes)
        self.local_csys = kargs.get("local_csys", False)
        self.dict_elm_type = kargs.get("dict_elm_type", {})
        self.associated_variables = kargs.get("associated_variables", {})

        _dict_elements[name] = self
        _dict_default_n_gp[name] = self.default_n_gp

    def set_variable_interpolation(
        self, variable_name, elm_type, associated_variables=None
    ):
        """Use a specific element to interpolate one variable.

        Parameters
        ----------
        variable_name : str
            Name of the variable, e.g. ``"PorePressure"`` or ``"Pressure"``.
        elm_type : str or element class
            Element used for this variable, e.g. ``"hex8"`` on a ``"hex20"``
            base element. Its nodes are the first nodes of the base element
            (the vertex nodes for the standard quadratic elements).
        associated_variables : dict, optional
            Associated variables of the interpolation, as for elements
            with coupled variables (beams). Not needed for scalar fields.
        """
        if isinstance(elm_type, str):
            elm_type = _dict_elements[elm_type]

        self.dict_elm_type[variable_name] = elm_type
        if associated_variables is not None:
            self.associated_variables[variable_name] = associated_variables

    def get_elm_type(self, variable_name):
        """Return the element interpolating ``variable_name``.

        The base element unless :py:meth:`set_variable_interpolation` set
        another one for this variable.
        """
        return self.dict_elm_type.get(variable_name, self.base_elm)

    def get_all_elm_type(self):
        """Return the set of elements used by the variables, base included."""
        list_elm_type = set(self.dict_elm_type.values())
        list_elm_type.add(self.base_elm)
        return list_elm_type
