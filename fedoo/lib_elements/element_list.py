import numpy as np
from numpy import linalg

from fedoo.lib_elements.finite_difference_1d import Node, Parameter

# from fedoo.lib_elements.beam import *
# from fedoo.lib_elements.cohesive import Cohesive1D
from fedoo.lib_elements.hexahedron import Hex8, Hex20, Hex8r, Hex8Hourglass
from fedoo.lib_elements.line import Lin2, Lin2Bubble, Lin3, Lin3Bubble
from fedoo.lib_elements.quadrangle import (
    Quad4,
    Quad8,
    Quad9,
    Quad4r,
    Quad4Hourglass,
)

# from fedoo.lib_elements.plate import *
from fedoo.lib_elements.tetrahedron import Tet4, Tet10
from fedoo.lib_elements.triangle import Tri3, Tri3Bubble, Tri6
from fedoo.lib_elements.wedge import Wed6, Wed15, Wed18

_dict_elements = {
    "lin2": Lin2,
    "lin3": Lin3,
    "lin2bubble": Lin2Bubble,
    "lin3bubble": Lin3Bubble,
    # 'cohesive1d':Cohesive1D, 'lin2interface':Lin2Interface, 'quad4interface':Quad4Interface,
    "tri3": Tri3,
    "tri6": Tri6,
    "tri3bubble": Tri3Bubble,
    "quad4": Quad4,
    "quad4r": Quad4r,
    "quad4hourglass": Quad4Hourglass,
    "quad8": Quad8,
    "quad9": Quad9,
    "tet4": Tet4,
    "tet10": Tet10,
    "hex8": Hex8,
    "hex8r": Hex8r,
    "hex8hourglass": Hex8Hourglass,
    "hex20": Hex20,
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
    "tri6": 4,
    "tri3bubble": 3,
    "quad4": 4,
    "quad4hourglass": 1,
    "quad8": 9,
    "quad9": 9,
    "tet4": 4,
    "tet10": 15,
    "hex8": 8,
    "hex8r": 1,
    "hex20": 27,
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
    "ptri6": 4,
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
    elif element in ["tri3", "tri6", "tri3bubble"]:
        if nNd_elm == 3:
            return np.c_[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
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
    elif element in ["tet4", "tet10"]:
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
    elif element in ["hex8", "hex20", "hex8r"]:
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
    # if not isinstance(element_str, str): return element_str
    # return _dict_elements[element_str.lower()]
    return _dict_elements[element_str]


class CombinedElement:  # element that use several interpolation depending on the variable
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
        if isinstance(elm_type, str):
            elm_type = _dict_elements[elm_type]

        self.dict_elm_type[variable_name] = elm_type
        if associated_variables is not None:
            self.associated_variables[variable_name] = associated_variables

    def get_elm_type(self, variable_name):
        return self.dict_elm_type.get(variable_name, self.base_elm)

    def get_all_elm_type(self):
        list_elm_type = set(self.dict_elm_type.values())
        list_elm_type.add(self.base_elm)
        return list_elm_type
