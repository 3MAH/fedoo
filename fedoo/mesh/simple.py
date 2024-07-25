"""This module contains functions to generate simple meshes"""

from fedoo.core.mesh import Mesh
import itertools
import numpy as np


# utility fuctions
# Only Functions are declared here !!
def stack(mesh1, mesh2, name=""):
    """
    Make the spatial stack of two mesh objects which have the same element shape.
    This function doesn't merge coindicent Nodes.
    For that purpose, use the Mesh methods 'find_coincident_nodes' and 'merge_nodes'
    on the resulting Mesh.

    Return
    ---------
    Mesh object with is the spacial stack of mesh1 and mesh2
    """
    return Mesh.stack(mesh1, mesh2, name)


def rectangle_mesh(
    nx=11,
    ny=11,
    x_min=0,
    x_max=1,
    y_min=0,
    y_max=1,
    elm_type="quad4",
    ndim=None,
    name="",
):
    """
    Create a rectangular Mesh

    Parameters
    ----------
    nx, ny : int
        Numbers of nodes in the x and y axes (default = 11).
    x_min, x_max, y_min, y_max : int,float
        The boundary of the square (default : 0, 1, 0, 1).
    elm_type : {'tri3', 'quad4', 'quad8', 'quad9'}
        The type of the element generated (default='quad4')

        * 'tri3' -- 3 node linear triangular mesh
        * 'tri6' -- 6 node linear triangular mesh
        * 'quad4' -- 4 node quadrangular mesh
        * 'quad8' -- 8 node quadrangular mesh (à tester)
        * 'quad9' -- 9 node quadrangular mesh

    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.

    See Also
    --------
    line_mesh : 1D mesh of a line
    rectangle_mesh : Surface mesh of a rectangle
    box_mesh : Volume mesh of a box
    grid_mesh_cylindric : Surface mesh of a grid in cylindrical coodrinate
    line_mesh_cylindric : Line mesh in cylindrical coordinate
    """

    if elm_type == "quad9" or elm_type == "tri6":
        nx = int(nx // 2 * 2 + 1)
        ny = int(ny // 2 * 2 + 1)  # pour nombre impair de noeuds
    X, Y = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    crd = np.c_[np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))]
    if elm_type == "quad8":
        dx = (x_max - x_min) / (nx - 1.0)
        dy = (y_max - y_min) / (ny - 1.0)
        X, Y = np.meshgrid(
            np.linspace(x_min + dx / 2.0, x_max - dx / 2.0, nx - 1),
            np.linspace(y_min, y_max, ny),
        )
        crd2 = np.c_[np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))]
        X, Y = np.meshgrid(
            np.linspace(x_min, x_max, nx),
            np.linspace(y_min + dy / 2, y_max - dy / 2, ny - 1),
        )
        crd3 = np.c_[np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))]
        crd = np.vstack((crd, crd2, crd3))
        elm = [
            [
                nx * j + i,
                nx * j + i + 1,
                nx * (j + 1) + i + 1,
                nx * (j + 1) + i,
                nx * ny + (nx - 1) * j + i,
                nx * ny + (nx - 1) * ny + nx * j + i + 1,
                nx * ny + (nx - 1) * (j + 1) + i,
                nx * ny + (nx - 1) * ny + nx * j + i,
            ]
            for j in range(0, ny - 1)
            for i in range(0, nx - 1)
        ]
    elif elm_type == "quad4":
        elm = [
            [nx * j + i, nx * j + i + 1, nx * (j + 1) + i + 1, nx * (j + 1) + i]
            for j in range(ny - 1)
            for i in range(nx - 1)
        ]
    elif elm_type == "quad9":
        elm = [
            [
                nx * j + i,
                nx * j + i + 2,
                nx * (j + 2) + i + 2,
                nx * (j + 2) + i,
                nx * j + i + 1,
                nx * (j + 1) + i + 2,
                nx * (j + 2) + i + 1,
                nx * (j + 1) + i,
                nx * (j + 1) + i + 1,
            ]
            for j in range(0, ny - 2, 2)
            for i in range(0, nx - 2, 2)
        ]
    elif elm_type == "tri3":
        elm = []
        for j in range(ny - 1):
            elm += [
                [nx * j + i, nx * j + i + 1, nx * (j + 1) + i] for i in range(nx - 1)
            ]
            elm += [
                [nx * j + i + 1, nx * (j + 1) + i + 1, nx * (j + 1) + i]
                for i in range(nx - 1)
            ]
    elif elm_type == "tri6":
        elm = []
        for j in range(0, ny - 2, 2):
            elm += [
                [
                    nx * j + i,
                    nx * j + i + 2,
                    nx * (j + 2) + i,
                    nx * j + i + 1,
                    nx * (j + 1) + i + 1,
                    nx * (j + 1) + i,
                ]
                for i in range(0, nx - 2, 2)
            ]
            elm += [
                [
                    nx * j + i + 2,
                    nx * (j + 2) + i + 2,
                    nx * (j + 2) + i,
                    nx * (j + 1) + i + 2,
                    nx * (j + 2) + i + 1,
                    nx * (j + 1) + i + 1,
                ]
                for i in range(0, nx - 2, 2)
            ]

    elm = np.array(elm)

    if elm_type != "quad8":
        N = len(crd)
        node_sets = {
            "bottom": [nd for nd in range(nx)],
            "top": [nd for nd in range(N - nx, N)],
            "left": [nd for nd in range(0, N, nx)],
            "right": [nd for nd in range(nx - 1, N, nx)],
        }
    else:
        node_sets = {}
        print("Warning: no boundary set of nodes defined for quad8 elements")

    return Mesh(crd, elm, elm_type, node_sets, {}, ndim, name)


def grid_mesh_cylindric(
    nr=11,
    nt=11,
    r_min=0,
    r_max=1,
    theta_min=0,
    theta_max=1,
    elm_type="quad4",
    init_rep_loc=0,
    ndim=None,
    name="",
):
    """
    Create a mesh as a regular grid in cylindrical coordinate

    Parameters
    ----------
    nr, nt : int
        Numbers of nodes in the r and theta axes (default = 11).
    x_min, x_max, y_min, y_max : int,float
        The boundary of the square (default : 0, 1, 0, 1).
    elm_type : {'tri3', 'quad4', 'quad8', 'quad9'}
        The type of the element generated (default='quad4')
        * 'tri3' -- 3 node linear triangular mesh
        * 'quad4' -- 4 node quadrangular mesh
        * 'quad8' -- 8 node quadrangular mesh (à tester)
        * 'quad9' -- 9 node quadrangular mesh
    init_rep_loc : {0, 1}
        if init_rep_loc is set to 1, the local basis is initialized with the global basis.

    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.

    See Also
    --------
    line_mesh : 1D mesh of a line
    rectangle_mesh : Surface mesh of a rectangle
    box_mesh : Volume mesh of a box
    line_mesh_cylindric : Line mesh in cylindrical coordinate
    """

    if theta_min < theta_max:
        m = rectangle_mesh(
            nr, nt, r_min, r_max, theta_min, theta_max, elm_type, ndim, name
        )
    else:
        m = rectangle_mesh(
            nr, nt, r_min, r_max, theta_max, theta_min, elm_type, ndim, name
        )

    r = m.nodes[:, 0]
    theta = m.nodes[:, 1]
    crd = np.c_[r * np.cos(theta), r * np.sin(theta)]
    returned_mesh = Mesh(crd, m.elements, elm_type, ndim=ndim, name=name)
    returned_mesh.local_frame = m.local_frame

    if theta_min < theta_max:
        returned_mesh.add_node_set(m.node_sets["left"], "bottom")
        returned_mesh.add_node_set(m.node_sets["right"], "top")
        returned_mesh.add_node_set(m.node_sets["bottom"], "left")
        returned_mesh.add_node_set(m.node_sets["top"], "right")
    else:
        returned_mesh.add_node_set(m.node_sets["left"], "bottom")
        returned_mesh.add_node_set(m.node_sets["right"], "top")
        returned_mesh.add_node_set(m.node_sets["top"], "left")
        returned_mesh.add_node_set(m.node_sets["bottom"], "right")

    return returned_mesh


def line_mesh_1D(n_nodes=11, x_min=0, x_max=1, elm_type="lin2", name=""):
    """
    Create the Mesh of a straight line with corrdinates in 1D.

    Parameters
    ----------
    n_nodes : int
        Numbers of nodes (default = 11).
    x_min, x_max : int,float
        The boundary of the line (default : 0, 1).
    elm_type : {'lin2', 'lin3', 'lin4'}
        The shape of the elements (default='lin2')

        * 'lin2' -- 2 node line
        * 'lin3' -- 3 node line
        * 'lin4' -- 4 node line

    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.

    See Also
    --------
    line_mesh : Mesh of a line whith choosen dimension
    rectangle_mesh : Surface mesh of a rectangle
    box_mesh : Volume mesh of a box
    grid_mesh_cylindric : Surface mesh of a grid in cylindrical coodrinate
    line_mesh_cylindric : Line mesh in cylindrical coordinate
    """
    if elm_type == "lin2":  # 1D element with 2 nodes
        crd = np.c_[np.linspace(x_min, x_max, n_nodes)]  # Nodes coordinates
        elm = np.c_[range(n_nodes - 1), np.arange(1, n_nodes)]  # Elements
    elif elm_type == "lin3":  # 1D element with 3 nodes
        n_nodes = n_nodes // 2 * 2 + 1  # In case N is not initially odd
        crd = np.c_[np.linspace(x_min, x_max, n_nodes)]  # Nodes coordinates
        elm = np.c_[
            np.arange(0, n_nodes - 2, 2),
            np.arange(1, n_nodes - 1, 2),
            np.arange(2, n_nodes, 2),
        ]  # Elements
    elif elm_type == "lin4":
        n_nodes = n_nodes // 3 * 3 + 1
        crd = np.c_[np.linspace(x_min, x_max, n_nodes)]  # Nodes coordinates
        elm = np.c_[
            np.arange(0, n_nodes - 3, 3),
            np.arange(1, n_nodes - 2, 3),
            np.arange(2, n_nodes - 1, 3),
            np.arange(3, n_nodes, 3),
        ]  # Elements

    node_sets = {"left": [0], "right": [n_nodes - 1]}
    return Mesh(crd, elm, elm_type, node_sets, name=name)


def line_mesh(n_nodes=11, x_min=0, x_max=1, elm_type="lin2", ndim=None, name=""):
    """
    Create the Mesh of a straight line

    Parameters
    ----------
    n_nodes : int
        Numbers of nodes (default = 11).
    x_min, x_max : int,float,list
        The boundary of the line as scalar (1D) or list (default : 0, 1).
    elm_type : {'lin2', 'lin3', 'lin4'}
        The shape of the elements (default='lin2')
        * 'lin2' -- 2 node line
        * 'lin3' -- 3 node line
        * 'lin4' -- 4 node line

    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.

    See Also
    --------
    rectangle_mesh : Surface mesh of a rectangle
    box_mesh : Volume mesh of a box
    grid_mesh_cylindric : Surface mesh of a grid in cylindrical coodrinate
    line_mesh_cylindric : Line mesh in cylindrical coordinate
    """
    if np.isscalar(x_min):
        m = line_mesh_1D(n_nodes, x_min, x_max, elm_type, name)
        crd = np.c_[m.nodes, np.zeros((n_nodes, ndim - 1))]
        return Mesh(crd, m.elements, elm_type, m.node_sets, name=name)
    else:
        m = line_mesh_1D(n_nodes, 0.0, 1.0, elm_type, name)
        crd = m.nodes
        crd = (np.array(x_max) - np.array(x_min)) * crd + np.array(x_min)
        return Mesh(crd, m.elements, elm_type, m.node_sets, name=name)


def line_mesh_cylindric(
    nt=11,
    r=1,
    theta_min=0,
    theta_max=3.14,
    elm_type="lin2",
    init_rep_loc=0,
    ndim=None,
    name="",
):
    """
    Create the mesh of a curved line based on cylindrical coordinates

    Parameters
    ----------
    nt : int
        Numbers of nodes along the angular coordinate (default = 11).
    theta_min, theta_max : int,float
        The boundary of the line defined by the angular coordinate (default : 0, 3.14).
    elm_type : {'lin2', 'lin3', 'lin4'}
        The shape of the elements (default='lin2')
        * 'lin2' -- 2 node line
        * 'lin3' -- 3 node line
        * 'lin4' -- 4 node line
    init_rep_loc : {0, 1}
        if init_rep_loc is set to 1, the local frame is initialized with the cylindrical local basis.

    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.

    See Also
    --------
    line_mesh : Mesh of a line whith choosen dimension
    rectangle_mesh : Surface mesh of a rectangle
    box_mesh : Volume mesh of a box
    grid_mesh_cylindric : Surface mesh of a grid in cylindrical coodrinate
    line_mesh_cylindric : Line mesh in cylindrical coordinate
    """
    # init_rep_loc = 1 si on veut initialiser le repère local (0 par défaut)
    m = line_mesh_1D(nt, theta_min, theta_max, elm_type, name)
    theta = m.nodes[:, 0]
    elm = m.elements

    crd = np.c_[r * np.cos(theta), r * np.sin(theta)]

    returned_mesh = Mesh(crd, elm, elm_type, m.node_sets, {}, ndim, name)

    if init_rep_loc:
        local_frame = np.array(
            [[[np.sin(t), -np.cos(t)], [np.cos(t), np.sin(t)]] for t in theta]
        )

    return returned_mesh


def box_mesh(
    nx=11,
    ny=11,
    nz=11,
    x_min=0,
    x_max=1,
    y_min=0,
    y_max=1,
    z_min=0,
    z_max=1,
    elm_type="hex8",
    name="",
):
    """
    Create the mesh of a box

    Parameters
    ----------
    nx, ny, nz : int
        Numbers of nodes in the x, y and z axes (default = 11).
    x_min, x_max, y_min, y_max, z_min, z_max : int,float
        The boundary of the box (default : 0, 1, 0, 1, 0, 1).
    elm_type : {'hex8', 'hex20'}
        The type of the element generated (default='hex8')
        * 'hex8' -- 8 node hexahedron
        * 'hex20' -- 20 node second order hexahedron

    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.

    See Also
    --------
    line_mesh : 1D mesh of a line
    rectangle_mesh : Surface mesh of a rectangle
    grid_mesh_cylindric : Surface mesh of a grid in cylindrical coodrinate
    line_mesh_cylindric : Line mesh in cylindrical coord
    """

    Y, Z, X = np.meshgrid(
        np.linspace(y_min, y_max, ny),
        np.linspace(z_min, z_max, nz),
        np.linspace(x_min, x_max, nx),
    )
    crd = np.c_[np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1)), np.reshape(Z, (-1, 1))]

    if elm_type == "hex20":
        dx = (x_max - x_min) / (nx - 1.0)
        dy = (y_max - y_min) / (ny - 1.0)
        dz = (z_max - z_min) / (nz - 1.0)
        Y, Z, X = np.meshgrid(
            np.linspace(y_min, y_max, ny),
            np.linspace(z_min, z_max, nz),
            np.linspace(x_min + dx / 2.0, x_max + dx / 2.0, nx - 1, endpoint=False),
        )
        crd2 = np.c_[
            np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1)), np.reshape(Z, (-1, 1))
        ]
        Y, Z, X = np.meshgrid(
            np.linspace(y_min, y_max, ny),
            np.linspace(z_min + dz / 2.0, z_max + dz / 2.0, nz - 1, endpoint=False),
            np.linspace(x_min, x_max, nx),
        )
        crd3 = np.c_[
            np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1)), np.reshape(Z, (-1, 1))
        ]
        Y, Z, X = np.meshgrid(
            np.linspace(y_min + dy / 2.0, y_max + dy / 2.0, ny - 1, endpoint=False),
            np.linspace(z_min, z_max, nz),
            np.linspace(x_min, x_max, nx),
        )
        crd4 = np.c_[
            np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1)), np.reshape(Z, (-1, 1))
        ]

        crd = np.vstack((crd, crd2, crd3, crd4))

        elm = [
            [
                nx * j + i + (k * nx * ny),
                nx * j + i + 1 + (k * nx * ny),
                nx * (j + 1) + i + 1 + (k * nx * ny),
                nx * (j + 1) + i + (k * nx * ny),
                nx * j + i + (k * nx * ny) + nx * ny,
                nx * j + i + 1 + (k * nx * ny) + nx * ny,
                nx * (j + 1) + i + 1 + (k * nx * ny) + nx * ny,
                nx * (j + 1) + i + (k * nx * ny) + nx * ny,
                nx * ny * nz + (nx - 1) * j + i + k * (nx - 1) * ny,
                nx * j
                + i
                + 1
                + nx * ny * nz
                + (nx - 1) * ny * nz
                + (nz - 1) * nx * ny
                + (k * nx * (ny - 1)),
                nx * ny * nz + (nx - 1) * (j + 1) + i + k * (nx - 1) * ny,
                nx * j
                + i
                + nx * ny * nz
                + (nx - 1) * ny * nz
                + (nz - 1) * nx * ny
                + (k * nx * (ny - 1)),
                nx * ny * nz + (nx - 1) * ny + (nx - 1) * j + i + k * (nx - 1) * ny,
                nx * j
                + i
                + 1
                + nx * ny * nz
                + (nx - 1) * ny * nz
                + (nz - 1) * nx * ny
                + nx * (ny - 1)
                + (k * nx * (ny - 1)),
                nx * ny * nz
                + (nx - 1) * ny
                + (nx - 1) * (j + 1)
                + i
                + k * (nx - 1) * ny,
                nx * j
                + i
                + nx * ny * nz
                + (nx - 1) * ny * nz
                + (nz - 1) * nx * ny
                + nx * (ny - 1)
                + (k * nx * (ny - 1)),
                nx * ny * nz + (nx - 1) * ny * nz + nx * j + i + k * ny * nx,
                nx * ny * nz + (nx - 1) * ny * nz + nx * j + i + 1 + k * ny * nx,
                nx * ny * nz + (nx - 1) * ny * nz + nx + nx * j + i + 1 + k * ny * nx,
                nx * ny * nz + (nx - 1) * ny * nz + nx + nx * j + i + k * ny * nx,
            ]
            for k in range(nz - 1)
            for j in range(ny - 1)
            for i in range(nx - 1)
        ]

        bottom = (
            [nd for nd in range(ny * nx)]
            + list(range(nx * ny * nz, (nx - 1) * ny + nx * ny * nz))
            + list(
                range(
                    nx * ny * nz + (nx - 1) * ny * nz + (nz - 1) * nx * ny,
                    nx * ny * nz
                    + (nx - 1) * ny * nz
                    + (nz - 1) * nx * ny
                    + (ny - 1) * nx,
                )
            )
        )
        top = (
            [nd for nd in range((nz - 1) * ny * nx, nz * nx * ny)]
            + list(
                range(
                    nx * ny * nz + (nx - 1) * ny * (nz - 1),
                    nx * ny * nz + (nx - 1) * ny * nz,
                )
            )
            + list(
                range(
                    nx * ny * nz
                    + (nx - 1) * ny * nz
                    + (nz - 1) * nx * ny
                    + (ny - 1) * nx * (nz - 1),
                    nx * ny * nz
                    + (nx - 1) * ny * nz
                    + (nz - 1) * nx * ny
                    + (ny - 1) * nx * nz,
                )
            )
        )
        left = (
            list(
                itertools.chain.from_iterable(
                    [range(i * nx * ny, i * nx * ny + nx * ny, nx) for i in range(nz)]
                )
            )
            + list(
                range(
                    nx * ny * nz + (nx - 1) * ny * nz,
                    nx * ny * nz + (nx - 1) * ny * nz + (nz - 1) * nx * ny,
                    nx,
                )
            )
            + list(
                range(
                    nx * ny * nz + (nx - 1) * ny * nz + (nz - 1) * nx * ny,
                    nx * ny * nz
                    + (nx - 1) * ny * nz
                    + (nz - 1) * nx * ny
                    + nx * (ny - 1) * nz,
                    nx,
                )
            )
        )
        right = (
            list(
                itertools.chain.from_iterable(
                    [
                        range(i * nx * ny + nx - 1, i * nx * ny + nx * ny, nx)
                        for i in range(nz)
                    ]
                )
            )
            + list(
                range(
                    nx * ny * nz + (nx - 1) * ny * nz + nx - 1,
                    nx * ny * nz + (nx - 1) * ny * nz + (nz - 1) * nx * ny,
                    nx,
                )
            )
            + list(
                range(
                    nx * ny * nz + (nx - 1) * ny * nz + (nz - 1) * nx * ny + nx - 1,
                    nx * ny * nz
                    + (nx - 1) * ny * nz
                    + (nz - 1) * nx * ny
                    + nx * (ny - 1) * nz,
                    nx,
                )
            )
        )
        front = (
            list(
                itertools.chain.from_iterable(
                    [range(i * nx * ny, i * nx * ny + nx) for i in range(nz)]
                )
            )
            + list(
                itertools.chain.from_iterable(
                    [
                        range(
                            i * (nx - 1) * ny + nx * ny * nz,
                            i * (nx - 1) * ny + nx * ny * nz + nx - 1,
                        )
                        for i in range(nz)
                    ]
                )
            )
            + list(
                itertools.chain.from_iterable(
                    [
                        range(
                            i * nx * ny + nx * ny * nz + (nx - 1) * ny * nz,
                            i * nx * ny + nx * ny * nz + (nx - 1) * ny * nz + nx,
                        )
                        for i in range(nz - 1)
                    ]
                )
            )
        )
        back = (
            list(
                itertools.chain.from_iterable(
                    [
                        range(i * nx * ny + nx * ny - nx, i * nx * ny + nx * ny)
                        for i in range(nz)
                    ]
                )
            )
            + list(
                itertools.chain.from_iterable(
                    [
                        range(
                            i * (nx - 1) * ny + nz * nx * ny + (nx - 1) * (ny - 1),
                            i * (nx - 1) * ny
                            + nz * nx * ny
                            + (nx - 1) * (ny - 1)
                            + nx
                            - 1,
                        )
                        for i in range(nz)
                    ]
                )
            )
            + list(
                itertools.chain.from_iterable(
                    [
                        range(
                            i * nx * ny
                            + nz * nx * ny
                            + (nx - 1) * ny * nz
                            + (ny - 1) * nx,
                            i * nx * ny
                            + nz * nx * ny
                            + (nx - 1) * ny * nz
                            + (ny - 1) * nx
                            + nx,
                        )
                        for i in range(nz - 1)
                    ]
                )
            )
        )

    elif elm_type == "hex8":
        elm = [
            [
                nx * j + i + (k * nx * ny),
                nx * j + i + 1 + (k * nx * ny),
                nx * (j + 1) + i + 1 + (k * nx * ny),
                nx * (j + 1) + i + (k * nx * ny),
                nx * j + i + (k * nx * ny) + nx * ny,
                nx * j + i + 1 + (k * nx * ny) + nx * ny,
                nx * (j + 1) + i + 1 + (k * nx * ny) + nx * ny,
                nx * (j + 1) + i + (k * nx * ny) + nx * ny,
            ]
            for k in range(nz - 1)
            for j in range(ny - 1)
            for i in range(nx - 1)
        ]

        front = list(
            itertools.chain.from_iterable(
                [range(i * nx * ny, i * nx * ny + nx) for i in range(nz)]
            )
        )  # [item for sublist in bas for item in sublist] #flatten a list
        back = list(
            itertools.chain.from_iterable(
                [
                    range(i * nx * ny + nx * ny - nx, i * nx * ny + nx * ny)
                    for i in range(nz)
                ]
            )
        )
        left = list(
            itertools.chain.from_iterable(
                [range(i * nx * ny, i * nx * ny + nx * ny, nx) for i in range(nz)]
            )
        )
        right = list(
            itertools.chain.from_iterable(
                [
                    range(i * nx * ny + nx - 1, i * nx * ny + nx * ny, nx)
                    for i in range(nz)
                ]
            )
        )
        bottom = [nd for nd in range(ny * nx)]
        top = [nd for nd in range((nz - 1) * ny * nx, nz * nx * ny)]

    else:
        raise NameError("Element not implemented. Only support hex8 and hex20 elements")

    N = np.shape(crd)[0]
    elm = np.array(elm)

    node_sets = {
        "right": right,
        "left": left,
        "top": top,
        "bottom": bottom,
        "front": front,
        "back": back,
    }
    return Mesh(crd, elm, elm_type, node_sets, name=name)


if __name__ == "__main__":
    import math

    a = line_mesh_cylindric(11, 1, 0, math.pi, "lin2", init_rep_loc=0)
    b = line_mesh_cylindric(11, 1, 0, math.pi, "lin4", init_rep_loc=1)

    print(b.nodes)
