import numpy as np
from fedoo.core.mesh import Mesh
from fedoo.mesh.simple import line_mesh, line_mesh_cylindric
from fedoo.mesh.functions import quad2tri, change_elm_type


def structured_mesh_2D(
    data, edge1, edge2, edge3, edge4, elm_type="quad4", method=0, ndim=None, name=""
):
    """Create a 2D structured grid from 4 edges.

    This function build a 2D structured mesh from the ordered position of nodes
    defining the 4 edges.

    Parameters
    ----------
    data: Mesh|np.ndarray[float]
        may be a fedoo Mesh object or a numpy array containing the node
        coordinates.
    edge1, edge2, edge3, edge4: list[int]|np.ndarray[int]
        ordered node indices of the 4 edges defining the structured surface.
        The node indices should be given considering the following rules:

        - the node indices must be sorted from the first node on the edge to
          the last in the order in which they are encountered.
        - edges 1 and 3 are opposite faces and must have the same length
          (same number of nodes).
        - edge2 and edge4 must also have the same length.
        - last node of edge1 should be the first of edge2, last node of
          edge2 should be the first of edge3 and so on...

    elm_type: {'quad4', 'quad9', 'tri3', 'tri6'}, default = 'quad4'
        The type of the element generated.
    method: int, default = 0
        The method used to generate inside nodes:

        - if method == 0: intesection of lines drawn between nodes of oposite edges.
        - if method == 1: nodes are regularly distributed between edge1 and edge3
          and moved in the perpendicular direction to be as closed as possible
          to a regular distribution between edge2 and edge4
        - if method == 2: nodes are regularly distributed between edge1 and edge3
          without correction for the perpendicular direction.
        - if method == 3: nodes are regularly distributed between edge2 and edge4
          without correction for the perpendicular direction.

    ndim: int, optional
        dimension of the generated mesh. Default = assert from the given
        data.
    name: str, optional
        Name of the returned mesh. If "" and data is a Mesh, the name is the same
        as data.name.
    """
    if hasattr(data, "elm_type"):  # data is a mesh
        if data.elements is None:
            elm = []
        else:
            elm = list(data.elements)
            elm_type = data.elm_type
        crd = data.nodes
        if name == "":
            name = data.name
    else:
        elm = []
        crd = data

    edge3 = edge3[::-1]
    edge4 = edge4[::-1]
    x1 = crd[edge1]
    x2 = crd[edge2]
    x3 = crd[edge3]
    x4 = crd[edge4]
    new_crd = list(crd.copy())
    grid = np.empty((len(x1), len(x2)))
    grid[0, :] = edge4
    grid[-1, :] = edge2
    grid[:, 0] = edge1
    grid[:, -1] = edge3

    N = len(new_crd)
    coef1 = np.linspace(0, 1, len(x1)).reshape(-1, 1)
    coef2 = np.linspace(0, 1, len(x2)).reshape(-1, 1)

    for i in range(1, len(x1) - 1):
        if method == 0:
            # intesection of lines drawn between nodes of oposite edges
            pos = (
                (x1[i, 0] * x3[i, 1] - x1[i, 1] * x3[i, 0]) * (x2 - x4)
                - (x1[i] - x3[i])
                * (x2[:, 0] * x4[:, 1] - x2[:, 1] * x4[:, 0]).reshape(-1, 1)
            ) / (
                (x1[i, 0] - x3[i, 0]) * (x2[:, 1] - x4[:, 1])
                - (x1[i, 1] - x3[i, 1]) * (x2[:, 0] - x4[:, 0])
            ).reshape(-1, 1)
            # px= ( (x1[i]*y3[i]-y1[i]*x3[i])*(x2-x4)-(x1[i]-x3[i])*(x2*y4-y2*x4) ) / ( (x1[i]-x3[i])*(y2-y4)-(y1[i]-y3[i])*(x2-x4) )
            # py= ( (x1[i]*y3[i]-y1[i]*x3[i])*(y2-y4)-(y1[i]-y3[i])*(x2*y4-y2*x4) ) / ( (x1[i]-x3[i])*(y2-y4)-(y1[i]-y3[i])*(x2-x4) )
        elif method == 1:
            # nodes are regularly distributed between edge1 and  edge3 and moved in the perpendicular direction to be
            # as closed as possible to a regular distribution between edge2 and edge4
            if i == 1:
                # pos = x4
                vec = (x2 - pos) / np.linalg.norm(x2 - pos, axis=1).reshape(-1, 1)

            # prediction of position
            pos1 = x1[i] * (1 - coef2) + x3[i] * coef2
            # uncentainty of position (pos1 - pos2) where pos2 is the position usging another method
            dpos = (
                x2 * (coef1[i]) + x4 * (1 - coef1[i]) - pos1
            )  # uncertainty about the best node position
            # direction vector normalized (pos1-pos_old)
            # vec = (x2 - pos)/np.linalg.norm(x2 - pos, axis=1).reshape(-1,1)
            # pos is modified only in the direction vec
            pos = np.sum(vec * dpos, axis=1).reshape(-1, 1) * vec + pos1
        elif method == 2:
            # nodes are regularly distributed between edge1 and edge3
            pos = x1[i] * (1 - coef2) + x3[i] * coef2
        elif method == 3:
            # nodes are regularly distributed between edge 2 and edge4
            pos = x2 * (coef1[i]) + x4 * (1 - coef1[i])

        new_crd += list(pos[1:-1])
        # new_crd += list(np.c_[px[1:-1],py[1:-1]])
        grid[i, 1:-1] = np.arange(N, len(new_crd), 1)
        N = len(new_crd)

    nx = grid.shape[0]
    ny = grid.shape[1]

    if elm_type == "quad4":
        elm += [
            [grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]]
            for j in range(ny - 1)
            for i in range(nx - 1)
        ]
    elif elm_type == "quad9":
        elm += [
            [
                grid[i, j],
                grid[i + 2, j],
                grid[i + 2, j + 2],
                grid[i, j + 2],
                grid[i + 1, j],
                grid[i + 2, j + 1],
                grid[i + 1, j + 2],
                grid[i, j + 1],
                grid[i + 1, j + 1],
            ]
            for j in range(0, ny - 2, 2)
            for i in range(0, nx - 2, 2)
        ]
    elif elm_type == "tri3":
        for j in range(ny - 1):
            elm += [[grid[i, j], grid[i + 1, j], grid[i, j + 1]] for i in range(nx - 1)]
            elm += [
                [grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]]
                for i in range(nx - 1)
            ]
    elif elm_type == "tri6":
        for j in range(0, ny - 2, 2):
            elm += [
                [
                    grid[i, j],
                    grid[i + 2, j],
                    grid[i, j + 2],
                    grid[i + 1, j],
                    grid[i + 1, j + 1],
                    grid[i, j + 1],
                ]
                for i in range(0, nx - 2, 2)
            ]
            elm += [
                [
                    grid[i + 2, j],
                    grid[i + 2, j + 2],
                    grid[i, j + 2],
                    grid[i + 2, j + 1],
                    grid[i + 1, j + 2],
                    grid[i + 1, j + 1],
                ]
                for i in range(0, nx - 2, 2)
            ]
    else:
        raise NameError("'{}' elements are not implemented".format(elm_type))

    elm = np.array(elm, dtype=int)
    return Mesh(np.array(new_crd), elm, elm_type, ndim=ndim, name=name)


def generate_nodes(mesh, N, data, type_gen="straight"):
    """
    Add regularly espaced nodes to an existing mesh between to existing nodes.

    This function serves to generated structured meshes.
    To create a 2D stuctured mesh:
        - Create and mesh with only sigular nodes that will serve to build the edges
        - Use the generate_nodes functions to add some nodes to the edge
        - Use the structured_mesh_2D from the set of nodes corresponding the egdes to build the final mesh.

    Parameters
    ----------
    mesh: Mesh
        the existing mesh
    N: int
        Number of generated nodes.
    data: list or tuple
        if type_gen == 'straight', data should contain the indices of the starting (data[0]) and ending (data[1]).
        if type_gen == 'circular', data should contain the indices
            of the starting (data[0]) and ending (data[1]) nodes and the coordinates of the center of the circle
            (data[2]). The nodes are generated using a trigonometric rotation.
    type_gen: str in {'straight', 'circular'}
        Type of line generated. The default is 'straight'.

    Returns
    -------
    np.ndarray[int]
        array containing indices of the new generated nodes

    """
    # if type_gen == 'straight' -> data = (node1, node2)
    # if type_gen == 'circular' -> data = (node1, node2, (center_x, center_y))
    crd = mesh.nodes
    if type_gen == "straight":
        node1 = data[0]
        node2 = data[1]
        listNodes = mesh.add_nodes(line_mesh(N, crd[node1], crd[node2]).nodes[1:-1])
        return np.array([node1] + list(listNodes) + [node2])
    if type_gen == "circular":
        nd1 = data[0]
        nd2 = data[1]
        c = data[2]
        c = np.array(c)
        R = np.linalg.norm(crd[nd1] - c)
        assert (
            np.abs(R - np.linalg.norm(crd[nd2] - c)) < R * 1e-4
        ), "Final nodes is not on the circle"
        # (crd[nd1]-c)
        theta_min = np.arctan2(crd[nd1, 1] - c[1], crd[nd1, 0] - c[0])
        theta_max = np.arctan2(crd[nd2, 1] - c[1], crd[nd2, 0] - c[0])
        # print(theta_min)
        # print(theta_max)
        if theta_max <= theta_min:
            theta_max += 2 * np.pi
        m = line_mesh_cylindric(N, R, theta_min, theta_max)  # circular mesh
        listNodes = mesh.add_nodes(m.nodes[1:-1] + c)
        return np.array([nd1] + list(listNodes) + [nd2])


def hole_plate_mesh(
    nr=11,
    nt=11,
    length=100,
    height=100,
    radius=20,
    elm_type="quad4",
    sym=False,
    include_node_sets=True,
    ndim=None,
    name="",
):
    """
    Create a mesh of a 2D plate with a hole

    Parameters
    ----------
    nr, nt: int
        Numbers of nodes in the radial and tangent direction from the hole (default = 11).
        nt is the number of nodes of the half of an exterior edge
    length, height : int,float
        The length and height of the plate (default : 100).
    radius : int, float, tuple
        The radius of the hole (default : 20).
        If tuple = (a,b), a and b are the ellipse radius along x and y axis.

    elm_type: {'quad4', 'quad9', 'tri3', 'tri6', 'quad8'}
        The type of the element generated (default='quad4')
    Sym: bool
        Sym = True, if only the returned mesh assume symetric condition and
        only the quarter of the plate is returned (default=False)
    include_node_sets : bool
        if True (default), the boundary nodes are included in the mesh node_sets dict.
    ndim: int, optional
        dimension of the generated mesh. By default, the returned mesh will be in 2d.
    name: str, optional
        Name of the returned mesh.

    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.

    See Also
    --------
    line_mesh: 1D mesh of a line
    rectangle_mesh: Surface mesh of a rectangle
    """
    if elm_type in ["quad9", "tri6"]:
        nr = nr // 2 * 2 + 1  # in case nr is not initially odd
        nt = nt // 2 * 2 + 1  # in case nt is not initially odd
    elif elm_type == "quad8":
        return change_elm_type(
            hole_plate_mesh(
                nr,
                nt,
                length,
                height,
                radius,
                "quad9",
                sym,
                include_node_sets,
                ndim,
                name,
            ),
            "quad8",
        )
    elif elm_type not in ["quad4", "tri3"]:
        raise NameError("Non compatible element shape")

    if isinstance(radius, tuple):
        ellipse = True
        ellipse_radius = np.array(radius)
        radius = 1
    else:
        ellipse = False

    L = length / 2
    h = height / 2
    m = Mesh(
        np.array(
            [
                [radius, 0],
                [L, 0],
                [L, h],
                [0, h],
                [0, radius],
                [radius * np.cos(np.pi / 4), radius * np.sin(np.pi / 4)],
            ]
        )
    )
    edge4 = generate_nodes(m, nt, (0, 5, (0, 0)), type_gen="circular")[::-1]
    edge7 = generate_nodes(m, nt, (5, 4, (0, 0)), type_gen="circular")
    if ellipse:
        m.nodes[edge4] *= ellipse_radius
        m.nodes[edge7[1:]] *= ellipse_radius

    edge1 = generate_nodes(m, nr, (0, 1))
    edge2 = generate_nodes(m, nt, (1, 2))
    edge3 = generate_nodes(m, nr, (2, 5))

    edge5 = generate_nodes(m, nr, (4, 3))
    edge6 = generate_nodes(m, nt, (3, 2))

    m = structured_mesh_2D(m, edge1, edge2, edge3, edge4, elm_type=elm_type, method=3)
    m = structured_mesh_2D(
        m, edge5, edge6, edge3, edge7, elm_type=elm_type, method=3, ndim=ndim, name=name
    )

    if sym:
        if include_node_sets:
            m.node_sets.update(
                {
                    "hole_edge": list(edge4[:0:-1]) + list(edge7),
                    "right": list(edge2),
                    "left_sym": list(edge5),
                    "top": list(edge6),
                    "bottom_sym": list(edge1),
                }
            )
    else:
        nnd = m.n_nodes
        crd = m.nodes.copy()
        crd[:, 0] = -m.nodes[:, 0]
        m2 = Mesh(crd, m.elements, m.elm_type)
        m = Mesh.stack(m, m2)

        crd = m.nodes.copy()
        crd[:, 1] = -m.nodes[:, 1]
        m2 = Mesh(crd, m.elements, m.elm_type)
        m = Mesh.stack(m, m2, name=name)

        if include_node_sets:
            m.node_sets["top"] = list((edge6 + nnd)[:0:-1]) + list(edge6)
            m.node_sets["bottom"] = list((edge6 + 3 * nnd)[:0:-1]) + list(
                edge6 + 2 * nnd
            )
            m.node_sets["right"] = list((edge2 + 2 * nnd)[:0:-1]) + list(edge2)
            m.node_sets["left"] = list((edge2 + 3 * nnd)[:0:-1]) + list(edge2 + nnd)
            edge_hole = np.hstack((edge4[:0:-1], edge7))
            m.node_sets["hole_edge"] = (
                list(edge_hole)
                + list(edge_hole[-2::-1] + nnd)
                + list(edge_hole[1:] + 3 * nnd)
                + list(edge_hole[-2:0:-1] + 2 * nnd)
            )

        node_to_merge = np.vstack(
            (
                np.c_[edge5, edge5 + nnd],
                np.c_[edge5 + 2 * nnd, edge5 + 3 * nnd],
                np.c_[edge1, edge1 + 2 * nnd],
                np.c_[edge1 + nnd, edge1 + 3 * nnd],
            )
        )

        m.merge_nodes(node_to_merge)

    return m


def disk_mesh(radius=1.0, nr=11, nt=11, elm_type="quad4", ndim=None, name=""):
    """
    Create a surface mesh of a disk or an ellipse

    Parameters
    ----------
    radius : float, tuple
        The radius of the disk (default : 1).
        If tuple = (a,b), a and b are the ellipse radius along x and y axis.
    nr, nt: int, default = 11.
        number of nodes in the radial (nr) and tangential (nt) directions used
        to build the mesh. nr and nt are not the total nodes in each direction.
    elm_type: {'quad4', 'quad9', 'tri3', 'tri6', 'quad8'}
        The type of the element generated (default='quad4')
    ndim: int, optional
        dimension of the generated mesh. By default, the returned mesh will be in 2d.
    name: str, optional
        Name of the returned mesh.

    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.

    See Also
    --------
    hole_plate_mesh: Mesh of a plate with a hole
    hollow_disk_mesh: Mesh of a hollow disk
    """
    if elm_type == "quad8":
        return change_elm_type(
            disk_mesh(radius, nr, nt, "quad9", ndim, name),
            "quad8",
        )

    if isinstance(radius, tuple):
        ellipse = True
        ellipse_radius = np.array(radius)
        radius = 1
    else:
        ellipse = False

    m = hole_plate_mesh(nr, nt, 0.5 * radius, 0.5 * radius, radius, elm_type)
    hole_edge = m.node_sets["hole_edge"]

    m = structured_mesh_2D(
        m,
        m.node_sets["right"],
        m.node_sets["top"][::-1],
        m.node_sets["left"][::-1],
        m.node_sets["bottom"],
        elm_type,
        ndim=ndim,
        name=name,
    )
    m.node_sets = {"boundary": hole_edge}

    if ellipse:
        m.nodes *= ellipse_radius

    return m


def hollow_disk_mesh(
    radius=1.0, thickness=0.1, nr=5, nt=41, elm_type="quad4", ndim=None, name=""
):
    """
    Create a surface mesh of an hollow disk or an ellipse

    Parameters
    ----------
    radius: float, default = 1
        The radius of the disk (default : 1).
    thickness: float, default = 0.1
        The thickness of the hollow disk
    nr, nt: int, default: nr = 5, nt=41.
        number of nodes in the radial (nr) and tangential (nt) directions.
    elm_type: {'quad4', 'quad9', 'tri3', 'tri6'}
        The type of the element generated (default='quad4')
    ndim: int, optional
        dimension of the generated mesh. By default, the returned mesh will be in 2d.
    name: str, optional
        Name of the returned mesh.

    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.

    See Also
    --------
    rectangle_mesh: Mesh of a retangle
    disk_mesh: Mesh of a disk or an ellipse
    """
    r_int = radius - thickness  # intern radius
    assert r_int > 0, "thickness should be lower than radius"

    m = Mesh(np.array([[radius, 0], [r_int, 0]]))

    nt = nt // 4 * 4 + 1  # to ensure a symetric mesh about x and y axes

    edge_ext = generate_nodes(m, nt, (0, 0, (0, 0)), type_gen="circular")
    edge_int = generate_nodes(m, nt, (1, 1, (0, 0)), type_gen="circular")[
        ::-1
    ]  # clockwise orientation
    edge_radius = generate_nodes(m, nr, (0, 1))

    m = structured_mesh_2D(
        m,
        edge_ext,
        edge_radius,
        edge_int,
        edge_radius[::-1],
        elm_type=elm_type,
        method=2,
    )
    return m


def I_shape_mesh(
    height=10,
    width=10,
    web_thickness=2,
    flange_thickness=2,
    size_elm=0.2,
    elm_type="quad4",
    ndim=None,
    name="",
):
    """
    Create the mesh of a I.

    Parameters
    ----------
    height: float, default = 10
        Total height of the I
    width: float, default = 10
        Total width of the I
    web_thickness: float, default = 2
        Web thickness
    flange_thickness: float, default = 2
        Flange thickness
    size_elm, default = 0.2
        Size of the edge of an element.
    elm_type: {'quad4', 'quad9', 'tri3', 'tri6'}
        The type of the element generated (default='quad4')
    ndim: int, optional
        dimension of the generated mesh. By default, the returned mesh will be in 2d.
    name: str, optional
        Name of the returned mesh.

    Returns
    -------
    Mesh
        The generated geometry in Mesh format. See the Mesh class for more details.

    See Also
    --------
    rectangle_mesh: Mesh of a retangle
    disk_mesh: Mesh of a disk or an ellipse
    """
    a = web_thickness / 2
    b = height / 2 - flange_thickness

    m = Mesh(
        np.array(
            [
                [-a, -b],
                [a, -b],
                [a, b],
                [-a, b],
                [-a, height / 2],
                [a, height / 2],
                [-a, -height / 2],
                [a, -height / 2],
                [-width / 2, height / 2],
                [-width / 2, b],
                [width / 2, height / 2],
                [width / 2, b],
                [-width / 2, -height / 2],
                [-width / 2, -b],
                [width / 2, -height / 2],
                [width / 2, -b],
            ]
        )
    )

    nx_web = int(2 * a // size_elm)
    ny_web = int(2 * b // size_elm)
    edges = [
        generate_nodes(m, nx_web, (0, 1)),
        generate_nodes(m, ny_web, (1, 2)),
        generate_nodes(m, nx_web, (2, 3)),
        generate_nodes(m, ny_web, (3, 0)),
    ]
    m = structured_mesh_2D(m, edges[0], edges[1], edges[2], edges[3], elm_type=elm_type)

    ny_flange = int(flange_thickness // size_elm)
    edges_top = [
        edges[2][::-1],  # (3,2)
        generate_nodes(m, ny_flange, (2, 5)),
        generate_nodes(m, nx_web, (5, 4)),
        generate_nodes(m, ny_flange, (4, 3)),
    ]
    m = structured_mesh_2D(
        m, edges_top[0], edges_top[1], edges_top[2], edges_top[3], elm_type=elm_type
    )

    edges_bottom = [
        edges[0],  # (0,1)
        generate_nodes(m, ny_flange, (1, 7)),
        generate_nodes(m, nx_web, (7, 6)),
        generate_nodes(m, ny_flange, (6, 0)),
    ]
    m = structured_mesh_2D(
        m,
        edges_bottom[0],
        edges_bottom[1],
        edges_bottom[2],
        edges_bottom[3],
        elm_type=elm_type,
    )

    nx_flange = int((width - web_thickness) // (2 * size_elm))
    edges_topl = [
        edges_top[3][::-1],  # (3,4)
        generate_nodes(m, nx_flange, (4, 8)),
        generate_nodes(m, ny_flange, (8, 9)),
        generate_nodes(m, nx_flange, (9, 3)),
    ]
    m = structured_mesh_2D(
        m, edges_topl[0], edges_topl[1], edges_topl[2], edges_topl[3], elm_type=elm_type
    )

    edges_topl = [
        edges_top[1],  # (2,5)
        generate_nodes(m, nx_flange, (5, 10)),
        generate_nodes(m, ny_flange, (10, 11)),
        generate_nodes(m, nx_flange, (11, 2)),
    ]
    m = structured_mesh_2D(
        m, edges_topl[0], edges_topl[1], edges_topl[2], edges_topl[3], elm_type=elm_type
    )

    edges_topb = [
        edges_bottom[1],  # (1,7)
        generate_nodes(m, nx_flange, (7, 14)),
        generate_nodes(m, ny_flange, (14, 15)),
        generate_nodes(m, nx_flange, (15, 1)),
    ]
    m = structured_mesh_2D(
        m, edges_topb[0], edges_topb[1], edges_topb[2], edges_topb[3], elm_type=elm_type
    )

    edges_topb = [
        edges_bottom[3],  # (6,0)
        generate_nodes(m, nx_flange, (0, 13)),
        generate_nodes(m, ny_flange, (13, 12)),
        generate_nodes(m, nx_flange, (12, 6)),
    ]
    m = structured_mesh_2D(
        m, edges_topb[0], edges_topb[1], edges_topb[2], edges_topb[3], elm_type=elm_type
    )

    return m
