"""Connected exterior surface and feature-edge selection without visualization."""

from collections import defaultdict
from numbers import Integral

import numpy as np

from fedoo.lib_elements.element_list import get_node_elm_coordinates


class AmbiguousSurfaceSeed(ValueError):
    """A seed touches distinct patches; choose one of ``face_ids`` explicitly."""

    def __init__(self, face_ids, normals):
        self.face_ids = np.asarray(face_ids, dtype=int)
        self.normals = np.asarray(normals)
        super().__init__(
            "The seed touches several surface patches. Choose a seed_face from "
            f"{self.face_ids.tolist()} (normals: {self.normals.tolist()})."
        )


def _face_templates(block, boundary_edges=None):
    kind = block.elm_type
    if kind.startswith("tet"):
        corners = [[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]]
        base = "tet10" if block.n_elm_nodes == 10 else "tet4"
    elif kind.startswith("hex"):
        corners = [
            [0, 1, 2, 3],
            [5, 4, 7, 6],
            [0, 4, 5, 1],
            [2, 6, 7, 3],
            [1, 5, 6, 2],
            [4, 0, 3, 7],
        ]
        base = "hex20" if block.n_elm_nodes == 20 else "hex8"
    elif kind.startswith("wed"):
        corners = [[0, 1, 2], [3, 4, 5], [0, 3, 4, 1], [1, 4, 5, 2], [2, 5, 3, 0]]
        base = kind
    elif kind.startswith(("tri", "quad")):
        count = 3 if kind.startswith("tri") else 4
        if block.nodes.shape[1] == 3 and not boundary_edges:
            return [(list(range(count)), list(range(block.n_elm_nodes)))], False
        corners = [[i, (i + 1) % count] for i in range(count)]
        base = ("tri" if count == 3 else "quad") + str(block.n_elm_nodes)
        if count == 3 and block.n_elm_nodes == 4:
            base = "tri3bubble"
    else:
        raise NotImplementedError(f"Surface selection does not support {kind!r}")
    try:
        reference = get_node_elm_coordinates(base)
    except (NameError, KeyError) as exc:
        raise NotImplementedError(f"No surface topology for {kind!r}") from exc
    if reference is None or len(reference) != block.n_elm_nodes:
        raise NotImplementedError(f"No surface topology for {kind!r}")
    templates = []
    for face in corners:
        points = reference[face]
        delta = reference - points[0]
        if len(face) == 2:
            tangent = points[1] - points[0]
            distances = delta[:, 0] * tangent[1] - delta[:, 1] * tangent[0]
        else:
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            distances = delta @ normal
        all_nodes = np.flatnonzero(np.isclose(distances, 0, atol=1e-12)).tolist()
        templates.append((face, all_nodes))
    return templates, True


class _SurfaceGraph:
    def __init__(self, mesh, boundary_edges=None):
        coordinates = np.asarray(mesh.nodes, dtype=float)
        coordinates = np.pad(coordinates, ((0, 0), (0, 3 - coordinates.shape[1])))
        occurrences = defaultdict(list)
        offset = 0
        for block in getattr(mesh, "submeshes", (mesh,)):
            templates, outward = _face_templates(block, boundary_edges)
            centers = coordinates[block.elements].mean(axis=1)
            for corners, all_nodes in templates:
                faces = block.elements[:, corners]
                complete = block.elements[:, all_nodes]
                points = coordinates[faces]
                if len(corners) == 2:
                    tangent = points[:, 1] - points[:, 0]
                    parent_points = coordinates[block.elements[:, :3]]
                    plane_normal = np.cross(
                        parent_points[:, 1] - parent_points[:, 0],
                        parent_points[:, 2] - parent_points[:, 0],
                    )
                    normals = np.cross(tangent, plane_normal)
                else:
                    # Translate first to keep the cross products accurate far
                    # from the origin (Newell's polygon normal).
                    shifted = points - points[:, :1]
                    normals = np.cross(shifted, np.roll(shifted, -1, axis=1)).sum(
                        axis=1
                    )
                magnitudes = np.linalg.norm(normals, axis=1)
                invalid = np.flatnonzero((magnitudes == 0) | ~np.isfinite(magnitudes))
                if len(invalid):
                    raise ValueError(
                        f"Degenerate surface face on element {offset + invalid[0]}"
                    )
                normals /= magnitudes[:, None]
                if outward:
                    reverse = (
                        np.einsum("ij,ij->i", normals, centers - points.mean(axis=1))
                        > 0
                    )
                    normals[reverse] *= -1
                for local_id, (face, nodes, normal) in enumerate(
                    zip(faces, complete, normals)
                ):
                    key = (outward, tuple(sorted(face)))
                    occurrences[key].append(
                        (face, nodes, offset + local_id, normal, outward)
                    )
            offset += block.n_elements
        # Count across submeshes too: shared volume faces are not exterior.
        self.faces = []
        for entries in occurrences.values():
            if not entries[0][4] or len(entries) == 1:
                self.faces.extend(entries)
        incidence = defaultdict(list)
        for index, (corners, _, _, _, _) in enumerate(self.faces):
            if len(corners) == 2:
                for node in corners:
                    incidence[(int(node),)].append((index, None))
            else:
                for a, b in zip(corners, np.roll(corners, -1)):
                    incidence[tuple(sorted((int(a), int(b))))].append(
                        (index, int(a) < int(b))
                    )
        self.neighbors = [[] for _ in self.faces]
        for entries in incidence.values():
            # Stop at open boundaries and non-manifold junctions.
            if len(entries) != 2:
                continue
            (a, direction_a), (b, direction_b) = entries
            face_a, face_b = self.faces[a], self.faces[b]
            cosine = np.dot(face_a[3], face_b[3])
            if not face_a[4] and not face_b[4] and direction_a == direction_b:
                cosine = -cosine  # tolerate reversed shell-element winding
            self.neighbors[a].append((b, cosine))
            self.neighbors[b].append((a, cosine))

    def grow(self, seed_face, angle):
        threshold = np.cos(np.deg2rad(angle)) - 1e-12
        visited, pending = {seed_face}, [seed_face]
        while pending:
            current = pending.pop()
            for neighbor, cosine in self.neighbors[current]:
                if neighbor not in visited and cosine >= threshold:
                    visited.add(neighbor)
                    pending.append(neighbor)
        return visited


def find_surface_nodes(
    mesh, seed, seed_type="Node", angle=30.0, *, seed_face=None, boundary_edges=None
):
    """Find the nodes of the connected exterior patch containing a seed.

    Parameters
    ----------
    mesh : Mesh or MultiMesh
        Reference geometry. Solid faces shared by cells (including across
        submeshes) are removed. Three-dimensional tri/quad meshes are treated
        as shells; two-dimensional tri/quad meshes expose their boundary edges.
    seed : int
        Node ID or global element ID, according to ``seed_type``.
    seed_type : {"Node", "Element"}
        Type of the starting entity. Interior entities cannot seed a patch.
    angle : float, default 30
        Maximum angle in degrees between adjacent face normals, in [0, 180].
        Zero joins coplanar neighbors; 90 also crosses right-angle corners.
        The comparison is local, so a smoothly curved surface may wrap around.
    seed_face : int, optional
        Exterior face ID used to resolve an ambiguous seed. If candidate faces
        belong to distinct patches, ``AmbiguousSurfaceSeed`` exposes their IDs
        and normals. IDs refer to this mesh and are stable until topology changes.
    boundary_edges : bool, optional
        Select boundary lines of tri/quad meshes even when coordinates have
        three components. By default, XYZ tri/quad meshes expose their faces.
        This option supports planar 2D meshes saved with XYZ coordinates.

    Returns
    -------
    numpy.ndarray
        Sorted unique node IDs, including midside nodes. On solid/2D meshes,
        pass these to ``extract_surface(mesh, node_set=ids)``. On a shell mesh,
        select its elements using ``mesh.get_elements_from_nodes(ids)`` instead.

    Notes
    -----
    Faces must share complete corner edges, not just one vertex. Traversal
    stops at non-manifold junctions and uses corner geometry for curved faces.
    Supported families: tet, hex, wedge, tri and quad. Coincident unmerged
    nodes are not connected. No mesh or named set is modified.
    """
    if seed_type not in ("Node", "Element"):
        raise ValueError("seed_type must be Node or Element")
    count = mesh.n_nodes if seed_type == "Node" else mesh.n_elements
    if (
        not isinstance(seed, Integral)
        or isinstance(seed, bool)
        or not 0 <= seed < count
    ):
        raise ValueError(f"{seed_type} seed ID outside the mesh")
    if not np.isscalar(angle) or not np.isfinite(angle) or not 0 <= angle <= 180:
        raise ValueError("angle must be finite and between 0 and 180 degrees")
    graph = _SurfaceGraph(mesh, boundary_edges)
    candidates = [
        index
        for index, face in enumerate(graph.faces)
        if (seed in face[1] if seed_type == "Node" else seed == face[2])
    ]
    if not candidates:
        raise ValueError("The seed does not belong to an exterior face")
    if seed_face is not None:
        if not isinstance(seed_face, Integral) or seed_face not in candidates:
            raise ValueError("seed_face must be an exterior face incident to the seed")
        selected = graph.grow(int(seed_face), angle)
    else:
        selected = graph.grow(candidates[0], angle)
        if not set(candidates).issubset(selected):
            raise AmbiguousSurfaceSeed(
                candidates, [graph.faces[i][3] for i in candidates]
            )
    return np.unique(np.concatenate([graph.faces[i][1] for i in sorted(selected)]))


class AmbiguousEdgeSeed(ValueError):
    """The seed touches separate edge chains; choose one of ``edge_ids``."""

    def __init__(self, edge_ids, endpoints, directions):
        self.edge_ids = np.asarray(edge_ids, dtype=int)
        self.endpoints = np.asarray(endpoints, dtype=int)
        self.directions = np.asarray(directions, dtype=float)
        super().__init__(
            f"The seed touches several exterior edges. Choose seed_edge from {self.edge_ids.tolist()}."
        )


def _reference_nodes(block):
    kind, count = block.elm_type, block.n_elm_nodes
    if kind.startswith("hex"):
        kind = "hex20" if count == 20 else "hex8"
    elif kind.startswith("tet"):
        kind = "tet10" if count == 10 else "tet4"
    elif kind.startswith("tri"):
        kind = "tri3bubble" if count == 4 else f"tri{count}"
    elif kind.startswith("quad"):
        kind = f"quad{count}"
    reference = get_node_elm_coordinates(kind)
    if reference is None or len(reference) != count:
        raise NotImplementedError(f"No edge topology for {block.elm_type!r}")
    return reference


class _FeatureEdges:
    def __init__(self, mesh, feature_angle):
        surface = _SurfaceGraph(mesh)
        blocks = getattr(mesh, "submeshes", (mesh,))
        ends = np.cumsum([block.n_elements for block in blocks])
        references = [_reference_nodes(block) for block in blocks]
        incidence = defaultdict(list)
        for corners, _, parent, normal, outward in surface.faces:
            if len(corners) < 3:
                raise ValueError(
                    "Exterior edge selection requires three-dimensional faces"
                )
            block_id = int(np.searchsorted(ends, parent, side="right"))
            local = parent - (ends[block_id - 1] if block_id else 0)
            element = blocks[block_id].elements[local]
            reference = references[block_id]
            ranks = {int(node): index for index, node in enumerate(element)}
            for a, b in zip(corners, np.roll(corners, -1)):
                start = reference[ranks[int(a)]]
                tangent = reference[ranks[int(b)]] - start
                delta = reference - start
                parameter = delta @ tangent / np.dot(tangent, tangent)
                residual = delta - parameter[:, None] * tangent
                on_edge = (
                    (np.linalg.norm(residual, axis=1) < 1e-12)
                    & (parameter >= -1e-12)
                    & (parameter <= 1 + 1e-12)
                )
                key = tuple(sorted((int(a), int(b))))
                incidence[key].append(
                    (normal, outward, int(a) < int(b), element[on_edge])
                )
        self.edges = []
        limit = np.cos(np.deg2rad(feature_angle)) + 1e-12
        for endpoints, entries in incidence.items():
            if len(entries) == 2:
                first, second = entries
                cosine = float(np.dot(first[0], second[0]))
                if not first[1] and not second[1] and first[2] == second[2]:
                    cosine = -cosine
                if cosine > limit:
                    continue  # smooth surface tessellation edge
            elif len(entries) != 1:
                continue  # non-manifold edge
            nodes = np.unique(np.concatenate([entry[3] for entry in entries]))
            if np.linalg.norm(mesh.nodes[endpoints[1]] - mesh.nodes[endpoints[0]]) == 0:
                raise ValueError("Degenerate exterior edge")
            self.edges.append((endpoints, nodes))
        vertices = defaultdict(list)
        for index, (endpoints, _) in enumerate(self.edges):
            for node in endpoints:
                vertices[node].append(index)
        self.neighbors = [[] for _ in self.edges]
        for vertex, attached in vertices.items():
            if len(attached) != 2:
                continue  # stop at a branch or endpoint
            a, b = attached
            other_a = next(node for node in self.edges[a][0] if node != vertex)
            other_b = next(node for node in self.edges[b][0] if node != vertex)
            first = mesh.nodes[other_a] - mesh.nodes[vertex]
            second = mesh.nodes[other_b] - mesh.nodes[vertex]
            lengths = np.linalg.norm(first) * np.linalg.norm(second)
            if lengths == 0:
                raise ValueError("Degenerate exterior edge")
            cosine = -float(np.dot(first, second)) / lengths
            self.neighbors[a].append((b, cosine))
            self.neighbors[b].append((a, cosine))

    def grow(self, seed_edge, angle):
        selected, pending = {seed_edge}, [seed_edge]
        limit = np.cos(np.deg2rad(angle)) - 1e-12
        while pending:
            for neighbor, cosine in self.neighbors[pending.pop()]:
                if neighbor not in selected and cosine >= limit:
                    selected.add(neighbor)
                    pending.append(neighbor)
        return selected


def find_edge_nodes(mesh, seed, angle=30.0, feature_angle=30.0, *, seed_edge=None):
    """Return node IDs along a connected exterior feature-edge chain.

    ``seed`` is a mesh-node ID on an exterior crease or open shell boundary.
    ``feature_angle`` is the minimum angle (degrees) between adjacent exterior
    face normals that identifies a crease. Open shell boundaries always qualify;
    smooth tessellation edges and solid internal faces are excluded. A threshold
    of zero includes coplanar tessellation edges too.

    ``angle`` is the maximum local turn (degrees) allowed while following the
    edge: zero follows a straight line. Traversal stops at branches and endpoints.
    If the seed touches distinct chains, ``AmbiguousEdgeSeed`` provides candidate
    IDs, endpoints and directions; repeat with ``seed_edge`` set to a candidate.
    Candidate IDs depend on the mesh topology and ``feature_angle``.

    Supports the solid and shell families of ``find_surface_nodes``, including
    MultiMesh interfaces and midside nodes. Angles use corner geometry. Returns
    sorted unique node IDs and does not modify the mesh. The seed must lie on an
    eligible edge; interior/face-interior nodes are not projected onto an edge.
    """
    if mesh.nodes.shape[1] != 3:
        raise ValueError(
            "Exterior edge selection requires a 3D mesh; use boundary-line selection for 2D meshes"
        )
    if (
        not isinstance(seed, Integral)
        or isinstance(seed, (bool, np.bool_))
        or not 0 <= seed < mesh.n_nodes
    ):
        raise ValueError("Node seed ID outside the mesh")
    for name, value in (("angle", angle), ("feature_angle", feature_angle)):
        if not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= 180:
            raise ValueError(f"{name} must be finite and between 0 and 180 degrees")
    graph = _FeatureEdges(mesh, feature_angle)
    candidates = [
        index for index, (_, nodes) in enumerate(graph.edges) if seed in nodes
    ]
    if not candidates:
        raise ValueError(
            "The seed is not on an exterior feature edge. Select a node on a crease or shell boundary, or lower the sharpness threshold."
        )
    if seed_edge is not None:
        if (
            not isinstance(seed_edge, Integral)
            or isinstance(seed_edge, (bool, np.bool_))
            or seed_edge not in candidates
        ):
            raise ValueError("seed_edge must be an exterior edge incident to the seed")
        selected = graph.grow(int(seed_edge), angle)
    else:
        selected = graph.grow(candidates[0], angle)
        if not set(candidates).issubset(selected):
            endpoints = [graph.edges[index][0] for index in candidates]
            directions = []
            for a, b in endpoints:
                direction = mesh.nodes[b] - mesh.nodes[a]
                directions.append(direction / np.linalg.norm(direction))
            raise AmbiguousEdgeSeed(candidates, endpoints, directions)
    return np.unique(
        np.concatenate([graph.edges[index][1] for index in sorted(selected)])
    )
