"""Shared local basis glyphs for DataSet.plot and the viewer (no Qt dependency)."""

import numpy as np
import pyvista as pv

from fedoo.core.mesh import MultiMesh
from fedoo.core.dataset import as_3d_coordinates
from fedoo.util.frame_fields import frame_field_names, frame_matrices, FRAME_COMPONENTS


def frame_samples(data, options, field=None):
    """Return global origins and row basis matrices for visible frame samples."""
    blocks = data.mesh.submeshes if isinstance(data.mesh, MultiMesh) else [data.mesh]
    associations = {
        "GaussPoint": "gausspoint_data",
        "Element": "element_data",
        "Node": "node_data",
    }
    selected = (
        options.get("element_set")
        if options.get("frame_element_sets") is None
        else None
    )
    if isinstance(selected, str):
        selected = data.mesh.element_sets[selected]
    origins, bases = [], []
    allowed = (
        set(data._resolve_submesh_indices(options.get("selected_submeshes")))
        if isinstance(data.mesh, MultiMesh)
        else {0}
    )
    offset = 0
    for index, mesh in enumerate(blocks):
        if index not in allowed:
            offset += mesh.n_elements
            continue
        block = (
            data._submesh_dataset(index) if isinstance(data.mesh, MultiMesh) else data
        )
        chosen = field or options.get("frame_field") or data.current_local_frame
        names = (
            [chosen]
            if chosen
            else sorted(
                frame_field_names(block),
                key=lambda name: (
                    0 if "ShellLocalFrame" in name or "BeamLocalFrame" in name else 1,
                    name,
                ),
            )
        )
        normals_only = options.get("geometric_normals", False)
        if normals_only:
            names = []
        if not chosen or normals_only:
            names = names + [None]
        if (
            chosen
            and not normals_only
            and chosen not in block.field_names()
            and chosen in ("ShellLocalFrame", "BeamLocalFrame", "MaterialLocalFrame")
        ):
            names = ["_" + chosen]
        for name in names:
            if not mesh.n_elements:
                continue
            geometry = name is None
            if geometry:
                # The reference derivative gives topological dimension without
                # guessing it from element names (including high-order types).
                from fedoo.lib_elements.element_list import get_element

                reference = get_element(mesh.elm_type)
                reference = getattr(reference, "geometry_elm", reference)(1)
                dimension = np.asarray(
                    reference.shape_function_derivative(
                        reference.get_gp_elm_coordinates(1)
                    )
                ).shape[1]
                supported = (
                    dimension == 1
                    and mesh.ndim in (2, 3)
                    or dimension == 2
                    and mesh.ndim == 3
                )
                if (
                    not supported
                    or normals_only
                    and not (
                        dimension == 1
                        and mesh.ndim == 2
                        or dimension == 2
                        and mesh.ndim == 3
                    )
                ):
                    continue
                association = (
                    "GaussPoint"
                    if options.get("frame_gauss_points", False)
                    else "Element"
                )
                values = None
            else:
                association = next(
                    (
                        kind
                        for kind, storage in associations.items()
                        if name in getattr(block, storage)
                    ),
                    None,
                )
                if association is None:
                    continue
                values = (
                    frame_matrices(
                        np.asarray(getattr(block, associations[association])[name])
                    )
                    .reshape(-1, 9)
                    .T
                )
            keep = np.ones(mesh.n_elements, dtype=bool)
            by_submesh = options.get("frame_element_sets")
            if by_submesh is not None:
                local_ids = by_submesh.get(index)
                if local_ids is not None:
                    keep = np.isin(np.arange(mesh.n_elements), local_ids)
            elif selected is not None:
                keep = np.isin(np.arange(mesh.n_elements) + offset, selected)
            if options.get("element_set_invert", False) and (
                selected is not None
                or by_submesh is not None
                and by_submesh.get(index) is not None
            ):
                keep = ~keep
            scale = options.get("scale", 1.0)
            displacement = block.node_data.get("Disp")
            points = np.asarray(mesh.nodes, dtype=float).copy()
            if displacement is not None and scale:
                disp = np.asarray(displacement)
                dimensions = min(points.shape[1], disp.shape[0])
                points[:, :dimensions] += scale * disp[:dimensions].T
            if geometry:
                geometric_mesh = mesh.copy()
                geometric_mesh.nodes = points
                from fedoo.lib_elements.element_list import get_default_n_gp

                n_gp = (
                    get_default_n_gp(mesh.elm_type, mesh)
                    if association == "GaussPoint"
                    else 1
                )
                frame = np.asarray(
                    geometric_mesh.get_element_local_frame(n_gp)
                ).reshape(-1, mesh.ndim, mesh.ndim)
                matrices = np.zeros((len(frame), 3, 3))
                matrices[:, : mesh.ndim, : mesh.ndim] = frame
                if normals_only:
                    matrices[:, : mesh.ndim - 1] = 0
                values = matrices.reshape(-1, 9).T
            if association == "Node":
                # Pick one actual orientation; never average matrix components.
                ids = np.unique(mesh.elements[keep].ravel())
                positions = points[ids]
                frames = values[:, ids].T.reshape(-1, 3, 3)
            else:
                count = values.shape[-1] // mesh.n_elements
                if count == 0 or values.shape[-1] % mesh.n_elements:
                    continue
                frames = values.T.reshape(count, mesh.n_elements, 3, 3)
                if association == "GaussPoint" and options.get(
                    "frame_gauss_points", False
                ):
                    positions = mesh._get_node2gausspoint_mat(count) @ points
                    mask = np.tile(keep, count)
                    positions = positions[mask]
                    frames = frames.reshape(-1, 3, 3)[mask]
                else:
                    positions = (mesh._get_node2gausspoint_mat(1) @ points)[keep]
                    frames = frames[0, keep]
            if mesh.ndim == 2:
                frames = frames.copy()
                frames[:, 2] = 0
            positions = as_3d_coordinates(positions)
            finite = np.isfinite(positions).all(axis=1) & np.isfinite(frames).all(
                axis=(1, 2)
            )
            origins.append(positions[finite])
            bases.append(frames[finite])
            break
        offset += mesh.n_elements
    if not origins:
        return np.empty((0, 3)), np.empty((0, 3, 3))
    origins, bases = np.concatenate(origins), np.concatenate(bases)
    limit = max(1, int(options.get("frame_max", 1000)))
    if len(origins) > limit:
        indices = np.linspace(0, len(origins) - 1, limit, dtype=int)
        origins, bases = origins[indices], bases[indices]
    return origins, bases


def plot_local_frames(data, plotter, options, field=None, component=None):
    if component in FRAME_COMPONENTS:
        options = dict(options)
        options["frame_axes"] = tuple(
            component == "Whole frame" or component == f"Local {axis + 1}"
            for axis in range(3)
        )
    origins, bases = frame_samples(data, options, field)
    if not len(origins):
        return
    cloud = pv.PolyData(origins)
    cloud.point_data["basis"] = bases.reshape(-1, 9)
    clip = options.get("clip_args")
    if clip is not None:
        clip = {key: value for key, value in clip.items() if key != "cell_ids"}
        cloud = cloud.clip(**clip)
    if not cloud.n_points:
        return
    extent = np.linalg.norm(np.ptp(as_3d_coordinates(data.mesh.nodes), axis=0))
    length = max(extent, np.finfo(float).eps) * options.get("frame_size", 2.0) / 100
    labels = []
    for axis, (enabled, color) in enumerate(
        zip(options.get("frame_axes", (True, True, True)), ("red", "green", "blue"))
    ):
        if not enabled:
            continue
        vectors = np.asarray(cloud.point_data["basis"]).reshape(-1, 3, 3)[:, axis]
        if not np.any(np.linalg.norm(vectors, axis=1) > 0):
            continue
        cloud.point_data["direction"] = np.array(
            np.asarray(cloud.point_data["basis"]).reshape(-1, 3, 3)[:, axis], copy=True
        )
        glyph = cloud.glyph(orient="direction", scale=False, factor=length)
        plotter.add_mesh(
            glyph,
            color=color,
            name=f"_local_frame_{axis}",
            show_scalar_bar=False,
            lighting=False,
            reset_camera=False,
        )
        labels.append([f"Local {axis + 1}", color])
    if labels and options.get("frame_legend", False):
        plotter.add_legend(labels, name="_local_frame_legend", size=(0.13, 0.13))
