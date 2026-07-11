"""
The Fedoo FDH5 format is a HDF5 file that is structured
to store multiple data computed from fedoo.
The structure of the file is the following:

mesh/
├── nodes
├── node_sets/
├── submesh_0/
│     ├── elements
│     ├── element_sets/
│     └── metadata/
│           └── element_type = "tri3"
├── submesh_1/
│     ├── elements
│     ├── element_sets/
│     └── metadata/
│           └── element_type = "tri3"
├── submesh_2/
│     ├── elements
│     ├── element_sets/
│     └── metadata/
│           └── element_type = "quad4"
└── ...

results/
├── iter_0/
│   ├── node_data/
│   ├── element_data/
│   │     ├── submesh_0/
│   │     │    ├── stress
│   │     │    ├── plasticity
│   │     │    └── ...
│   │     ├── submesh_1/
│   │     └── ...
│   ├── gausspoint_data/
│   │     ├── submesh_0/
│   │     ├── submesh_1/
│   │     └── ...
│   ├── scalars/
│   └── metadata/
├── iter_1/
└── ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Union, Any, Literal, List, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
import contextlib

import numpy as np
import h5py

PathLike = Union[str, Path]
NDArray = np.ndarray


def submesh_id(index: int) -> str:
    """Return the canonical FDH5 submesh id for a Fedoo submesh index."""
    return f"submesh_{index}"


def iteration_name(iteration: int) -> str:
    """Return the canonical FDH5 group name for a result iteration."""
    return f"iter_{iteration}"


@dataclass(frozen=True)
class CompressionConfig:
    """HDF5 dataset storage options used by :class:`FDH5Writer`.

    Parameters
    ----------
    compression : {"gzip", "lzf"} or None, optional
        Compression filter applied to non-scalar datasets; ``None`` disables
        compression. Default is ``"gzip"``.
    compression_opts : int or None, optional
        Compression level for the gzip filter (0-9). Default is 4.
    chunks : bool or tuple of int or None, optional
        Chunk shape passed to h5py; ``True`` enables auto-chunking and ``None``
        disables chunking. Default is ``True``.
    """

    compression: Optional[Literal["gzip", "lzf"]] = "gzip"
    compression_opts: Optional[int] = 4
    chunks: Optional[Union[bool, tuple[int, ...]]] = True


class FDH5Writer:
    """Writer for the Fedoo HDF5 (FDH5) result format.

    Creates or appends to an FDH5 file following the layout described in the
    module docstring: a ``mesh/`` group holding the nodes and submeshes, and a
    ``results/`` group holding one subgroup per result iteration.
    """

    FILE_VERSION = "1.0"

    def __init__(
        self,
        file_path: PathLike,
        *,
        compression: CompressionConfig = CompressionConfig(),
        validate: bool = True,
        create_parents: bool = True,
    ) -> None:
        """Open (or create) an FDH5 file for writing.

        Parameters
        ----------
        file_path : str or Path
            Path of the FDH5 file to create or append to.
        compression : CompressionConfig, optional
            HDF5 storage options applied to the written datasets.
        validate : bool, optional
            If True (default), check array shapes and dtypes before writing.
        create_parents : bool, optional
            If True (default), create the parent directories of ``file_path``
            when they do not already exist.
        """
        self.path = Path(file_path)
        if create_parents:
            self.path.parent.mkdir(parents=True, exist_ok=True)

        self.compression = compression
        self.validate = validate

        with h5py.File(self.path, "a") as f:
            if "format" not in f.attrs:
                f.attrs["format"] = "FDH5"
                f.attrs["version"] = self.FILE_VERSION
                f.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()

    def _create_dataset(
        self,
        parent: h5py.Group,
        name: str,
        data: NDArray,
        *,
        overwrite: bool = False,
    ) -> h5py.Dataset:
        if name in parent:
            if overwrite:
                del parent[name]
            else:
                raise ValueError(f"Dataset '{parent.name}/{name}' already exists")

        kwargs: Dict[str, Any] = {}
        is_scalar = np.asarray(data).shape == ()
        if self.compression.compression and not is_scalar:
            kwargs["compression"] = self.compression.compression
            if self.compression.compression == "gzip":
                kwargs["compression_opts"] = self.compression.compression_opts
        if self.compression.chunks is not None and not is_scalar:
            kwargs["chunks"] = self.compression.chunks

        return parent.create_dataset(name, data=data, **kwargs)

    def _next_submesh_id(self, f: h5py.File) -> str:
        existing = [k for k in f["mesh"].keys() if k.startswith("submesh_")]
        if not existing:
            return "submesh_0"
        nums = [int(k.split("_")[1]) for k in existing]
        return f"submesh_{max(nums) + 1}"

    def write_mesh(
        self,
        nodes: NDArray,
        *,
        node_sets: Optional[Mapping[str, NDArray]] = None,
        overwrite: bool = False,
    ) -> None:
        """Write the global node coordinates and optional node sets.

        Parameters
        ----------
        nodes : numpy.ndarray
            Node coordinates with shape ``(n_nodes, dim)``.
        node_sets : mapping of str to numpy.ndarray, optional
            Named sets of node indices stored under ``mesh/node_sets``.
        overwrite : bool, optional
            If True, replace existing node data and node sets. If False
            (default), writing over existing data raises ``ValueError``.
        """
        nodes = np.asarray(nodes)

        if self.validate:
            if nodes.ndim != 2:
                raise ValueError("nodes must have shape (n_nodes, dim)")
            if not np.issubdtype(nodes.dtype, np.floating):
                raise TypeError("nodes must be floating point")

        with h5py.File(self.path, "a") as f:
            mesh = f.require_group("mesh")

            if "nodes" in mesh and overwrite:
                del mesh["nodes"]

            self._create_dataset(mesh, "nodes", nodes, overwrite=overwrite)

            if node_sets:
                ns = mesh.require_group("node_sets")
                if overwrite:
                    for k in list(ns.keys()):
                        del ns[k]

                for name, idx in node_sets.items():
                    arr = np.asarray(idx)
                    self._create_dataset(ns, name, arr, overwrite=overwrite)

    def add_submesh(
        self,
        element_type: str,
        elements: NDArray,
        *,
        element_sets: Optional[Mapping[str, NDArray]] = None,
        name: Optional[str] = None,
        submesh_id: Optional[str] = None,
        overwrite: bool = False,
    ) -> str:
        """Add a submesh (a single element type) to the mesh group.

        Parameters
        ----------
        element_type : str
            Fedoo element type stored in the submesh metadata (e.g. ``"tri3"``).
        elements : numpy.ndarray
            Connectivity table ``(n_elements, nodes_per_element)`` of 0-based
            node indices.
        element_sets : mapping of str to numpy.ndarray, optional
            Named sets of element indices stored under the submesh.
        name : str, optional
            Human-readable submesh name stored in the metadata.
        submesh_id : str, optional
            Target submesh group id. If None (default), the next free
            ``submesh_<n>`` id is used.
        overwrite : bool, optional
            If True, replace an existing submesh with the same id. If False
            (default), a collision raises ``ValueError``.

        Returns
        -------
        str
            The submesh id under which the data was written.
        """
        elements = np.asarray(elements)

        if self.validate:
            if elements.ndim != 2:
                raise ValueError("elements must be 2D (n_elements, nodes_per_element)")
            if not np.issubdtype(elements.dtype, np.integer):
                raise TypeError("elements must be integer indices (0-based)")

        with h5py.File(self.path, "a") as f:
            mesh = f.require_group("mesh")
            sid = submesh_id or self._next_submesh_id(f)

            if sid in mesh:
                if overwrite:
                    del mesh[sid]
                else:
                    raise ValueError(f"Submesh '{sid}' already exists")

            sm = mesh.create_group(sid)
            self._create_dataset(sm, "elements", elements)

            meta = sm.create_group("metadata")
            meta.attrs["element_type"] = element_type
            meta.attrs["n_elements"] = elements.shape[0]
            meta.attrs["nodes_per_element"] = elements.shape[1]
            if name:
                meta.attrs["name"] = name

            if element_sets:
                es = sm.create_group("element_sets")
                for set_name, idx in element_sets.items():
                    self._create_dataset(es, set_name, np.asarray(idx))

            return sid

    def write_iteration(
        self,
        iteration: int,
        *,
        node_data: Optional[Mapping[str, NDArray]] = None,
        element_data: Optional[Mapping[str, Mapping[str, NDArray]]] = None,
        gausspoint_data: Optional[Mapping[str, Mapping[str, NDArray]]] = None,
        scalars: Optional[Mapping[str, Union[int, float, NDArray]]] = None,
        time: Optional[float] = None,
        dt: Optional[float] = None,
        overwrite: bool = False,
    ) -> str:
        """Write all result data for a single iteration.

        Parameters
        ----------
        iteration : int
            Iteration index; stored as the ``iter_<n>`` results subgroup.
        node_data : mapping of str to numpy.ndarray, optional
            Nodal fields, each with shape ``(n_nodes, ...)``.
        element_data : mapping, optional
            Element fields keyed by submesh id, then by field name.
        gausspoint_data : mapping, optional
            Gauss-point fields keyed by submesh id, then by field name. Each
            field uses a GP-major flattened layout of shape
            ``(n_elements * n_gauss_points, n_components)``.
        scalars : mapping of str to (int, float or numpy.ndarray), optional
            Scalar quantities stored under the iteration ``scalars`` group.
        time : float, optional
            Physical time of the iteration, stored in the metadata.
        dt : float, optional
            Time increment of the iteration, stored in the metadata.
        overwrite : bool, optional
            If True, replace an existing iteration. If False (default), a
            collision raises ``ValueError``.

        Returns
        -------
        str
            The HDF5 path of the written iteration group.
        """
        iter_name = iteration_name(iteration)

        with h5py.File(self.path, "a") as f:
            results = f.require_group("results")

            if iter_name in results:
                if overwrite:
                    del results[iter_name]
                else:
                    raise ValueError(f"Iteration '{iter_name}' already exists")

            it = results.create_group(iter_name)

            md = it.create_group("metadata")
            md.attrs["iteration"] = iteration
            if time is not None:
                md.attrs["time"] = float(time)
            if dt is not None:
                md.attrs["dt"] = float(dt)
            md.attrs["created_utc"] = datetime.now(timezone.utc).isoformat()

            if node_data:
                nd = it.create_group("node_data")
                for name, arr in node_data.items():
                    self._create_dataset(nd, name, np.asarray(arr))

            if element_data:
                ed = it.create_group("element_data")
                for sid, fields in element_data.items():
                    smg = ed.create_group(sid)
                    for fname, arr in fields.items():
                        self._create_dataset(smg, fname, np.asarray(arr))

            if gausspoint_data:
                gd = it.create_group("gausspoint_data")
                for sid, fields in gausspoint_data.items():
                    smg = gd.create_group(sid)
                    n_elements = f[f"mesh/{sid}/elements"].shape[0]
                    for fname, arr in fields.items():
                        arr = np.asarray(arr)
                        ds = self._create_dataset(smg, fname, arr)
                        ds.attrs["gp_order"] = "gp-major"
                        ds.attrs["n_elements"] = n_elements
                        if arr.shape[0] % n_elements == 0:
                            ds.attrs["n_gauss_points"] = arr.shape[0] // n_elements
                        if arr.ndim == 1:
                            ds.attrs["n_components"] = 1
                        else:
                            ds.attrs["n_components"] = arr.shape[1]

            if scalars:
                sc = it.create_group("scalars")
                for name, val in scalars.items():
                    self._create_dataset(sc, name, np.asarray(val))

            return it.name


class FDH5Reader:
    """
    Reader for FDH5 (Finite Element HDF5) files.

    Supports:
      - eager reads (NumPy arrays)
      - lazy reads (h5py.Dataset)

    Lazy access MUST be used inside `with reader.open():`.
    """

    def __init__(self, file_path: PathLike) -> None:
        self.path = Path(file_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    @contextlib.contextmanager
    def open(self) -> Iterator[h5py.File]:
        """
        Open the HDF5 file.

        Required for lazy access (datasets remain valid only
        while the file is open).
        """
        f = h5py.File(self.path, "r")
        try:
            yield f
        finally:
            f.close()

    def _open(self) -> h5py.File:
        """Internal helper for eager reads."""
        return h5py.File(self.path, "r")

    @staticmethod
    def _decode(value: Any) -> Any:
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8")
        return value

    def _require_lazy_file(self, file: h5py.File | None) -> h5py.File:
        if file is None:
            raise ValueError(
                "Lazy reads require an open HDF5 file: "
                "use `with reader.open() as f:` and pass `file=f`."
            )
        return file

    def read_nodes(
        self,
        *,
        lazy: bool = False,
        file: h5py.File | None = None,
    ) -> Union[NDArray, h5py.Dataset]:
        """Read the global node coordinates.

        Parameters
        ----------
        lazy : bool, optional
            If True, return the open ``h5py.Dataset`` instead of a NumPy array;
            this requires ``file`` and that the file remains open while it is
            used.
        file : h5py.File, optional
            Open file handle, required when ``lazy`` is True.

        Returns
        -------
        numpy.ndarray or h5py.Dataset
            Node coordinates ``(n_nodes, dim)``, eager or lazy.
        """
        if not lazy:
            with self._open() as f:
                return f["mesh/nodes"][...]

        f = self._require_lazy_file(file)
        return f["mesh/nodes"]

    def read_node_sets(
        self,
        *,
        lazy: bool = False,
        file: h5py.File | None = None,
    ) -> Dict[str, Union[NDArray, h5py.Dataset]]:
        """Read the named node sets.

        Parameters
        ----------
        lazy : bool, optional
            If True, return open ``h5py.Dataset`` objects; the file must stay
            open while they are used.
        file : h5py.File, optional
            Open file handle, required when ``lazy`` is True.

        Returns
        -------
        dict of str to (numpy.ndarray or h5py.Dataset)
            Node index arrays keyed by set name; empty if no node sets exist.
        """
        if not lazy:
            out: Dict[str, NDArray] = {}
            with self._open() as f:
                grp = f.get("mesh/node_sets")
                if grp is None:
                    return out
                for name, ds in grp.items():
                    out[name] = ds[...]
            return out

        f = self._require_lazy_file(file)
        grp = f.get("mesh/node_sets")
        return {} if grp is None else {name: ds for name, ds in grp.items()}

    def list_submeshes(self, *, file: h5py.File | None = None) -> List[str]:
        """Return the sorted submesh ids stored in the mesh group.

        Parameters
        ----------
        file : h5py.File, optional
            Open file handle to reuse; a temporary one is opened if omitted.

        Returns
        -------
        list of str
            Submesh ids (``submesh_<n>``) in ascending order.
        """
        if file is None:
            with self._open() as f:
                return self.list_submeshes(file=f)

        mesh = file.get("mesh")
        if mesh is None:
            return []
        return sorted(name for name in mesh if name.startswith("submesh_"))

    def read_submesh(
        self,
        submesh_id: str,
        *,
        lazy: bool = False,
        file: h5py.File | None = None,
    ) -> Dict[str, Any]:
        """
        Read a submesh definition.
        """
        if not lazy:
            with self._open() as f:
                sm = f[f"mesh/{submesh_id}"]
                meta = sm["metadata"]

                data: Dict[str, Any] = {
                    "id": submesh_id,
                    "element_type": self._decode(meta.attrs.get("element_type")),
                    "elements": sm["elements"][...],
                    "element_sets": {},
                }

                if "name" in meta.attrs:
                    data["name"] = self._decode(meta.attrs["name"])

                eset_grp = sm.get("element_sets")
                if eset_grp is not None:
                    for name, ds in eset_grp.items():
                        data["element_sets"][name] = ds[...]

                return data

        f = self._require_lazy_file(file)
        sm = f[f"mesh/{submesh_id}"]
        meta = sm["metadata"]

        return {
            "id": submesh_id,
            "element_type": self._decode(meta.attrs.get("element_type")),
            "name": self._decode(meta.attrs.get("name"))
            if "name" in meta.attrs
            else None,
            "elements": sm["elements"],
            "element_sets": (
                {name: ds for name, ds in sm["element_sets"].items()}
                if "element_sets" in sm
                else {}
            ),
        }

    def read_mesh(
        self,
        *,
        lazy: bool = False,
        file: h5py.File | None = None,
    ) -> Dict[str, Any]:
        """
        Read the full mesh (nodes + submeshes).
        """
        mesh: Dict[str, Any] = {}
        mesh["nodes"] = self.read_nodes(lazy=lazy, file=file)
        mesh["node_sets"] = self.read_node_sets(lazy=lazy, file=file)

        subs: Dict[str, Any] = {}
        for sid in self.list_submeshes(file=file):
            subs[sid] = self.read_submesh(sid, lazy=lazy, file=file)

        mesh["submeshes"] = subs
        return mesh

    def list_iterations(self, *, file: h5py.File | None = None) -> List[int]:
        """Return the sorted result iteration indices stored in the file.

        Parameters
        ----------
        file : h5py.File, optional
            Open file handle to reuse; a temporary one is opened if omitted.

        Returns
        -------
        list of int
            Iteration indices in ascending order.
        """
        if file is None:
            with self._open() as f:
                return self.list_iterations(file=f)

        grp = file.get("results")
        if grp is None:
            return []
        return sorted(
            int(name.split("_")[1]) for name in grp if name.startswith("iter_")
        )

    def read_iteration_metadata(self, iteration: int) -> Dict[str, Any]:
        """Read the metadata attributes of one result iteration.

        Parameters
        ----------
        iteration : int
            Iteration index to read.

        Returns
        -------
        dict of str to Any
            Metadata attributes (e.g. ``iteration``, ``time``, ``dt``).
        """
        with self._open() as f:
            md = f[f"results/{iteration_name(iteration)}/metadata"]
            return {k: self._decode(v) for k, v in md.attrs.items()}

    def read_iteration(
        self,
        iteration: int,
        *,
        lazy: bool = False,
        file: h5py.File | None = None,
    ) -> Dict[str, Any]:
        """Read all data stored for one result iteration."""
        if lazy:
            self._require_lazy_file(file)

        return {
            "metadata": self.read_iteration_metadata(iteration),
            "node_data": self.read_node_data(iteration, lazy=lazy, file=file),
            "element_data": self.read_element_data(iteration, lazy=lazy, file=file),
            "gausspoint_data": self.read_gausspoint_data(
                iteration, lazy=lazy, file=file
            ),
            "scalars": self.read_scalars(iteration, lazy=lazy, file=file),
        }

    def read_scalars(
        self,
        iteration: int,
        *,
        lazy: bool = False,
        file: h5py.File | None = None,
    ) -> Dict[str, Union[NDArray, h5py.Dataset]]:
        """Read the scalar quantities stored for one iteration.

        Parameters
        ----------
        iteration : int
            Iteration index to read.
        lazy : bool, optional
            If True, return open ``h5py.Dataset`` objects; the file must stay
            open while they are used.
        file : h5py.File, optional
            Open file handle, required when ``lazy`` is True.

        Returns
        -------
        dict of str to (numpy.ndarray or h5py.Dataset)
            Scalar values keyed by name; empty if none are stored.
        """
        if not lazy:
            out: Dict[str, NDArray] = {}
            with self._open() as f:
                grp = f.get(f"results/{iteration_name(iteration)}/scalars")
                if grp is None:
                    return out
                for name, ds in grp.items():
                    out[name] = ds[...]
            return out

        f = self._require_lazy_file(file)
        grp = f.get(f"results/{iteration_name(iteration)}/scalars")
        return {} if grp is None else {name: ds for name, ds in grp.items()}

    def read_node_data(
        self,
        iteration: int,
        *,
        lazy: bool = False,
        file: h5py.File | None = None,
    ) -> Dict[str, Union[NDArray, h5py.Dataset]]:
        """Read the nodal fields stored for one iteration.

        Parameters
        ----------
        iteration : int
            Iteration index to read.
        lazy : bool, optional
            If True, return open ``h5py.Dataset`` objects; the file must stay
            open while they are used.
        file : h5py.File, optional
            Open file handle, required when ``lazy`` is True.

        Returns
        -------
        dict of str to (numpy.ndarray or h5py.Dataset)
            Nodal field arrays keyed by field name; empty if none are stored.
        """
        if not lazy:
            out: Dict[str, NDArray] = {}
            with self._open() as f:
                grp = f.get(f"results/{iteration_name(iteration)}/node_data")
                if grp is None:
                    return out
                for name, ds in grp.items():
                    out[name] = ds[...]
            return out

        f = self._require_lazy_file(file)
        grp = f.get(f"results/{iteration_name(iteration)}/node_data")
        return {} if grp is None else {name: ds for name, ds in grp.items()}

    def read_element_data(
        self,
        iteration: int,
        *,
        lazy: bool = False,
        file: h5py.File | None = None,
    ) -> Dict[str, Dict[str, Union[NDArray, h5py.Dataset]]]:
        """Read the element fields stored for one iteration.

        Parameters
        ----------
        iteration : int
            Iteration index to read.
        lazy : bool, optional
            If True, return open ``h5py.Dataset`` objects; the file must stay
            open while they are used.
        file : h5py.File, optional
            Open file handle, required when ``lazy`` is True.

        Returns
        -------
        dict of str to (dict of str to (numpy.ndarray or h5py.Dataset))
            Element fields keyed by submesh id, then by field name; empty if
            none are stored.
        """
        if not lazy:
            out: Dict[str, Dict[str, NDArray]] = {}
            with self._open() as f:
                grp = f.get(f"results/{iteration_name(iteration)}/element_data")
                if grp is None:
                    return out
                for sid, sm_grp in grp.items():
                    out[sid] = {name: ds[...] for name, ds in sm_grp.items()}
            return out

        f = self._require_lazy_file(file)
        grp = f.get(f"results/{iteration_name(iteration)}/element_data")
        if grp is None:
            return {}
        return {
            sid: {name: ds for name, ds in sm_grp.items()}
            for sid, sm_grp in grp.items()
        }

    def read_gausspoint_data(
        self,
        iteration: int,
        *,
        lazy: bool = False,
        file: h5py.File | None = None,
    ) -> Dict[str, Dict[str, Union[NDArray, h5py.Dataset]]]:
        """Read the Gauss-point fields stored for one iteration.

        Each field uses a GP-major flattened layout of shape
        ``(n_elements * n_gauss_points, n_components)``.

        Parameters
        ----------
        iteration : int
            Iteration index to read.
        lazy : bool, optional
            If True, return open ``h5py.Dataset`` objects; the file must stay
            open while they are used.
        file : h5py.File, optional
            Open file handle, required when ``lazy`` is True.

        Returns
        -------
        dict of str to (dict of str to (numpy.ndarray or h5py.Dataset))
            Gauss-point fields keyed by submesh id, then by field name; empty
            if none are stored.
        """
        if not lazy:
            out: Dict[str, Dict[str, NDArray]] = {}
            with self._open() as f:
                grp = f.get(f"results/{iteration_name(iteration)}/gausspoint_data")
                if grp is None:
                    return out
                for sid, sm_grp in grp.items():
                    out[sid] = {name: ds[...] for name, ds in sm_grp.items()}
            return out

        f = self._require_lazy_file(file)
        grp = f.get(f"results/{iteration_name(iteration)}/gausspoint_data")
        if grp is None:
            return {}
        return {
            sid: {name: ds for name, ds in sm_grp.items()}
            for sid, sm_grp in grp.items()
        }


def mesh_to_fedoo(mesh_data: dict):
    """Build a Fedoo Mesh or MultiMesh from FDH5 reader mesh data."""
    from fedoo.core.mesh import Mesh, MultiMesh

    nodes = mesh_data["nodes"]
    node_sets = mesh_data.get("node_sets", {})
    submeshes = mesh_data.get("submeshes", {})

    if len(submeshes) == 0:
        return Mesh(nodes, node_sets=node_sets)

    ordered_submeshes = [
        submeshes[sid]
        for sid in sorted(submeshes, key=lambda name: int(name.split("_")[1]))
    ]

    if len(ordered_submeshes) == 1:
        submesh = ordered_submeshes[0]
        return Mesh(
            nodes,
            submesh["elements"],
            submesh["element_type"],
            node_sets=node_sets,
            element_sets=submesh.get("element_sets", {}),
            name=submesh.get("name", ""),
            register_name=False,
        )

    # Reconstruct submeshes by order, not by name: two submeshes may share the
    # same name (e.g. duplicate element types with empty names, stored with
    # ``name = elm_type``). Keying a dict by name would silently drop the
    # collision, so build ordered Mesh objects and preserve their stored names.
    submesh_list = [
        Mesh(
            nodes,
            submesh["elements"],
            submesh["element_type"],
            element_sets=submesh.get("element_sets", {}),
            name=submesh.get("name", ""),
            register_name=False,
        )
        for submesh in ordered_submeshes
    ]

    return MultiMesh.from_mesh_list(
        submesh_list,
        node_sets=node_sets,
        register_name=False,
    )


def fields_to_submesh_dict(dataset, fields: dict) -> dict:
    """Convert Fedoo field dictionaries to the FDH5 submesh-first layout."""
    out = {}

    if dataset._is_multimesh():
        for field, value in fields.items():
            data = dataset._as_multimesh_data(value)
            for sid, block in data.items():
                out.setdefault(submesh_id(sid), {})[field] = block
    else:
        for field, value in fields.items():
            out.setdefault("submesh_0", {})[field] = value

    return out


def fields_from_submesh_dict(mesh, fields: dict) -> dict:
    """Convert FDH5 submesh-first data to Fedoo field dictionaries."""
    from fedoo.core.mesh import MultiMesh

    if not isinstance(mesh, MultiMesh):
        if "submesh_0" in fields:
            submesh_fields = fields["submesh_0"]
        elif fields:
            first_key = sorted(fields)[0]
            submesh_fields = fields[first_key]
        else:
            submesh_fields = {}
        return dict(submesh_fields)

    out = {}
    for sid, submesh_fields in fields.items():
        submesh_index = int(sid.split("_")[1])
        for field, value in submesh_fields.items():
            out.setdefault(field, {})[submesh_index] = value
    return out


def scalar_value(value):
    """Return a scalar value as a Python scalar when possible."""
    arr = np.asarray(value)
    if arr.shape == ():
        return arr.item()
    return arr


def load_dataset_iteration(dataset, filename: str, iteration: int = 0) -> None:
    """Load one FDH5 iteration into an existing DataSet object."""
    reader = FDH5Reader(filename)
    if dataset.mesh is None:
        dataset.mesh = mesh_to_fedoo(reader.read_mesh())

    iter_data = reader.read_iteration(iteration)
    dataset.node_data = iter_data["node_data"]
    dataset.element_data = fields_from_submesh_dict(
        dataset.mesh,
        iter_data["element_data"],
    )
    dataset.gausspoint_data = fields_from_submesh_dict(
        dataset.mesh,
        iter_data["gausspoint_data"],
    )
    dataset.scalar_data = {
        key: scalar_value(value) for key, value in iter_data["scalars"].items()
    }


def write_dataset(dataset, filename: str, iteration: int = 0, overwrite: bool = False):
    """Write a Fedoo DataSet iteration to a FDH5 file."""
    from fedoo.core.mesh import MultiMesh

    if dataset.mesh is None:
        raise TypeError("Mesh should be defined before writing a FDH5 file.")

    path = Path(filename)
    if path.suffix == "":
        path = path.with_suffix(".fdh5")

    file_exists = path.exists()
    if overwrite and file_exists:
        path.unlink()
        file_exists = False

    writer = FDH5Writer(path)
    if not file_exists:
        writer.write_mesh(
            dataset.mesh.nodes,
            node_sets=dataset.mesh.node_sets,
            overwrite=True,
        )

        submeshes = (
            dataset.mesh.submeshes
            if isinstance(dataset.mesh, MultiMesh)
            else (dataset.mesh,)
        )
        for i, mesh in enumerate(submeshes):
            writer.add_submesh(
                mesh.elm_type,
                mesh.elements,
                element_sets=mesh.element_sets,
                name=mesh.name or mesh.elm_type,
                submesh_id=submesh_id(i),
                overwrite=True,
            )

    time = dataset.scalar_data.get("Time", None)
    writer.write_iteration(
        iteration,
        node_data=dataset.node_data,
        element_data=fields_to_submesh_dict(dataset, dataset.element_data),
        gausspoint_data=fields_to_submesh_dict(dataset, dataset.gausspoint_data),
        scalars=dataset.scalar_data,
        time=time,
        overwrite=True,
    )


def read_fdh5(filename: str):
    """Read a FDH5 file as a DataSet or MultiFrameDataSet."""
    from fedoo.core.dataset import DataSet, MultiFrameDataSet

    path = Path(filename)
    if path.suffix == "":
        path = path.with_suffix(".fdh5")

    assert path.is_file(), "File not found"

    reader = FDH5Reader(path)
    mesh = mesh_to_fedoo(reader.read_mesh())
    iterations = reader.list_iterations()

    if len(iterations) == 0:
        return DataSet(mesh)

    if len(iterations) == 1:
        dataset = DataSet(mesh)
        load_dataset_iteration(dataset, str(path), iterations[0])
        return dataset

    dataset = MultiFrameDataSet(mesh)
    dataset.list_data = [("fdh5", str(path), iteration) for iteration in iterations]
    return dataset
