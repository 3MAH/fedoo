"""Data containers for values attached to ``MultiMesh`` submeshes."""

from __future__ import annotations

import numpy as np

from fedoo.core.mesh import MultiMesh
from fedoo.lib_elements.element_list import get_default_n_gp


class MultiMeshData:
    """Array-like data container associated with a ``MultiMesh``.

    ``MultiMeshData`` stores one optional data block per ordered submesh of a
    ``MultiMesh``. It is returned by :meth:`DataSet.get_data` for element and
    Gauss point fields attached to a ``MultiMesh``. The object behaves like the
    data block of its active submesh when converted to a NumPy array or indexed
    directly, while still allowing explicit submesh selection.

    Submeshes may be selected by integer id, submesh name, element type, or a
    list combining these selectors:

    >>> data = dataset.get_data("Stress", "XX")
    >>> data.active          # data for dataset.active_submesh
    >>> data.submesh(1)      # data for submesh id 1
    >>> data.submesh("tri3") # data for all tri3 submeshes

    Parameters
    ----------
    mesh : MultiMesh
        Mesh defining the ordered submeshes.
    data : dict, sequence, MultiMeshData or array-like
        Data to associate with submeshes. Dictionaries may use integer
        submesh ids, submesh names, or unique element types as keys. A sequence
        must have one entry per submesh. A plain array is attached only to the
        active submesh.
    active_submesh : int or str, default=0
        Submesh used when ``MultiMeshData`` is accessed as an array.

    Notes
    -----
    Missing submesh data are stored as ``None``. ``to_global`` concatenates
    selected submesh blocks in mesh order and fills missing blocks with
    ``fill_value``.
    """

    def __init__(self, mesh: MultiMesh, data, active_submesh=0) -> None:
        self.mesh = mesh
        self.active_submesh = active_submesh
        self._data = self._normalize_data(data)

    def _normalize_data(self, data) -> tuple:
        n_submesh = len(self.mesh.submeshes)
        if isinstance(data, MultiMeshData):
            return data._data
        if isinstance(data, dict):
            return tuple(self._value_from_dict(data, i) for i in range(n_submesh))
        if isinstance(data, (list, tuple)) and len(data) == n_submesh:
            return tuple(data)
        return tuple(
            data if i == self._resolve_single(self.active_submesh) else None
            for i in range(n_submesh)
        )

    def _value_from_dict(self, data: dict, index: int):
        mesh = self.mesh[index]
        if index in data:
            return data[index]
        if mesh.name and mesh.name in data:
            return data[mesh.name]
        matches = [
            i
            for i, submesh in enumerate(self.mesh.submeshes)
            if submesh.elm_type == mesh.elm_type
        ]
        if len(matches) == 1 and mesh.elm_type in data:
            return data[mesh.elm_type]
        return None

    def _resolve_indices(self, selector=None) -> list[int]:
        if selector is None:
            return [i for i, data in enumerate(self._data) if data is not None]
        if isinstance(selector, (list, tuple, set, np.ndarray)):
            indices = []
            for item in selector:
                indices.extend(self._resolve_indices(item))
            return list(dict.fromkeys(indices))
        if isinstance(selector, int):
            return [selector]
        if isinstance(selector, str):
            name_matches = [
                i for i, mesh in enumerate(self.mesh.submeshes) if mesh.name == selector
            ]
            if name_matches:
                return name_matches
            type_matches = [
                i
                for i, mesh in enumerate(self.mesh.submeshes)
                if mesh.elm_type == selector
            ]
            if type_matches:
                return type_matches
        raise KeyError(selector)

    def _resolve_single(self, selector=None) -> int:
        indices = self._resolve_indices(selector)
        if len(indices) != 1:
            raise KeyError(
                f"Selector {selector!r} matches {len(indices)} submeshes, expected one."
            )
        return indices[0]

    @property
    def active(self):
        """Data block of the currently active submesh (see ``active_submesh``)."""
        return self.submesh(self.active_submesh)

    def submesh(self, selector):
        """Return data associated with one submesh or a filtered view.

        If ``selector`` resolves to a single submesh, the raw data block is
        returned. If it resolves to several submeshes, a new ``MultiMeshData``
        view backed by a filtered ``MultiMesh`` is returned.
        """
        indices = self._resolve_indices(selector)
        if len(indices) == 1:
            return self._data[indices[0]]
        mesh = MultiMesh.from_mesh_list(
            [self.mesh[i] for i in indices],
            name=self.mesh.name,
            node_sets=self.mesh.node_sets,
            register_name=False,
        )
        return MultiMeshData(
            mesh,
            {
                j: self._data[i]
                for j, i in enumerate(indices)
                if self._data[i] is not None
            },
            active_submesh=0,
        )

    def map(self, func) -> "MultiMeshData":
        """Return a new ``MultiMeshData`` with ``func`` applied to each non-empty
        submesh block."""
        return MultiMeshData(
            self.mesh,
            {i: func(data) for i, data in enumerate(self._data) if data is not None},
            active_submesh=self.active_submesh,
        )

    def keys(self):
        """Submesh ids that hold a (non-empty) data block."""
        return [i for i, data in enumerate(self._data) if data is not None]

    def items(self):
        """``(submesh_id, data_block)`` pairs for submeshes that hold data."""
        return [(i, data) for i, data in enumerate(self._data) if data is not None]

    def global_element_location(self, element_id: int) -> tuple[int, int]:
        """Return ``(submesh_id, local_element_id)`` for a global element id.

        Global element ids follow the same concatenated order as
        :meth:`to_global`: all elements of submesh 0, then all elements of
        submesh 1, and so on.
        """
        element_id = int(element_id)
        offset = 0
        for submesh_id, submesh in enumerate(self.mesh.submeshes):
            stop = offset + submesh.n_elements
            if offset <= element_id < stop:
                return submesh_id, element_id - offset
            offset = stop
        raise IndexError(element_id)

    def global_element_id(self, submesh_id: int, local_element_id: int) -> int:
        """Return the global element id for a submesh-local element id."""
        submesh_id = int(submesh_id)
        local_element_id = int(local_element_id)
        if submesh_id < 0 or submesh_id >= len(self.mesh.submeshes):
            raise IndexError(submesh_id)
        submesh = self.mesh[submesh_id]
        if local_element_id < 0 or local_element_id >= submesh.n_elements:
            raise IndexError(local_element_id)
        return (
            sum(mesh.n_elements for mesh in self.mesh.submeshes[:submesh_id])
            + local_element_id
        )

    def global_element_value(self, element_id: int):
        """Return the value attached to one global element id.

        The value is read from the corresponding submesh block. For vector or
        tensor element fields stored with elements on the last axis, the
        returned value is ``block[..., local_element_id]``.
        """
        submesh_id, local_id = self.global_element_location(element_id)
        block = self._data[submesh_id]
        if block is None:
            return None
        return np.asarray(block)[..., local_id]

    def global_element_values(self, element_ids, fill_value=0.0):
        """Return values attached to several global element ids.

        Missing submesh blocks are filled with ``fill_value``. The returned
        array preserves the order of ``element_ids``.
        """
        element_ids = np.asarray(element_ids, dtype=int)
        scalar_input = element_ids.ndim == 0
        if scalar_input:
            return self.global_element_value(int(element_ids))

        values = []
        template = next((data for data in self._data if data is not None), None)
        for element_id in element_ids:
            submesh_id, local_id = self.global_element_location(int(element_id))
            block = self._data[submesh_id]
            if block is None:
                if template is None or np.asarray(template).ndim == 1:
                    values.append(fill_value)
                else:
                    values.append(np.full(np.asarray(template).shape[:-1], fill_value))
            else:
                values.append(np.asarray(block)[..., local_id])
        if not values:
            return np.array([])
        if np.asarray(values[0]).ndim == 0:
            return np.asarray(values)
        return np.stack(values, axis=-1)

    def to_global(self, indices=None, fill_value=0.0):
        """Concatenate per-submesh data in the selected submesh order."""
        if indices is None:
            indices = list(range(len(self.mesh.submeshes)))
        else:
            indices = self._resolve_indices(indices)
        template_entry = next(
            ((i, self._data[i]) for i in indices if self._data[i] is not None),
            None,
        )
        arrays = []
        for i in indices:
            data = self._data[i]
            n_elements = self.mesh[i].n_elements
            if data is None:
                if (
                    template_entry is not None
                    and np.asarray(template_entry[1]).ndim > 1
                ):
                    template_id, template = template_entry
                    template = np.asarray(template)
                    n_template_elements = self.mesh[template_id].n_elements
                    factor = template.shape[-1] // n_template_elements
                    if factor > 1:
                        n_missing = n_elements * get_default_n_gp(
                            self.mesh[i].elm_type, self.mesh[i]
                        )
                    else:
                        n_missing = n_elements
                    shape = template.shape[:-1] + (n_missing,)
                else:
                    shape = (n_elements,)
                arrays.append(np.full(shape, fill_value))
            else:
                arrays.append(np.asarray(data))
        if not arrays:
            return np.array([])
        if arrays[0].ndim == 1:
            return np.concatenate(arrays)
        return np.concatenate(arrays, axis=-1)

    def __array__(self, dtype=None):
        data = np.asarray(self.active)
        if dtype is not None:
            data = data.astype(dtype)
        return data

    def __getitem__(self, item):
        return self.active[item]

    @property
    def shape(self):
        """Shape of the active submesh block only (not the whole MultiMesh)."""
        return np.asarray(self.active).shape

    @property
    def ndim(self):
        """Number of dimensions of the active submesh block only."""
        return np.asarray(self.active).ndim


def copy_data_value(value, deep: bool = False):
    """Copy a dataset value, including per-submesh dictionaries."""
    if isinstance(value, MultiMeshData):
        value = value._data

    if isinstance(value, dict):
        return {key: copy_data_value(item, deep=deep) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        copied = [copy_data_value(item, deep=deep) for item in value]
        return type(value)(copied)

    if deep and hasattr(value, "copy"):
        return value.copy()
    return value
