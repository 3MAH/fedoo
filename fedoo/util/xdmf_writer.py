from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Any
from lxml import etree

import h5py

from .fdh5 import iteration_name


class XDMFExporter:
    """
    Export FDH5 files to XDMF (XDMF 3.0, HDF5-backed).

    - One XDMF file for the full time series
    - Multiple submeshes supported
    - Node, element and Gauss-point data
    - Gauss-point data exported in flattened GP-major form
    """

    XDMF_VERSION = "3.0"

    # FE element type -> (XDMF topology type, nodes per element)
    ELEMENT_TYPE_MAP = {
        "tri3": ("Triangle", 3),
        "quad4": ("Quadrilateral", 4),
        "tet4": ("Tetrahedron", 4),
        "hex8": ("Hexahedron", 8),
    }

    def __init__(
        self,
        h5_path: Path,
        xdmf_path: Optional[Path] = None,
    ) -> None:
        """
        Parameters
        ----------
        h5_path : Path
            Path to the source FDH5 file to describe.
        xdmf_path : Path, optional
            Output path for the ``.xdmf`` file. Defaults to ``h5_path`` with a
            ``.xdmf`` suffix.
        """
        self.h5_path = Path(h5_path)
        self.xdmf_path = xdmf_path or self.h5_path.with_suffix(".xdmf")

    def export(self) -> Path:
        """
        Generate the XDMF file for all iterations (time series).
        """
        root = etree.Element("Xdmf", Version=self.XDMF_VERSION)
        domain = etree.SubElement(root, "Domain")

        temporal = etree.SubElement(
            domain,
            "Grid",
            Name="TimeSeries",
            GridType="Collection",
            CollectionType="Temporal",
        )

        with h5py.File(self.h5_path, "r") as f:
            for it in self._list_iterations(f):
                temporal.append(self._build_iteration_grid(f, it))

        tree = etree.ElementTree(root)
        tree.write(
            str(self.xdmf_path),
            pretty_print=True,
            xml_declaration=True,
            encoding="UTF-8",
        )

        return self.xdmf_path

    def _list_iterations(self, f: h5py.File) -> Iterable[int]:
        grp = f.get("results")
        if grp is None:
            return []
        return sorted(
            int(name.split("_")[1]) for name in grp if name.startswith("iter_")
        )

    def _build_iteration_grid(self, f: h5py.File, iteration: int) -> etree.Element:
        it_name = iteration_name(iteration)
        it_grp = f[f"results/{it_name}"]

        time_value = it_grp["metadata"].attrs.get("time", iteration)

        it_grid = etree.Element(
            "Grid",
            Name=it_name,
            GridType="Collection",
            CollectionType="Spatial",
        )
        etree.SubElement(it_grid, "Time", Value=str(time_value))

        for submesh_id in self._list_submeshes(f):
            it_grid.append(self._build_submesh_grid(f, it_grp, submesh_id))

        return it_grid

    def _list_submeshes(self, f: h5py.File) -> Iterable[str]:
        mesh = f.get("mesh")
        if mesh is None:
            return []
        return sorted(name for name in mesh if name.startswith("submesh_"))

    def _build_submesh_grid(
        self,
        f: h5py.File,
        it_grp: h5py.Group,
        submesh_id: str,
    ) -> etree.Element:
        sm = f[f"mesh/{submesh_id}"]
        meta = sm["metadata"]

        element_type = meta.attrs.get("element_type")
        if isinstance(element_type, (bytes, bytearray)):
            element_type = element_type.decode("utf-8")

        if element_type not in self.ELEMENT_TYPE_MAP:
            raise ValueError(f"Unsupported element type '{element_type}'")

        topo_type, nodes_per_elem = self.ELEMENT_TYPE_MAP[element_type]

        elements = sm["elements"]
        n_elems = elements.shape[0]

        grid = etree.Element("Grid", Name=submesh_id, GridType="Uniform")

        # Topology
        topo = etree.SubElement(
            grid,
            "Topology",
            TopologyType=topo_type,
            NumberOfElements=str(n_elems),
        )

        etree.SubElement(
            topo,
            "DataItem",
            Dimensions=f"{n_elems} {nodes_per_elem}",
            NumberType="Int",
            Format="HDF",
        ).text = f"{self.h5_path.name}:/mesh/{submesh_id}/elements"

        # Geometry (global nodes)
        nodes = f["mesh/nodes"]
        n_nodes, dim = nodes.shape
        geom_type = "XYZ" if dim == 3 else "XY"

        geom = etree.SubElement(
            grid,
            "Geometry",
            GeometryType=geom_type,
        )

        etree.SubElement(
            geom,
            "DataItem",
            Dimensions=f"{n_nodes} {dim}",
            NumberType="Float",
            Precision="8",
            Format="HDF",
        ).text = f"{self.h5_path.name}:/mesh/nodes"

        # Node data (global)
        node_grp = it_grp.get("node_data")
        if node_grp is not None:
            for name, ds in node_grp.items():
                self._add_attribute(
                    grid,
                    name=name,
                    center="Node",
                    dataset_path=f"/results/{it_grp.name.split('/')[-1]}/node_data/{name}",
                    shape=ds.shape,
                )

        # Element data (per submesh)
        elem_grp = it_grp.get(f"element_data/{submesh_id}")
        if elem_grp is not None:
            for name, ds in elem_grp.items():
                self._add_attribute(
                    grid,
                    name=name,
                    center="Cell",
                    dataset_path=(
                        f"/results/{it_grp.name.split('/')[-1]}/"
                        f"element_data/{submesh_id}/{name}"
                    ),
                    shape=ds.shape,
                )

        gp_grp = it_grp.get(f"gausspoint_data/{submesh_id}")
        if gp_grp is not None:
            for name, ds in gp_grp.items():
                # Only single-Gauss-point fields map onto a cell-centered
                # attribute; multi-GP fields have no per-cell value and are
                # skipped (XDMF has no native multi-value-per-cell attribute).
                n_gauss_points = int(ds.attrs.get("n_gauss_points", 1))
                if n_gauss_points != 1:
                    continue
                self._add_attribute(
                    grid,
                    name=name,
                    center="Cell",
                    dataset_path=(
                        f"/results/{it_grp.name.split('/')[-1]}/"
                        f"gausspoint_data/{submesh_id}/{name}"
                    ),
                    shape=ds.shape,
                )

        return grid

    def _add_attribute(
        self,
        grid: etree.Element,
        *,
        name: str,
        center: str,
        dataset_path: str,
        shape: Any,
    ) -> None:
        """
        Add an Attribute element referencing an HDF5 dataset.
        """
        attr_type = "Scalar"
        if len(shape) == 2 and shape[1] == 3:
            attr_type = "Vector"
        elif len(shape) == 2 and shape[1] == 6:
            attr_type = "Tensor6"

        attr = etree.SubElement(
            grid,
            "Attribute",
            Name=name,
            AttributeType=attr_type,
            Center=center,
        )

        etree.SubElement(
            attr,
            "DataItem",
            Dimensions=" ".join(str(s) for s in shape),
            NumberType="Float",
            Precision="8",
            Format="HDF",
        ).text = f"{self.h5_path.name}:{dataset_path}"
