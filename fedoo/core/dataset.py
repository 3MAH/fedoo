"""Fedoo DataSet object."""

from __future__ import annotations

import numpy as np
import os
from zipfile import ZipFile, Path as ZipPath
from fedoo.core.mesh import Mesh, MultiMesh
from fedoo.core.multimeshdata import MultiMeshData, copy_data_value
from fedoo.lib_elements.element_list import get_default_n_gp
from fedoo.util.voigt_tensors import StressTensorList, StrainTensorList

try:
    from matplotlib import pylab as plt

    USE_MPL = True
except ImportError:
    USE_MPL = False

try:
    import pyvista as pv

    USE_PYVISTA = True
except ImportError:
    USE_PYVISTA = False
try:
    import pyvistaqt as pvqt

    USE_PYVISTA_QT = True
except ImportError:
    USE_PYVISTA_QT = False

try:
    import pandas

    USE_PANDAS = True
except ImportError:
    USE_PANDAS = False


def _as_3d_points(points: np.ndarray) -> np.ndarray:
    """Return point coordinates padded or truncated to 3D."""
    if points.shape[1] == 3:
        return points
    if points.shape[1] < 3:
        return np.column_stack(
            (points, np.zeros((points.shape[0], 3 - points.shape[1])))
        )
    return points[:, :3]


def _array_to_pyvista_data(data: np.ndarray) -> np.ndarray:
    """Return data in the component-last shape expected by pyvista."""
    data = np.asarray(data)
    if data.ndim == 1:
        return data
    return data.T


def _component_from_array(field: str, data: np.ndarray, component=None):
    """Extract one component from a Fedoo data array."""
    if (
        component is None
        or np.isscalar(data)
        or not hasattr(data, "shape")
        or len(data.shape) <= 1
    ):
        return data

    if isinstance(component, str):
        component = {
            "X": 0,
            "Y": 1,
            "Z": 2,
            "XX": 0,
            "YY": 1,
            "ZZ": 2,
            "XY": 3,
            "XZ": 4,
            "YZ": 5,
        }.get(component, component)

    if component == "norm":
        return np.linalg.norm(data, axis=0)

    if isinstance(component, str):
        if field == "Stress":
            data = StressTensorList(data)
        elif field == "Strain":
            data = StrainTensorList(data)
    return data[component]


class DataSet:
    """
    Object to store, save, load and plot data associated to a mesh.

    DataSet have a multiframe version :py:class:`fedoo.MultiFrameDataSet` that
    is a class that encapsulate several DataSet mainly usefull for time
    dependent data.

    Attributes
    ----------
    mesh : fd.Mesh
        Mesh object associated to the data
    node_data : dict
        Dictionnary of data fields defined at mesh nodes.
    element_data : dict
        Dictionnary of data fields defined at mesh elements.
    gausspoint_data : dict
        Dictionnary of data fields defined at gauss points.
    res.scalar_data : dict
        Dictionnary of scalar data.

    Notes
    -----
    Data in the node_data, element_data or gauspoint_data dictionarries
    should be provided as 1D or 2D NumPy arrays, where the last
    dimension matches the number of nodes, elements or integration
    points in the associated mesh, respectively.

    For gausspoint_data, the number of Gauss points per element is assumed
    to be constant and is inferred from the array's shape.

    If 2D NumPy arrays are provided, or 1D for scalar data,
    the first dimension corresponds to the data components.

    To access data, it is recommended to use the
    :py:meth:`fedoo.DataSet.get_data` method as it supports automatic
    conversion between different data types.

    Parameters
    ----------
    mesh : Mesh, optional
        Mesh object associated to the data. The default is None.
    data : dict, optional
        dict containing the data. The default is None.
    data_type : str in {'node', 'element', 'gausspoint', 'scalar', 'all'}
        type of data. The default is 'node'.
    """

    def __init__(
        self,
        mesh: Mesh | None = None,
        data: dict | None = None,
        data_type: str = "node",
    ) -> None:
        self.mesh = mesh
        self.node_data = {}
        self.element_data = {}
        self.gausspoint_data = {}
        self.scalar_data = {}
        self.active_submesh = 0
        """Active submesh used by MultiMesh data access.

        For ``MultiMesh`` datasets, element and Gauss point fields are returned
        as ``MultiMeshData`` objects. Plain array access to these objects points
        to ``active_submesh``. Node fields stay plain arrays; by default,
        ``get_data`` fills nodes unused by the active submesh with zero.
        """

        if isinstance(data, dict):
            data_type = data_type.lower()
            if data_type == "node":
                self.node_data = data
            elif data_type == "element":
                self.element_data = data
            elif data_type == "gausspoint":
                self.gausspoint_data = data
            elif data_type == "scalar":
                self.scalar_data = data
            elif data_type == "all":
                self.node_data = {k: v for k, v in data.items() if k[-2:] == "nd"}
                self.element_data = {k: v for k, v in data.items() if k[-2:] == "el"}
                self.gausspoint_data = {k: v for k, v in data.items() if k[-2:] == "gp"}
                self.scalar_data = {k: v for k, v in data.items() if k[-2:] == "sc"}

        self.meshplot = None
        self.meshplot_gp = None  # a mesh with discontinuity between each element to plot gauss points field

    def _is_multimesh(self) -> bool:
        return isinstance(self.mesh, MultiMesh)

    def _resolve_submesh_indices(self, selector=None) -> list[int]:
        if not self._is_multimesh():
            return [0]
        if selector is None:
            return list(range(len(self.mesh.submeshes)))
        if isinstance(selector, (list, tuple, set, np.ndarray)):
            indices = []
            for item in selector:
                indices.extend(self._resolve_submesh_indices(item))
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

    def _active_submesh_index(self) -> int:
        return self._resolve_submesh_indices(self.active_submesh)[0]

    def global_element_location(self, element_id: int) -> tuple[int, int]:
        """Return ``(submesh_id, local_element_id)`` for a global element id.

        For regular ``Mesh`` datasets, the returned submesh id is always 0.
        For ``MultiMesh`` datasets, global element ids follow the same
        concatenated order used by ``to_pyvista`` and multimesh plotting: all
        elements of submesh 0, then all elements of submesh 1, and so on.
        """
        element_id = int(element_id)
        if not self._is_multimesh():
            if element_id < 0 or element_id >= self.mesh.n_elements:
                raise IndexError(element_id)
            return 0, element_id

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

        if not self._is_multimesh():
            if submesh_id != 0:
                raise IndexError(submesh_id)
            if local_element_id < 0 or local_element_id >= self.mesh.n_elements:
                raise IndexError(local_element_id)
            return local_element_id

        if submesh_id < 0 or submesh_id >= len(self.mesh.submeshes):
            raise IndexError(submesh_id)
        submesh = self.mesh[submesh_id]
        if local_element_id < 0 or local_element_id >= submesh.n_elements:
            raise IndexError(local_element_id)
        return (
            sum(mesh.n_elements for mesh in self.mesh.submeshes[:submesh_id])
            + local_element_id
        )

    def split_global_element_indices(self, element_ids) -> dict[int, np.ndarray]:
        """Split global element ids into submesh-local element ids.

        Parameters
        ----------
        element_ids : array-like of int
            Element ids in the concatenated global numbering used by
            multimesh plotting and viewer selections.

        Returns
        -------
        dict[int, numpy.ndarray]
            Mapping ``submesh_id -> local_element_ids``. Empty submeshes are
            omitted.
        """
        element_ids = np.asarray(element_ids, dtype=int)
        if element_ids.ndim == 0:
            element_ids = element_ids.reshape(1)

        if not self._is_multimesh():
            if np.any((element_ids < 0) | (element_ids >= self.mesh.n_elements)):
                raise IndexError(element_ids)
            return {0: element_ids}

        split_indices = {}
        offset = 0
        for submesh_id, submesh in enumerate(self.mesh.submeshes):
            stop = offset + submesh.n_elements
            local_ids = (
                element_ids[(element_ids >= offset) & (element_ids < stop)] - offset
            )
            if len(local_ids):
                split_indices[submesh_id] = local_ids
            offset = stop

        if len(split_indices) == 0 and len(element_ids) > 0:
            raise IndexError(element_ids)
        return split_indices

    def _selected_multimesh(self, selected_submeshes=None):
        if not self._is_multimesh():
            return self.mesh, [0]
        indices = self._resolve_submesh_indices(selected_submeshes)
        mesh = MultiMesh.from_mesh_list(
            [self.mesh[i] for i in indices],
            name=self.mesh.name,
            node_sets=self.mesh.node_sets,
            register_name=False,
        )
        return mesh, indices

    def _as_multimesh_data(self, data) -> MultiMeshData:
        return MultiMeshData(
            self.mesh,
            data,
            active_submesh=self._active_submesh_index(),
        )

    def _convert_multimesh_data(self, data, convert_from: str, convert_to: str):
        """Convert MultiMesh data block-by-block on each submesh."""
        active_submesh = self._active_submesh_index()

        if convert_from == convert_to:
            if convert_to in ["Element", "GaussPoint"]:
                return self._as_multimesh_data(data)
            return data

        if isinstance(data, MultiMeshData):
            if convert_to == "Node":
                block = data.submesh(active_submesh)
                if block is None:
                    raise NameError("Field data not found on the active submesh.")
                return self.mesh[active_submesh].convert_data(
                    np.asarray(block),
                    convert_from=convert_from,
                    convert_to=convert_to,
                )

            converted = {
                submesh_id: self.mesh[submesh_id].convert_data(
                    np.asarray(block),
                    convert_from=convert_from,
                    convert_to=convert_to,
                )
                for submesh_id, block in data.items()
                if block is not None
            }
            return MultiMeshData(
                self.mesh,
                converted,
                active_submesh=active_submesh,
            )

        if convert_from == "Node" and convert_to in ["Element", "GaussPoint"]:
            converted = {
                submesh_id: submesh.convert_data(
                    data,
                    convert_from=convert_from,
                    convert_to=convert_to,
                )
                for submesh_id, submesh in enumerate(self.mesh.submeshes)
            }
            return MultiMeshData(
                self.mesh,
                converted,
                active_submesh=active_submesh,
            )

        return self.mesh[active_submesh].convert_data(
            data,
            convert_from=convert_from,
            convert_to=convert_to,
        )

    def __getitem__(self, items):
        if isinstance(items, tuple):
            return self.get_data(*items)
        else:
            return self.get_data(items)

    def add_data(self, data_set: "DataSet") -> None:
        """
        Update the DataSet object including all the node, element and gausspoint
        data from antoher DataSet object data_set. The associated mesh is not
        modified.
        """
        self.node_data.update(data_set.node_data)
        self.element_data.update(data_set.element_data)
        self.gausspoint_data.update(data_set.gausspoint_data)
        self.scalar_data.update(data_set.scalar_data)

    def _build_mesh_gp(self):
        # define a new mesh for the plot to gauss point (duplicate nodes between element)
        crd = self.mesh.nodes
        elm = self.mesh.elements
        nodes_gp = crd[elm.ravel()]
        element_gp = np.arange(elm.shape[0] * elm.shape[1]).reshape(-1, elm.shape[1])
        self.mesh_gp = self.mesh.__class__(nodes_gp, element_gp, self.mesh.elm_type)
        self.meshplot_gp = self.mesh_gp.to_pyvista()

    def plot(
        self,
        field: str | None = None,
        component: int | str = 0,
        data_type: str | None = None,
        scale: float = 1,
        show: bool = True,
        show_edges: bool = True,
        clim: list[float] | None = None,
        node_labels: bool | list = False,
        element_labels: bool | list = False,
        show_nodes: bool | float = False,
        show_normals: bool | float = False,
        plotter: object = None,
        screenshot: str | None = None,
        azimuth: float = 30.0,
        elevation: float = 15.0,
        roll: float = 0,
        title: str | None = None,
        title_size: float = 18.0,
        window_size: list = None,
        multiplot: bool | None = None,
        element_set: str | np.ndarray[int] | None = None,
        element_set_invert: bool = False,
        selected_submeshes=None,
        clip_args: tuple | None = None,
        lock_view: bool = False,
        iteration: int | None = None,
        **kargs,
    ) -> None:
        """Plot a field on the surface of the associated mesh.

        Parameters
        ----------
        field : str (optional)
            The name of the field to plot. If no name is given, plot only the
            mesh.

        component : int | str, default = 0
            The data component to plot in case of vector data.
            The available str components are:

            * 'X', 'Y' and 'Z'; respectively equivalent to 0, 1 and 2
              for vector components.
            * 'XX', 'YY', 'ZZ', 'XY', 'XZ' and 'YZ' are respectively
              equivalent to 0, 1, 2, 3, 4 and 5 for tensor using
              the voigt notations.
            * 'vm' to plot the von-mises stress from a stress field.
            * 'pressure' to extract the hydrostatic pressure of a stress field.
            * 'norm' to compute the vector euclidean norm.

        data_type : str in {'Node', 'Element', 'GaussPoint'} - Optional
            The type of data to plot (defined at nodes, elements au gauss
            integration points). If the existing data doesn't match to the
            specified one, the data are converted before plotted.
            For instance data_type = 'Node' make en average of data from
            adjacent elements at nodes. This allow a more smooth plot.
            It the type is not specified, look for any type of data and, if the
            data is found, draw the field without conversion.

        scale : float, default = 1
            The scale factor used for the nodes displacement, using the 'Disp'
            vector field.
            If scale = 0, the field is plotted on the underformed shape.

        show : bool, default = True

            * If show = True, the plot is rendered in a new window.
            * If show = False, the current pyvista plotter is returned without
              rendering.
            * show = False allow to customize the plot with pyvista before
              rendering it.

        show_edges : bool, default = True
            if True, the mesh edges are shown

        clim : sequence[float], optional
            Sequence of two float to define data boundaries for color bar.
            Defaults to minimum and maximum of data.

        node_labels : bool | list, default = False
            If True, show node labels (node indexe)
            If a list is given, print the label given in node_labels[i] for
            each node i.

        element_labels : bool | list, default = False
            If True, show element labels (element indexe)
            If a list is given, print the label given in element_labels[i] for
            each element i.

        show_nodes : bool|float, default = False
            Plot the nodes. If True, the nodes are shown with a default size.
            If float, show_nodes is the required size.

        show_normals : bool|float, default = False
            Plot the face normals. If True,
            the vectors are shown with a default magnitude.
            If float, show_normals is the required magnitude.
            Only available for 1D or 2D mesh.

        plotter : pyvista.Plotter object or str in {'qt', 'pv'}

            * If pyvista.Plotter object, plot the mesh in the given plotter
            * If 'qt': use the background plotter of pyvistaqt (need the lib
              pyvistaqt)
            * If 'pv': use the standard pyvista plotter
            * If None: use the background plotter if available, or pyvista
              plotter if not.

        screenshot : str, optional
            If defined, indicated a filename to save the plot.

        azimuth : float, default = 30.
            Azimuth angle of the camera around the scene
            (not used for 2D scene)

        elevation : float, default = 15.
            Elevaltion angle of the camera around the scene
            (not used for 2D scene).

        roll : float, default = 0
            Roll angle of the camera. The default state (roll angle = 0.) is
            set with the y direction on the up.

        title : str | None, default = None
            Title of the plot. By default the title is field name
            and the component is printed.

        title_size : float, default = 18
            Size of the title

        window_size : tuple, default = (1024, 768)
            Window size in pixels.

        multiplot : bool | None, default = None
            If True, the pyvista mesh is copied to force a separated scalar
            bar. This is usefull when ploting several figures at the same time.
            If multiplot si False, the same scalarbar will be applied to
            all the plots.
            If None, uses separated scalarbars only if the pyvista plotter uses
            subplot.

        element_set : str or array[int], optional
            Name of an element set associated to the mesh (str), or list of
            element set indices. If specified, plot only the given elements
            or hide the element set if element_set_invert == True.

        element_set_invert : bool, optional
            Used only if element_set is defined. Invert element set.

        selected_submeshes : int, str or list, optional
            Only used with MultiMesh objects. Restrict the plot to the selected
            submesh ids, submesh names or element types.

        clip_args : dict, optional
            Dictionary of arguments to pass to the pyvista clip filter in
            order to clip the current plot.
            If clip_args["cell_ids"] exist and is set to True, the "cell_ids"
            array is passed to the pyvista mesh to track the original element
            IDs in the clipped mesh.

        lock_view : bool, default = False
            If ``True``, the camera position and background color are not
            modified. In this mode, any view‑modifying arguments such as
            ``azimuth``, ``elevation``, or ``roll`` are ignored.

        iteration: int, optional
            ignored if the object is not a MultiFrameDataSet.
            Index of the iteration to plot. If None, the current iteration is
            plotted. If no current iteration is defined, the last iteration
            is loaded and plotted.

        **kwargs : dict
            See pyvista.Plotter.add_mesh() in the document of pyvista for
            additional usefull options.

        Notes
        -----
        If the package pyvistaqt is installed, the BackgroundPlotter is used
        by default. To desactivate pyvistaqt, set the fedoo config:

            >>> fedoo.get_config()['USE_PYVISTA_QT'] = False
        """
        if not (USE_PYVISTA):
            raise ImportError("Pyvista not installed.")

        if self.mesh is None:
            raise NameError(
                "Can't generate a plot without an associated mesh. "
                "Set the mesh attribute first."
            )

        if hasattr(self, "loaded_iter"):
            if iteration is None:
                if self.loaded_iter is None:
                    self.load(-1)  # load last iteration
            else:
                self.load(iteration)

        ndim = self.mesh.ndim

        field = kargs.pop(
            "scalars", field
        )  # kargs scalars can be used instead of field

        if self._is_multimesh():
            return self._plot_multimesh(
                field=field,
                component=component,
                data_type=data_type,
                scale=scale,
                show=show,
                show_edges=show_edges,
                clim=clim,
                node_labels=node_labels,
                element_labels=element_labels,
                show_nodes=show_nodes,
                plotter=plotter,
                screenshot=screenshot,
                azimuth=azimuth,
                elevation=elevation,
                roll=roll,
                title=title,
                title_size=title_size,
                window_size=window_size,
                multiplot=multiplot,
                element_set=element_set,
                element_set_invert=element_set_invert,
                selected_submeshes=selected_submeshes,
                clip_args=clip_args,
                lock_view=lock_view,
                return_cpos=kargs.pop("return_cpos", False),
                **kargs,
            )

        if field is not None:
            data, data_type = self.get_data(field, component, data_type, True)
        else:
            data_type = None

        if screenshot is None:
            screenshot = False  # not used if show = False

        return_cpos = kargs.pop("return_cpos", False)
        cmap = kargs.pop("cmap", "jet")  # if cmap not defined, default to "jet"
        extra_cell_data = kargs.pop("_extra_cell_data", None)

        if data_type == "GaussPoint":
            if self.meshplot_gp is None:
                self._build_mesh_gp()
            meshplot = self.meshplot_gp

            data = self.mesh_gp.convert_data(
                data,
                convert_from="GaussPoint",
                convert_to="Node",
                n_elm_gp=len(data) // self.mesh.n_elements,
            )
            if "Disp" in self.node_data and scale != 0:
                ndim = self.mesh.ndim
                U = (
                    (
                        self.node_data["Disp"]
                        .reshape(ndim, -1)
                        .T[self.mesh.elements.ravel()]
                    ).T
                ).T
                # meshplot.point_data['Disp'] = U
                meshplot.points = as_3d_coordinates(self.mesh_gp.nodes + scale * U)

                if show_nodes:
                    # compute center (dont use meshplot to compute center because
                    # isolated nodes are removed -> may be annoying with show_nodes)
                    crd = self.mesh.nodes + scale * self.node_data["Disp"].T
                    center = 0.5 * (crd.min(axis=0) + crd.max(axis=0))
                    if len(center) < 3:
                        center = np.hstack((center, np.zeros(3 - len(center))))
                else:
                    meshplot.ComputeBounds()
                    center = meshplot.center
            else:
                meshplot.points = as_3d_coordinates(self.mesh_gp.nodes)
                center = self.mesh.as_3d().bounding_box.center

        else:
            if self.meshplot is None:
                meshplot = self.meshplot = self.mesh.to_pyvista()
            else:
                meshplot = self.meshplot

            if "Disp" in self.node_data and scale != 0:
                meshplot.points = as_3d_coordinates(
                    self.mesh.nodes + scale * self.node_data["Disp"].T
                )
            else:
                meshplot.points = as_3d_coordinates(self.mesh.nodes)

            center = 0.5 * (meshplot.points.min(axis=0) + meshplot.points.max(axis=0))

        if extra_cell_data:
            for key, value in extra_cell_data.items():
                meshplot.cell_data[key] = np.asarray(value)

        backgroundplotter = True
        if USE_PYVISTA_QT and (plotter is None or plotter == "qt"):
            # use pyvistaqt plotter
            pl = pvqt.BackgroundPlotter(window_size=window_size)
        elif plotter is None or plotter == "pv":
            # default pyvista plotter
            backgroundplotter = False
            if screenshot:
                pl = pv.Plotter(off_screen=True, window_size=window_size)
            else:
                pl = pv.Plotter(window_size=window_size)
        else:
            # try to use the given plotter
            # dont show
            pl = plotter

        if "name" not in kargs:
            # add default name = "data{i}"
            i = 1
            while f"data{i}" in pl.actors.keys():
                i += 1
            kargs["name"] = f"data{i}"

        if multiplot is None:
            if pl.renderers.shape == (1, 1):
                multiplot = False
            else:
                multiplot = True

        if not lock_view:
            pl.set_background("White")
            # camera position
            # meshplot.ComputeBounds()
            # center = meshplot.center
            pl.camera.SetFocalPoint(center)
            pl.camera.position = tuple(center + np.array([0, 0, 2 * meshplot.length]))
            pl.camera.up = tuple([0, 1, 0])
            if roll != 0:
                pl.camera.Roll(roll)

            if ndim == 3:
                pl.camera.Azimuth(azimuth)
                pl.camera.Elevation(elevation)

        # default sargs values
        # if sargs is None and field is not None:  # default value
        if multiplot:
            # scalarbar can't be interactive in multiplot
            sargs = dict(
                label_font_size=int(
                    pl.window_size[1] / pl.renderers.shape[1] * 0.6 / 22
                ),
                color="Black",
                position_x=0.2,
                width=0.6,
                # n_colors= 10
            )
        else:
            sargs = dict(
                interactive=True,
                title_font_size=20,
                label_font_size=16,
                color="Black",
                # n_colors= 10
            )
        sargs.update(kargs.pop("scalar_bar_args", {}))

        if multiplot and "title" not in sargs:
            # title use as scalar_bar id required to plot several scalar bar
            sargs["title"] = f"{pl.renderers.active_index}"
            sargs["title_font_size"] = 1

        mesh_to_show = meshplot
        if element_set is not None or clip_args is not None:
            # add data field to mesh object to clip data with the mesh
            if data_type == "Element":
                mesh_to_show.cell_data["Data"] = data
            elif data_type:
                mesh_to_show.point_data["Data"] = data

            if element_set is not None:
                if len(element_set) == 0:
                    return pl
                if isinstance(element_set, str):
                    element_set = self.mesh.element_sets[element_set]
                mesh_to_show = mesh_to_show.extract_cells(
                    element_set,
                    invert=element_set_invert,
                )

            if clip_args is not None:
                if clip_args.pop("cell_ids", False):
                    if "vtkOriginalCellIds" in mesh_to_show.cell_data:
                        mesh_to_show.cell_data["cell_ids"] = mesh_to_show.cell_data[
                            "vtkOriginalCellIds"
                        ]
                    else:
                        mesh_to_show.cell_data["cell_ids"] = np.arange(
                            self.mesh.n_elements
                        )
                mesh_to_show = mesh_to_show.clip(**clip_args)

            if data_type == "Element":
                data = mesh_to_show.cell_data["Data"]
            elif data_type:
                data = mesh_to_show.point_data["Data"]

        edges = None
        if multiplot:
            # The cached plotting mesh is updated in place when another result
            # iteration is loaded. Keep each actor's geometry independent.
            mesh_to_show = mesh_to_show.copy(deep=True)

        if show_edges and self.mesh.elm_type in [
            "tri6",
            "quad8",
            "quad9",
            "hex20",
            "tet10",
            "wed15",
            "wed18",
        ]:
            # patch to correct edges visualization in 2nd ordre elements
            show_edges = False
            edges = (
                mesh_to_show.separate_cells()
                .extract_surface(nonlinear_subdivision=4)
                .extract_feature_edges()
            )

        if mesh_to_show.is_empty:
            return pl

        if field is None:
            mesh_to_show.active_scalars_name = None
            pl.add_mesh(
                mesh_to_show,
                show_edges=show_edges,
                **kargs,
            )
            if title is None:
                title = ""
        else:
            pl.add_mesh(
                mesh_to_show,
                scalars=data,
                show_edges=show_edges,
                scalar_bar_args=sargs,
                cmap=cmap,
                clim=clim,
                **kargs,
            )
            if title is None:
                title = f"{field}_{component}"

        if edges:
            pl.add_mesh(edges, color="black", line_width=1.7, name="edges1")

        pl.add_text(title, name="name", color="Black", font_size=title_size)

        if not lock_view:
            pl.add_axes(color="Black", interactive=True)

        # Node and Element Labels and plot points
        if node_labels or show_nodes:  # extract nodes coordinates
            if data_type == "GaussPoint":
                if "Disp" in self.node_data:
                    crd_labels = as_3d_coordinates(
                        self.mesh.nodes + self.node_data["Disp"].T
                    )
                else:
                    crd_labels = as_3d_coordinates(self.mesh.nodes)
            else:
                crd_labels = meshplot.points

        if node_labels:
            if node_labels == True:
                node_labels = list(range(self.mesh.n_nodes))
            pl.add_point_labels(crd_labels, node_labels)

        if element_labels:
            if element_labels == True:
                if "cell_ids" in mesh_to_show.cell_data:
                    element_labels = mesh_to_show.cell_data["cell_ids"]
                    pl.add_point_labels(mesh_to_show.cell_centers(), element_labels)
                else:
                    element_labels = list(range(self.mesh.n_elements))
                    pl.add_point_labels(meshplot.cell_centers(), element_labels)

        if show_nodes:
            if show_nodes == True:
                show_nodes = 5
            pl.add_points(
                crd_labels,
                render_points_as_spheres=True,
                point_size=show_nodes,
            )

        if show_normals:
            if show_normals == True:
                show_normals = 1.0  # normal magnitude

            centers = self.mesh.element_centers
            if self.mesh.elm_type[:3] not in ["lin", "tri", "qua"]:
                raise NameError(
                    "Can't plot normals for volume meshes. Use fedoo.mesh.extract_surface to get a compatible mesh."
                )
            normals = self.mesh.get_element_local_frame()[:, -1]

            if ndim < 3:
                normals = np.column_stack(
                    (normals, np.zeros((self.mesh.n_elements, 3 - ndim)))
                )
                centers = np.column_stack(
                    (
                        self.mesh.element_centers,
                        np.zeros((self.mesh.n_elements, 3 - ndim)),
                    )
                )

            pl.add_arrows(centers, normals, mag=show_normals, show_scalar_bar=False)

        # required to avoid bug for non adapted clipping range
        # pl.camera.reset_clipping_range()
        pl.renderer.ResetCameraClippingRange()

        if screenshot:
            ext = os.path.splitext(screenshot)[1]
            ext = ext.lower()
            if ext in [".pdf", ".svg", ".eps", ".ps", ".tex"]:
                pl.save_graphic(screenshot)
            else:
                pl.screenshot(screenshot)

            return pl

        if not (backgroundplotter) and show:
            return pl.show(return_cpos=return_cpos)

        return pl

    def _multimesh_block_for_submesh(
        self,
        data,
        submesh_id: int,
        n_items: int,
        fill_missing: bool = True,
    ):
        """Return one submesh data block without copying when possible."""
        multimesh_data = self._as_multimesh_data(data)
        block = multimesh_data.submesh(submesh_id)
        if block is not None or not fill_missing:
            return block

        template_entry = next(
            (item for item in multimesh_data.items() if item[1] is not None),
            None,
        )
        if template_entry is None:
            return None

        template_id, template = template_entry
        template = np.asarray(template)
        if template.ndim > 1:
            shape = template.shape[:-1] + (n_items,)
        else:
            shape = (n_items,)
        return np.zeros(shape, dtype=template.dtype)

    def _submesh_dataset(self, submesh_id: int) -> "DataSet":
        """Build a lightweight DataSet view attached to one submesh."""
        submesh = self.mesh[submesh_id]
        dataset = DataSet(submesh)
        dataset.node_data = self.node_data
        dataset.scalar_data = self.scalar_data
        dataset.element_data = {
            field: block
            for field, value in self.element_data.items()
            if (
                block := self._multimesh_block_for_submesh(
                    value,
                    submesh_id,
                    submesh.n_elements,
                )
            )
            is not None
        }
        dataset.gausspoint_data = {
            field: block
            for field, value in self.gausspoint_data.items()
            if (
                block := self._multimesh_block_for_submesh(
                    value,
                    submesh_id,
                    submesh.n_elements * get_default_n_gp(submesh.elm_type, submesh),
                )
            )
            is not None
        }
        return dataset

    def _submesh_element_set(
        self,
        element_set,
        submesh_id: int,
        offsets: dict[int, int],
        global_element_set: bool,
        element_set_invert: bool,
    ):
        """Map a MultiMesh element selection to one submesh."""
        if element_set is None:
            return None, True

        submesh = self.mesh[submesh_id]
        if isinstance(element_set, str):
            if element_set in submesh.element_sets:
                return element_set, True
            return None, bool(element_set_invert)

        element_ids = np.asarray(element_set, dtype=int)
        if not global_element_set:
            return element_ids, True

        offset = offsets[submesh_id]
        local_ids = (
            element_ids[
                (element_ids >= offset) & (element_ids < offset + submesh.n_elements)
            ]
            - offset
        )
        if len(local_ids) == 0 and not element_set_invert:
            return None, False
        if len(local_ids) == 0 and element_set_invert:
            return None, True
        return local_ids, True

    def _plot_multimesh(
        self,
        *,
        field,
        component,
        data_type,
        scale,
        show,
        show_edges,
        clim,
        node_labels,
        element_labels,
        show_nodes,
        plotter,
        screenshot,
        azimuth,
        elevation,
        roll,
        title,
        title_size,
        window_size,
        multiplot,
        element_set,
        element_set_invert,
        selected_submeshes,
        clip_args,
        lock_view,
        return_cpos,
        **kargs,
    ):
        """Plot a MultiMesh by reusing the single-mesh plotting path."""
        global_element_set = kargs.pop("global_element_set", False)
        submesh_indices = self._resolve_submesh_indices(selected_submeshes)
        if (
            element_set is not None
            and not isinstance(element_set, str)
            and not global_element_set
        ):
            submesh_indices = [self._active_submesh_index()]

        offsets = {}
        offset = 0
        for i, submesh in enumerate(self.mesh.submeshes):
            offsets[i] = offset
            offset += submesh.n_elements

        if screenshot and (plotter is None or plotter == "pv"):
            pl = pv.Plotter(off_screen=True, window_size=window_size)
            backgroundplotter = False
        else:
            pl = plotter
            backgroundplotter = not (plotter is None or plotter == "pv")
            if USE_PYVISTA_QT and (plotter is None or plotter == "qt"):
                backgroundplotter = True

        base_name = kargs.pop("name", None)
        scalar_bar_added = False
        plotted = False

        for submesh_id in submesh_indices:
            local_element_set, should_plot = self._submesh_element_set(
                element_set,
                submesh_id,
                offsets,
                global_element_set,
                element_set_invert,
            )
            if not should_plot:
                continue

            subdataset = self._submesh_dataset(submesh_id)
            if field is not None:
                try:
                    subdataset.get_data(field, component, data_type)
                except NameError:
                    continue
            sub_kargs = dict(kargs)
            if base_name is not None:
                sub_kargs["name"] = f"{base_name}_{submesh_id}"
            if field is not None and scalar_bar_added:
                sub_kargs["show_scalar_bar"] = False
            sub_kargs["_extra_cell_data"] = {
                "_fedoo_global_cell_ids": (
                    np.arange(subdataset.mesh.n_elements, dtype=int)
                    + offsets[submesh_id]
                ),
                "_fedoo_submesh_id": np.full(
                    subdataset.mesh.n_elements,
                    submesh_id,
                    dtype=int,
                ),
            }
            if clip_args is not None:
                sub_clip_args = dict(clip_args)
            else:
                sub_clip_args = None

            pl = subdataset.plot(
                field=field,
                component=component,
                data_type=data_type,
                scale=scale,
                show=False,
                show_edges=show_edges,
                clim=clim,
                node_labels=False,
                element_labels=element_labels,
                show_nodes=False,
                plotter=pl,
                screenshot=None,
                azimuth=azimuth,
                elevation=elevation,
                roll=roll,
                title="",
                title_size=title_size,
                window_size=window_size,
                multiplot=multiplot,
                element_set=local_element_set,
                element_set_invert=element_set_invert,
                clip_args=sub_clip_args,
                lock_view=True,
                return_cpos=return_cpos,
                **sub_kargs,
            )
            plotted = True
            if field is not None:
                scalar_bar_added = True

        if pl is None:
            pl = pv.Plotter(window_size=window_size)
            backgroundplotter = False

        if field is not None and not plotted:
            raise NameError(f"Field data {field!r} not found on any selected submesh.")

        if title is None:
            title = "" if field is None else f"{field}_{component}"
        pl.add_text(title, name="name", color="Black", font_size=title_size)

        if not lock_view:
            pl.set_background("White")
            points = self.mesh.nodes
            if "Disp" in self.node_data and scale != 0:
                points = points + scale * np.asarray(self.node_data["Disp"]).T
            points = _as_3d_points(points)
            center = 0.5 * (points.min(axis=0) + points.max(axis=0))
            length = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
            if length == 0:
                length = 1
            pl.camera.SetFocalPoint(center)
            pl.camera.position = tuple(center + np.array([0, 0, 2 * length]))
            pl.camera.up = tuple([0, 1, 0])
            if roll != 0:
                pl.camera.Roll(roll)
            if self.mesh.ndim == 3:
                pl.camera.Azimuth(azimuth)
                pl.camera.Elevation(elevation)
            pl.add_axes(color="Black", interactive=True)

        if plotted and (node_labels or show_nodes):
            crd_labels = self.mesh.nodes
            if "Disp" in self.node_data and scale != 0:
                crd_labels = crd_labels + scale * np.asarray(self.node_data["Disp"]).T
            crd_labels = _as_3d_points(crd_labels)
            if node_labels:
                if node_labels is True:
                    node_labels = list(range(self.mesh.n_nodes))
                pl.add_point_labels(crd_labels, node_labels)
            if show_nodes:
                if show_nodes is True:
                    show_nodes = 5
                pl.add_points(
                    crd_labels,
                    render_points_as_spheres=True,
                    point_size=show_nodes,
                )

        pl.renderer.ResetCameraClippingRange()

        if screenshot:
            ext = os.path.splitext(screenshot)[1].lower()
            if ext in [".pdf", ".svg", ".eps", ".ps", ".tex"]:
                pl.save_graphic(screenshot)
            else:
                pl.screenshot(screenshot)
            return pl

        if not backgroundplotter and show:
            return pl.show(return_cpos=return_cpos)

        return pl

    def get_data(
        self,
        field,
        component=None,
        data_type=None,
        return_data_type=False,
        *,
        fill_unused_nodes=0.0,
    ):
        """Retrieve data from the DataSet for a given field.

        This method is equivalent to the `DataSet.__getitem__` magic method.
        One may prefer to use the shorthand syntax:
        `dataset[field, component, data_type]`.

        Parameters
        ----------
        field : str
            Name of the data field to retrieve.
        component : int, str or None, optional
            Index or label of the component to extract if the data is
            multi-dimensional.
            If None, all components are returned.
        data_type : str or None, optional
            Desired data type to convert to. Can be one of 'Node', 'Element',
            or 'GaussPoint'.
            If None, the original data type is preserved.
        return_data_type : bool, optional
            If True, the method returns a tuple `(data, data_type)`
            where `data_type` is the type of the returned data. If False, only
            the data is returned.
        fill_unused_nodes : scalar, optional
            Value used for nodes that are not used by the active submesh when a
            MultiMesh node field is restricted to one submesh. Default is 0.
            If ``None``, the original node array is returned unchanged.

        Returns
        -------
        data : np.ndarray or MultiMeshData
            The requested data, possibly converted to the specified type. For a
            ``MultiMesh`` dataset, element and Gauss point fields are returned
            as ``MultiMeshData`` objects. Node fields remain NumPy arrays.
        data_type : str, optional
            Returned only if `return_data_type` is True.
            Indicates the type of the returned data.

        Notes
        -----
        This method supports automatic conversion between node, element,
        and Gauss point data types when applicable.

        With a ``MultiMesh``, element and Gauss point fields may be stored as a
        dictionary keyed by submesh id, submesh name, or unique element type.
        The active submesh is controlled by ``dataset.active_submesh``.

        Element ids in a ``MultiMesh`` use a global, concatenated numbering:
        all elements of submesh 0, then all elements of submesh 1, and so on.
        Retrieve values using those ids from the returned ``MultiMeshData``::

            stress = dataset.get_data("Stress", "vm", "Element")
            value = stress.global_element_value(global_element_id)
            values = stress.global_element_values(global_element_ids)

        Use ``stress.to_global()`` when a single NumPy array in this global
        order is needed. ``np.asarray(stress)`` instead returns the data of
        the active submesh.
        """

        if data_type is None:  # search if field exist somewhere
            if field in self.node_data:
                data_type = "Node"
            elif field in self.element_data:
                data_type = "Element"
            elif field in self.gausspoint_data:
                data_type = "GaussPoint"
            elif field in self.scalar_data:
                data_type = "Scalar"
            else:
                raise NameError("Field data not found.")
            data = self.dict_data[data_type][field]
        else:
            if field in self.dict_data[data_type]:
                data = self.dict_data[data_type][field]
            else:  # if field is not present whith the given data_type search if it exist elsewhere and convert it
                # Fetch the source field with all nodes intact: the fill applies
                # only to a Node request restricted to the active submesh. When
                # converting to Element/GaussPoint data every submesh is used,
                # so masking the source nodes here would zero the other
                # submeshes' converted values.
                data, current_data_type = self.get_data(
                    field,
                    component,
                    return_data_type=True,
                    fill_unused_nodes=None,
                )
                if current_data_type != "Scalar":
                    if self._is_multimesh():
                        data = self._convert_multimesh_data(
                            data,
                            convert_from=current_data_type,
                            convert_to=data_type,
                        )
                    else:
                        data = self.mesh.convert_data(
                            data,
                            convert_from=current_data_type,
                            convert_to=data_type,
                        )

        if self._is_multimesh() and data_type in ["Element", "GaussPoint"]:
            data = self._as_multimesh_data(data)

        if isinstance(data, MultiMeshData):
            data = data.map(lambda val: _component_from_array(field, val, component))
        else:
            data = _component_from_array(field, data, component)

        if (
            self._is_multimesh()
            and data_type == "Node"
            and fill_unused_nodes is not None
            and field in self.node_data
        ):
            active_mesh = self.mesh[self._active_submesh_index()]
            used_nodes = np.unique(active_mesh.elements)
            unused_nodes = np.setdiff1d(np.arange(self.mesh.n_nodes), used_nodes)
            if len(unused_nodes):
                data = np.array(data, copy=True)
                data[..., unused_nodes] = fill_unused_nodes

        if return_data_type:
            return data, data_type
        else:
            return data

    def field_names(self):
        return list(
            set(
                list(self.gausspoint_data.keys())
                + list(self.node_data.keys())
                + list(self.element_data.keys())
            )
        )

    def save(
        self, filename: str, save_mesh: bool = False, compressed: bool = False
    ) -> None:
        """Save data to a file.
        File type is inferred from the extension of the filename.

        The available file types are:
            * 'fdz': A zipped archive containing the mesh using the 'vtk' format named '_mesh_.vtk',
              and data from several iterations named 'iter_x.npz' where x is the iteration number
              (x=0 for the 1st iteration).
            * 'vtk': The vtk format contains the mesh and the data in a single files. The gauss
              points data are not included in the file.
              This format is efficient for a linear problem when we need only one time
              iteration. In case of multiple saved iterations, a directory is created and
              one vtk file is saved per iteration. The mesh is included in every file
              which is not memory efficient.
            * 'msh': Format associated to gmsh. Have the same drawback as the vtk format for
              time depend results and missing gauss points data. The vtk format should be prefered.
            * 'npz': Save data in a numpy file npz which doesn't include the mesh. The mesh
              is generally saved beside in a raw vtk files without results.
            * 'csv': Save DataSet that contains only one type of data
              (ie Node, Element or Gauss point data) in a csv file (need the library
              pandas installed).
              The mesh is not included and may be saved beside in a vtk file.
            * 'xlsx': Same as csv but with the excel format.
            * 'fdh5': HDF5 format for Fedoo meshes and results, including
              MultiMesh element and Gauss point fields.

        Parameters
        ----------
        filename : str
            Name of the file including the path.
        save_mesh : bool, default = False
            If True, the mesh is also saved in a vtk file using the same filename with a '.vtk' extention.
            For vtk and msh file, the mesh is always included in the file and save_mesh have no effect.
        compressed : bool, default = False
            If True, the file is compressed if available (only for npz and fdz files)
        """
        ext = os.path.splitext(filename)[1]
        ext = ext.lower()
        if ext == "":
            ext = ".fdh5"
            filename = filename + ext
        if ext == ".vtk":
            self.to_vtk(filename)
        elif ext == ".msh":
            self.to_msh(filename)
        elif ext == ".npz":
            if compressed:
                self.savez_compressed(filename, save_mesh)
            else:
                self.savez(filename, save_mesh)
        elif ext == ".csv":
            self.to_csv(filename, save_mesh)
        elif ext == ".xlsx":
            self.to_excel(filename, save_mesh)
        elif ext == ".fdz":
            self.to_fdz(
                filename, save_mesh=True, compressed=compressed
            )  # create a new file and add the mesh
        elif ext == ".fdh5":
            self.to_fdh5(filename, iteration=0, overwrite=True)

    def save_mesh(self, filename: str):
        """Save the mesh using a vtk file. The extension of filename is ignored and modified to '.vtk'."""
        name = os.path.splitext(filename)[0]
        self.mesh.save(name)

    def load(self, data: object, load_mesh: bool = False, iteration: int = 0):
        """Load data from a data object.

        This method replace the current data with new data.
        The old data are erased.

        Parameters
        ----------
        data : dict or DataSet or pyvista.UnstructuredGrid or str
            Input data to load.

            * **dict** :
                Load data using the ``load_dict`` method.
            * **DataSet** :
                Load data from another ``DataSet`` object without copying.
            * **pyvista.UnstructuredGrid** :
                Load data from a PyVista ``UnstructuredGrid`` object without
                copy.
            * **str** :
                Path to a data file. Supported file extensions are
                ``'vtk'``, ``'msh'``, ``'fdz'``, ``'fdh5'`` and ``'npz'``.

        load_mesh : bool, optional
            If ``True``, the mesh is loaded from the file (when the file
            contains a mesh). If ``False``, only the data are loaded.
            Default is ``False``.

        iteration : int, optional
            Iteration index to load when ``data`` refers to an ``fdz`` file.
        """
        if isinstance(data, dict):
            self.load_dict(data)
        elif isinstance(data, DataSet):
            self.node_data = data.node_data
            self.element_data = data.element_data
            self.gausspoint_data = data.gausspoint_data
            self.scalar_data = data.scalar_data
            if load_mesh:
                self.mesh = data.mesh
        elif USE_PYVISTA and isinstance(data, pv.UnstructuredGrid):
            self.meshplot = data
            self.node_data = {k: v.T for k, v in data.point_data.items()}
            self.element_data = {k: v.T for k, v in data.cell_data.items()}
            if load_mesh:
                self.mesh = Mesh.from_pyvista(data)
        elif isinstance(data, ZipPath):
            # used to load one iteration in fdz file
            data = np.load(data.open("rb"))
            self.load_dict(data)
        elif isinstance(data, str):
            # load from a file
            filename = data
            ext = os.path.splitext(filename)[1]
            ext = ext.lower()
            if ext == ".vtk":
                # load_mesh ignored because the mesh already in the vtk file
                if not (USE_PYVISTA):
                    raise NameError(
                        "Pyvista not installed. Pyvista required to load vtk meshes."
                    )
                DataSet.load(self, pv.read(filename))
            elif ext == ".msh":
                return NotImplemented
            elif ext == ".fdh5":
                from fedoo.util.fdh5 import load_dataset_iteration

                load_dataset_iteration(self, filename, iteration)
            elif ext in [".npz", ".fdz"]:
                if ext == ".fdz":
                    file = ZipFile(filename, "r")
                    if f"iter_{iteration}.npz" in file.namelist():
                        data = np.load(file.open(f"iter_{iteration}.npz"))
                        # pyvista cant read file object. So copy to disk read and remove.
                        file.extract("_mesh_.vtk")
                        self.mesh = Mesh.read("_mesh_.vtk")
                        os.remove("_mesh_.vtk")
                    else:
                        raise NameError(
                            f"Specified iteration not found in the fdz {filename}."
                        )
                else:
                    if load_mesh:
                        self.mesh = Mesh.read(os.path.splitext(filename)[0] + ".vtk")
                    data = np.load(filename)

                self.load_dict(data)

            elif ext == ".csv":
                return NotImplemented
            elif ext == ".xlsx":
                return NotImplemented

            else:
                raise NameError("Can't load data -> Data not understood")
        else:
            raise NameError("Can't load data -> Data not understood")

    def load_dict(self, data: dict) -> None:
        """Load data from a dict generated with the to_dict method.

        The old data are erased."""
        self.node_data = {k[:-3]: v for k, v in data.items() if k[-2:] == "nd"}
        self.element_data = {k[:-3]: v for k, v in data.items() if k[-2:] == "el"}
        self.gausspoint_data = {k[:-3]: v for k, v in data.items() if k[-2:] == "gp"}
        self.scalar_data = {
            k[:-3]: v if np.size(v) > 1 else v.item()
            for k, v in data.items()
            if k[-2:] == "sc"
        }
        # self.scalar_data = {k[:-3]:v for k,v in data.items() if k[-2:] == 'sc'}

    def to_pandas(self) -> pandas.DataFrame:
        if USE_PANDAS:
            out = {}
            n_data_type = (
                (self.node_data != {})
                + (self.element_data != {})
                + (self.gausspoint_data != {})
            )
            if n_data_type > 1:
                raise NameError(
                    "Can't convert to pandas DataSet with with several different data type."
                )

            for k, v in self.node_data.items():
                if len(v.shape) == 1:
                    out[k] = v
                elif len(v.shape) == 2:
                    out.update({k + "_" + str(i): v[i] for i in range(v.shape[0])})
                else:
                    return NotImplemented

            for k, v in self.element_data.items():
                if len(v.shape) == 1:
                    out[k] = v
                elif len(v.shape) == 2:
                    out.update({k + "_" + str(i): v[i] for i in range(v.shape[0])})
                else:
                    return NotImplemented

            for k, v in self.element_data.items():
                if len(v.shape) == 1:
                    out[k] = v
                elif len(v.shape) == 2:
                    out.update({k + "_" + str(i): v[i] for i in range(v.shape[0])})
                else:
                    return NotImplemented

            return pandas.DataFrame.from_dict(out)
        else:
            raise NameError("Pandas lib is not installed.")

    def to_csv(self, filename: str, save_mesh: bool = False) -> None:
        """Write data in a csv file.

        This method require the installation of pandas library
        and is available only if 1 type of data (node, element, gausspoint) is defined.

        Parameters
        ----------
        filename : str
            Name of the file including the path.
        save_mesh : bool (default = False)
            If True, the mesh is also saved in a vtk file using the same filename with a '.vtk' extention.
        """
        if USE_PANDAS:
            self.to_pandas().to_csv(filename)
            if save_mesh:
                self.save_mesh(filename)
        else:
            raise NameError("Pandas lib need to be installed for csv export.")

    def to_excel(self, filename: str, save_mesh: bool = False) -> None:
        """Write data in a xlsx file (excel format).

        This method require the installation of pandas and openpyxl libraries
        and is available only if 1 type of data (node, element, gausspoint) is defined.

        Parameters
        ----------
        filename : str
            Name of the file including the path.
        save_mesh : bool (default = False)
            If True, the mesh is also saved in a vtk file using the same filename with a '.vtk' extention.
        """
        if USE_PANDAS:
            self.to_pandas().to_excel(filename)
            if save_mesh:
                self.save_mesh(filename)
        else:
            raise NameError("Pandas lib need to be installed for excel export.")

    def to_vtk(
        self, filename: str, binary: bool = True, gp_data_to_node: bool = True
    ) -> None:
        """Write vtk file with the mesh and associated data.

        Gauss Point data are interpolated as Node data because
        vtk don't support gauss point data.

        Parameters
        ----------
        filename : str
            Name of the file including the path.
        binary : bool, optional
            If True, write as binary. Otherwise, write as ASCII.
        gp_data_to_node : bool, default = True
            If True, the Gauss Point data are interpolated as Node data.
            If False, the Gauss Point data are ignored (vtk file don't have Gauss Point Data)
        """
        if USE_PYVISTA:
            binary = True
            ext = os.path.splitext(filename)[1]
            if ext == "":
                filename = filename + ".vtk"
            self.to_pyvista(gp_data_to_node).save(filename, binary=binary)
        else:
            from fedoo.util.mesh_writer import write_vtk

            write_vtk(self, filename, gp_data_to_node)

    def to_pyvista(self, gp_data_to_node: bool = True, selected_submeshes=None):
        """Convert the dataset to a PyVista unstructured grid.

        Parameters
        ----------
        gp_data_to_node : bool, default=True
            For a single ``Mesh``, convert Gauss point data to node data before
            export. For a ``MultiMesh``, Gauss point conversion is currently
            skipped.
        selected_submeshes : int, str or sequence, optional
            Only used with ``MultiMesh`` datasets. Restrict the exported grid
            and element data to the selected submesh ids, names, or element
            types.

        Returns
        -------
        pyvista.UnstructuredGrid
            Mesh with point, cell and field data attached.
        """
        if self.mesh is not None:
            if self._is_multimesh():
                mesh, submesh_indices = self._selected_multimesh(selected_submeshes)
            else:
                mesh, submesh_indices = self.mesh, None

            pv_data = mesh.to_pyvista()

            for key, val in self.node_data.items():
                pv_data.point_data[key] = val.T

            for key, val in self.element_data.items():
                if self._is_multimesh():
                    val = self._as_multimesh_data(val).to_global(submesh_indices)
                pv_data.cell_data[key] = _array_to_pyvista_data(val)

            for key, val in self.scalar_data.items():
                pv_data.field_data[key] = np.array(val)

            if gp_data_to_node and not self._is_multimesh():
                for key in self.gausspoint_data:
                    pv_data.point_data[key] = self.get_data(key, data_type="Node").T

            return pv_data
        else:
            raise TypeError("Mesh should be defined befort converted to pyvista object")

    def to_msh(self, filename: str) -> None:
        """Write a msh (gmsh format) file with mesh and associated data.

        Warning: gausspoint data are not included in the saved file.

        Parameters
        ----------
        filename : str
            Name of the file including the path.
        """
        from fedoo.util.mesh_writer import write_msh

        write_msh(self, filename)

    def to_dict(self) -> dict:
        """Return a dict with all the node, element and gausspoint data."""
        out = {k + "_nd": v for k, v in self.node_data.items()}
        out.update({k + "_el": v for k, v in self.element_data.items()})
        out.update({k + "_gp": v for k, v in self.gausspoint_data.items()})
        out.update({k + "_sc": np.array(v) for k, v in self.scalar_data.items()})

        return out

    def to_fdz(
        self,
        filename: str,
        save_mesh: bool = False,
        iteration: int = 0,
        compressed: bool = False,
    ) -> None:
        """Write a fdz file from the dataset.

        Parameters
        ----------
        filename : str
            Name of the file including the path.
        """

        name, ext = os.path.splitext(filename)
        if ext == "":
            filename = filename + ".fdz"
        if compressed:
            self.savez_compressed("_mesh_", save_mesh)
        else:
            self.savez("_mesh_", save_mesh)
        if save_mesh:
            file = ZipFile(filename, "w")
        else:
            file = ZipFile(filename, "a")

        file.write("_mesh_.npz", "iter_" + str(iteration) + ".npz")
        os.remove("_mesh_.npz")
        if save_mesh:
            file.write("_mesh_.vtk")
            os.remove("_mesh_.vtk")
        file.close()

    def to_fdh5(
        self,
        filename: str,
        iteration: int = 0,
        overwrite: bool = False,
    ) -> None:
        """Write the dataset to a FDH5 file.

        Parameters
        ----------
        filename : str
            Name of the FDH5 file. If no extension is provided, ``.fdh5`` is
            appended.
        iteration : int, default=0
            Result iteration id written under ``results/iter_<iteration>``.
        overwrite : bool, default=False
            If True, an existing file is removed before writing. If False, the
            mesh is kept and only the requested iteration is added or replaced.

        Notes
        -----
        Element and Gauss point fields attached to a ``MultiMesh`` are written
        under their matching ``submesh_X`` groups. Single ``Mesh`` datasets are
        written under ``submesh_0``.
        """
        from fedoo.util.fdh5 import write_dataset

        write_dataset(self, filename, iteration=iteration, overwrite=overwrite)

    def savez(self, filename: str, save_mesh: bool = False) -> None:
        """Write a npz file using the numpy savez function.

        Parameters
        ----------
        filename : str
            Name of the file including the path.
        save_mesh : bool (default = False)
            If True, the mesh is also saved in a vtk file using the same filename with a '.vtk' extention.
        """
        np.savez(filename, **self.to_dict())

        if save_mesh:
            self.save_mesh(filename)

    def savez_compressed(self, filename: str, save_mesh: bool = False) -> None:
        """Write a compressed npz file using the numpy savez_compressed function.

        Parameters
        ----------
        filename : str
            Name of the file including the path.
        save_mesh : bool (default = False)
            If True, the mesh is also saved in a vtk file using the same filename with a '.vtk' extention.
        """
        np.savez_compressed(filename, **self.to_dict())

        if save_mesh:
            self.save_mesh(filename)

    @staticmethod
    def read(filename: str, file_format: str = "fdh5") -> DataSet | MultiFrameDataSet:
        """Read a file from disk.

        Same as :py:func:`fedoo.read_data`.
        """
        return read_data(filename, file_format=file_format)

    @property
    def dict_data(self) -> dict:
        return {
            "Node": self.node_data,
            "Element": self.element_data,
            "GaussPoint": self.gausspoint_data,
            "Scalar": self.scalar_data,
        }

    def copy(self):
        """Make a copy of the dataset.

        This method make a shallow copy, ie every arrays (node coordinates,
        element table, data arrays, ...) are alias of the former arrays.
        To make a copy of all data, use the deepcopy method.

        Returns
        -------
        The copied DataSet object.
        """
        copy = DataSet()
        copy.mesh = self.mesh.copy()
        copy.node_data = dict(self.node_data)
        copy.element_data = dict(self.element_data)
        copy.gausspoint_data = dict(self.gausspoint_data)
        copy.scalar_data = dict(self.scalar_data)
        copy.active_submesh = self.active_submesh
        return copy

    def deepcopy(self):
        """Make a deep copy of the dataset.

        Returns
        -------
        The copied DataSet object.
        """
        copy = DataSet()
        copy.mesh = self.mesh.deepcopy()
        copy.node_data = {
            key: copy_data_value(value, deep=True)
            for key, value in self.node_data.items()
        }
        copy.element_data = {
            key: copy_data_value(value, deep=True)
            for key, value in self.element_data.items()
        }
        copy.gausspoint_data = {
            key: copy_data_value(value, deep=True)
            for key, value in self.gausspoint_data.items()
        }
        copy.scalar_data = {
            key: value if np.isscalar(value) else np.array(value).copy()
            for key, value in self.scalar_data.items()
        }
        copy.active_submesh = self.active_submesh
        return copy


class MultiFrameDataSet(DataSet):
    def __init__(self, mesh=None, list_data=None):
        if list_data is None:
            self.list_data = []
        elif isinstance(list_data, list):
            self.list_data = list_data
        else:
            self.list_data = [list_data]

        self.loaded_iter = None
        DataSet.__init__(self, mesh)

    def __getitem__(self, items):
        if self.loaded_iter is None:
            self.load()
        return DataSet.__getitem__(self, items)

    def save_all(
        self, filename: str, file_format: str = "fdh5", compressed: bool = False
    ):
        """Save all data from MultiFrameDataSet.

        If filename has no extension, the format is given in the parameter file_format
        (default = 'fdh5').
        If format is not 'fdz' or 'fdh5', the data files are saved using the given filename and format
        simply adding the iteration number to the file name. The mesh is also saved in vtk format in the same directory.
        For 'fdh5', all iterations are written into one HDF5 file.
        """
        dirname = os.path.dirname(filename)
        extension = os.path.splitext(filename)[1]
        if extension == "":
            file_format = file_format.lower()
            if file_format not in ["fdz", "fdh5"]:
                dirname = filename + "/"
                filename = dirname + os.path.basename(filename)
        else:
            # use extension as file format
            file_format = extension[1:].lower()
            filename = os.path.splitext(filename)[
                0
            ]  # remove extension for the base name

        if dirname and not (os.path.isdir(dirname)):
            os.mkdir(dirname)
        if file_format == "fdz":
            self.load(0)
            self.to_fdz(filename, True, 0, compressed)
            for i in range(1, len(self.list_data)):
                self.load(i)
                self.to_fdz(filename, False, i, compressed)
        elif file_format == "fdh5":
            if os.path.splitext(filename)[1] == "":
                filename += ".fdh5"
            for i in range(len(self.list_data)):
                self.load(i)
                self.to_fdh5(filename, iteration=i, overwrite=(i == 0))
        else:
            for i in range(len(self.list_data)):
                self.load(i)
                self.save(
                    filename + "_" + str(i) + "." + file_format,
                    compressed=compressed,
                )
            if file_format not in ["vtk", "msh"]:
                self.save_mesh(filename + ".vtk")

    def load(self, data=-1, load_mesh=False):
        if isinstance(data, int):
            # data is an iteration to load
            # iteration = self.list_data.index(self.list_data[data])
            iteration = data
            if iteration < 0:
                iteration += len(self.list_data)
            if self.loaded_iter == iteration:
                return
            if iteration > len(self.list_data) or iteration < 0:
                raise NameError("Number of iteration out of bounds")
            data_ref = self.list_data[iteration]
            if (
                isinstance(data_ref, tuple)
                and len(data_ref) == 3
                and data_ref[0] == "fdh5"
            ):
                DataSet.load(self, data_ref[1], load_mesh, iteration=data_ref[2])
            else:
                DataSet.load(self, data_ref)
            self.loaded_iter = iteration

        elif isinstance(data, tuple) and len(data) == 3 and data[0] == "fdh5":
            DataSet.load(self, data[1], load_mesh, iteration=data[2])
            self.loaded_iter = data[2]

        elif data:
            DataSet.load(self, data, load_mesh)

    def write_movie(
        self,
        filename: str = "test",
        field: str | None = None,
        component: int | str = 0,
        data_type: str | None = None,
        **kargs,
    ):
        """Create a video from the data.

        Generate a video by loading iteratively every frame.
        This method rely on the :py:meth:`fedoo.DataSet.plot` and then
        accept the all its arguments as keyword arguments.

        Parameters
        ----------
        filename : str
            Name of the video file to write. The type of video generated
            depend on the file extension. If no extension provided, a 'mp4'
            file will be written.
        field : str (optional)
            Name of the field to plot. If no field provided, only the
            mesh is ploted.
        component : int, str (default = 0)
            The data component to plot in case of vector data
        data_type : str in {'Node', 'Element' or 'GaussPoint'}, optional
            Type of the data. By default, the data_type is determined
            automatically by scanning the data arrays.
        **kargs: dict
            Other optional parameters. See notes below.

        Notes
        -----
        Many options are available as keyword args. Refer to the documentation
        of :py:meth:`fedoo.DataSet.plot` method for details.
        Below, only the specific arguments are presented.

        Available keyword arguments are:

        * clim : sequence[float|None] or None
            Sequence of two float to define data boundaries for colorbar.
            If clim is None, clim change at each iteration with the min and
            max. If one of the boundary is set to None, the value is replace
            by the min or max of data for the all iterations sequence.
            Defaults to minimum and maximum of data for the all iterations
            sequence, ie clim =[None,None].
        * framerate : int, default = 24
            Number of frames per second
        * quality : int between 1 and 10, default = 5
            Define the quality of the writen movie if the movie writer accept
            this parameter.
            Higher is better but take more place.
        * azimuth : scalar, default = 30
            Angle of azimuth (degree) at the begining of the video.
        * elevation : scalar, default = 15
            Angle of elevation (degree) at the begining of the video.
        * rot_azimuth : scalar, default = 0
            Angle of azimuth rotation that is made at each new frame.
            Used to make easy video with camera moving around the scene.
        * rot_elevation : scalar, default = 0
            Angle of elevation rotation that is made at each new frame.
            Used to make easy video with camera moving around the scene.
        * window_size : list of int (default = [1024, 768])
            Size of the video in pixel
        """
        if not (USE_PYVISTA):
            raise NameError("Pyvista not installed.")

        if self.mesh is None:
            raise NameError(
                "Can't generate a plot without an associated mesh. Set the mesh attribute first."
            )

        field = kargs.pop("scalars", field)

        framerate = kargs.pop("framerate", 24)
        quality = kargs.pop("quality", 5)
        rot_azimuth = kargs.pop("rot_azimuth", 0)
        rot_elevation = kargs.pop("rot_elevation", 0)
        window_size = kargs.pop("window_size", (1024, 768))

        azimuth = kargs.pop("azimuth", 30)
        elevation = kargs.pop("elevation", 15)

        # auto compute boundary
        Xmin, Xmax, clim_data = self.get_all_frame_lim(
            field, component, data_type, kargs.get("scale", 1)
        )
        clim = kargs.pop("clim", [None, None])
        if clim is not None:
            if clim[0] is None and clim_data is not None:
                clim[0] = clim_data[0]
            if clim[1] is None and clim_data is not None:
                clim[1] = clim_data[1]

        kargs["title"] = kargs.get("title", "")
        kargs["clim"] = kargs.get("clim", clim)

        ext = os.path.splitext(filename)[1].lower()
        if ext == "":
            ext = ".mp4"
            filename += ext

        if "plotter" not in kargs:
            pl = pv.Plotter(window_size=window_size, off_screen=True)
            lock_view = False
        else:
            pl = kargs["plotter"]
            lock_view = True  # don't change the current view

        self.plot(
            field,
            component,
            data_type,
            iteration=0,
            plotter=pl,
            lock_view=lock_view,
            **kargs,
        )

        if not lock_view and "cpos" not in kargs:
            # set initial camera position
            center = (Xmin + Xmax) / 2
            if len(center) < 3:
                center = np.hstack((center, np.zeros(3 - len(center))))
            length = np.linalg.norm(Xmax - Xmin)
            pl.camera.SetFocalPoint(center)
            pl.camera.position = tuple(center + np.array([0, 0, 2 * length]))
            pl.camera.up = tuple([0, 1, 0])
            if self.mesh.ndim == 3:
                pl.camera.Azimuth(azimuth)
                pl.camera.Elevation(elevation)

        if ext == ".gif":
            pl.open_gif(filename, fps=framerate)
        else:
            pl.open_movie(
                filename,
                framerate=framerate,
                quality=quality,
            )
        pl.write_frame()
        for iteration in range(1, self.n_iter):
            if rot_azimuth != 0:
                pl.camera.Azimuth(rot_azimuth)
            if rot_elevation != 0:
                pl.camera.Elevation(rot_elevation)
            self.plot(
                field,
                component,
                data_type,
                iteration=iteration,
                plotter=pl,
                lock_view=True,
                **kargs,
            )
            pl.write_frame()

        pl.close()
        self.meshplot = None

    def get_history(
        self, field, indices=None, component=None, data_type="Node", return_list=False
    ):
        """Retrieve history data from the MultiFrameDataSet.

        This method load every iteration and save the requested data
        in array(s).

        Parameters
        ----------
        field : str or list[str]
            Name(s) of the data field(s) to retrieve. If a list of str is given
            the function return a dict whose keys are each requested fields.
            If field is a str, only the requested field array is returned.
        indices : int, list[int], list[list[int]] optional
            The node/elements/gauss point indice(s) over which the solution is
            extracted.
            When many fields are required, indices should be a list of same
            length than field.
        component : int, str, None or list[int|str|None], optional
            Index or label of the component to extract if the data is
            multi-dimensional.
            If None, all components are returned.
        data_type : str, None or list[str|None], default = "Node"
            Desired data type to convert to. Can be one of 'Node', 'Element',
            or 'GaussPoint'. If None, the original data type is preserved.
            A list may be given to use different data type for many fields.

        Returns
        -------
        history_data : dict|list[array] or array
            If multiple data fields are requested, a dict whose keys are the
            field names is returned by default, or if return_list is True, an
            ordered list of array is returned instead of a dict.
            The returned arrays shape = (nb iterations, [nb_indices], [nb_comp])
            If only one indice or component are extracted, the coresponding
            dimensions are removed.

        Notes
        -----
        This method may be costly, because it needed to load every iterations
        in memory. It is generaly more efficient to extract several fields in
        a single operation than to extract one field many times.
        """
        if isinstance(field, str):
            list_fields = [field]
            list_indices = [indices]
            component = [component]
            data_type = [data_type]
            return_many_fields = False
        else:
            list_fields = field
            return_many_fields = True
            # if needed, convert indices, component and data_type to list
            if np.isscalar(indices) or indices is None:
                list_indices = [indices for f in list_fields]
            else:
                list_indices = indices
            if (
                component is None
                or np.isscalar(component)
                or isinstance(component, str)
            ):
                component = [component for f in list_fields]
            if data_type is None or isinstance(data_type, str):
                data_type = [data_type for f in list_fields]

        history = [[] for f in list_fields]
        for it in range(self.n_iter):
            self.load(it)
            for i, field in enumerate(list_fields):
                data = self.get_data(field, component[i], data_type[i])
                if isinstance(data, MultiMeshData):
                    data = data.to_global()
                if list_indices[i] is None or np.isscalar(
                    data
                ):  # modify for allowing scalar_data
                    history[i].append(data)
                else:
                    history[i].append(data[..., list_indices[i]])

        if return_many_fields:
            if return_list:
                return [np.array(field_hist) for field_hist in history]
            else:
                return {
                    field: np.array(field_hist)
                    for field, field_hist in zip(list_fields, history)
                }
        else:
            return np.array(history[0])

    def plot_history(
        self,
        field: str,
        indices: int | list[int],
        component: int | str = 0,
        data_type: str | None = "Node",
        show_legend: bool = True,
        **kargs,
    ) -> None:
        """Plot history data from the MultiFrameDataSet.

        Parameters
        ----------
        field : str
            Name of the data field to plot.
        indices : int, list[int]
            The node/elements/gauss point indice(s) over which the solution is
            extracted.
            If several indices as given, data extrated from each node/cell
            indices are plot on the same graph.
        component : int, str, None, optional
            Index or label of the component to extract if the data is
            multi-dimensional. If None, all components are ploted on
            the same graph.
        data_type : str, None, default = "Node"
            Desired data type to convert to. Can be one of 'Node', 'Element',
            or 'GaussPoint'. If None, the original data type is preserved.
        show_legend : bool, default = True
            Whereas the legend should be plotted.

        Notes
        -----
        The GaussPoint indices are arange in gausspoint major ordering. For
        instance, if n_elements = 3 and n_gp = 2 the stored values are:
        [elem0_gp0, elem1_gp0, elem2_gp0, elem0_gp1, elem1_gp1, elem2_gp1]
        """
        if USE_MPL:
            data = self.get_history(
                ["Time", field], [None, indices], [None, component], data_type
            )
            # if data[field].ndim>2:
            ydata = data[field].reshape(data[field].shape[0], -1)
            plt.plot(data["Time"], ydata)
            if show_legend:
                if ydata.shape[1] > 1:
                    if data_type is None:
                        _, data_type = self.get_data(field, return_data_type=True)
                    if data[field].ndim == 2:
                        if np.size(indices) > 1:
                            legend = [f"{data_type} {ind}" for ind in indices]
                        else:
                            legend = [
                                f"{field}_{comp}"
                                for comp in range(data[field].shape[1])
                            ]
                    else:
                        legend = [
                            f"{field}_{comp} at {data_type} {ind}"
                            for comp in range(data[field].shape[1])
                            for ind in indices
                        ]

                    plt.gca().legend(legend)
                    plt.xlabel("Time")
                    plt.ylabel(field)
            # else:
            # plt.plot(data['Time'], data[field])
        else:
            raise NameError("Matplotlib should be installed to plot the data history")

    def get_all_frame_lim(self, field, component=0, data_type=None, scale=1):
        ndim = self.mesh.ndim
        clim = [np.inf, -np.inf]
        crd = self.mesh.nodes
        current_iter = self.loaded_iter

        for i in range(0, self.n_iter):
            self.load(i)
            if field is not None:
                data = self.get_data(field, component, data_type)
                if isinstance(data, MultiMeshData):
                    data = data.to_global()
                clim = [
                    np.nanmin([np.nanmin(data), clim[0]]),
                    np.nanmax([np.nanmax(data), clim[1]]),
                ]

            if "Disp" in self.node_data:
                new_crd = crd + scale * self.node_data["Disp"].T

                new_Xmin = new_crd.min(axis=0)
                new_Xmax = new_crd.max(axis=0)
                if i == 0:
                    Xmin = new_Xmin
                    Xmax = new_Xmax
                else:
                    Xmin = [np.min([Xmin[i], new_Xmin[i]]) for i in range(ndim)]
                    Xmax = [np.max([Xmax[i], new_Xmax[i]]) for i in range(ndim)]

        self.load(current_iter)
        if "Disp" not in self.node_data:
            Xmin = self.mesh.bounding_box[0]
            Xmax = self.mesh.bounding_box[1]
        if field is None:
            clim = None

        return np.array(Xmin), np.array(Xmax), clim

    def copy(self):
        """Make a copy of the dataset.

        This method make a shallow copy of the mesh.
        The current iteration data is reloaded.

        Returns
        -------
        The copied MultiFrameDataSet object.
        """
        copy = MultiFrameDataSet(self.mesh.copy(), list(self.list_data))
        copy.active_submesh = self.active_submesh
        if self.loaded_iter is not None:
            copy.load(self.loaded_iter)
        return copy

    def deepcopy(self):
        """Make a deep copy of the dataset.

        This method make a deep copy of the mesh.
        The current iteration data is reloaded.

        Returns
        -------
        The copied MultiFrameDataSet object.
        """
        copy = MultiFrameDataSet(self.mesh.deepcopy(), list(self.list_data))
        copy.active_submesh = self.active_submesh
        if self.loaded_iter is not None:
            copy.load(self.loaded_iter)
        return copy

    @property
    def n_iter(self):
        return len(self.list_data)


def read_data(filename: str, file_format: str = "fdh5"):
    """Read a file from disk.

    The file may be a single file or a directory containing files from
    several iterations. The file format may be specified either by the
    filename extension or by the ``file_format`` parameter (default is
    ``"fdh5"``) when the filename has no extension.

    Supported file formats are ``"fdz"``, ``"fdh5"``, ``"vtk"``, and
    ``"npz"``. For ``"npz"`` files, a VTK mesh with the same base name is
    also searched.

    Parameters
    ----------
    filename : str
        Path to the file or directory to read.
    file_format : str, optional
        File format identifier to use when the filename has no extension.
        Default is ``"fdh5"``.

    Returns
    -------
    DataSet or MultiFrameDataSet
        The loaded dataset.
    """
    extension = os.path.splitext(filename)[1]
    if extension != "":
        # use extension as file format
        file_format = extension[1:].lower()

    if file_format == "fdz":
        return read_fdz(filename)
    if file_format == "fdh5":
        from fedoo.util.fdh5 import read_fdh5

        return read_fdh5(filename)

    dirname = os.path.dirname(filename)
    if extension == "":
        dirname = filename + "/"
        filename = dirname + os.path.basename(filename)
        file_format = file_format.lower()
    else:
        filename = os.path.splitext(filename)[0]  # remove extension for the base name

    assert dirname == "" or (os.path.isdir(dirname)), "File not found"
    if file_format[:3] in ["npz", "vtk"] and os.path.isfile(filename + ".vtk"):
        mesh = Mesh.read(filename + ".vtk")
    else:
        mesh = None

    if os.path.isfile(filename + "." + file_format):
        dataset = DataSet(mesh)
        dataset.load(filename + "." + file_format)
        return dataset

    if os.path.isfile(filename + "_0." + file_format):
        iter0 = 0
    elif os.path.isfile(filename + "_1." + file_format):
        iter0 = 1
    else:
        raise NameError("File not found")

    if file_format == "vtk":  # read the mesh from the 1st iteration
        mesh = Mesh.read(filename + "_" + str(iter0) + ".vtk")
    dataset = MultiFrameDataSet(mesh)
    i = iter0
    while os.path.isfile(filename + "_" + str(i) + "." + file_format):
        dataset.list_data.append(filename + "_" + str(i) + "." + file_format)
        i += 1

    return dataset


def read_fdz(filename: str):
    """Read a fdz file unto a MultiFrameDataSet file."""
    extension = os.path.splitext(filename)[1]
    if extension == "":
        filename += ".fdz"

    assert os.path.isfile(filename), "File not found"
    file = ZipFile(filename, "r")
    # pyvista cant read file object. So copy to disk read and remove.
    file.extract("_mesh_.vtk")
    mesh = Mesh.read("_mesh_.vtk")
    os.remove("_mesh_.vtk")
    list_iter = file.namelist()
    file.close()

    dataset = MultiFrameDataSet(mesh)
    i = 0
    while "iter_" + str(i) + ".npz" in list_iter:
        dataset.list_data.append(ZipPath(filename, "iter_" + str(i) + ".npz"))
        i += 1

    return dataset


def as_3d_coordinates(crd):
    if crd.shape[1] < 3:
        return np.c_[crd, np.zeros((len(crd), 3 - crd.shape[1]))]
    else:
        return crd
