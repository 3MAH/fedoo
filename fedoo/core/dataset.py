"""Fedoo DataSet object."""

from __future__ import annotations

import numpy as np
import os
from zipfile import ZipFile, Path
from fedoo.core.mesh import Mesh
from fedoo.util.voigt_tensors import StressTensorList

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


class DataSet:
    """
    Object to store, save, load and plot data associated to a mesh.

    DataSet have a multiframe version :py:class:`fedoo.MultiFrameDataSet` that
    is a class that encapsulate several DataSet mainly usefull for time dependent data.

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

        screenshot: str, optional
            If defined, indicated a filename to save the plot.

        azimuth: float, default = 30.
            Azimuth angle of the camera around the scene
            (not used for 2D scene)

        elevation: float, default = 15.
            Elevaltion angle of the camera around the scene
            (not used for 2D scene).

        roll: float, default = 0
            Roll angle of the camera. The default state (roll angle = 0.) is
            set with the y direction on the up.

        title: str | None, default = None
            Title of the plot. By default the title is field name
            and the component is printed.

        title_size: float, default = 18
            Size of the title

        window_size: tuple, default = (1024, 768)
            Window size in pixels.

        multiplot: bool | None, default = None
            If True, the pyvista mesh is copied to force a separated scalar
            bar. This is usefull when ploting several figures at the same time.
            If multiplot si False, the same scalarbar will be applied to
            all the plots.
            If None, uses separated scalarbars only if the pyvista plotter uses
            subplot.


        **kwargs: dict, default = 15.
            See pyvista.Plotter.add_mesh() in the document of pyvista for
            additional usefull options.
        """

        if not (USE_PYVISTA):
            raise NameError("Pyvista not installed.")

        if self.mesh is None:
            raise NameError(
                "Can't generate a plot without an associated mesh. Set the mesh attribute first."
            )

        ndim = self.mesh.ndim
        n_physical_nodes = self.mesh.n_physical_nodes

        field = kargs.pop(
            "scalars", field
        )  # kargs scalars can be used instead of field

        if field is not None:
            data, data_type = self.get_data(field, component, data_type, True)
        else:
            data_type = None

        if screenshot is None:
            screenshot = False  # not used if show = False

        return_cpos = kargs.pop("return_cpos", False)
        sargs = kargs.pop("scalar_bar_args", None)

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
                    crd = (
                        self.mesh.physical_nodes
                        + scale * self.node_data["Disp"].T[:n_physical_nodes]
                    )
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
                    self.mesh.physical_nodes
                    + scale * self.node_data["Disp"].T[:n_physical_nodes]
                )
            else:
                meshplot.points = as_3d_coordinates(self.mesh.physical_nodes)

            if data_type == "Node":
                data = data[:n_physical_nodes]

            center = 0.5 * (meshplot.points.min(axis=0) + meshplot.points.max(axis=0))

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

        if multiplot is None:
            if pl.renderers.shape == (1, 1):
                multiplot = False
            else:
                multiplot = True

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
        if sargs is None and field is not None:  # default value
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

        if multiplot and "title" not in sargs:
            # title use as scalar_bar id required to plot several scalar bar
            sargs["title"] = f"{pl.renderers.active_index}"
            sargs["title_font_size"] = 1

        if field is None:
            meshplot.active_scalars_name = None
            if multiplot:
                pl.add_mesh(meshplot.copy(), show_edges=show_edges, **kargs)
            else:
                pl.add_mesh(meshplot, show_edges=show_edges, **kargs)
        else:
            if multiplot:
                pl.add_mesh(
                    meshplot.copy(),
                    scalars=data,
                    show_edges=show_edges,
                    scalar_bar_args=sargs,
                    cmap="jet",
                    clim=clim,
                    **kargs,
                )
            else:
                pl.add_mesh(
                    meshplot,
                    scalars=data,
                    show_edges=show_edges,
                    scalar_bar_args=sargs,
                    cmap="jet",
                    clim=clim,
                    **kargs,
                )

            if title is None:
                title = f"{field}_{component}"

            pl.add_text(title, name="name", color="Black", font_size=title_size)

        pl.add_axes(color="Black", interactive=True)

        # Node and Element Labels and plot points
        if node_labels or show_nodes:  # extract nodes coordinates
            if data_type == "GaussPoint":
                if "Disp" in self.node_data:
                    crd_labels = as_3d_coordinates(
                        self.mesh.physical_nodes
                        + self.node_data["Disp"].T[:n_physical_nodes]
                    )
                else:
                    crd_labels = as_3d_coordinates(self.mesh.physical_nodes)
            else:
                crd_labels = meshplot.points

        if node_labels:
            if node_labels == True:
                node_labels = list(range(n_physical_nodes))
            pl.add_point_labels(crd_labels, node_labels[:n_physical_nodes])

        if element_labels:
            if element_labels == True:
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
        pl.camera.reset_clipping_range()

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

    def get_data(self, field, component=None, data_type=None, return_data_type=False):
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
                data, current_data_type = self.get_data(
                    field, component, return_data_type=True
                )
                data = self.mesh.convert_data(
                    data, convert_from=current_data_type, convert_to=data_type
                )

        if (
            component is not None and not (np.isscalar(data)) and len(data.shape) > 1
        ):  # if data is scalar or 1d array, component ignored
            if component == "norm":
                data = np.linalg.norm(data, axis=0)
            elif component == "vm":
                # Try to compute the von mises stress
                data = StressTensorList(data).von_mises()
            elif component == "pressure":
                # Try to compute the pressure stress
                data = StressTensorList(data).pressure()
            else:
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
                    }[component]
                data = data[component]

        if return_data_type:
            return data, data_type
        else:
            return data

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

        Parameters
        ----------
        filename : str
            Name of the file including the path.
        save_mesh : bool, default = False
            If True, the mesh is also saved in a vtk file using the same filename with a '.vtk' extention.
            For vtk and msh file, the mesh is always included in the file and save_mesh have no effect.
        compressed: bool, default = False
            If True, the file is compressed if available (only for npz and fdz files)
        """
        ext = os.path.splitext(filename)[1]
        ext = ext.lower()
        if ext == "":
            ext = ".fdz"
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

    def save_mesh(self, filename: str):
        """Save the mesh using a vtk file. The extension of filename is ignored and modified to '.vtk'."""
        name = os.path.splitext(filename)[0]
        self.mesh.save(name)

    def load(self, data: object, load_mesh: bool = False, iteration: int = 0):
        """Load data from a data object.
        The old data are erased.

        Parameters
        ----------
        data :
        * if type(data) is dict:
            load data using the load_dict method
        * if type(data) is DataSet:
            load data from another DataSet object without copy
        * if type(data) is pyvista.UnstructuredGrid
            load data from a pyvista UnstructuredGrid object without copy
        * if type(data) is str
            load data from a file. Available extention are 'vtk', 'msh', 'fdz', 'npz'
        load_mesh : bool (default = False)
            If True, the mesh is loaded from the file (if the file contans a mesh).
            If False, only the data are loaded.
        iteration : int
            iteration loaded if data is a fdz object.
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
                Mesh.from_pyvista(data)
        elif isinstance(data, Path):
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
        self.scalar_data = {k[:-3]: v.item() for k, v in data.items() if k[-2:] == "sc"}
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
            self.to_pyvista(gp_data_to_node).save(filename, binary)
        else:
            from fedoo.util.mesh_writer import write_vtk

            write_vtk(self, filename, gp_data_to_node)

    def to_pyvista(self, gp_data_to_node: bool = True):
        if self.mesh is not None:
            pv_data = self.mesh.to_pyvista()

            for key, val in self.node_data.items():
                pv_data.point_data[key] = val.T

            for key, val in self.element_data.items():
                pv_data.cell_data[key] = val.T

            for key, val in self.scalar_data.items():
                pv_data.field_data[key] = val

            if gp_data_to_node:
                for key in self.gausspoint_data:
                    pv_data.point_data[key] = self.get_data(key, data_type="Node").T

            return pv_data
        else:
            raise TypeError("Mesh should be defined befort converted to pyvista object")

    def to_msh(self, filename: str) -> None:
        """Write a msh (gmsh format) file with the mesh and associated data
        (gausspoint data not included).

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
    def read(filename: str, file_format: str = "fdz") -> DataSet | MultiFrameDataSet:
        return read_data(filename, file_format="fdz")

    @property
    def dict_data(self) -> dict:
        return {
            "Node": self.node_data,
            "Element": self.element_data,
            "GaussPoint": self.gausspoint_data,
            "Scalar": self.scalar_data,
        }


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
        self, filename: str, file_format: str = "fdz", compressed: bool = False
    ):
        """Save all data from MultiFrameDataSet.

        If filename has no extension, the format is given in the parameter file_format
        (default = 'fdz').
        If format is not 'fdz', the data files are saved using the given filename and format
        simply adding the iteration number to the file name. The mesh is also saved in vtk format in the same directory.
        """
        dirname = os.path.dirname(filename)
        extension = os.path.splitext(filename)[1]
        if extension == "":
            file_format = file_format.lower()
            if file_format != "fdz":
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
            DataSet.load(self, self.list_data[iteration])
            self.loaded_iter = iteration

        else:
            DataSet.load(self, data, load_mesh)

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
        iteration: int | None = None,
        **kargs,
    ) -> None:
        """Plot a field on the surface of the associated mesh.

        Same function as DataSet.plot, with an addition iteration parameter to
        select the iteration from which the data should be plotted.

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
            It the type is not specified, look for any type of data and,
            if the data is found, draw the field without conversion.

        scale : float (default = 1)
            The scale factor used for the nodes displacement, using the 'Disp'
            vector field.
            If scale = 0, the field is plotted on the underformed shape.

        show : bool (default = True)

            * If show = True, the plot is rendered in a new window.
            * If show = False, the current pyvista plotter is returned
              without rendering.
            * show = False allow to customize the plot with pyvista before
              rendering it.

        show_edges : bool (default = True)
            if True, the mesh edges are shown

        clim: sequence[float], optional
            Sequence of two float to define data boundaries for color bar.
            Defaults to minimum and maximum of data.

        node_labels : bool | list (default = False)
            If True, show node labels (node indexe)
            If a list is given, print the label given in node_labels[i] for
            each node i.

        element_labels : bool | list (default = False)
            If True, show element labels (element indexe)
            If a list is given, print the label given in element_labels[i] for
            each element i.

        show_nodes : bool|float (default = False)
            Plot the nodes. If True, the nodes are shown with a default size.
            If float, show_nodes is the required size.

        show_normals : bool|float (default = False)
            Plot the face normals. If True,
            the vectors are shown with a default magnitude.
            If float, show_normals is the required magnitude.
            Only available for 1D or 2D mesh.

        plotter : pyvista.Plotter object or str in {'qt', 'pv'}

            * If pyvista.Plotter object, plot the mesh in the given plotter
            * If 'qt': use the background plotter of pyvistaqt (need the lib pyvistaqt)
            * If 'pv': use the standard pyvista plotter
            * If None: use the background plotter if available, or pyvista plotter if not.

        screenshot: str, optional
            If defined, indicated a filename to save the plot.

        azimuth: float, default = 30.
            Azimuth angle of the camera around the scene
            (not used for 2D scene)

        elevation: float, default = 15.
            Elevaltion angle of the camera around the scene
            (not used for 2D scene).

        roll: float, default = 0
            Roll angle of the camera. The default state (roll angle = 0.)
            is set with the y direction on the up.

        iteration : int (Optional)
            num of the iteration to plot. If None, the current iteration is
            plotted. If no current iteration is defined, the last iteration
            is loaded and plotted.

        **kwargs: dict, optional
            See pyvista.Plotter.add_mesh() in the document of pyvista for
            additional usefull options.
        """
        if iteration is None:
            if self.loaded_iter is None:
                self.load(-1)  # load last iteration
        else:
            self.load(iteration)

        return DataSet.plot(
            self,
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
            show_normals,
            plotter,
            screenshot,
            azimuth,
            elevation,
            roll,
            **kargs,
        )

    def write_movie(
        self,
        filename: str = "test",
        field: str | None = None,
        component: int | str = 0,
        data_type: str | None = None,
        scale: float = 1,
        show_edges: bool = True,
        clim: list[float | None] | None = [None, None],
        show_nodes: bool | float = False,
        **kargs,
    ):
        """
        Generate a video of the MultiFrameDataSet object by loading iteratively every frame.

        Parameters
        ----------
        filename : str
            Name of the videofile to write.
        field : str (optional)
            Name of the field to plot
        component : int, str (default = 0)
            The data component to plot in case of vector data
        data_type : str in {'Node', 'Element' or 'GaussPoint'}, optional
            Type of the data. By default, the data_type is determined automatically by scanning the data arrays.
        scale : scalar (default = 1)
            The scale factor used for the nodes displacement, using the 'Disp' vector field
        show_edges : bool (default = True)
            if True, the mesh edges are shown
        clim: sequence[float|None] or None
            Sequence of two float to define data boundaries for color bar.
            If clim is None, clim change at each iteration with the min and max.
            If one of the  boundary is set to None, the value is replace by the min or max.
            of data for the all iterations sequence.
            Defaults to minimum and maximum of data for the all iterations sequence
            (clim =[None,None]).
        show_nodes : bool|float (default = False)
            Plot the nodes. If True, the nodes are shown with a default size.
            If float, show_nodes is the required size.
        **kargs: dict
            Other optional parameters (see notes below)

        Notes
        -----------
        Many options are available as keyword args. Some of these options are
        directly related to pyvista options (for instance in the pyvista.plotter.add_mesh method).
        Please, refer to the documentation of pyvista for more details.

        Available keyword arguments are :
        * framerate : int (default = 24)
            Number of frames per second
        * quality : int between 1 and 10 (default = 5)
            Define the quality of the writen movie. Higher is better but take more place.
        * azimuth: scalar (default = 30)
            Angle of azimuth (degree) at the begining of the video.
        * elevation: scalar (default = 15)
            Angle of elevation (degree) at the begining of the video.
        * rot_azimuth : scalar (default = 0)
            Angle of azimuth rotation that is made at each new frame. Used to make easy video with camera moving around the scene.
        * rot_elevation :  scalar (default = 0)
            Angle of elevation rotation that is made at each new frame. Used to make easy video with camera moving around the scene.
        * scalar_bar_args' :
            dict containing the arguments related to scalar bar.
        * window_size : list of int (default = [1024, 768])
            Size of the video in pixel
        """

        if not (USE_PYVISTA):
            raise NameError("Pyvista not installed.")

        if self.mesh is None:
            raise NameError(
                "Can't generate a plot without an associated mesh. Set the mesh attribute first."
            )

        ndim = self.mesh.ndim
        n_physical_nodes = self.mesh.n_physical_nodes

        field = kargs.pop("scalars", field)

        framerate = kargs.pop("framerate", 24)
        quality = kargs.pop("quality", 5)

        sargs = kargs.pop("scalar_bar_args", None)

        rot_azimuth = kargs.pop("rot_azimuth", 0)
        rot_elevation = kargs.pop("rot_elevation", 0)

        azimuth = kargs.pop("azimuth", 30)
        elevation = kargs.pop("elevation", 15)

        if show_nodes == True:
            show_nodes = 5  # default size of nodes

        # auto compute boundary
        Xmin, Xmax, clim_data = self.get_all_frame_lim(
            field, component, data_type, scale
        )
        center = (Xmin + Xmax) / 2
        if len(center) < 3:
            center = np.hstack((center, np.zeros(3 - len(center))))
        length = np.linalg.norm(Xmax - Xmin)

        if clim is not None:
            if clim[0] is None:
                clim[0] = clim_data[0]
            if clim[1] is None:
                clim[1] = clim_data[1]

        window_size = kargs.pop("window_size", [1024, 768])

        self.load(0)

        data, data_type = self.get_data(field, None, data_type, True)

        if data_type == "GaussPoint":
            if self.meshplot_gp is None:
                self._build_mesh_gp()
            meshplot = self.meshplot_gp
            crd = self.mesh_gp.nodes

        elif data is not None:
            if self.meshplot is None:
                meshplot = self.meshplot = self.mesh.to_pyvista()
            else:
                meshplot = self.meshplot
            crd = self.mesh.physical_nodes
        else:
            return NotImplemented

        pl = pv.Plotter(window_size=window_size)

        # pl = pv.Plotter()
        pl.set_background("White")

        if sargs is None:  # default value
            sargs = dict(
                interactive=True,
                title_font_size=20,
                label_font_size=16,
                color="Black",
                # n_colors= 10
            )

        pl.add_axes(color="Black", interactive=True)

        pl.open_movie(filename + ".mp4", framerate=framerate, quality=quality)
        # pl.open_movie(filename, framerate=framerate, quality = quality)

        # pl.show(auto_close=False)  # only necessary for an off-screen movie

        # camera position
        pl.camera.SetFocalPoint(center)
        pl.camera.position = tuple(center + np.array([0, 0, 2 * length]))
        pl.camera.up = tuple([0, 1, 0])

        if ndim == 3:
            pl.camera.Azimuth(azimuth)
            pl.camera.Elevation(elevation)

        for i in range(0, self.n_iter):
            self.load(i)
            data = self.get_data(field, component, data_type)

            if data_type == "GaussPoint":
                data = self.mesh_gp.convert_data(
                    data,
                    convert_from="GaussPoint",
                    convert_to="Node",
                    n_elm_gp=len(data) // self.mesh.n_elements,
                )
                if "Disp" in self.node_data:
                    U = (
                        (
                            self.node_data["Disp"]
                            .reshape(ndim, -1)
                            .T[self.mesh.elements.ravel()]
                        ).T
                    ).T
                    meshplot.points = as_3d_coordinates(crd + scale * U)

                if show_nodes:
                    if "Disp" in self.node_data:
                        crd_points = as_3d_coordinates(
                            self.mesh.physical_nodes
                            + scale * self.node_data["Disp"].T[:n_physical_nodes]
                        )
            else:
                if "Disp" in self.node_data:
                    meshplot.points = as_3d_coordinates(
                        crd + scale * self.node_data["Disp"].T[:n_physical_nodes]
                    )
                    crd_points = meshplot.points  # alias

                if data_type == "Node":
                    data = data[:n_physical_nodes]

            if i == 0:
                pl.add_mesh(
                    meshplot,
                    scalars=data,
                    show_edges=show_edges,
                    scalar_bar_args=sargs,
                    cmap="jet",
                    clim=clim,
                    **kargs,
                )
                if show_nodes:
                    pl.add_points(
                        crd_points,
                        render_points_as_spheres=True,
                        point_size=show_nodes,
                    )
                    mesh_points = pl.mesh
            else:
                if clim is None:
                    pl.update_scalar_bar_range([data.min(), data.max()])

                meshplot[meshplot.active_scalars_name] = data
                # pl.update_scalars(data, meshplot) #deprecated
                if show_nodes:
                    mesh_points.points = crd_points

            if rot_azimuth:
                pl.camera.Azimuth(rot_azimuth)
            if rot_elevation:
                pl.camera.Elevation(rot_elevation)

            # Run through each frame
            # pl.add_text(f"Iteration: {i}", name='time-label', color='Black')
            pl.write_frame()

        pl.close()
        self.meshplot = None

    def get_history(self, list_fields, list_indices=None, **kargs):
        data_type = kargs.pop("data_type", None)
        component = kargs.pop("component", 0)

        if isinstance(list_fields, str):
            list_fields = [list_fields]
            list_indices = [list_indices]
        else:
            if list_indices is None:
                list_indices = [None for field in list_fields]

        history = [[] for field in list_fields]
        for it in range(self.n_iter):
            self.load(it)
            for i, field in enumerate(list_fields):
                data = self.get_data(field, component, data_type)
                if list_indices[i] is None or np.isscalar(data):
                    history[i].append(data)
                else:
                    history[i].append(data[list_indices[i]])

        return tuple(np.array(field_hist) for field_hist in history)

    def plot_history(
        self,
        field: str,
        indice: int,
        data_type: str | None = None,
        component: int | str = 0,
        **kargs,
    ) -> None:
        if USE_MPL:
            t, data = self.get_history(["Time", field], [None, indice])
            plt.plot(t, data)
        else:
            raise NameError("Matplotlib should be installed to plot the data history")

    def get_all_frame_lim(self, field, component=0, data_type=None, scale=1):
        ndim = self.mesh.ndim
        clim = [np.inf, -np.inf]
        crd = self.mesh.physical_nodes
        n_physical_nodes = self.mesh.n_physical_nodes

        for i in range(0, self.n_iter):
            self.load(i)
            data = self.get_data(field, component, data_type)[:n_physical_nodes]
            clim = [
                np.min([data.min(), clim[0]]),
                np.max([data.max(), clim[1]]),
            ]

            if "Disp" in self.node_data:
                new_crd = crd + scale * self.node_data["Disp"].T[:n_physical_nodes]

                new_Xmin = new_crd.min(axis=0)
                new_Xmax = new_crd.max(axis=0)
                if i == 0:
                    Xmin = new_Xmin
                    Xmax = new_Xmax
                else:
                    Xmin = [np.min([Xmin[i], new_Xmin[i]]) for i in range(ndim)]
                    Xmax = [np.max([Xmax[i], new_Xmax[i]]) for i in range(ndim)]

        if "Disp" not in self.node_data:
            Xmin = self.mesh.bounding_box[0]
            Xmax = self.mesh.bounding_box[1]

        return np.array(Xmin), np.array(Xmax), clim

    @property
    def n_iter(self):
        return len(self.list_data)


def read_data(filename: str, file_format: str = "fdz"):
    """Read a file from a disk.

    The file may be a directory containing files from several iterations.
    The file format may be specified in the filename extension or using
    the file_format parameter (default = fdz) if the filename has no extension.

    Available file format are 'fdz', 'vtk' and 'npz'.
    For 'npz' a vtk mesh with the same base name is also searched.
    """
    extension = os.path.splitext(filename)[1]
    if extension != "":
        # use extension as file format
        file_format = extension[1:].lower()

    if file_format == "fdz":
        return read_fdz(filename)

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
        dataset.list_data.append(Path(filename, "iter_" + str(i) + ".npz"))
        i += 1

    return dataset


def as_3d_coordinates(crd):
    if crd.shape[1] < 3:
        return np.c_[crd, np.zeros((len(crd), 3 - crd.shape[1]))]
    else:
        return crd
