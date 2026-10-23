import vtk
from fedoo.util.beam_recovery import BeamOptions, SavedBeamSection, stored_point_indices
from fedoo.util.shell_recovery import (
    ShellOptions,
    sample_positions,
    stored_shell_point_index,
    default_shell_position,
)

import fedoo as fd
import numpy as np
import sys
from contextlib import contextmanager
from qtpy import QtWidgets, QtGui
from qtpy.QtWidgets import (
    QDockWidget,
    QToolBar,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QPushButton,
    QDoubleSpinBox,
    QCheckBox,
    QWidget,
    QShortcut,
)
from qtpy.QtCore import (
    Qt,
    Signal,
    QSignalBlocker,
    QTimer,
    QEvent,
    QSize,
    QObject,
    QRect,
)
import matplotlib as mpl  # only for colormap
import pyvista as pv
from pyvistaqt import QtInteractor
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
import os
import re
import tempfile

from fedoo.core.multimeshdata import MultiMeshData
from fedoo.core.mesh import MultiMesh
from fedoo.core.base import ProblemBase
from fedoo.core.boundary_conditions import BoundaryCondition, ListBC
from fedoo.core.problem_dataset import ProblemDataSet, _is_problem
from fedoo.core.dataset import as_3d_coordinates
from fedoo.util._viewer.object_inspector import ObjectInspector
from fedoo.mesh.selection import (
    find_surface_nodes,
    find_edge_nodes,
    AmbiguousSurfaceSeed,
    AmbiguousEdgeSeed,
)
from fedoo.util.fdh5 import CompressionConfig, FDH5Reader, zstd_available
from fedoo.util.field_components import get_field_components
from fedoo.util.expressions import (
    evaluate_expression,
    selection_mask,
    mesh_positions,
    nodes_to_elements,
    elements_to_nodes,
)

USE_PYVISTA_QT = True
SOLID_COLOR_FIELD = "Solid color"


def _viewer_field_name(data, field):
    """Present recovered tensor aliases as a single field in the UI."""
    if field in data.section_derived_fields():
        return data.section_derived_fields()[field][0]
    return field


def _viewer_field_names(data):
    return sorted(
        {
            _viewer_field_name(data, field)
            for field in data.field_names()
            if getattr(data, "show_internal_fields", False) or not field.startswith("_")
        }
    )


def _viewer_tensor_field(data, field, coordinates="global"):
    """Resolve a displayed field and coordinate choice to its dataset field."""
    field = _viewer_field_name(data, field)
    candidate = f"{field}_{coordinates}"
    derived = data.section_derived_fields()
    if candidate in derived:
        return candidate
    if f"{field}_local" in derived:
        return f"{field}_local"
    return field


def _is_multimesh(mesh):
    return isinstance(mesh, MultiMesh)


def _global_data_array(data):
    if isinstance(data, MultiMeshData):
        return data.to_global()
    return data


def _data_n_iter(data):
    return getattr(data, "n_iter", 1)


def _picked_node_value(dock, point, picker):
    """Read the clicked actor's displayed value and original physical node id."""
    mesh = pv.wrap(picker.GetDataSet())
    displayed_id = picker.GetPointId()
    if mesh is None or displayed_id < 0:
        return None, None, None
    if dock.opts.get("clip_args") or "_fedoo_node_ids" not in mesh.point_data:
        coordinates = dock.data.mesh.nodes.astype(float, copy=True)
        if "Disp" in dock.data.node_data:
            coordinates += dock.opts["scale"] * dock.data.node_data["Disp"].T
        coordinates = np.pad(coordinates, ((0, 0), (0, 3 - coordinates.shape[1])))
        node_id = int(np.linalg.norm(coordinates - point, axis=1).argmin())
    else:
        node_id = int(mesh.point_data["_fedoo_node_ids"][displayed_id])
    if dock.current_field is None:
        return node_id, None, None
    if dock.current_data_type == "Element":
        if "Data" not in mesh.cell_data:
            return node_id, None, None
        cells = vtk.vtkIdList()
        mesh.GetPointCells(displayed_id, cells)
        values = mesh.cell_data["Data"][
            [cells.GetId(i) for i in range(cells.GetNumberOfIds())]
        ]
        return node_id, values, "adjacent elements"
    if "Data" in mesh.point_data:
        return node_id, mesh.point_data["Data"][displayed_id], None
    return node_id, None, None


def _refresh_result_frames(data, reload_current=False):
    """Discover FDH5 frames without reading their fields or computing histories.

    A single-frame file may become a sequence while its simulation runs.
    Keep loaded fields and selection options until an explicit reload/load.
    """
    if isinstance(data, ProblemDataSet):
        if reload_current:
            data.refresh()
        return data
    filename = getattr(data, "_fdh5_filename", None)
    refs = getattr(data, "list_data", [])
    if refs and isinstance(refs[0], tuple) and refs[0][0] == "fdh5":
        filename = refs[0][1]
    if filename is None:
        return data
    iterations = FDH5Reader(filename).list_iterations()
    if not iterations:
        return data
    current_id = getattr(data, "_fdh5_iteration", None)
    if refs and data.loaded_iter is not None:
        current_id = refs[data.loaded_iter][2]
    new_refs = [("fdh5", filename, iteration) for iteration in iterations]
    if not hasattr(data, "list_data"):
        sequence = fd.MultiFrameDataSet(data.mesh)
        sequence.__dict__.update(data.__dict__)
        data = sequence
    data.list_data = new_refs
    data.loaded_iter = (
        iterations.index(current_id) if current_id in iterations else None
    )
    if reload_current:
        index = data.loaded_iter if data.loaded_iter is not None else 0
        previous_iter = data.loaded_iter
        data.loaded_iter = None
        try:
            data.load(index)
        except (OSError, KeyError):
            data.loaded_iter = previous_iter
            raise
    return data


@contextmanager
def _preserve_camera(plotter, enabled):
    """Keep zoom and projection as well as position across scene rebuilds."""
    camera = plotter.camera.copy() if enabled else None
    try:
        yield
    finally:
        if camera is not None:
            plotter.camera.DeepCopy(camera)
            plotter.reset_camera_clipping_range()


@contextmanager
def _defer_rendering(plotter, enabled):
    """Render a rebuilt scene only after all its actors are ready."""
    if not enabled or not hasattr(plotter, "suppress_rendering"):
        yield
        return

    was_suppressed = plotter.suppress_rendering
    plotter.suppress_rendering = True
    try:
        yield
    finally:
        plotter.suppress_rendering = was_suppressed
        if not was_suppressed:
            plotter.render()


def _as_3d_tuple(values, fill=0.0):
    if values is None:
        return (fill, fill, fill)
    values = tuple(values)
    if len(values) >= 3:
        return values[:3]
    return values + (fill,) * (3 - len(values))


def _element_data_value(data, field, component, data_type, element_id):
    values = data.get_data(field, component, data_type)
    if isinstance(values, MultiMeshData):
        return values.global_element_value(element_id)
    return values[..., element_id]


def _gausspoint_values_for_element(data, field, component, element_id):
    values = data.get_data(field, component, "GaussPoint")
    if isinstance(values, MultiMeshData):
        submesh_id, local_id = data.global_element_location(element_id)
        block = values.submesh(submesh_id)
        if block is None:
            return np.array([])
        block = np.asarray(block)
        n_elements = data.mesh[submesh_id].n_elements
        return block.reshape(-1, n_elements)[:, local_id]

    values = np.asarray(values)
    return values.reshape(-1, data.mesh.n_elements)[:, element_id]


def _gausspoint_global_indices_for_element(data, field, component, element_id):
    values = data.get_data(field, component, "GaussPoint")
    if isinstance(values, MultiMeshData):
        submesh_id, local_id = data.global_element_location(element_id)
        offset = 0
        for i in range(submesh_id):
            block = values.submesh(i)
            if block is None:
                offset += data.mesh[i].n_elements
            else:
                offset += np.asarray(block).shape[-1]

        block = values.submesh(submesh_id)
        if block is None:
            return []
        block = np.asarray(block)
        n_elements = data.mesh[submesh_id].n_elements
        n_gp = block.shape[-1] // n_elements
        return [offset + local_id + i * n_elements for i in range(n_gp)]

    values = np.asarray(values)
    n_gp = values.shape[-1] // data.mesh.n_elements
    return [element_id + i * data.mesh.n_elements for i in range(n_gp)]


class DockTitleBar(QtWidgets.QWidget):
    clicked = Signal()

    def __init__(self, dock: QtWidgets.QDockWidget, parent=None):
        super().__init__(parent)
        self._dock = dock
        self._active = False

        # --- Left: title label
        self.label = QtWidgets.QLabel(dock.windowTitle())
        self.label.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.label.setTextInteractionFlags(Qt.NoTextInteraction)

        # --- Right: close button
        self.btnClose = QtWidgets.QToolButton(self)
        self.btnClose.setAutoRaise(True)
        self.btnClose.setIconSize(QSize(10, 10))
        self.btnClose.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_TitleBarCloseButton)
        )
        self.btnClose.setToolTip("Close")

        # Layout
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(8, 2, 8, 2)
        lay.addWidget(self.label, 1)
        lay.addWidget(self.btnClose, 0)

        # Keep label synced with dock title
        dock.windowTitleChanged.connect(self.label.setText)

        # Connect buttons
        self.btnClose.clicked.connect(dock.close)
        self._applyStyle()

    # Click anywhere on title → signal (to activate dock)
    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)

    def _set_active(self, active: bool):
        if self._active != active:
            self._active = active
            self._applyStyle()

    def _applyStyle(self):
        # Use palette highlight to match platform themes
        pal = self.palette()
        if self._active:
            bg = pal.color(QtGui.QPalette.Highlight)
            fg = pal.color(QtGui.QPalette.HighlightedText)
            self.setStyleSheet(f"""
                QWidget {{ background-color: {bg.name()}; }}
                QLabel {{ color: {fg.name()}; font-weight: bold; }}
                QToolButton {{ color: {fg.name()}; }}
            """)
        else:
            # Use explicit light gray for inactive titlebar (Windows default)
            bg = "#CCCCCC"
            text = "#000000"
            self.setStyleSheet(f"""
                QWidget {{ background-color: {bg}; }}
                QLabel {{ color: {text}; font-weight: normal; }}
            """)


class IterationSpinBox(QSpinBox):
    """Editable frame number with chronological left/right step buttons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setButtonSymbols(QtWidgets.QAbstractSpinBox.NoButtons)
        self.setStyleSheet("QSpinBox { padding-left: 20px; padding-right: 20px; }")
        self.lineEdit().setAlignment(Qt.AlignCenter)
        self.previous_button = QtWidgets.QToolButton(self)
        self.next_button = QtWidgets.QToolButton(self)
        for button, direction, text, step in (
            (self.previous_button, Qt.LeftArrow, "Previous iteration", self.stepDown),
            (self.next_button, Qt.RightArrow, "Next iteration", self.stepUp),
        ):
            button.setArrowType(direction)
            button.setToolTip(text)
            button.setAutoRepeat(True)
            button.setFocusPolicy(Qt.NoFocus)
            button.clicked.connect(step)
        self.valueChanged.connect(self._update_step_buttons)
        self._update_step_buttons()

    def _update_step_buttons(self, *args):
        enabled = self.stepEnabled()
        self.previous_button.setEnabled(
            bool(enabled & QtWidgets.QAbstractSpinBox.StepDownEnabled)
        )
        self.next_button.setEnabled(
            bool(enabled & QtWidgets.QAbstractSpinBox.StepUpEnabled)
        )

    def setRange(self, minimum, maximum):
        super().setRange(minimum, maximum)
        self._update_step_buttons()

    def setValue(self, value):
        super().setValue(value)
        self._update_step_buttons()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        rect = self.contentsRect().adjusted(2, 2, -2, -2)
        self.previous_button.setGeometry(rect.left(), rect.top(), 18, rect.height())
        self.next_button.setGeometry(rect.right() - 17, rect.top(), 18, rect.height())


class PlotDock(QDockWidget):
    _n_created_dock = 0  # total dock created

    def __init__(self, data, title, parent=None, opts=None, filename=None):
        PlotDock._n_created_dock += 1
        self._dock_index = PlotDock._n_created_dock
        self.title = title
        self.filename = filename
        title = f"{self._dock_index}: " + str(title)
        super().__init__(title, parent)
        self.data = data
        self.line_points = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
        self.element_sets = None
        self.unsaved_element_set_names = set()
        self.node_sets = None
        self.unsaved_node_set_names = set()
        # visibility_mode to keep track on current_selection for element_sets dialog box
        #  -> None for show all, "show" or "hide"
        self.visibility_mode = None

        container = QWidget()

        # disable activate/desactivate capability
        self.setContextMenuPolicy(Qt.PreventContextMenu)
        action = self.toggleViewAction()
        action.setEnabled(False)
        action.setVisible(False)
        self.setAttribute(Qt.WA_DeleteOnClose, True)

        layout = QVBoxLayout(container)
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        container.setLayout(layout)
        self.setWidget(container)

        self.setAllowedAreas(Qt.AllDockWidgetAreas)

        self.setFeatures(
            QDockWidget.DockWidgetMovable
            | QDockWidget.DockWidgetFloatable
            | QDockWidget.DockWidgetClosable
        )

        # if hasattr(dock, "plotter") and hasattr(dock.plotter, "interactor"):
        # container.setFocusPolicy(Qt.NoFocus)
        self.plotter.interactor.setFocusPolicy(Qt.StrongFocus)

        #####################################################################
        # Parameters / Plot options
        #####################################################################
        if opts:
            self.opts = opts
            self.opts.setdefault("color", "lightgray")
            self.opts.setdefault("color_missing_fields", True)
        else:
            self.opts = {
                # Scalar bar options
                "clim_mode": "current",  # 'current' | 'all' | 'manual'
                "clim": None,  # tuple (vmin, vmax) or None for auto
                "n_colors": 256,  # nb of colors (LUT)
                "n_labels": 5,  # nb of colorbar labels
                "cmap_name": "jet",  # current cmap name
                "cmap": "jet",  # cmap object
                # Other plot options
                "scale": 1.0,
                "scale_save": 1.0,
                "shell_thickness": False,
                "show_edges": True,
                "show_scalar_bar": True,
                "node_labels": False,
                "element_labels": False,
                "title_plot": "",
                # Renderer options
                "opacity": 1.0,
                "pbr": False,  # Physical based rendering
                "metallic": 1.0,
                "roughness": 0.5,
                "diffuse": 1.0,
                "color": "lightgray",
                "color_missing_fields": True,
                # Clip plane parameters
                "clip_args": None,
                "clip_origin": None,
                "clip_normal": None,
                "clip_invert": False,
                # Element sets
                "element_set": None,
                # "element_set_invert": False,
            }

        if hasattr(data, "loaded_iter"):
            if data.loaded_iter is None:
                data.load(-1)
            self.current_iter = data.loaded_iter
        else:
            self.current_iter = (
                (data.current_mode or 0) if isinstance(data, ProblemDataSet) else 0
            )

        field_names = _viewer_field_names(data)
        self.beam_csys = "global"
        if "Stress" in field_names:
            self.current_field = "Stress"
            self.current_comp = "vm"
        elif "Disp" in field_names:
            self.current_field = "Disp"
            self.current_comp = "X"
        else:
            try:
                self.current_field = field_names[0]
            except:
                self.current_field = None
            self.current_comp = None
        self.current_data_type = "Node"
        self.current_field = _viewer_tensor_field(
            data, self.current_field, self.beam_csys
        )

        titlebar = DockTitleBar(self)
        titlebar.clicked.connect(lambda d=self: parent._set_active(d))
        self.setTitleBarWidget(titlebar)
        self._titlebar = titlebar  # keep reference

        parent.all_docks.append(self)
        if parent.active_dock is None:
            for action in parent.actions_requiring_data:
                action.setEnabled(True)
        parent._set_active(self)

    @property
    def pv_mesh(self, id_mesh=1):
        mesh_name = "data" + str(id_mesh)
        if mesh_name in self.plotter.actors:
            return pv.wrap(self.plotter.actors[mesh_name].GetMapper().GetInput())

        def actor_sort_key(name):
            try:
                return (0, int(name.rsplit("_", 1)[1]))
            except (IndexError, ValueError):
                return (1, name)

        multimesh_actor_names = sorted(
            (name for name in self.plotter.actors if name.startswith("data_")),
            key=actor_sort_key,
        )
        if not multimesh_actor_names:
            return None

        meshes = [
            pv.wrap(self.plotter.actors[name].GetMapper().GetInput())
            for name in multimesh_actor_names
        ]
        if len(meshes) == 1:
            return meshes[0]
        return pv.MultiBlock(meshes).combine()

    def picked_actor_mesh(self, actor):
        """Return the actor name and PyVista mesh for a picked VTK actor."""
        for name, candidate in self.plotter.actors.items():
            if candidate is actor:
                return name, pv.wrap(candidate.GetMapper().GetInput())
            try:
                if candidate.GetAddressAsString("") == actor.GetAddressAsString(""):
                    return name, pv.wrap(candidate.GetMapper().GetInput())
            except AttributeError:
                pass
        return None, None

    def get_components(self, field):
        if field == "":
            return [""]
        data = self.data
        if field in data.section_derived_fields():
            return get_field_components(field, 6)
        if np.isscalar(data[field]):
            return []

        if data[field].ndim == 1:
            comps = get_field_components(field, 1)
        else:
            comps = get_field_components(field, data[field].shape[0])

        return comps

    def update_plot(self, val=None, iteration=None, lock_view=True, plotter=None):
        if self.data.mesh is None:
            # don't plot anything without a mesh
            return

        if plotter is None:
            plotter = self.plotter

        if iteration is not None and (
            hasattr(self.data, "loaded_iter")
            or (isinstance(self.data, ProblemDataSet) and self.data.n_modes)
        ):
            try:
                if isinstance(self.data, ProblemDataSet):
                    self.data.set_mode(iteration)
                else:
                    self.data.load(iteration)
            except (OSError, KeyError, ValueError, IndexError) as exc:
                self.parent().statusBar().showMessage(
                    f"Results temporarily unavailable: {exc}"
                )
                return

        scalar_dialog = getattr(self.parent(), "scalar_expression_dialog", None)
        if scalar_dialog is not None and scalar_dialog.isVisible():
            scalar_dialog.refresh()

        if self.opts["clim_mode"] == "current":
            clim = None
        else:
            clim = self.opts["clim"]

        sargs = {
            # "n_colors": self.opts['n_colors'],
            "n_labels": self.opts["n_labels"],
        }
        is_multimesh = _is_multimesh(self.data.mesh)
        multimesh_kargs = {}
        if is_multimesh:
            multimesh_kargs["global_element_set"] = True
            multimesh_kargs["name"] = "data"
            if self.current_field is not None and self.opts.get(
                "color_missing_fields", True
            ):
                multimesh_kargs["missing_field_color"] = self.opts["color"]

        with (
            _defer_rendering(plotter, is_multimesh or lock_view),
            _preserve_camera(plotter, lock_view),
        ):
            # plotter.clear()  # not compatible with pbr ???
            plotter.renderer.clear_actors()
            plot_kargs = {}
            if self.current_field is None:
                plot_kargs["color"] = self.opts["color"]
            self.data.plot(
                field=self.current_field,
                component=self.current_comp,
                data_type=self.current_data_type,
                clim=clim,
                plotter=plotter,
                show=False,
                show_edges=self.opts["show_edges"],
                show_scalar_bar=self.opts["show_scalar_bar"],
                scalar_bar_args=sargs,
                node_labels=self.opts["node_labels"],
                element_labels=self.opts["element_labels"],
                scale=self.opts["scale"],
                shell_thickness=self.opts.get("shell_thickness", False),
                opacity=self.opts["opacity"],
                pbr=self.opts["pbr"],
                metallic=self.opts["metallic"],
                roughness=self.opts["roughness"],
                diffuse=self.opts["diffuse"],
                clip_args=self.opts["clip_args"],
                lock_view=lock_view,
                title=self.opts["title_plot"],
                element_set=self.opts["element_set"],
                # element_set_invert = self.opts["element_set_invert"],
                cmap=self.opts["cmap"],
                **multimesh_kargs,
                **plot_kargs,
            )

            if self.parent()._plane_widget_enabled:
                self.parent().enable_plane_widget()

            if self.parent()._line_widget_enabled:
                self.parent()._rebuild_line_widget()

            if self.opts.get("show_time", False) and "Time" in self.data.scalar_data:
                time_value = np.asarray(self.data.scalar_data["Time"])
                if time_value.size == 1 and np.isfinite(time_value.item()):
                    plotter.add_text(
                        f"Time = {time_value.item():.6g}",
                        position="upper_right",
                        font_size=10,
                        color="black",
                        name="_fedoo_time",
                    )
        if self is self.parent().active_dock:
            self.parent()._refresh_object_inspector()

    def closeEvent(self, event):
        if self is self.parent().active_dock:
            self.parent()._clip_redraw_timer.stop()
            for dialog in (
                self.parent().node_sets_dialog,
                self.parent().element_sets_dialog,
            ):
                if dialog is not None:
                    dialog.close()
        self.plotter.close()
        if self is self.parent().active_dock:
            self.parent().active_dock = None
        self.parent().all_docks.remove(self)
        self.parent()._update_dock_selector()
        self.parent()._refresh_object_inspector()
        if self.parent().active_dock is None:
            self.parent().field_combo.clear()
            for action in self.parent().actions_requiring_data:
                action.setEnabled(False)
        super().closeEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, data=None):
        super().__init__()

        self.all_docks = []
        self.active_dock = None
        self._sync_field = (
            False  # if True, all docks will share the same field/component/data_type
        )
        self._sync_iter = False  # if True, all docks will share the same iteration
        self._clim_dialog = None  # clim windows if open
        self._clip_dialog = None  # clip windows if open
        self._plot_over_line_dialog = None
        self._pol_results_dialog = None
        self._renderer_dialog = None
        self._plot_dialog = None
        self.element_sets_dialog = None
        self.node_sets_dialog = None
        self.scalar_expression_dialog = None
        self.boundary_conditions_dialog = None
        self.object_inspector = None
        self._problem_dialog = None
        self._window_index = 1
        self._picking_target = -1  # internal arg for picking tools. -1 = No target
        self._syncing_line_widget = False
        self.window_layout = self._distribute_auto

        # if plane_widget or line wiget should be shown in the active dock
        self._plane_widget_enabled = False
        self._plane_widget = None
        self._plane_widget_plotter = None
        self._clip_redraw_timer = QTimer(self)
        self._clip_redraw_timer.setSingleShot(True)
        self._clip_redraw_timer.timeout.connect(self.update_plot)
        self._line_widget_enabled = False

        self.apply_options_to_all = True

        # fullscreen options
        self._is_fullscreen = False
        self._hidden_ui_when_fullscreen = []

        self.setTabPosition(Qt.AllDockWidgetAreas, QtWidgets.QTabWidget.North)

        #####################################################################
        # Tool bars
        #####################################################################
        # ------------------------------------------------
        # Toolbar 1: Field / Component / Data Type
        # ------------------------------------------------
        toolbar_fields = QToolBar("Fields")
        toolbar_fields.setMovable(True)

        toolbar_fields.addWidget(QLabel("Field: "))
        self.field_combo = QtWidgets.QComboBox()
        # if data is not None:
        #     self.field_combo.addItems(data.field_names())

        toolbar_fields.addWidget(self.field_combo)
        toolbar_fields.addSeparator()  # <-- adds a small gap

        toolbar_fields.addWidget(QLabel("Component: "))
        self.comp_combo = QtWidgets.QComboBox()
        toolbar_fields.addWidget(self.comp_combo)

        self.beam_toolbar = QToolBar("Beam")
        self.beam_toolbar.addWidget(QLabel("Coordinates: "))
        self.csys_combo = QtWidgets.QComboBox()
        self.csys_combo.addItem("Stored", "stored")
        self.csys_combo.setEnabled(False)
        self.csys_combo.setToolTip(
            "Coordinate system for recovered beam Stress and Strain"
        )
        self.csys_combo.currentIndexChanged.connect(self.on_coordinates_changed)
        self.beam_toolbar.addWidget(self.csys_combo)

        toolbar_fields.addSeparator()  # <-- adds a small gap

        toolbar_fields.addWidget(QLabel("Data type: "))
        self.avg_combo = QtWidgets.QComboBox()
        self.avg_combo.addItems(["Node", "GaussPoint", "Element"])
        toolbar_fields.addWidget(self.avg_combo)

        self.beam_toolbar.addSeparator()
        self.beam_section_button = QPushButton("Beam…")
        self.beam_section_button.setToolTip(
            "Select the section point or a sampled envelope for derived beam tensors"
        )
        self.beam_section_button.clicked.connect(self.open_beam_section_dialog)
        self.beam_section_button.setEnabled(False)
        self.beam_toolbar.addWidget(self.beam_section_button)

        self.shell_toolbar = QToolBar("Shell")
        self.shell_toolbar.addWidget(QLabel("Coordinates: "))
        self.shell_csys_combo = QtWidgets.QComboBox()
        self.shell_csys_combo.addItem("Stored", "stored")
        self.shell_csys_combo.setEnabled(False)
        self.shell_csys_combo.setToolTip(
            "Coordinate system for recovered shell Stress and Strain"
        )
        self.shell_csys_combo.currentIndexChanged.connect(
            self.csys_combo.setCurrentIndex
        )
        self.shell_toolbar.addWidget(self.shell_csys_combo)
        self.shell_toolbar.addSeparator()
        self.shell_section_button = QPushButton("Shell…")
        self.shell_section_button.setToolTip(
            "Select the thickness point or an envelope for derived shell tensors"
        )
        self.shell_section_button.clicked.connect(self.open_shell_section_dialog)
        self.shell_section_button.setEnabled(False)
        self.shell_toolbar.addWidget(self.shell_section_button)

        self.addToolBar(Qt.TopToolBarArea, toolbar_fields)
        self.addToolBar(Qt.TopToolBarArea, self.beam_toolbar)
        self.addToolBar(Qt.TopToolBarArea, self.shell_toolbar)
        self.beam_toolbar.hide()
        self.shell_toolbar.hide()

        # ------------------------------------------------
        # Toolbar 2: Iteration controls
        # ------------------------------------------------
        toolbar_iter = QToolBar("Iteration")
        toolbar_iter.setMovable(True)

        toolbar_iter.addWidget(QLabel("Iteration:"))
        self.iter_slider = QSlider(Qt.Horizontal)
        self.iter_spin = IterationSpinBox()
        self.iter_slider.setTracking(False)
        self.iter_slider.setTickPosition(QSlider.TicksBelow)

        toolbar_iter.addWidget(self.iter_slider)
        toolbar_iter.addWidget(self.iter_spin)

        self.reload_btn = QPushButton("Reload")
        self.reload_btn.setToolTip(
            "Discover newly saved frames and reload the current frame. "
            "Reapply All frames Min/Max or recreate history plots to update them."
        )
        self.reload_btn.clicked.connect(self.reload_results)
        toolbar_iter.addWidget(self.reload_btn)

        self.animate_btn = QPushButton("▶ Play")
        self.animate_btn.setCheckable(True)
        toolbar_iter.addWidget(self.animate_btn)

        self.anim_fps_spin = QDoubleSpinBox()
        self.anim_fps_spin.setRange(0.1, 60.0)
        self.anim_fps_spin.setSingleStep(0.5)
        self.anim_fps_spin.setValue(5.0)
        self.anim_fps_spin.setSuffix(" fps")
        toolbar_iter.addWidget(self.anim_fps_spin)

        self.anim_wrap_check = QCheckBox("Loop")
        self.anim_wrap_check.setChecked(True)
        toolbar_iter.addWidget(self.anim_wrap_check)

        self.addToolBar(Qt.TopToolBarArea, toolbar_iter)

        # Force a line break on the TOP area
        self.addToolBarBreak(Qt.TopToolBarArea)

        # ------------------------------------------------
        # Toolbar 3 Window
        # ------------------------------------------------
        window_toolbar = QtWidgets.QToolBar("Dock Selector")
        window_toolbar.setMovable(True)
        self.addToolBar(Qt.TopToolBarArea, window_toolbar)

        window_toolbar.addWidget(QtWidgets.QLabel("Active window: "))
        self.dock_selector_combo = QtWidgets.QComboBox()
        self.dock_selector_combo.setMinimumWidth(150)

        # Allow user text input
        self.dock_selector_combo.setEditable(True)

        # Prevent adding arbitrary new items
        self.dock_selector_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)

        window_toolbar.addWidget(self.dock_selector_combo)

        # When user selects a dock from combo
        self.dock_selector_combo.currentIndexChanged.connect(
            lambda idx: self._set_active(self.all_docks[idx])
        )

        self.dock_selector_combo.lineEdit().editingFinished.connect(
            self.rename_active_dock_from_combo
        )

        # ------------------------------------------------
        # Toolbar 4 Selection
        # ------------------------------------------------
        select_toolbar = QtWidgets.QToolBar("Select")
        select_toolbar.setMovable(True)
        self.addToolBar(Qt.TopToolBarArea, select_toolbar)

        select_toolbar.addWidget(QtWidgets.QLabel("Select: "))

        # Create a checkable action (toggle)
        self.select_node_action = QtWidgets.QAction("Node", self)
        self.select_node_action.setCheckable(True)
        self.select_node_action.setChecked(False)
        self.select_elm_action = QtWidgets.QAction("Element", self)
        self.select_elm_action.setCheckable(True)
        self.select_elm_action.setChecked(False)

        select_toolbar.addAction(self.select_node_action)
        select_toolbar.addAction(self.select_elm_action)

        self.select_node_action.toggled.connect(self._on_select_node_changed)
        self.select_elm_action.toggled.connect(self._on_select_elm_changed)

        # ------------------------------------------------
        # Toolbar 5 View
        # ------------------------------------------------
        view_toolbar = QtWidgets.QToolBar("View")
        view_toolbar.setMovable(True)
        self.addToolBar(Qt.TopToolBarArea, view_toolbar)

        # Define actions for tool bars and menu
        view_top_action = QtWidgets.QAction("Top (Z-)", self)
        view_top_action.setShortcut("Ctrl+1")
        view_top_action.triggered.connect(self.view_top)

        view_bottom_action = QtWidgets.QAction("Bottom (Z+)", self)
        view_bottom_action.setShortcut("Ctrl+2")
        view_bottom_action.triggered.connect(self.view_bottom)

        view_front_action = QtWidgets.QAction("Front (Y-)", self)
        view_front_action.setShortcut("Ctrl+3")
        view_front_action.triggered.connect(self.view_front)

        view_back_action = QtWidgets.QAction("Back (Y+)", self)
        view_back_action.setShortcut("Ctrl+4")
        view_back_action.triggered.connect(self.view_back)

        view_left_action = QtWidgets.QAction("Left (X-)", self)
        view_left_action.setShortcut("Ctrl+5")
        view_left_action.triggered.connect(self.view_left)

        view_right_action = QtWidgets.QAction("Right (X+)", self)
        view_right_action.setShortcut("Ctrl+6")
        view_right_action.triggered.connect(self.view_right)

        view_isometric_action = QtWidgets.QAction("Isometric", self)
        view_isometric_action.setShortcut("Ctrl+0")
        view_isometric_action.triggered.connect(self.view_isometric)

        reset_camera_action = QtWidgets.QAction("Reset", self)
        reset_camera_action.setToolTip("Reset camera to fit all visible actors")
        reset_camera_action.triggered.connect(self.reset_camera)

        # add view actions to toolbars
        view_toolbar.addAction(view_top_action)
        view_toolbar.addAction(view_bottom_action)
        view_toolbar.addAction(view_front_action)
        view_toolbar.addAction(view_back_action)
        view_toolbar.addAction(view_left_action)
        view_toolbar.addAction(view_right_action)
        view_toolbar.addAction(view_isometric_action)
        view_toolbar.addAction(reset_camera_action)

        # -------------------------
        # Connections
        # -------------------------
        self.iter_slider.valueChanged.connect(self._on_slider_changed)
        self.iter_spin.valueChanged.connect(self._on_spin_changed)
        self.animate_btn.clicked.connect(self.toggle_animation)
        self.anim_fps_spin.valueChanged.connect(self._on_anim_fps_changed)
        self.field_combo.currentTextChanged.connect(self.on_field_changed)
        self.comp_combo.currentTextChanged.connect(self.update_plot_with_clim)
        self.avg_combo.currentTextChanged.connect(self.update_plot_with_clim)

        # Timer for animation
        self.anim_timer = QTimer(self)
        self.anim_timer.setTimerType(Qt.PreciseTimer)
        self.anim_timer.timeout.connect(self._on_anim_tick)
        self._on_anim_fps_changed(self.anim_fps_spin.value())

        #####################################################################
        # Menu bar
        #####################################################################

        # --- Main Menu ---
        menubar = self.menuBar()

        # --- Menu file ---
        file_menu = menubar.addMenu("File")

        # Action "Open"
        open_action = QtWidgets.QAction("Open...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)

        # Save Data
        save_action = QtWidgets.QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_current_data)
        file_menu.addAction(save_action)

        save_as_action = QtWidgets.QAction("Save as...", self)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(self.save_current_data_as)
        file_menu.addAction(save_as_action)

        file_menu.addSeparator()

        # Save Image
        save_image_action = QtWidgets.QAction("Save Image...", self)
        save_image_action.triggered.connect(self.save_image_dialog)
        file_menu.addAction(save_image_action)

        # Save Movie
        save_movie_action = QtWidgets.QAction("Write movie...", self)
        save_movie_action.triggered.connect(self.save_movie_dialog)
        file_menu.addAction(save_movie_action)

        # Action "Quit"
        quitAction = QtWidgets.QAction("Quit", self)
        quitAction.setShortcut("Ctrl+Q")
        quitAction.triggered.connect(self.close)
        file_menu.addAction(quitAction)

        # --- Menu "Options"---
        options_menu = menubar.addMenu("Options")
        plot_options_action = QtWidgets.QAction("Plot...", self)
        plot_options_action.setShortcut("Ctrl+P")
        plot_options_action.triggered.connect(self.open_plot_dialog)
        options_menu.addAction(plot_options_action)

        clim_action = QtWidgets.QAction("Scalar bar…", self)
        clim_action.setShortcut("Ctrl+L")
        clim_action.triggered.connect(self.open_clim_dialog)
        options_menu.addAction(clim_action)

        renderer_options_action = QtWidgets.QAction("Renderer…", self)
        renderer_options_action.triggered.connect(self.open_renderer_dialog)
        options_menu.addAction(renderer_options_action)

        self.beam_options_action = QtWidgets.QAction("Beam…", self)
        self.beam_options_action.triggered.connect(self.open_beam_section_dialog)
        self.beam_options_action.setEnabled(False)
        options_menu.addAction(self.beam_options_action)
        self.shell_options_action = QtWidgets.QAction("Shell…", self)
        self.shell_options_action.triggered.connect(self.open_shell_section_dialog)
        self.shell_options_action.setEnabled(False)
        options_menu.addAction(self.shell_options_action)
        internal_fields_action = QtWidgets.QAction(
            "Show internal fields", self, checkable=True
        )
        internal_fields_action.toggled.connect(self.show_internal_fields)
        options_menu.addAction(internal_fields_action)

        apply_opt_to_menu = options_menu.addMenu("Apply options to")
        group = QtWidgets.QActionGroup(self)
        group.setExclusive(True)
        apply_opt_to_active = QtWidgets.QAction("Active window", self, checkable=True)
        apply_opt_to_all = QtWidgets.QAction("All windows", self, checkable=True)
        apply_opt_to_all.setChecked(True)
        group.addAction(apply_opt_to_active)
        group.addAction(apply_opt_to_all)
        group.triggered.connect(self._on_apply_options_target_changed)

        apply_opt_to_menu.addActions(group.actions())

        # --- Menu View ---
        view_menu = menubar.addMenu("View")
        self.object_inspector = ObjectInspector(self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.object_inspector)
        self.object_inspector.hide()
        self.object_inspector_action = self.object_inspector.toggleViewAction()
        # add same actions as view toolbar
        self._link_views_action = QtWidgets.QAction("Link views", self)
        self._link_views_action.setCheckable(True)
        self._link_views_action.triggered.connect(self._toggle_link_views)
        view_menu.addAction(self._link_views_action)

        view_menu.addAction(view_top_action)
        view_menu.addAction(view_bottom_action)
        view_menu.addAction(view_front_action)
        view_menu.addAction(view_back_action)
        view_menu.addAction(view_left_action)
        view_menu.addAction(view_right_action)
        view_menu.addAction(view_isometric_action)
        view_menu.addAction(reset_camera_action)

        # --- Menu Tools ---
        tools_menu = menubar.addMenu("Tools")

        clipAction = QtWidgets.QAction("Clip...", self)
        clipAction.setShortcut("Ctrl+K")
        clipAction.triggered.connect(self.open_clip_dialog)
        tools_menu.addAction(clipAction)
        self.clip_action = clipAction

        plotOverLineAction = QtWidgets.QAction("Plot over line...", self)
        plotOverLineAction.triggered.connect(self.open_plot_over_line_dialog)
        tools_menu.addAction(plotOverLineAction)

        element_sets_action = QtWidgets.QAction("Element sets...", self)
        element_sets_action.triggered.connect(self.open_element_sets_dialog)
        tools_menu.addAction(element_sets_action)

        node_sets_action = QtWidgets.QAction("Node sets...", self)
        node_sets_action.triggered.connect(self.open_node_sets_dialog)
        tools_menu.addAction(node_sets_action)
        scalar_expression_action = QtWidgets.QAction("Compute scalar...", self)
        scalar_expression_action.triggered.connect(self.open_scalar_expression_dialog)
        tools_menu.addAction(scalar_expression_action)

        problem_menu = menubar.addMenu("Problem")
        self.open_problem_action = QtWidgets.QAction("Open existing problem...", self)
        self.open_problem_action.triggered.connect(self.open_existing_problem_dialog)
        problem_menu.addAction(self.open_problem_action)
        self.boundary_conditions_action = QtWidgets.QAction(
            "Boundary conditions...", self
        )
        self.boundary_conditions_action.triggered.connect(
            self.open_boundary_conditions_dialog
        )
        problem_menu.addAction(self.boundary_conditions_action)

        # --- History Plot Dialog ---
        history_plot_action = QtWidgets.QAction("History plot...", self)
        history_plot_action.triggered.connect(self.open_history_plot_dialog)
        tools_menu.addAction(history_plot_action)
        self._history_plot_dialog = None

        # --- Menu Windows ---
        windows_menu = menubar.addMenu("Windows")
        windows_menu.addAction(self.object_inspector_action)

        toolbars_menu = windows_menu.addMenu("Toolbars")  # Show/Hide Toolbars
        for tb in self.findChildren(QtWidgets.QToolBar):
            action = tb.toggleViewAction()  # built-in QAction to show/hide toolbar
            toolbars_menu.addAction(action)

        distribute_menu = windows_menu.addMenu("Arrange Windows")

        # Actions
        tabify_action = QtWidgets.QAction("Tabify All", self)
        tabify_action.triggered.connect(self._distribute_tabified)
        distribute_menu.addAction(tabify_action)

        vertical_action = QtWidgets.QAction("Split Vertically", self)
        vertical_action.triggered.connect(self._distribute_vertical)
        distribute_menu.addAction(vertical_action)

        horizontal_action = QtWidgets.QAction("Split Horizontally", self)
        horizontal_action.triggered.connect(self._distribute_horizontal)
        distribute_menu.addAction(horizontal_action)

        auto_action = QtWidgets.QAction("Automatic Layout", self)
        auto_action.triggered.connect(self._distribute_auto)
        distribute_menu.addAction(auto_action)

        # --- Link Windows Menu ---
        link_windows_menu = windows_menu.addMenu("Link windows")

        # Link field action
        self._link_field_action = QtWidgets.QAction("Link field", self)
        self._link_field_action.setCheckable(True)
        self._link_field_action.setChecked(self._sync_field)
        self._link_field_action.triggered.connect(self._toggle_sync_field)
        link_windows_menu.addAction(self._link_field_action)

        # Link iteration action
        self._link_iter_action = QtWidgets.QAction("Link iteration", self)
        self._link_iter_action.setCheckable(True)
        self._link_iter_action.setChecked(self._sync_iter)
        self._link_iter_action.triggered.connect(self._toggle_sync_iter)
        link_windows_menu.addAction(self._link_iter_action)

        # Link view action
        self._link_view_action = QtWidgets.QAction("Link view", self)
        self._link_view_action.setCheckable(True)
        self._link_view_action.triggered.connect(self._toggle_link_views)
        link_windows_menu.addAction(self._link_view_action)

        link_windows_menu.addSeparator()

        # Link all action
        link_all_action = QtWidgets.QAction("Link all", self)
        link_all_action.triggered.connect(self._link_all)
        link_windows_menu.addAction(link_all_action)

        # Unlink all action
        unlink_all_action = QtWidgets.QAction("Unlink all", self)
        unlink_all_action.triggered.connect(self._unlink_all)
        link_windows_menu.addAction(unlink_all_action)

        act_fullscreen = QtWidgets.QAction("Full Screen\tF11", self)
        act_fullscreen.triggered.connect(self.toggle_fullscreen)
        windows_menu.addAction(act_fullscreen)  # or put it in View/Window menu
        self._act_fullscreen = act_fullscreen

        copy_action = QtWidgets.QAction("Copy Window", self)
        copy_action.setShortcut("Ctrl+D")  # optional
        copy_action.triggered.connect(self.copy_active_dock)
        windows_menu.addAction(copy_action)

        # Define escape and F11 shortcuts to exit fullscreen mode
        self.esc_shortcut = QShortcut(QtGui.QKeySequence("Esc"), self)
        self.esc_shortcut.setContext(Qt.ApplicationShortcut)
        self.esc_shortcut.activated.connect(self._exit_fullscreen)
        self.f11_shortcut = QShortcut(QtGui.QKeySequence("F11"), self)
        self.f11_shortcut.setContext(Qt.ApplicationShortcut)
        self.f11_shortcut.activated.connect(self.toggle_fullscreen)

        self.actions_requiring_data = [
            plot_options_action,
            clim_action,
            renderer_options_action,
            self.beam_options_action,
            self.shell_options_action,
            copy_action,
            save_action,
            save_as_action,
            clipAction,
            plotOverLineAction,
            element_sets_action,
            node_sets_action,
            scalar_expression_action,
            self.boundary_conditions_action,
            history_plot_action,
            view_top_action,
            view_bottom_action,
            view_left_action,
            view_right_action,
            view_front_action,
            view_back_action,
            view_isometric_action,
            reset_camera_action,
        ]
        # -------------------------
        # Dockable PyVista Widget
        # -------------------------
        if data is not None:
            filename = None
            if isinstance(data, str):
                filename = data
                title = os.path.basename(data)
                data = fd.read_data(data)
            else:
                title = "Data"

            if _is_problem(data):
                title = data.name or "Problem"
                data = ProblemDataSet(data)

            self.add_dataset_dock(data, title=title, filename=filename)
            # self.plotter = QtInteractor(self)
            # dock_plotter = QDockWidget("3D View", self)
            # dock_plotter.setWidget(self.plotter.interactor)
            # dock_plotter.setAllowedAreas(Qt.AllDockWidgetAreas)
            # self.addDockWidget(Qt.RightDockWidgetArea, dock_plotter)
        else:
            self.setRange(0, 0)
            for action in self.actions_requiring_data:
                (action.setEnabled(False),)

        # Initialisation
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        ICON_PATH = os.path.join(BASE_DIR, "_viewer", "fedoo_logo_simple.png")
        self.setWindowIcon(QtGui.QIcon(ICON_PATH))
        self.setWindowTitle("Fedoo Viewer")

    @property
    def opts(self):
        if self.active_dock:
            return self.active_dock.opts
        else:
            return None

    @property
    def data(self):
        return self.active_dock.data

    def update_plot(self, *args, **kargs):
        if self.active_dock:
            return self.active_dock.update_plot(*args, **kargs)

    def add_dataset_dock(self, data, title, opts=None, filename=None):
        if _is_problem(data):
            data = ProblemDataSet(data)
        dock = PlotDock(data, title, self, opts, filename=filename)

        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.visibilityChanged.connect(
            lambda v, d=dock: self._set_active(d) if v else None  # self._hide_dock
        )
        dock.widget().installEventFilter(self)
        self.window_layout()
        self._update_dock_selector()
        self.update_plot(lock_view=False)
        return dock

    # def _hide_dock(self, dock):
    #     dock.setFloating(True)
    #     dock.move(-10000, -10000)

    def copy_active_dock(self):
        dock = self.active_dock
        if dock:
            copied_dock = self.add_dataset_dock(
                dock.data.copy(), dock.title, opts=dict(dock.opts)
            )
            copied_dock.line_points = dock.line_points
            if self._line_widget_enabled and self.active_dock is copied_dock:
                self._rebuild_line_widget()
            copied_dock.plotter.camera_position = dock.plotter.camera_position

    def eventFilter(self, obj, event):
        et = event.type()
        if et in (QEvent.FocusIn, QEvent.MouseButtonPress):
            dock = self._dock_for_widget(obj)
            if dock and dock is not self.active_dock:
                self._set_active(dock)
        return super().eventFilter(obj, event)

    def _on_apply_options_target_changed(self, action):
        self.apply_options_to_all = action.text() == "All windows"

    def _update_dock_selector(self):
        # Block signals while updating the combo
        blocker = QSignalBlocker(self.dock_selector_combo)

        self.dock_selector_combo.clear()
        for dock in self.all_docks:
            self.dock_selector_combo.addItem(dock.windowTitle())
        if self.active_dock not in self.all_docks:
            if len(self.all_docks) > 0:
                self._set_active(self.all_docks[0])
            else:
                self.active_dock = None
        if self.active_dock:
            self.dock_selector_combo.setCurrentIndex(
                self.all_docks.index(self.active_dock)
            )

    def rename_active_dock_from_combo(self):
        dock = self.active_dock
        if not dock:
            return

        new_name = self.dock_selector_combo.currentText().strip()
        if not new_name:
            return
        # extract window number if any
        match = re.match(r"^(\d+)\s*:\s*(.*)$", new_name)
        if match:
            dock._dock_index = int(match.group(1))
            if dock._dock_index > PlotDock._n_created_dock:
                PlotDock._n_created_dock = dock._dock_index
            dock.title = match.group(2)
        else:
            dock.title = new_name

        dock.setWindowTitle(f"{dock._dock_index}: " + dock.title)
        self._update_dock_selector()

    def _dock_for_widget(self, w):
        # Walk up parents until the QDockWidget (or your PlotDock subclass)
        while w:
            if isinstance(w, PlotDock):
                return w
            w = w.parentWidget()
        return None

    def _distribute_tabified(self):
        """redistribute docks"""
        # Simple example: tabify all docks together
        active_dock = self.active_dock
        if len(self.all_docks) > 1:
            # first = self.active_dock
            first = self.all_docks[0]
            for dock in self.all_docks:
                if dock is not first:
                    self.tabifyDockWidget(first, dock)
        self._set_active(active_dock)
        self.active_dock.raise_()
        self.window_layout = self._distribute_tabified

    def _distribute_vertical(self):
        self._distribute_split(Qt.Vertical)
        self.window_layout = self._distribute_vertical

    def _distribute_horizontal(self):
        self._distribute_split(Qt.Horizontal)
        self.window_layout = self._distribute_horizontal

    def _distribute_split(self, orientation):
        if not self.all_docks:
            return
        first = self.all_docks[0]
        self.addDockWidget(Qt.RightDockWidgetArea, first)
        for dock in self.all_docks[1:]:
            self.splitDockWidget(first, dock, orientation)

    def _distribute_auto(self):
        n = len(self.all_docks)
        if n == 0:
            return

        # Compute grid dimensions
        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        # # Start with the first dock
        # main_dock = self.all_docks[0]
        # self.addDockWidget(Qt.RightDockWidgetArea, main_dock)

        # Arrange remaining docks
        dock_index = 0
        start_dock = []
        for r in range(rows):
            for c in range(cols):
                if dock_index >= n:
                    break
                dock = self.all_docks[dock_index]
                if r == 0:
                    start_dock.append(dock)
                if c == 0 and r == 0:
                    self.addDockWidget(Qt.RightDockWidgetArea, dock)
                    # current_row_start = dock
                elif r == 0:
                    # First row: split horizontally
                    self.splitDockWidget(start_dock[0], dock, Qt.Horizontal)
                    # current_row_start = dock

                else:
                    # split vertically
                    self.splitDockWidget(start_dock[c], dock, Qt.Vertical)
                    # current_row_start = dock
                # else:
                #     # Subsequent columns: split horizontally
                #     self.splitDockWidget(current_row_start, dock, Qt.Horizontal)
                dock_index += 1
        self.window_layout = self._distribute_auto

    def _clip_is_supported(self, dock=None):
        if dock is None:
            dock = self.active_dock
        mesh = getattr(getattr(dock, "data", None), "mesh", None)
        return mesh is not None and getattr(mesh, "ndim", 3) == 3

    def _toggle_link_views(self, checked):
        # Synchronize both menu items
        self._link_views_action.setChecked(checked)
        self._link_view_action.setChecked(checked)

        if checked:
            # Sync all cameras to the active dock
            if len(self.all_docks) > 1:
                ref = self.active_dock
                for dock in self.all_docks:
                    if dock is not ref:
                        ref.plotter.link_views_across_plotters(dock.plotter)
        else:
            if len(self.all_docks) > 0:
                # Unlink when disabled
                cam = self.active_dock.plotter.camera_position
                for dock in self.all_docks:
                    try:
                        dock.plotter.unlink_views()
                        dock.plotter.camera_position = cam
                    except Exception:
                        pass

    def _toggle_sync_field(self, checked):
        """Toggle field synchronization across windows."""
        self._sync_field = checked
        if checked and self.active_dock:
            # Synchronize all docks to active dock's current state
            for dock in self.all_docks:
                if dock is not self.active_dock:
                    field = _viewer_field_name(
                        self.active_dock.data, self.active_dock.current_field
                    )
                    if field in _viewer_field_names(dock.data):
                        dock.beam_csys = getattr(self.active_dock, "beam_csys", "local")
                        dock.current_field = _viewer_tensor_field(
                            dock.data, field, dock.beam_csys
                        )
                        comps = dock.get_components(dock.current_field)
                        if self.active_dock.current_comp in comps:
                            dock.current_comp = self.active_dock.current_comp
                    dock.current_data_type = self.active_dock.current_data_type
                    dock.update_plot()

    def _toggle_sync_iter(self, checked):
        """Toggle iteration synchronization across windows."""
        self._sync_iter = checked
        if checked and self.active_dock:
            # Synchronize all docks to active dock's current iteration
            active_iter = self.active_dock.current_iter
            for dock in self.all_docks:
                if (
                    dock is not self.active_dock
                    and _data_n_iter(dock.data) > active_iter
                ):
                    dock.current_iter = active_iter
                    dock.update_plot(iteration=active_iter)

    def _link_all(self):
        """Enable all link options: field, iteration, and view."""
        if not self._sync_field:
            self._toggle_sync_field(True)
        if not self._sync_iter:
            self._toggle_sync_iter(True)
        self._toggle_link_views(True)
        self._link_field_action.setChecked(True)
        self._link_iter_action.setChecked(True)
        self._link_view_action.setChecked(True)

    def _unlink_all(self):
        """Disable all link options: field, iteration, and view."""
        self._sync_field = False
        self._sync_iter = False
        self._link_field_action.setChecked(False)
        self._link_iter_action.setChecked(False)
        self._link_view_action.setChecked(False)
        self._toggle_link_views(False)

    def _set_active(self, dock):
        self._clip_redraw_timer.stop()
        # block all signals to avoid replot
        blockers = [
            QSignalBlocker(self.iter_slider),
            QSignalBlocker(self.iter_spin),
            QSignalBlocker(self.field_combo),
            QSignalBlocker(self.avg_combo),
            QSignalBlocker(self.dock_selector_combo),
        ]
        if self._plane_widget_enabled and self.active_dock:
            # remove plane widget from previous active dock
            # try:
            self.plotter.clear_plane_widgets()
            self._plane_widget = None
            self._plane_widget_plotter = None
            # except Exception:
            #     self.plotter.clear_widgets()
        if self._line_widget_enabled and self.active_dock:
            self.plotter.clear_line_widgets()
        if self.active_dock:
            self.plotter.remove_actor("_picked_cell")  # if picked cell mesh exist
            self.plotter.remove_actor("_picked_node_set")
            self.plotter.disable_picking()
        if self.element_sets_dialog:
            self.element_sets_dialog._stop_picking()
        if self.node_sets_dialog:
            self.node_sets_dialog._cancel_surface_pick()
            self.node_sets_dialog._stop_picking()
        if self._history_plot_dialog:
            self._stop_element_picking()

        # save node/element selector current values
        node_selector_ischecked = self.select_node_action.isChecked()
        elm_selector_ischecked = self.select_elm_action.isChecked()
        if node_selector_ischecked:
            self._on_select_node_changed(False)
            self.statusBar().clearMessage()
        if elm_selector_ischecked:
            self._on_select_elm_changed(False)
            self.statusBar().clearMessage()

        self.active_dock = dock

        if hasattr(self, "clip_action"):
            clip_supported = self._clip_is_supported(dock)
            self.clip_action.setEnabled(clip_supported)
            if not clip_supported:
                dock.opts["clip_args"] = None
                self.disable_plane_widget()

        for d in self.all_docks:
            d._titlebar._set_active(d is dock)

        max_iter = _data_n_iter(dock.data) - 1
        if isinstance(dock.data, ProblemDataSet) and dock.data.current_mode is not None:
            dock.current_iter = dock.data.current_mode
        current_iter = dock.current_iter  # may be modified by setRange
        self.setRange(0, max_iter)
        self.iter_slider.setValue(current_iter)
        self.iter_spin.setValue(current_iter)

        # old_state = self.field_combo.blockSignals(True)

        # update_field_combo
        dock = self.active_dock
        current_field = dock.current_field
        self.beam_section_button.setEnabled(bool(dock.data.beam_derived_fields()))
        field_blocker = QSignalBlocker(self.field_combo)
        self.field_combo.clear()
        self.field_combo.addItem(SOLID_COLOR_FIELD)
        if self.data is not None:
            self.field_combo.addItems(_viewer_field_names(self.data))
        if dock.current_field is not None:
            self.field_combo.setCurrentText(
                _viewer_field_name(dock.data, dock.current_field)
            )
        else:
            self.field_combo.setCurrentText(SOLID_COLOR_FIELD)
        del field_blocker
        # self.field_combo.blockSignals(old_state)

        # update data_type value
        # old_state = self.avg_combo.blockSignals(True)
        self.avg_combo.setCurrentText(dock.current_data_type)

        # update component combo
        self.update_coordinate_choices(current_field)
        self.update_components(_viewer_field_name(dock.data, current_field))

        # self.avg_combo.blockSignals(old_state)
        if self._plot_dialog:  # if plot dialog exist
            self._plot_dialog.update_values()
        if self._renderer_dialog:
            self._renderer_dialog.update_values()
        if self._clim_dialog:
            self._clim_dialog.update_values()
        if self._plane_widget_enabled:
            self.enable_plane_widget()
        if self._line_widget_enabled:
            self._rebuild_line_widget()
        if self._clip_dialog:
            self._clip_dialog.update_values()
        if self.element_sets_dialog:
            self.element_sets_dialog._refresh_sets_list()
            if self.active_dock.visibility_mode:
                if self.active_dock.visibility_mode == "show":
                    self.element_sets_dialog.current_selection = set(
                        self.opts["element_set"]
                    )
                else:
                    self.element_sets_dialog.current_selection = (
                        self.element_sets_dialog.all_elements()
                        - set(self.opts["element_set"])
                    )
            else:
                self.element_sets_dialog.current_selection = set()
            self.element_sets_dialog._update_selection_info()
        if self.node_sets_dialog:
            self.node_sets_dialog.current_selection = set()
            self.node_sets_dialog._refresh_sets_list()
            self.node_sets_dialog._update_selection_info()
        if self.scalar_expression_dialog:
            self.scalar_expression_dialog.refresh()
        self.boundary_conditions_action.setEnabled(
            isinstance(dock.data, ProblemDataSet)
        )
        if self.boundary_conditions_dialog:
            self.boundary_conditions_dialog.refresh()
        if self._history_plot_dialog:
            self._history_plot_dialog._on_change_active()
        if self.active_dock in self.all_docks:
            self.dock_selector_combo.setCurrentIndex(
                self.all_docks.index(self.active_dock)
            )
        if node_selector_ischecked:
            self._on_select_node_changed(True)
        elif elm_selector_ischecked:
            self._on_select_elm_changed(True)
        self._refresh_object_inspector()

    def _refresh_object_inspector(self):
        if self.object_inspector is not None:
            self.object_inspector.schedule_refresh()

    def _on_anim_tick(self):
        self._refresh_iterations()
        # go to next iteration
        new_iter = self.iteration + 1
        if new_iter > self.iter_slider.maximum():
            if self.anim_wrap_check.isChecked():
                new_iter = self.iter_slider.minimum()
            else:
                # Arrêter l'anim au bout
                self.stop_animation()
                return

        self.iter_slider.setValue(new_iter)

    def _on_slider_changed(self, val: int):
        self._change_iteration(val, self.iter_spin)

    def _on_spin_changed(self, val: int):
        self._change_iteration(val, self.iter_slider)

    def _change_iteration(self, val, other_control):
        self._refresh_iterations()
        if other_control.value() != val:
            blocker = QSignalBlocker(other_control)
            other_control.setValue(val)
            del blocker

        if self.active_dock:
            self.active_dock.current_iter = val

        if self._sync_iter:
            for dock in self.all_docks:
                if _data_n_iter(dock.data) > val:
                    dock.current_iter = val
                    dock.update_plot(iteration=val)
        else:
            self.update_plot(iteration=self.iteration)

    def _refresh_iterations(self, reload_current=False):
        """Refresh only frame references; keep expensive all-frame outputs."""
        docks = self.all_docks if self._sync_iter else [self.active_dock]
        refreshed = {}
        for dock in docks:
            if dock is None:
                continue
            try:
                key = id(dock.data)
                if key not in refreshed:
                    refreshed[key] = _refresh_result_frames(
                        dock.data, reload_current=reload_current
                    )
                dock.data = refreshed[key]
                if (
                    reload_current
                    and isinstance(dock.data, ProblemDataSet)
                    and dock.data.current_mode is not None
                ):
                    dock.current_iter = dock.data.current_mode
                dock.current_iter = min(
                    dock.current_iter, max(0, _data_n_iter(dock.data) - 1)
                )
            except (OSError, KeyError) as exc:
                # The simulation may have the file open for writing. Retry on
                # the next navigation/reload rather than interrupting it.
                self.statusBar().showMessage(f"Results temporarily unavailable: {exc}")
        if self.active_dock:
            blockers = [
                QSignalBlocker(self.iter_slider),
                QSignalBlocker(self.iter_spin),
            ]
            self.setRange(0, max(0, _data_n_iter(self.data) - 1))
            if reload_current and isinstance(self.data, ProblemDataSet):
                self.iter_slider.setValue(self.active_dock.current_iter)
                self.iter_spin.setValue(self.active_dock.current_iter)
            del blockers

    def reload_results(self):
        self._refresh_iterations(reload_current=True)
        docks = self.all_docks if self._sync_iter else [self.active_dock]
        for dock in docks:
            if dock is not None:
                dock.update_plot(iteration=dock.current_iter)

    # set slider and iteration spinbox min/max values
    def setRange(self, min_iter: int, max_iter: int):
        self.iter_slider.setRange(min_iter, max_iter)
        self.iter_spin.setRange(min_iter, max_iter)
        mode_navigation = (
            self.active_dock
            and isinstance(self.data, ProblemDataSet)
            and self.data.n_modes
        )
        hint = (
            "Select a stored eigenmode (zero-based index, as for FDH5 frames)."
            if mode_navigation
            else "Select a result iteration."
        )
        self.iter_slider.setToolTip(hint)
        self.iter_spin.setToolTip(hint)
        # Ajuster les ticks si besoin
        self.iter_slider.setTickInterval(max(1, (max_iter - min_iter) // 10))

    # --- Animation ---
    def toggle_animation(self, checked: bool):
        if checked:
            self.start_animation()
        else:
            self.stop_animation()

    def start_animation(self):
        self._refresh_iterations()
        if self.anim_timer.isActive():
            return
        if self.iter_slider.maximum() <= self.iter_slider.minimum():
            self.animate_btn.setChecked(False)
            return
        if self.iter_slider.value() == self.iter_slider.maximum():
            self.iter_slider.setValue(self.iter_slider.minimum())

        self.animate_btn.setText("⏸ Pause")
        self.anim_timer.start()

    def stop_animation(self):
        if not self.anim_timer.isActive():
            return
        self.anim_timer.stop()
        self.animate_btn.setText("▶ Play")
        self.animate_btn.setChecked(False)

    def _on_anim_fps_changed(self, fps: float):
        # Protection: fps peut valoir 0.1–60
        interval_ms = int(1000.0 / max(0.1, fps))
        self.anim_timer.setInterval(interval_ms)

    @property
    def iteration(self) -> int:
        return self.iter_slider.value()

    def on_field_changed(self, field):
        field = None if field == SOLID_COLOR_FIELD else field
        self.update_coordinate_choices(field)
        self.update_components(field)
        if self.active_dock:
            self.update_plot_with_clim(lock_view=True)

    def update_coordinate_choices(self, field):
        blocker = QSignalBlocker(self.csys_combo)
        self.csys_combo.clear()
        dock = self.active_dock
        if dock is not None:
            field = _viewer_field_name(dock.data, field)
        derived = dock.data.section_derived_fields() if dock is not None else {}
        beams = dock.data.beam_derived_fields() if dock is not None else {}
        shells = dock.data.shell_derived_fields() if dock is not None else {}
        self.beam_options_action.setEnabled(bool(beams))
        self.beam_section_button.setEnabled(bool(beams))
        self.shell_options_action.setEnabled(bool(shells))
        self.shell_section_button.setEnabled(bool(shells))
        if f"{field}_local" in derived:
            self.csys_combo.addItem("Local", "local")
            if f"{field}_global" in derived:
                self.csys_combo.addItem("Global", "global")
            selected = self.csys_combo.findData(getattr(dock, "beam_csys", "local"))
            self.csys_combo.setCurrentIndex(max(0, selected))
            dock.beam_csys = self.csys_combo.currentData()
            self.csys_combo.setEnabled(self.csys_combo.count() > 1)
        else:
            self.csys_combo.addItem("Stored", "stored")
            self.csys_combo.setEnabled(False)
        shell_blocker = QSignalBlocker(self.shell_csys_combo)
        self.shell_csys_combo.clear()
        for index in range(self.csys_combo.count()):
            self.shell_csys_combo.addItem(
                self.csys_combo.itemText(index), self.csys_combo.itemData(index)
            )
        self.shell_csys_combo.setCurrentIndex(self.csys_combo.currentIndex())
        self.shell_csys_combo.setEnabled(bool(shells) and self.csys_combo.isEnabled())
        del shell_blocker
        del blocker

    def on_coordinates_changed(self, index):
        shell_blocker = QSignalBlocker(self.shell_csys_combo)
        self.shell_csys_combo.setCurrentIndex(index)
        del shell_blocker
        if self.active_dock is None:
            return
        field = self.field_combo.currentText()
        coordinates = self.csys_combo.currentData()
        if coordinates not in ("local", "global"):
            return
        apply = (
            self.apply_shell_options
            if self.active_dock.data.shell_derived_fields()
            and not self.active_dock.data.beam_derived_fields()
            else self.apply_beam_options
        )
        if not apply({}, coordinates, field):
            self.update_coordinate_choices(field)
            return
        self.update_components(field)

    def show_internal_fields(self, enabled):
        for dock in self.all_docks:
            if dock.data is not None:
                dock.data.show_internal_fields = enabled
        if self.active_dock is not None:
            field = self.field_combo.currentText()
            blocker = QSignalBlocker(self.field_combo)
            self.field_combo.clear()
            self.field_combo.addItem(SOLID_COLOR_FIELD)
            self.field_combo.addItems(_viewer_field_names(self.active_dock.data))
            if self.field_combo.findText(field) >= 0:
                self.field_combo.setCurrentText(field)
            del blocker
            if field.startswith("_") and not enabled:
                self.on_field_changed(SOLID_COLOR_FIELD)

    def open_beam_section_dialog(self):
        dock = self.active_dock
        if dock is None or not dock.data.beam_derived_fields():
            return
        options = dock.data.beam_options
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Beam options")
        layout = QtWidgets.QFormLayout(dialog)
        help_label = QLabel(
            "Recover Stress/Strain at a section point or compute sampled extrema."
        )
        help_label.setToolTip(
            "Shear uses parabolic approximations for rectangles/disks, otherwise Q/A. "
            "Torsion recovery supports circles, pipes and rectangles."
        )
        layout.addRow(help_label)
        section_data = dock.data
        submesh_choice = None
        if section_data._is_multimesh():
            submesh_choice = QtWidgets.QComboBox()
            submesh_choice.setObjectName("beam_submesh")
            for index, mesh in enumerate(dock.data.mesh.submeshes):
                candidate = dock.data._submesh_dataset(index)
                if candidate.beam_derived_fields():
                    label = mesh.name or mesh.elm_type
                    submesh_choice.addItem(f"{index}: {label}", index)
            if not submesh_choice.count():
                return
            current = submesh_choice.findData(dock.data.active_submesh)
            submesh_choice.setCurrentIndex(max(0, current))
            section_data = dock.data._submesh_dataset(submesh_choice.currentData())
            submesh_choice.setToolTip(
                "Choose the beam section used for sampling and preview. "
                "Its selected normalized coordinates apply to all beam submeshes."
            )
            layout.addRow("Beam submesh:", submesh_choice)
        displayed = _viewer_field_name(dock.data, dock.current_field)
        kind = (
            displayed
            if displayed in ("Stress", "Strain")
            else (
                "Stress"
                if "Stress_local" in dock.data.beam_derived_fields()
                else "Strain"
            )
        )
        if f"{kind}_local" not in section_data.beam_derived_fields():
            kind = (
                "Stress"
                if "Stress_local" in section_data.beam_derived_fields()
                else "Strain"
            )
        source_choice = QtWidgets.QComboBox()
        source_choice.setObjectName("beam_field_source")
        source_choice.addItem("Stored", "stored")
        source_choice.addItem("Recomputed", "recomputed")
        source_choice.setToolTip(
            "Required on the selected beam submesh. Other beam submeshes use this "
            "source when available, otherwise the other source. Stored positions "
            "use the nearest saved point."
        )
        has_stored = "stored_points" in section_data.field_metadata.get(kind, {})
        has_recomputed = (
            ("BeamStress" if kind == "Stress" else "BeamStrain")
            in section_data.gausspoint_data
            and section_data.beam_section_description().get("recovery", "linear")
            != "stored_only"
        )
        source_choice.model().item(0).setEnabled(has_stored)
        source_choice.model().item(1).setEnabled(has_recomputed)
        requested = options.get("source", "auto")
        source_choice.setCurrentIndex(
            0 if has_stored and (requested != "recomputed" or not has_recomputed) else 1
        )
        layout.addRow("Source:", source_choice)
        reduction = QtWidgets.QComboBox()
        reduction.setObjectName("beam_section_reduction")
        modes = [
            ("Section point index", "point_index"),
            ("Normalized coordinates (y, z)", "position"),
            ("Maximum", "max"),
            ("Minimum", "min"),
            ("Largest magnitude", "abs_max"),
            ("Largest magnitude (signed)", "signed_abs_max"),
        ]
        for label, value in modes:
            reduction.addItem(label, value)
        reduction.setCurrentIndex(
            reduction.findData(
                options.get("reduction")
                or (
                    "point_index"
                    if options.get("point_index") is not None
                    else "position"
                )
            )
        )
        layout.addRow("Value:", reduction)
        position = options.get("position")
        if position is None:
            position = (0.0, 0.0)
        y, z = QDoubleSpinBox(), QDoubleSpinBox()
        for spin in (y, z):
            spin.setDecimals(10)
            spin.setRange(-1e12, 1e12)
        y.setValue(float(position) if np.isscalar(position) else position[0])
        z.setValue(0.0 if np.isscalar(position) else position[1])
        layout.addRow("Normalized y:", y)
        layout.addRow("Normalized z:", z)
        point_index = QSpinBox()
        point_index.setRange(0, 100000)
        point_index.setValue(options.get("point_index") or 0)
        layout.addRow("Section point index (from 0):", point_index)
        samples = QSpinBox()
        samples.setObjectName("beam_sample_points_number")
        samples.setRange(1, 1000000)
        samples.setKeyboardTracking(False)
        samples.setValue(options.get("samples", 32))
        samples.setToolTip(
            "Snaps to the closest supported total point count when editing finishes. "
            "Stored and supplied custom point sets have a fixed count."
        )
        layout.addRow("Sample points number:", samples)
        # The longitudinal location only affects the section preview and the
        # displayed coordinates of indexed samples, not stress extraction.
        beam_location = 0
        source = next(
            (
                name
                for name in ("BeamStress", "BeamStrain", "Stress", "Strain")
                if name in section_data.gausspoint_data
            ),
            None,
        )
        count = section_data.gausspoint_data[source].shape[-1] if source else 1
        point_coordinates_label = QLabel()
        point_coordinates_label.setWordWrap(True)
        layout.addRow(point_coordinates_label)
        y.setObjectName("beam_section_y")
        z.setObjectName("beam_section_z")
        point_index.setObjectName("beam_section_point_index")
        manual_position = [y.value(), z.value()]
        was_indexed = False
        cached_points = {}

        def show_point_coordinates():
            nonlocal was_indexed
            indexed = reduction.currentData() == "point_index"
            if indexed and not was_indexed:
                manual_position[:] = [y.value(), z.value()]
            elif not indexed and was_indexed:
                y.setValue(manual_position[0])
                z.setValue(manual_position[1])
            was_indexed = indexed
            if not indexed:
                point_coordinates_label.clear()
                return
            try:
                budget = samples.value()
                location = beam_location
                key = (source_choice.currentData(), budget, location)
                if key not in cached_points:
                    description = section_data.beam_section_description()
                    for name, value in description["properties"].items():
                        values = np.asarray(value).reshape(-1)
                        description["properties"][name] = values[
                            0 if values.size == 1 else location
                        ]
                    section = SavedBeamSection(description)
                    cached_points.clear()
                    if source_choice.currentData() == "stored":
                        points = np.asarray(
                            section_data.field_metadata[kind]["stored_points"]
                        )
                        cached_points[key] = (
                            points if points.ndim == 2 else points[:, :, location]
                        )
                    else:
                        cached_points[key] = section.section.normalized_output_points(
                            budget
                        )
                points = cached_points[key]
                blocker = QSignalBlocker(point_index)
                point_index.setMaximum(len(points) - 1)
                del blocker
                physical = points[point_index.value()]
                values = [np.asarray(value).reshape(-1) for value in physical]
                y.setValue(float(values[0][0]))
                z.setValue(float(values[1][0]))
                point_coordinates_label.clear()
            except (ValueError, NotImplementedError, IndexError) as error:
                point_coordinates_label.setText(
                    f"Point coordinates unavailable: {error}"
                )
                y.clear()
                z.clear()

        point_index.valueChanged.connect(show_point_coordinates)
        samples.valueChanged.connect(show_point_coordinates)
        reduction.currentIndexChanged.connect(show_point_coordinates)
        source_choice.currentIndexChanged.connect(show_point_coordinates)
        show_point_coordinates()
        preview_button = QPushButton("Section geometry and sampling…")
        layout.addRow(preview_button)

        def preview_section():
            data = section_data
            description = data.beam_section_description()
            preview = QtWidgets.QDialog(dialog)
            preview_title = f"Beam section: {description['geometry']}"
            if submesh_choice is not None:
                preview_title += f" — submesh {submesh_choice.currentData()}"
            preview.setWindowTitle(preview_title)
            box = QVBoxLayout(preview)
            selector = QSpinBox()
            selector.setObjectName("beam_preview_location")
            preview_count = max(
                count,
                max(
                    (np.asarray(v).size for v in description["properties"].values()),
                    default=1,
                ),
            )
            selector.setRange(0, preview_count - 1)
            selector.setPrefix("Longitudinal Gauss-point index: ")
            selector.setValue(min(beam_location, preview_count - 1))
            selector.setToolTip(
                "Preview the section properties and output points at this beam location."
            )
            box.addWidget(selector)
            preview_coordinates = QtWidgets.QComboBox()
            preview_coordinates.setObjectName("beam_preview_coordinates")
            preview_coordinates.addItem("Physical coordinates (y, z)", "physical")
            preview_coordinates.addItem("Normalized coordinates (y, z)", "normalized")
            preview_coordinates.setToolTip(
                "Physical coordinates use the section dimensions at the selected "
                "Gauss point. Normalized coordinates use the section bounds."
            )
            box.addWidget(preview_coordinates)
            property_label = QLabel()
            property_label.setWordWrap(True)
            box.addWidget(property_label)
            figure = Figure(figsize=(5, 5))
            canvas = FigureCanvas(figure)
            box.addWidget(canvas)

            def draw():
                nonlocal beam_location
                figure.clear()
                axes = figure.add_subplot(111)
                try:
                    index = selector.value()
                    beam_location = index
                    show_point_coordinates()
                    local_description = dict(description)
                    local_description["properties"] = {
                        name: np.asarray(value).reshape(-1)[
                            min(index, np.asarray(value).size - 1)
                        ]
                        for name, value in description["properties"].items()
                    }
                    section = SavedBeamSection(local_description)
                    if source_choice.currentData() == "stored":
                        normalized = np.asarray(
                            data.field_metadata[kind]["stored_points"]
                        )
                        if normalized.ndim == 3:
                            normalized = normalized[:, :, index]
                    else:
                        try:
                            normalized = section.section.normalized_output_points(
                                samples.value()
                            )
                        except (ValueError, NotImplementedError):
                            if reduction.currentData() != "position":
                                raise
                            normalized = np.empty((0, 2))
                    use_normalized = preview_coordinates.currentData() == "normalized"
                    points = (
                        normalized
                        if use_normalized
                        else [
                            section.normalized_section_coordinates(tuple(point))
                            for point in normalized
                        ]
                    )
                    xy = np.asarray(
                        [
                            [
                                np.asarray(y).reshape(-1)[
                                    min(index, np.asarray(y).size - 1)
                                ],
                                np.asarray(z).reshape(-1)[
                                    min(index, np.asarray(z).size - 1)
                                ],
                            ]
                            for y, z in points
                        ]
                    ).reshape(-1, 2)
                    axes.scatter(xy[:, 0], xy[:, 1], s=10, label="Output points")
                    if reduction.currentData() == "point_index":
                        selected = min(point_index.value(), len(xy) - 1)
                        current_point = xy[selected]
                    elif reduction.currentData() == "position":
                        current_point = (y.value(), z.value())
                        if source_choice.currentData() == "stored":
                            nearest = int(
                                stored_point_indices(normalized, current_point)
                            )
                            current_point = xy[nearest]
                        elif not use_normalized:
                            current_point = section.normalized_section_coordinates(
                                current_point
                            )
                            current_point = [
                                np.asarray(value).reshape(-1)[
                                    min(index, np.asarray(value).size - 1)
                                ]
                                for value in current_point
                            ]
                    else:
                        current_point = None
                    if current_point is not None:
                        axes.scatter(
                            *current_point,
                            s=75,
                            facecolors="none",
                            edgecolors="red",
                            label="Nearest stored point"
                            if source_choice.currentData() == "stored"
                            and reduction.currentData() == "position"
                            else "Current point",
                        )
                    axes.plot(0, 0, "+", color="red", label="Centroid / beam axis")
                    axes.set_aspect("equal")
                    label = "Normalized" if use_normalized else "Physical local"
                    axes.set_xlabel(f"{label} y")
                    axes.set_ylabel(f"{label} z")
                    axes.set_title(description["geometry"])
                    axes.legend()
                    property_label.setText(
                        "; ".join(
                            f"{name} = {np.asarray(value).reshape(-1)[min(index, np.asarray(value).size - 1)]:.6g}"
                            for name, value in description["properties"].items()
                        )
                    )
                except (ValueError, NotImplementedError) as error:
                    axes.text(
                        0.05, 0.5, str(error), transform=axes.transAxes, wrap=True
                    )
                canvas.draw()

            selector.valueChanged.connect(draw)
            preview_coordinates.currentIndexChanged.connect(draw)
            draw()
            preview.resize(600, 600)
            preview.exec()

        preview_button.clicked.connect(preview_section)
        derived = section_data.beam_derived_fields()
        frame_choice = QtWidgets.QComboBox()
        frame_choice.setObjectName("beam_coordinate_system")
        frame_choice.addItem("Local", "local")
        if f"{kind}_global" in derived:
            frame_choice.addItem("Global", "global")
        previous_csys = getattr(
            dock,
            "beam_csys",
            "global",
        )
        frame_choice.setCurrentIndex(max(0, frame_choice.findData(previous_csys)))
        frame_choice.setEnabled(frame_choice.count() > 1)
        layout.addRow("Coordinate system:", frame_choice)
        layout.addRow(
            QLabel(
                "Normalized (y, z): 0 is the centroid; ±1 selects each bounding-box face.\n"
                "Extrema depend on the available section samples."
            )
        )

        def enable_controls():
            stored = source_choice.currentData() == "stored"
            for spin in (y, z):
                spin.setToolTip(
                    "Stored results use the nearest saved point in normalized (y, z) "
                    "at each beam location. The preview highlights that point."
                    if stored
                    else "Requested normalized section coordinate."
                )
            try:
                available = (
                    len(section_data.field_metadata[kind]["stored_points"])
                    if stored
                    else len(
                        SavedBeamSection(
                            section_data.beam_section_description()
                        ).section.normalized_output_points(max(8, samples.value()))
                    )
                )
            except (ValueError, NotImplementedError, KeyError):
                available = 0
            for index in range(reduction.count()):
                reduction.model().item(index).setEnabled(
                    bool(available) or reduction.itemData(index) == "position"
                )
            if not available and reduction.currentData() != "position":
                reduction.setCurrentIndex(reduction.findData("position"))
            point = reduction.currentData() in ("point_index", "position")
            y.setEnabled(point)
            z.setEnabled(point)
            y.setReadOnly(reduction.currentData() == "point_index")
            z.setReadOnly(reduction.currentData() == "point_index")
            point_index.setEnabled(reduction.currentData() == "point_index")
            fixed = stored or (
                section_data.beam_section_description().get("output_points") is not None
                or "section_points" in section_data.beam_section_description()
            )
            samples.setEnabled(not fixed and bool(available))
            if available:
                blocker = QSignalBlocker(samples)
                samples.setMaximum(max(samples.maximum(), available))
                samples.setValue(available)
                del blocker

        def snap_samples():
            enable_controls()
            cached_points.clear()
            show_point_coordinates()

        samples.editingFinished.connect(snap_samples)

        reduction.currentIndexChanged.connect(enable_controls)
        source_choice.currentIndexChanged.connect(enable_controls)
        enable_controls()
        show_point_coordinates()
        if submesh_choice is not None:

            def select_submesh():
                nonlocal section_data, count, beam_location, kind
                section_data = dock.data._submesh_dataset(submesh_choice.currentData())
                available_fields = section_data.beam_derived_fields()
                if f"{kind}_local" not in available_fields:
                    kind = "Stress" if "Stress_local" in available_fields else "Strain"
                stored = "stored_points" in section_data.field_metadata.get(kind, {})
                recomputed = (
                    ("BeamStress" if kind == "Stress" else "BeamStrain")
                    in section_data.gausspoint_data
                    and section_data.beam_section_description().get(
                        "recovery", "linear"
                    )
                    != "stored_only"
                )
                blocker = QSignalBlocker(source_choice)
                source_choice.model().item(0).setEnabled(stored)
                source_choice.model().item(1).setEnabled(recomputed)
                if source_choice.currentData() == "stored" and not stored:
                    source_choice.setCurrentIndex(1)
                elif source_choice.currentData() == "recomputed" and not recomputed:
                    source_choice.setCurrentIndex(0)
                del blocker
                count = max(
                    (
                        np.asarray(values).shape[-1]
                        for name, values in section_data.gausspoint_data.items()
                        if name in ("BeamStress", "BeamStrain", "Stress", "Strain")
                    ),
                    default=1,
                )
                beam_location = 0
                cached_points.clear()
                previous = frame_choice.currentData()
                blocker = QSignalBlocker(frame_choice)
                frame_choice.clear()
                frame_choice.addItem("Local", "local")
                if f"{kind}_global" in available_fields:
                    frame_choice.addItem("Global", "global")
                frame_choice.setCurrentIndex(max(0, frame_choice.findData(previous)))
                frame_choice.setEnabled(frame_choice.count() > 1)
                del blocker
                enable_controls()
                show_point_coordinates()

            submesh_choice.currentIndexChanged.connect(select_submesh)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        if reduction.currentData() == "point_index":
            show_point_coordinates()
            if point_coordinates_label.text():
                QtWidgets.QMessageBox.warning(
                    self, "Beam section point", point_coordinates_label.text()
                )
                return
        updated = dict(
            source=source_choice.currentData(),
            position=(y.value(), z.value())
            if reduction.currentData() in ("point_index", "position")
            else None,
            reduction=None
            if reduction.currentData() in ("point_index", "position")
            else reduction.currentData(),
            samples=samples.value(),
            point_index=None,
            stored_position="nearest",
            source_fallback=submesh_choice is not None,
            reference_submesh=submesh_choice.currentData()
            if submesh_choice is not None
            else 0,
        )
        if not self.apply_beam_options(updated, frame_choice.currentData(), kind):
            return
        if submesh_choice is not None:
            dock.data.active_submesh = submesh_choice.currentData()
        blocker = QSignalBlocker(self.field_combo)
        self.field_combo.setCurrentText(kind)
        del blocker
        self.update_coordinate_choices(kind)
        self.update_components(kind)

    def apply_beam_options(self, updated, coordinates, kind):
        """Apply settings to the selected scope after validating every beam pane."""
        active = self.active_dock
        targets = [active]
        if self.apply_options_to_all:
            targets += [
                target
                for target in self.all_docks
                if target is not active
                and target.data is not None
                and target.data.beam_derived_fields()
            ]
        prepared = []
        for target in targets:
            displayed = _viewer_field_name(target.data, target.current_field)
            derived = target.data.beam_derived_fields()
            target_kind = displayed if displayed in ("Stress", "Strain") else kind
            if f"{target_kind}_local" not in derived:
                target_kind = "Stress" if "Stress_local" in derived else "Strain"
            old_options = target.data.beam_options
            try:
                selection = BeamOptions(old_options)
                selection.update(updated)
                if target is not active and selection.get("source_fallback"):
                    selection["reference_submesh"] = target.data.active_submesh
                if not target.data._is_multimesh():
                    selection["source_fallback"] = False
                target.data.beam_options = selection
                if coordinates == "global" and f"{target_kind}_global" not in derived:
                    raise ValueError("Global coordinates require BeamLocalFrame")
                field = _viewer_tensor_field(target.data, target_kind, coordinates)
                component = target.current_comp if displayed == target_kind else "XX"
                target.data.get_data(field, component or "XX")
            except (ValueError, NotImplementedError) as error:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Beam section recovery",
                    f"Cannot apply beam options to {getattr(target, 'title', 'this pane')}: {error}\n"
                    "No panes were changed. Use Apply options to → Active window to limit the scope.",
                )
                return False
            finally:
                target.data.beam_options = old_options
            prepared.append((target, field, component, displayed, selection))
        for target, field, component, displayed, selection in prepared:
            if any(
                other is not target and other.data is target.data
                for other in self.all_docks
            ):
                target.data = target.data.copy()
            target.data.beam_options = selection
            target.opts["clim"] = None
            target.beam_csys = coordinates
            if target is active or displayed in ("Stress", "Strain"):
                target.current_field = field
                target.current_comp = component or "XX"
            self.update_plot_with_clim(dock=target)
        return True

    def open_shell_section_dialog(self):
        dock = self.active_dock
        if dock is None or not dock.data.shell_derived_fields():
            return
        data = dock.data
        submesh_choice = None
        if data._is_multimesh():
            submesh_choice = QtWidgets.QComboBox()
            submesh_choice.setObjectName("shell_submesh")
            for mesh_index, mesh in enumerate(dock.data.mesh.submeshes):
                if dock.data._submesh_dataset(mesh_index).shell_derived_fields():
                    submesh_choice.addItem(
                        f"{mesh_index}: {mesh.name or mesh.elm_type}", mesh_index
                    )
            if not submesh_choice.count():
                return
            submesh_choice.setCurrentIndex(
                max(0, submesh_choice.findData(data.active_submesh))
            )
            data = dock.data._submesh_dataset(submesh_choice.currentData())
        if not data.shell_derived_fields():
            return
        options = data.shell_options
        displayed = _viewer_field_name(data, dock.current_field)
        kind = (
            displayed
            if displayed in ("Stress", "Strain")
            else (
                "Stress" if "Stress_local" in data.shell_derived_fields() else "Strain"
            )
        )
        if f"{kind}_local" not in data.shell_derived_fields():
            kind = (
                "Stress" if "Stress_local" in data.shell_derived_fields() else "Strain"
            )
        description = data.shell_section_description()
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Shell options")
        layout = QtWidgets.QFormLayout(dialog)
        show_thickness = QCheckBox("Show saved shell thickness")
        show_thickness.setObjectName("shell_display_thickness")
        show_thickness.setChecked(bool(dock.opts.get("shell_thickness", False)))
        show_thickness.setEnabled(data.mesh.elm_type in ("tri3", "quad4"))
        show_thickness.setToolTip(
            "Display the saved section thickness on tri3/quad4 shells. "
            "Each shell submesh uses its own thickness; the selected field is "
            "repeated through it without additional face-stress recovery."
        )
        layout.addRow(show_thickness)
        help_label = QLabel("Thickness position: −1 bottom, 0 midsurface, +1 top.")
        help_label.setToolTip(
            "Only homogeneous linear shell stresses are recovered from resultants. Other stresses are interpolated within each layer from saved material-point values. At interfaces, a coordinate selects the lower layer; use a point index to select a particular side."
        )
        layout.addRow(help_label)
        if submesh_choice is not None:
            layout.addRow("Shell submesh:", submesh_choice)
        source = QtWidgets.QComboBox()
        source.setObjectName("shell_field_source")
        source.addItem("Stored", "stored")
        source.addItem("Recomputed", "recomputed")
        source.setToolTip(
            "Required on the selected shell submesh. Other shell submeshes use this source when available, otherwise the other source."
        )
        has_stored, has_recomputed = data.shell_sources(kind)
        source.model().item(0).setEnabled(has_stored)
        source.model().item(1).setEnabled(has_recomputed)
        source.setCurrentIndex(
            0
            if has_stored and (options["source"] != "recomputed" or not has_recomputed)
            else 1
        )
        layout.addRow("Source:", source)
        frame = QtWidgets.QComboBox()
        frame.setObjectName("shell_coordinates")
        frame.addItem("Local", "local")
        if f"{kind}_global" in data.shell_derived_fields():
            frame.addItem("Global", "global")
        frame.setCurrentIndex(
            max(0, frame.findData(getattr(dock, "beam_csys", "global")))
        )
        reduction = QtWidgets.QComboBox()
        reduction.setObjectName("shell_section_reduction")
        for label, mode in (
            ("Normalized position", "position"),
            ("Point index", "point_index"),
            ("Maximum", "max"),
            ("Minimum", "min"),
            ("Largest magnitude", "abs_max"),
            ("Largest magnitude (signed)", "signed_abs_max"),
        ):
            reduction.addItem(label, mode)
        reduction.setCurrentIndex(
            reduction.findData(
                options["reduction"]
                or (
                    "point_index"
                    if options["point_index"] is not None
                    or options.get("reference_point_index") is not None
                    else "position"
                )
            )
        )
        layout.addRow("Value:", reduction)
        position = QDoubleSpinBox()
        position.setObjectName("shell_section_position")
        position.setDecimals(10)
        position.setRange(-1, 1)
        position.setSingleStep(0.1)
        position.setValue(
            options["position"]
            if options["position"] is not None
            else default_shell_position(data.field_metadata.get(kind))
        )
        layout.addRow("Normalized position:", position)
        index = QSpinBox()
        index.setObjectName("shell_section_point_index")
        index.setRange(0, 100000)
        index.setValue(
            options.get("reference_point_index")
            if options.get("reference_point_index") is not None
            else options["point_index"] or 0
        )
        layout.addRow("Point index (from 0):", index)
        samples = QSpinBox()
        samples.setObjectName("shell_section_samples")
        samples.setRange(2, 100000)
        samples.setValue(options["samples"])
        layout.addRow("Samples per layer:", samples)
        note = QLabel()
        note.setWordWrap(True)
        layout.addRow(note)
        frame.setEnabled(frame.count() > 1)
        layout.addRow("Coordinate system:", frame)
        manual_position = [position.value()]
        was_indexed = [False]

        def refresh(*args):
            stored = source.currentData() == "stored"
            indexed = reduction.currentData() == "point_index"
            if indexed and not was_indexed[0]:
                manual_position[0] = position.value()
            if not indexed and was_indexed[0]:
                position.setValue(manual_position[0])
            was_indexed[0] = indexed
            points = (
                np.asarray(data.field_metadata[kind]["stored_points"])
                if stored
                else sample_positions(description, samples.value())[0]
            )
            with QSignalBlocker(index):
                index.setMaximum(len(points) - 1)
            if indexed:
                position.setValue(float(points[index.value()]))
            point = reduction.currentData() in ("position", "point_index")
            position.setEnabled(point)
            position.setReadOnly(indexed)
            index.setEnabled(point and indexed)
            samples.setEnabled(not stored)
            note.setText(
                f"{len(points)} thickness points. "
                + (
                    "Stored sampling is fixed."
                    if stored
                    else "Sampling includes both faces of each layer."
                )
            )
            if stored and reduction.currentData() == "position":
                nearest = stored_shell_point_index(
                    points, position.value(), data.field_metadata[kind]
                )
                note.setText(
                    f"{len(points)} stored thickness points. Nearest stored position: {points[nearest]:.6g}."
                )

        for control in (source, reduction):
            control.currentIndexChanged.connect(refresh)
        index.valueChanged.connect(refresh)
        samples.valueChanged.connect(refresh)
        position.valueChanged.connect(
            lambda: refresh() if reduction.currentData() == "position" else None
        )
        refresh()
        if submesh_choice is not None:

            def select_submesh():
                nonlocal data, description, kind
                data = dock.data._submesh_dataset(submesh_choice.currentData())
                if f"{kind}_local" not in data.shell_derived_fields():
                    kind = (
                        "Stress"
                        if "Stress_local" in data.shell_derived_fields()
                        else "Strain"
                    )
                description = data.shell_section_description()
                stored, recomputed = data.shell_sources(kind)
                with QSignalBlocker(source):
                    source.model().item(0).setEnabled(stored)
                    source.model().item(1).setEnabled(recomputed)
                    if source.currentData() == "stored" and not stored:
                        source.setCurrentIndex(1)
                    elif source.currentData() == "recomputed" and not recomputed:
                        source.setCurrentIndex(0)
                previous = frame.currentData()
                with QSignalBlocker(frame):
                    frame.clear()
                    frame.addItem("Local", "local")
                    if f"{kind}_global" in data.shell_derived_fields():
                        frame.addItem("Global", "global")
                    frame.setCurrentIndex(max(0, frame.findData(previous)))
                frame.setEnabled(frame.count() > 1)
                refresh()

            submesh_choice.currentIndexChanged.connect(select_submesh)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addRow(buttons)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        updated = dict(
            source=source.currentData(),
            reduction=None
            if reduction.currentData() in ("position", "point_index")
            else reduction.currentData(),
            samples=samples.value(),
            position=position.value()
            if reduction.currentData() in ("position", "point_index")
            else None,
            point_index=None,
            stored_position="nearest",
            source_fallback=submesh_choice is not None,
            reference_submesh=submesh_choice.currentData()
            if submesh_choice is not None
            else 0,
            reference_point_index=index.value()
            if reduction.currentData() == "point_index"
            else None,
        )
        if not self.apply_shell_options(
            updated, frame.currentData(), kind, show_thickness.isChecked()
        ):
            return
        if submesh_choice is not None:
            dock.data.active_submesh = submesh_choice.currentData()
        with QSignalBlocker(self.field_combo):
            self.field_combo.setCurrentText(kind)
        self.update_coordinate_choices(kind)
        self.update_components(kind)

    def apply_shell_options(self, updated, coordinates, kind, show_thickness=None):
        """Validate all eligible panes before applying thickness settings."""

        active = self.active_dock
        targets = [active]
        if self.apply_options_to_all:
            targets += [
                target
                for target in self.all_docks
                if target is not active
                and target.data is not None
                and target.data.shell_derived_fields()
            ]
        prepared = []
        for target in targets:
            derived = target.data.shell_derived_fields()
            displayed = _viewer_field_name(target.data, target.current_field)
            target_kind = displayed if displayed in ("Stress", "Strain") else kind
            if f"{target_kind}_local" not in derived:
                target_kind = "Stress" if "Stress_local" in derived else "Strain"
            old = target.data.shell_options
            try:
                selection = ShellOptions(old)
                selection.update(updated)
                if target is not active and selection.get("source_fallback"):
                    selection["reference_submesh"] = target.data.active_submesh
                if not target.data._is_multimesh():
                    selection["source_fallback"] = False
                target.data.shell_options = selection
                if coordinates == "global" and f"{target_kind}_global" not in derived:
                    raise ValueError("Global coordinates require ShellLocalFrame")
                field = _viewer_tensor_field(target.data, target_kind, coordinates)
                component = target.current_comp if displayed == target_kind else "XX"
                target.data.get_data(field, component or "XX")
            except (ValueError, NotImplementedError) as error:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Shell thickness recovery",
                    f"Cannot apply shell options to {getattr(target, 'title', 'this pane')}: {error}\nNo panes were changed. Use Apply options to → Active window to limit the scope.",
                )
                return False
            finally:
                target.data.shell_options = old
            prepared.append((target, field, component, displayed, selection))
        for target, field, component, displayed, selection in prepared:
            if any(
                other is not target and other.data is target.data
                for other in self.all_docks
            ):
                target.data = target.data.copy()
            target.data.shell_options = selection
            if show_thickness is not None:
                target.opts["shell_thickness"] = show_thickness
            target.opts["clim"] = None
            target.beam_csys = coordinates
            if target is active or displayed in ("Stress", "Strain"):
                target.current_field, target.current_comp = field, component or "XX"
            self.update_plot_with_clim(dock=target)
        return True

    def get_components(self, field):
        field = _viewer_tensor_field(
            self.active_dock.data,
            field,
            getattr(self.active_dock, "beam_csys", "local"),
        )
        return self.active_dock.get_components(field)

    def update_components(self, field):
        if not field or field == SOLID_COLOR_FIELD or not self.active_dock:
            comps = [""]
            if self.active_dock:
                self.active_dock.current_field = None
        else:
            field = _viewer_tensor_field(
                self.active_dock.data,
                field,
                getattr(self.active_dock, "beam_csys", "local"),
            )
            comps = self.get_components(field)
            self.active_dock.current_field = field

        blocker = QSignalBlocker(self.comp_combo)
        self.comp_combo.clear()
        self.comp_combo.addItems(comps)
        try:
            self.comp_combo.setCurrentText(self.active_dock.current_comp)
        except:
            self.comp_combo.setCurrentIndex(0)

    @property
    def current_field(self):
        field = self.field_combo.currentText()
        if field in ("", SOLID_COLOR_FIELD):
            return None
        else:
            return _viewer_tensor_field(
                self.data, field, getattr(self.active_dock, "beam_csys", "local")
            )

    @property
    def current_component(self):
        component = self.comp_combo.currentText()
        try:
            component = int(component)
        except:
            pass
        return component

    @property
    def current_data_type(self):
        return self.avg_combo.currentText()

    def get_current_data(self):
        return self.data.get_data(
            field=self.current_field,
            component=self.current_component,
            data_type=self.current_data_type,
        )

    def open_file(self):
        # Ouvre une boîte de dialogue pour choisir un fichier
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open file",
            "",
            "Fedoo files (*.fdz *.fdh5) ;;VTK Files (*.vtk);;CSV Files (*.csv) ;; All Files (*)",
        )
        if fname:
            data = fd.read_data(fname)
            title = os.path.basename(fname)
            self.add_dataset_dock(data, title=title, filename=fname)

    def _fdh5_save_filename(self, filename):
        root, ext = os.path.splitext(filename)
        if ext == "":
            return root + ".fdh5"
        if ext.lower() != ".fdh5":
            return None
        return filename

    def _default_element_sets_for_dock(self, dock):
        element_sets = {"All": None}
        unsaved = {"All"}

        mesh = dock.data.mesh
        if _is_multimesh(mesh):
            offset = 0
            for submesh_id, submesh in enumerate(mesh.submeshes):
                label = submesh.name or submesh.elm_type
                name = f"submesh_{submesh_id}: {label}"
                if submesh.elm_type not in name:
                    name = f"{name} ({submesh.elm_type})"
                base_name = name
                counter = 1
                while name in element_sets or name in mesh.element_sets:
                    counter += 1
                    name = f"{base_name} #{counter}"
                element_sets[name] = list(range(offset, offset + submesh.n_elements))
                unsaved.add(name)
                offset += submesh.n_elements

        element_sets.update(mesh.element_sets)
        return element_sets, unsaved

    def _ensure_dock_element_sets(self, dock):
        if dock.element_sets is None:
            element_sets, unsaved = self._default_element_sets_for_dock(dock)
            dock.element_sets = element_sets
            dock.unsaved_element_set_names = unsaved

    def _element_sets_for_save(self, dock):
        self._ensure_dock_element_sets(dock)
        excluded = getattr(dock, "unsaved_element_set_names", set())
        return {
            name: np.asarray(values, dtype=int)
            for name, values in dock.element_sets.items()
            if name not in excluded and values is not None
        }

    def _ensure_dock_node_sets(self, dock):
        if dock.node_sets is None:
            dock.node_sets = {"All": None, **dock.data.mesh.node_sets}
            dock.unsaved_node_set_names = (
                {"All"} if "All" not in dock.data.mesh.node_sets else set()
            )

    def _sync_viewer_sets(self, dock):
        """Keep named sets available to expressions and the FDH5 save path."""
        self._ensure_dock_node_sets(dock)
        mesh = dock.data.mesh
        node_sets = {
            name: np.asarray(values, dtype=int)
            for name, values in dock.node_sets.items()
            if name not in dock.unsaved_node_set_names and values is not None
        }
        mesh.node_sets = node_sets
        if _is_multimesh(mesh):
            for submesh in mesh.submeshes:
                submesh.node_sets = node_sets
        self._apply_saved_element_sets_to_mesh(mesh, self._element_sets_for_save(dock))
        if isinstance(dock.data, ProblemDataSet):
            dock.data.sync_problem_sets()
        self._refresh_object_inspector()

    def _apply_saved_element_sets_to_mesh(self, mesh, element_sets):
        if not _is_multimesh(mesh):
            mesh.element_sets = {
                name: np.asarray(values, dtype=int)
                for name, values in element_sets.items()
            }
            return

        submesh_sets = [{} for _ in mesh.submeshes]
        dataset = fd.DataSet(mesh)
        for name, values in element_sets.items():
            for submesh_id, local in dataset.split_global_element_indices(
                values
            ).items():
                submesh_sets[submesh_id][name] = local

        for submesh, sets in zip(mesh.submeshes, submesh_sets):
            submesh.element_sets = sets

    def _data_for_fdh5_save(self, dock):
        self._sync_viewer_sets(dock)
        data = dock.data.deepcopy()
        self._apply_saved_element_sets_to_mesh(
            data.mesh,
            self._element_sets_for_save(dock),
        )
        return data

    def _save_data_to_fdh5(self, filename, compressed=None):
        if self.active_dock is None:
            return False

        filename = self._fdh5_save_filename(filename)
        if filename is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Save data",
                "Only FDH5 files (*.fdh5) are supported.",
            )
            return False

        if compressed is None:
            compressed = self.active_dock.opts.get("fdh5_compression", True)
        try:
            CompressionConfig.from_compressed(compressed).dataset_kwargs()
        except ImportError as exc:
            QtWidgets.QMessageBox.warning(self, "Save data", str(exc))
            return False
        data = self._data_for_fdh5_save(self.active_dock)
        # Read all source frames before replacing the destination, which may
        # itself be the file backing this dataset (e.g. when recompressing).
        with tempfile.TemporaryDirectory(
            prefix=".fedoo-save-", dir=os.path.dirname(os.path.abspath(filename))
        ) as temporary:
            destination = os.path.join(temporary, "data.fdh5")
            if isinstance(data, fd.MultiFrameDataSet):
                data.save_all(destination, file_format="fdh5", compressed=compressed)
            else:
                data.save(destination, compressed=compressed)
            os.replace(destination, filename)

        self.active_dock.opts["fdh5_compression"] = compressed
        self.active_dock.filename = filename
        self.active_dock.title = os.path.basename(filename)
        self.active_dock.setWindowTitle(
            f"{self.active_dock._dock_index}: {self.active_dock.title}"
        )
        QtWidgets.QMessageBox.information(self, "Save data", f"Saved:\n{filename}")
        return True

    def save_current_data(self):
        if self.active_dock is None:
            return
        filename = self.active_dock.filename
        if not filename or os.path.splitext(filename)[1].lower() != ".fdh5":
            self.save_current_data_as()
            return
        self._save_data_to_fdh5(filename)

    def save_current_data_as(self):
        if self.active_dock is None:
            return
        start = self.active_dock.filename or self.active_dock.title
        if not start or os.path.splitext(start)[1].lower() != ".fdh5":
            start = os.path.splitext(start)[0] + ".fdh5"

        dlg = QtWidgets.QFileDialog(self, "Save data as")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setNameFilter("Fedoo HDF5 files (*.fdh5)")
        dlg.setDefaultSuffix("fdh5")
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        dlg.selectFile(start)

        options = QtWidgets.QWidget(dlg)
        options_layout = QtWidgets.QHBoxLayout(options)
        options_layout.setContentsMargins(0, 0, 0, 0)
        compression = QtWidgets.QComboBox(options)
        compression.setObjectName("fdh5_compression")
        compression.addItem("None", None)
        compression.addItem("gzip", "gzip")
        compression.addItem("Zstandard", "zstd")
        available = zstd_available()
        compression.model().item(2).setEnabled(available)
        compression.setToolTip(
            "Lossless FDH5 compression. Zstandard requires hdf5plugin on readers "
            "and writers; install with pip install fedoo[zstd]."
        )
        level = QtWidgets.QSpinBox(options)
        level.setObjectName("fdh5_compression_level")
        saved = self.active_dock.opts.get("fdh5_compression", True)
        config = CompressionConfig.from_compressed(saved)

        def update_level():
            codec = compression.currentData()
            level.setEnabled(codec is not None)
            level.setRange(1, 22 if codec == "zstd" else 9)
            level.setValue(3 if codec == "zstd" else 4)
            level.setToolTip(
                "1: fastest save; higher levels favor compression. "
                + ("Zstandard default: 3." if codec == "zstd" else "gzip default: 4.")
            )

        compression.currentIndexChanged.connect(update_level)
        selected = compression.findData(config.compression)
        compression.setCurrentIndex(max(0, selected))
        update_level()
        if config.compression_opts is not None:
            level.setValue(config.compression_opts)
        options_layout.addWidget(QtWidgets.QLabel("Compression:", options))
        options_layout.addWidget(compression)
        options_layout.addWidget(QtWidgets.QLabel("Level:", options))
        options_layout.addWidget(level)
        options_layout.addStretch(1)
        grid = dlg.layout()
        grid.addWidget(options, grid.rowCount(), 0, 1, grid.columnCount())

        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self._save_data_to_fdh5(
                dlg.selectedFiles()[0],
                compressed=(
                    False
                    if compression.currentData() is None
                    else CompressionConfig(compression.currentData(), level.value())
                ),
            )

    def open_clim_dialog(self):
        if self._clim_dialog is None:
            self._clim_dialog = ClimOptionsDialog(parent=self)
            self._clim_dialog.applyRequested.connect(self.apply_clim_from_dialog)
            self._clim_dialog.accepted.connect(self.apply_clim_and_close)
            self._clim_dialog.rejected.connect(self._clim_dialog.close)

        self._clim_dialog.setWindowModality(Qt.NonModal)
        self._clim_dialog.show()
        self._clim_dialog.raise_()

    def apply_clim_from_dialog(self):
        """Called by the Apply button."""
        clim_opts = {}
        (
            clim_opts["clim_mode"],
            clim_opts["clim"],
            clim_opts["n_colors"],
            clim_opts["n_labels"],
            clim_opts["cmap_name"],
            clim_opts["cmap"],
        ) = self._clim_dialog.get_values()
        if self.apply_options_to_all:
            list_docks = self.all_docks
        else:
            list_docks = [self.active_dock]
        for dock in list_docks:
            dock.opts.update(clim_opts)
            dock.update_plot(lock_view=True)

    def apply_clim_and_close(self):
        """OK: applique puis ferme."""
        self.apply_clim_from_dialog()
        if self._clim_dialog:
            self._clim_dialog.close()

    def open_plot_dialog(self):
        # Create once and keep a reference so it isn't garbage-collected
        if not self._plot_dialog:
            self._plot_dialog = PlotOptionsDialog(self)
            self._plot_dialog.optionsChanged.connect(self._apply_plot_options)
        self._plot_dialog.update_values()
        # Show non-modally (no exec())
        self._plot_dialog.show()
        self._plot_dialog.raise_()
        self._plot_dialog.activateWindow()

    def _apply_plot_options(self):
        if self.apply_options_to_all:
            for dock in self.all_docks:
                self.apply_plot_options_to_dock(dock)
                dock.update_plot()
        else:
            self.apply_plot_options_to_dock(self.active_dock)
            self.update_plot()

    def apply_plot_options_to_dock(self, dock):
        opts = dock.opts
        if self._plot_dialog.scale_cb.isChecked():
            opts["scale"] = float(self._plot_dialog.scale_spin.value())
        else:
            opts["scale"] = 0.0
        # save scale value in case of active dock change
        opts["scale_save"] = float(self._plot_dialog.scale_spin.value())

        opts["show_edges"] = self._plot_dialog.edges_cb.isChecked()
        opts["show_scalar_bar"] = self._plot_dialog.scalarbar_cb.isChecked()
        opts["show_time"] = self._plot_dialog.show_time_cb.isChecked()
        if self._plot_dialog.axes_cb.isChecked():
            dock.plotter.show_axes()
        else:
            dock.plotter.hide_axes()

        opts["node_labels"] = self._plot_dialog.node_labels_cb.isChecked()
        opts["element_labels"] = self._plot_dialog.element_labels_cb.isChecked()

        if not (self._plot_dialog.show_title_cb.isChecked()):
            opts["title_plot"] = ""
        elif self._plot_dialog.auto_title_rb.isChecked():
            opts["title_plot"] = None
        else:
            opts["title_plot"] = self._plot_dialog.title_edit.text()

    def open_renderer_dialog(self):
        # # Create once and keep a reference so it isn't garbage-collected
        if not self._renderer_dialog:
            self._renderer_dialog = RedererOptionsDialog(self)
            self._renderer_dialog.optionsChanged.connect(self._apply_renderer_options)
        # Show non-modally (no exec())
        self._renderer_dialog.show()
        self._renderer_dialog.raise_()
        self._renderer_dialog.activateWindow()

    def _apply_renderer_options(self):
        if self.apply_options_to_all:
            list_docks = self.all_docks
        else:
            list_docks = [self.active_dock]
        for dock in list_docks:
            dock.opts["opacity"] = float(self._renderer_dialog.opacity_spin.value())
            dock.opts["color"] = self._renderer_dialog.color
            dock.opts["color_missing_fields"] = (
                self._renderer_dialog.missing_fields_cb.isChecked()
            )
            dock.opts["pbr"] = self._renderer_dialog.pbr_cb.isChecked()
            dock.opts["metallic"] = float(self._renderer_dialog.metallic_spin.value())
            dock.opts["roughness"] = float(self._renderer_dialog.roughness_spin.value())
            dock.opts["diffuse"] = float(self._renderer_dialog.diffuse_spin.value())
            dock.update_plot()

    def update_plot_with_clim(
        self, val=None, iteration=None, lock_view=True, dock=None
    ):
        if dock is None:
            if self._sync_field:
                for d in self.all_docks:
                    if self.current_field is None:
                        d.current_field = None
                    elif self.field_combo.currentText() in _viewer_field_names(d.data):
                        d.beam_csys = getattr(self.active_dock, "beam_csys", "local")
                        d.current_field = _viewer_tensor_field(
                            d.data, self.field_combo.currentText(), d.beam_csys
                        )
                        # Only set component if field exists in this dock
                        comps = d.get_components(d.current_field)
                        if self.current_component in comps:
                            d.current_comp = self.current_component
                    self.update_plot_with_clim(val, iteration, lock_view, dock=d)
                return
            else:
                dock = self.active_dock
                if dock:
                    dock.current_comp = self.current_component
                else:
                    return

        dock.current_data_type = self.current_data_type

        if dock.current_field is not None and dock.opts["clim_mode"] == "all":
            if hasattr(dock.data, "get_all_frame_lim"):
                dock.opts["clim"] = dock.data.get_all_frame_lim(
                    field=dock.current_field,
                    component=dock.current_comp,
                    data_type=dock.current_data_type,
                )[2]
            else:
                dock.opts["clim"] = None
        dock.update_plot(val=val, iteration=iteration, lock_view=lock_view)

    def save_image_dialog(self):
        """Open a dialogbox to save an image."""
        dlg = QtWidgets.QFileDialog(self, "Save Image")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        dlg.setNameFilters(
            [
                "PNG (*.png)",
                "JPEG (*.jpg *.jpeg)",
                "TIFF (*.tif *.tiff)",
                "BMP (*.bmp)",
                "PDF (*.pdf)",
                "EPS (*.eps)",
                "PS (*.ps)",
                "TEX (*.tex)",
            ]
        )
        # Use non native dialog to allow modification
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)

        # --- Custom additional widgets ---
        options_widget = QtWidgets.QWidget(dlg)
        options_layout = QtWidgets.QHBoxLayout(options_widget)
        options_layout.setContentsMargins(0, 0, 0, 0)

        transparent_cb = QtWidgets.QCheckBox("Transparent background", options_widget)

        options_layout.addWidget(transparent_cb)
        options_layout.addStretch(1)

        grid = dlg.layout()  # type: QGridLayout
        last_row = grid.rowCount()
        grid.addWidget(options_widget, last_row, 0, 1, grid.columnCount())

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return None

        fname = dlg.selectedFiles()[0]
        selected_filter = dlg.selectedNameFilter()
        transparent = transparent_cb.isChecked()

        # Catch extension
        ext = os.path.splitext(fname)[1].lower()
        if ext == "":
            ext = {
                "PNG": ".png",
                "JPE": ".jpg",
                "TIF": ".tif",
                "BMP": ".bmp",
                "PDF": ".pdf",
                "EPS": ".eps",
                "PS ": ".ps",
                "TEX": ".tex",
            }.get(selected_filter[:3])
            # If no extension given, take the one of the filter
            fname += ext

        if ext in [".pdf", ".eps", ".ps", ".tex"]:
            self.plotter.save_graphic(fname)
        else:
            self.plotter.screenshot(fname, transparent_background=transparent)

        QtWidgets.QMessageBox.information(self, "Save Image", f"Saved:\n{fname}")

    def save_movie_dialog(self):
        """Open a dialogbox to write a movie."""
        dlg = QtWidgets.QFileDialog(self, "Write movie")
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        # Filter
        dlg.setNameFilters(
            [
                "MP4 (*.mp4)",
                "AVI (*.avi)",
                "MKV (*.mkv)",
                "MOV (*.mov)",
                "GIF (*.gif)",
                "WebM (*.webm)",
                "All files (*)",
            ]
        )
        # Use non native dialog to allow customization
        dlg.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)

        # ---------------- Option widget ----------------
        options_widget = QtWidgets.QWidget(dlg)
        options_layout = QtWidgets.QGridLayout(options_widget)
        options_layout.setContentsMargins(6, 6, 6, 6)
        options_layout.setHorizontalSpacing(12)
        options_layout.setVerticalSpacing(6)

        # FPS
        fps_spin = QtWidgets.QDoubleSpinBox(options_widget)
        fps_spin.setRange(0.1, 240.0)
        fps_spin.setDecimals(1)
        # fps_spin.setSingleStep(1.0)
        fps_spin.setValue(self.anim_fps_spin.value())

        # Quality
        quality_spin = QtWidgets.QDoubleSpinBox(options_widget)
        quality_spin.setRange(0, 10)
        quality_spin.setDecimals(0)
        # quality_spin.setSingleStep(1)
        quality_spin.setValue(6)

        # Window size (width/height) + “Use current”
        use_current_cb = QtWidgets.QCheckBox("Use current window size", options_widget)
        width_spin = QtWidgets.QSpinBox(options_widget)
        height_spin = QtWidgets.QSpinBox(options_widget)
        width_spin.setRange(64, 8192)
        height_spin.setRange(64, 8192)

        # Init window size to current renderer
        try:
            width_spin.setValue(self.plotter.window_size[0])
            height_spin.setValue(self.plotter.window_size[1])
            use_current_cb.setChecked(True)
        except Exception:
            width_spin.setValue(1280)
            height_spin.setValue(720)
            use_current_cb.setChecked(False)

        # Transparent background
        # transparent_cb = QtWidgets.QCheckBox("Transparent background", options_widget)
        # transparent_cb.setToolTip("Utile si ton encodeur/format gère l’alpha (sinon ignoré).")

        # Rotations (par frame) : azimuth / elevation (float)
        rot_azimuth_spin = QtWidgets.QDoubleSpinBox(options_widget)
        rot_elevation_spin = QtWidgets.QDoubleSpinBox(options_widget)
        for s in (rot_azimuth_spin, rot_elevation_spin):
            s.setRange(-360.0, 360.0)
            s.setDecimals(3)
            s.setSingleStep(1.0)
            s.setValue(0.0)

        # Activate/Desactivate width/height if "use current"
        def on_use_current_toggled(checked: bool):
            width_spin.setEnabled(not checked)
            height_spin.setEnabled(not checked)

        use_current_cb.toggled.connect(on_use_current_toggled)
        on_use_current_toggled(use_current_cb.isChecked())

        # Grid to place options
        row = 0
        options_layout.addWidget(QtWidgets.QLabel("Framerate (fps):"), row, 0)
        options_layout.addWidget(fps_spin, row, 1)
        row += 1

        options_layout.addWidget(QtWidgets.QLabel("Quality (0-10):"), row, 0)
        options_layout.addWidget(quality_spin, row, 1)
        row += 1

        options_layout.addWidget(QtWidgets.QLabel("Window Size:"), row, 0)
        options_layout.addWidget(use_current_cb, row, 1)
        row += 1
        options_layout.addWidget(QtWidgets.QLabel("    Width:"), row, 0)
        options_layout.addWidget(width_spin, row, 1)
        row += 1
        options_layout.addWidget(QtWidgets.QLabel("    Height:"), row, 0)
        options_layout.addWidget(height_spin, row, 1)
        row += 1

        # options_layout.addWidget(transparent_cb, row, 0, 1, 2); row += 1
        options_layout.addWidget(QtWidgets.QLabel("Camera rotation (°/frame):"), row, 0)
        row += 1
        options_layout.addWidget(QtWidgets.QLabel("    Azimuth:"), row, 0)
        options_layout.addWidget(rot_azimuth_spin, row, 1)
        row += 1

        options_layout.addWidget(QtWidgets.QLabel("    Elevation:"), row, 0)
        options_layout.addWidget(rot_elevation_spin, row, 1)
        row += 1

        # Instert the options_widget on the botom of QFileDialog
        grid = dlg.layout()  # QGridLayout
        grid.addWidget(options_widget, grid.rowCount(), 0, 1, grid.columnCount())

        # Show
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return  # Cancel

        # Catch file and filter
        fname = dlg.selectedFiles()[0]
        selected_filter = dlg.selectedNameFilter()
        ext = os.path.splitext(fname)[1].lower()
        if ext == "":
            ext = {
                "MP4": ".mp4",
                "AVI": ".avi",
                "MKV": ".mkv",
                "MOV": ".mov",
                "GIF": ".gif",
                "Web": ".webm",
            }.get(selected_filter[:3])
            # If no extension given, take the one of the filter
            fname += ext

        # Apply the current window size if required
        if use_current_cb.isChecked():
            window_size = self.plotter.window_size
        else:
            window_size = [int(width_spin.value()), int(height_spin.value())]

        # transparent = bool(transparent_cb.isChecked()),
        pl = pv.Plotter(window_size=tuple(window_size), off_screen=True)
        self.update_plot(iteration=0, plotter=pl, lock_view=False)

        pl.camera_position = self.plotter.camera_position
        pl.set_background(self.plotter.background_color)
        if not (self.plotter.renderer.axes_enabled):
            pl.hide_axes()
            # don't know how to catch the current axes position.
            # not very important imo

        if ext == ".gif":
            pl.open_gif(fname, fps=float(fps_spin.value()))
        else:
            pl.open_movie(
                fname,
                framerate=float(fps_spin.value()),
                quality=int(quality_spin.value()),
            )
        pl.write_frame()

        for iteration in range(1, _data_n_iter(self.data)):
            rot_azimuth = float(rot_azimuth_spin.value())
            rot_elevation = float(rot_elevation_spin.value())
            if rot_azimuth != 0:
                pl.camera.Azimuth(rot_azimuth)
            if rot_elevation != 0:
                pl.camera.Elevation(rot_elevation)
            self.update_plot(iteration=iteration, plotter=pl)
            pl.write_frame()

        pl.close()
        QtWidgets.QMessageBox.information(self, "Create Movie", f"Saved:\n{fname}")

    def view_top(self):
        self.plotter.view_vector((0, 0, -1), viewup=(0, 1, 0))
        self.plotter.render()

    def view_bottom(self):
        self.plotter.view_vector((0, 0, 1), viewup=(0, 1, 0))
        self.plotter.render()

    def view_front(self):
        self.plotter.view_vector((0, -1, 0), viewup=(0, 0, 1))
        self.plotter.render()

    def view_back(self):
        self.plotter.view_vector((0, 1, 0), viewup=(0, 0, 1))
        self.plotter.render()

    def view_left(self):
        self.plotter.view_vector((-1, 0, 0), viewup=(0, 0, 1))
        self.plotter.render()

    def view_right(self):
        self.plotter.view_vector((1, 0, 0), viewup=(0, 0, 1))
        self.plotter.render()

    def view_isometric(self):
        self.plotter.view_isometric()
        self.plotter.render()

    def reset_camera(self):
        self.plotter.reset_camera()
        self.plotter.render()

    # Clip dialog : Open and close
    # ----------------------------
    def open_clip_dialog(self):
        if not self._clip_is_supported():
            self.statusBar().showMessage("Clip is only available for 3D meshes.", 3000)
            return

        # Create clip dialog only if not already exist
        if self._clip_dialog is None:
            origin = (
                _as_3d_tuple(self.data.mesh.bounding_box.center)
                if self.data.mesh is not None
                else (0.0, 0.0, 0.0)
            )
            normal = (1.0, 0.0, 0.0)
            self.opts["clip_origin"] = origin
            self.opts["clip_normal"] = normal
            self._clip_dialog = ClipDialog(
                self,
                default_origin=origin,
                default_normal=normal,
                invert=self.opts["clip_invert"],
            )
            # connection to sync dialog with clip & widget
            self._clip_dialog.clipPlaneChanged.connect(self._on_dialog_clip_changed)
            # if dialog closed, remove plane widget
            self._clip_dialog.finished.connect(self._on_clip_dialog_closed)
            self._clip_dialog.destroyed.connect(self._on_clip_dialog_closed)

        # Show dialog
        self._clip_dialog.show()
        self._clip_dialog.raise_()
        self._clip_dialog.activateWindow()

        # Activate plane widget et sync its values
        self.enable_plane_widget()

    def _on_clip_dialog_closed(self, *args, **kwargs):
        """Quand la fenêtre est fermée : on retire le widget plan."""
        self.disable_plane_widget()

    def _on_dialog_clip_changed(
        self, enabled, origin, normal, invert, *, defer_render=False
    ):
        """Dialog → Apply clip and update clip plane widget."""
        self.opts["clip_invert"] = bool(invert)
        self.opts["clip_normal"] = normal
        self.opts["clip_origin"] = origin
        if enabled:
            self.opts["clip_args"] = {
                "normal": normal,
                "origin": origin,
                "invert": self.opts["clip_invert"],
                "cell_ids": True,  # add 'cell_ids' to cell_data
                # "crinkle": True,  # keep entire elements and add 'cell_ids'
            }
        else:
            self.opts["clip_args"] = None

        if defer_render:
            # Leave VTK's interaction callback before rebuilding its scene.
            # Multiple pending movements only need the newest plane position.
            self._clip_redraw_timer.start(0)
        else:
            self._clip_redraw_timer.stop()
            self.update_plot()

    # ----------------------------
    # Plane widget gestion
    # ----------------------------
    def enable_plane_widget(self):
        # compute bounds
        try:
            if self.current_data_type == "GaussPoint":
                bounds = self.data.meshplot_gp.bounds
            else:
                bounds = self.data.meshplot.bounds
        except AttributeError:
            bounds = tuple(np.array(self.data.mesh.as_3d().bounding_box).T.ravel())

        if self.opts["clip_origin"] is None:
            self.opts["clip_origin"] = _as_3d_tuple(self.data.mesh.bounding_box.center)
            self.opts["clip_normal"] = (1.0, 0.0, 0.0)

        origin = self.opts["clip_origin"]
        normal = self.opts["clip_normal"]
        widget = self._plane_widget
        if (
            widget is not None
            and self._plane_widget_plotter is self.plotter
            and widget.GetEnabled()
        ):
            widget.PlaceWidget(bounds)
            widget.SetOrigin(origin)
            widget.SetNormal(normal)
            widget.UpdatePlacement()
            return
        dock = self.active_dock
        # bounds = self._bounds

        def _cb(normal_cb, origin_cb):
            """Widget → Dialog + Clip (live)."""
            if (
                self.active_dock is dock
                and self._plane_widget_enabled
                and self._clip_dialog is not None
            ):
                self._clip_dialog.set_values_from_widget(
                    origin_cb, normal_cb, invert=self.opts["clip_invert"]
                )
                self._on_dialog_clip_changed(
                    self._clip_dialog.enableClipChk.isChecked(),
                    origin_cb,
                    normal_cb,
                    self.opts["clip_invert"],
                    defer_render=True,
                )

        def clamp_to_bounds(pt, b):
            # to avoid origin outside the bounds
            x = min(max(pt[0], b[0]), b[1])
            y = min(max(pt[1], b[2]), b[3])
            z = min(max(pt[2], b[4]), b[5])
            return (x, y, z)

        self.plotter.clear_plane_widgets()
        self._plane_widget = self.plotter.add_plane_widget(
            callback=_cb,
            bounds=bounds,
            origin=clamp_to_bounds(origin, bounds),
            normal=normal,
            implicit=True,
            color="red",
            # interaction_event = 'end',
            outline_translation=False,
            normal_rotation=True,
            origin_translation=True,
            test_callback=False,
        )
        self._plane_widget_enabled = True
        self._plane_widget_plotter = self.plotter

        # Init dialog si ouverte
        # if self._clip_dialog is not None:
        #     self._clip_dialog.set_values_from_widget(origin, normal, invert=self.opts['clip_invert'])

    def disable_plane_widget(self):
        self._clip_redraw_timer.stop()
        if not self._plane_widget_enabled:
            return
        try:
            self.plotter.clear_plane_widgets()
        except Exception:
            self.plotter.clear_widgets()
        self._plane_widget = None
        self._plane_widget_plotter = None
        self._plane_widget_enabled = False

    #
    # Dialog to show the results over a line
    #
    def open_plot_over_line_dialog(self):
        # Create dialog only if not already exist
        if self._plot_over_line_dialog is None:
            self._plot_over_line_dialog = PlotOverLineDialog(self)
            self._plot_over_line_dialog.plotRequested.connect(self.plot_over_line)
            self._plot_over_line_dialog.requestPick.connect(
                self._start_pick
            )  # 0->P1, 1->P2

            # connection to sync dialog with widget
            self._plot_over_line_dialog.lineChanged.connect(self._on_pol_dialog_changed)
            # if dialog closed, remove widget
            self._plot_over_line_dialog.finished.connect(self._on_pol_dialog_closed)
            self._plot_over_line_dialog.destroyed.connect(self._on_pol_dialog_closed)

        p1, p2 = self.active_dock.line_points
        self._line_widget_enabled = True
        # Line widget
        self._syncing_line_widget = True
        try:
            self._line_widget = self.plotter.add_line_widget(
                callback=self.on_line_changed, use_vertices=True
            )
        finally:
            self._syncing_line_widget = False
        self._on_pol_dialog_changed(p1, p2)
        self._plot_over_line_dialog.update_line(p1, p2)
        # Show dialog
        self._plot_over_line_dialog.show()
        self._plot_over_line_dialog.raise_()

    def _on_pol_dialog_closed(self, *args, **kwargs):
        # remove line widget
        self._line_widget_enabled = False
        self.plotter.clear_line_widgets()

    def _on_pol_dialog_changed(self, p1, p2):
        self.active_dock.line_points = self._set_line_widget_points(p1, p2)

    def on_line_changed(self, p1, p2):
        """Called interactively while moving the widget"""
        if self._syncing_line_widget:
            return
        p1, p2 = self._set_line_widget_points(p1, p2)
        self.active_dock.line_points = (p1, p2)
        self._plot_over_line_dialog.update_line(p1, p2)

    def _line_points_for_active_mesh(self, p1, p2):
        p1, p2 = tuple(p1), tuple(p2)
        if self.data.mesh.ndim == 2:
            p1, p2 = (p1[0], p1[1], 0.0), (p2[0], p2[1], 0.0)
        return p1, p2

    def _set_line_widget_points(self, p1, p2):
        p1, p2 = self._line_points_for_active_mesh(p1, p2)
        self._syncing_line_widget = True
        try:
            self._line_widget.SetPoint1(p1)
            self._line_widget.SetPoint2(p2)
        finally:
            self._syncing_line_widget = False
        return p1, p2

    def _rebuild_line_widget(self):
        if self._plot_over_line_dialog:
            p1, p2 = self.active_dock.line_points
            self.plotter.clear_line_widgets()
            self._syncing_line_widget = True
            try:
                self._line_widget = self.plotter.add_line_widget(
                    callback=self.on_line_changed, use_vertices=True
                )
            finally:
                self._syncing_line_widget = False
            # value of p1 and p2 are changed by the callback function
            # force values to the initial ones
            self._on_pol_dialog_changed(p1, p2)
            self.on_line_changed(p1, p2)  # update dialog values

    # ---------- Picking ----------
    def _start_pick(self, which: int):
        """Start one-shot picking for endpoint 'which' (0=P1, 1=P2)."""
        self._picking_target = which

        def _picked(point):
            # Retrieve current other endpoint
            p1 = tuple(
                self._plot_over_line_dialog.p1_edits[i].value() for i in range(3)
            )
            p2 = tuple(
                self._plot_over_line_dialog.p2_edits[i].value() for i in range(3)
            )
            if self._picking_target == 0:
                p1 = tuple(point)
            else:
                p2 = tuple(point)
            p1, p2 = self._line_points_for_active_mesh(p1, p2)

            # Sync UI and scene
            self._on_pol_dialog_changed(p1, p2)  # update widget
            self.on_line_changed(p1, p2)  # update dialog values

            # Stop picking
            self.plotter.disable_picking()
            self._picking_target = -1

        # Remove active click actions
        self.select_node_action.setChecked(False)
        self.select_elm_action.setChecked(False)
        # Enable one-shot point picking (left click)
        self.plotter.enable_point_picking(
            callback=_picked,
            left_clicking=True,
            show_message=False,  # shows hint in the render window
        )

    def _on_select_node_changed(self, checked: bool):
        if checked:
            self.select_elm_action.setChecked(False)
            self.statusBar().setVisible(True)
            self._pick_node()
        else:
            self.plotter.remove_actor("_picked_point")
            self.plotter.disable_picking()
            self.statusBar().setVisible(False)

    def _on_select_elm_changed(self, checked: bool):
        if checked:
            self.select_node_action.setChecked(False)
            self.statusBar().setVisible(True)
            self._pick_element()
        else:
            self._stop_element_picking()
            self.plotter.remove_actor("_picked_cell")
            # self.plotter.disable_picking()
            self.statusBar().setVisible(False)

    def _stop_element_picking(self):
        if getattr(self, "_lbp_tag", None) is not None:
            self.plotter.interactor.RemoveObserver(self._lbp_tag)
            self._lbp_tag = None

    def _pick_node(self, on_pick=None):
        """Start point picking with an optional consumer of physical node IDs."""
        self._stop_element_picking()

        def _picked(point, picker):
            pid, value, association = _picked_node_value(
                self.active_dock, point, picker
            )
            if pid is not None:
                msg = f"Node id={pid} | "
            else:
                msg = ""
            msg += f"x={point[0]:.3g}, y={point[1]:.3g}, z={point[2]:.3g}"
            name = self.current_field
            if name and value is not None:
                msg += f" | {name}={value}"
                if association:
                    msg += f" ({association})"

            self.statusBar().showMessage(msg)
            if pid is not None and on_pick is not None:
                on_pick(int(pid))

        # Enable one-shot point picking (left click)
        self.plotter.disable_picking()
        self.plotter.enable_point_picking(
            callback=_picked,
            left_clicking=True,
            show_message=False,
            picker="point",
            use_picker=True,
        )

    def _pick_element(self, on_pick=None):
        self._stop_element_picking()
        self.plotter.disable_picking()
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)

        def _on_click(obj, event):
            x, y = self.plotter.interactor.GetEventPosition()
            picker.Pick(x, y, 0, self.plotter.renderer)

            cell_id = picker.GetCellId()
            if cell_id < 0:
                return
            display_cell_id = cell_id
            actor_name, mesh = self.active_dock.picked_actor_mesh(picker.GetActor())
            if mesh is None:
                mesh = self.active_dock.pv_mesh
            if "vtkOriginalCellIds" in mesh.cell_data:
                save_original_cell_ids = mesh.cell_data["vtkOriginalCellIds"]
                cell = mesh.extract_cells(cell_id)
                mesh.cell_data["vtkOriginalCellIds"] = save_original_cell_ids
            else:
                cell = mesh.extract_cells(cell_id)
            self.plotter.remove_actor("_picked_cell")
            self.plotter.add_mesh(
                cell, style="wireframe", color="pink", name="_picked_cell"
            )

            if self.opts["clip_args"]:
                # retrieve true cell ids
                cell_id = mesh.cell_data["cell_ids"][cell_id]
            elif "vtkOriginalCellIds" in mesh.cell_data:
                # don't work !!!!
                cell_id = mesh.cell_data["vtkOriginalCellIds"][cell_id]
            if "_fedoo_global_cell_ids" in mesh.cell_data:
                cell_id = mesh.cell_data["_fedoo_global_cell_ids"][display_cell_id]
            cell_id = int(cell_id)

            msg = ""
            if mesh is not None and cell_id is not None:
                msg = f"Element id={cell_id} | "
            # msg += f"x={point[0]:.3g}, y={point[1]:.3g}, z={point[2]:.3g}"
            name = self.current_field
            if name:
                try:
                    elem_value = _element_data_value(
                        self.data,
                        name,
                        self.current_component,
                        "Element",
                        cell_id,
                    )
                    msg += f"{name}_{self.current_component}={elem_value}"
                except Exception:
                    pass

                try:
                    data_gp = _gausspoint_values_for_element(
                        self.data, name, self.current_component, cell_id
                    )
                    msg += f" | gp vals: {data_gp}"
                except Exception:
                    pass

            self.statusBar().showMessage(msg)

            if on_pick is not None:
                on_pick(cell_id)

        self._lbp_tag = self.plotter.interactor.AddObserver(
            "LeftButtonPressEvent", _on_click
        )

    def plot_over_line(self, p1, p2, resolution):
        """
        Compute plot_over_line and push data to the Matplotlib dialog.
        Set live=True if you call this frequently (dragging) and want to avoid titles/reflows.
        """
        # Compute line result
        pv_mesh = self.active_dock.pv_mesh
        if pv_mesh is None:
            QtWidgets.QMessageBox.information(
                self, "Error", "No compatible data found."
            )
            return

        res = pv_mesh.sample_over_line(p1, p2, resolution=resolution)
        x = res["Distance"]
        # PyVista gives a new name (Data-1, Data-2, ...) to scalars passed
        # on successive plots of the same mesh. Use the array displayed by
        # the actor instead of the first, now stale, "Data" array.
        actors = self.active_dock.plotter.actors
        actor = actors.get("data1")
        if actor is None:
            actor = next(
                (actors[name] for name in actors if name.startswith("data_")), None
            )
        scalar_name = actor.mapper.array_name if actor is not None else None
        if scalar_name is None or scalar_name not in res.array_names:
            QtWidgets.QMessageBox.information(
                self, "Error", "No plotted scalar data found along the line."
            )
            return
        y = res[scalar_name]
        try:
            ylabel = (
                self.active_dock.current_field + "_" + self.active_dock.current_comp
            )
        except:
            ylabel = "Data"

        # Update Matplotlib dialog
        if self._pol_results_dialog is None:
            self._pol_results_dialog = MplLinePlotDialog(
                self, title="Plot Over Line - Result"
            )
        self._pol_results_dialog.update_curve(x, y, p1=p1, p2=p2, ylabel=ylabel)

        # If you prefer to bring the dialog to front only on manual plot:
        # if not live and not self.result_dialog.isVisible():
        self._pol_results_dialog.show()
        self._pol_results_dialog.raise_()
        self._pol_results_dialog.activateWindow()

    def open_element_sets_dialog(self):
        if self.element_sets_dialog is None:
            self.element_sets_dialog = ElementVisibilityDialog(parent=self)
        self.element_sets_dialog._refresh_sets_list()
        self.element_sets_dialog.show()
        self.element_sets_dialog.raise_()

    def open_node_sets_dialog(self):
        if self.node_sets_dialog is None:
            self.node_sets_dialog = NodeSetsDialog(parent=self)
        self.node_sets_dialog._refresh_sets_list()
        self.node_sets_dialog.show()
        self.node_sets_dialog.raise_()

    def open_scalar_expression_dialog(self):
        if self.scalar_expression_dialog is None:
            self.scalar_expression_dialog = ScalarExpressionDialog(self)
        self.scalar_expression_dialog.refresh()
        self.scalar_expression_dialog.show()
        self.scalar_expression_dialog.raise_()

    def open_existing_problem_dialog(self):
        if self._problem_dialog is None:
            self._problem_dialog = OpenProblemDialog(self)
        self._problem_dialog.refresh()
        self._problem_dialog.show()
        self._problem_dialog.raise_()
        self._problem_dialog.activateWindow()

    def open_workspace_problem(self, problem, title):
        """Open a live session object, or refresh its existing viewer pane."""
        for dock in self.all_docks:
            if isinstance(dock.data, ProblemDataSet) and dock.data.problem is problem:
                dock.data.refresh()
                self._set_active(dock)
                dock.update_plot()
                return dock
        snapshot = ProblemDataSet(problem)
        return self.add_dataset_dock(snapshot, title=problem.name or title)

    def open_boundary_conditions_dialog(self):
        if self.active_dock is None or not isinstance(self.data, ProblemDataSet):
            return
        if self.boundary_conditions_dialog is None:
            self.boundary_conditions_dialog = BoundaryConditionsDialog(self)
        self.boundary_conditions_dialog.refresh()
        self.boundary_conditions_dialog.show()
        self.boundary_conditions_dialog.raise_()

    def open_history_plot_dialog(self):
        if self._history_plot_dialog is None:
            self._history_plot_dialog = HistoryPlotDialog(self)
        self._history_plot_dialog.show()
        self._history_plot_dialog.raise_()
        self._history_plot_dialog.activateWindow()

    def toggle_fullscreen(self):
        # Hide chrome (optional): menu + toolbars when in fullscreen
        if not (self.isFullScreen()):
            # Remember current visible toolbars/menubar to restore later
            self._hidden_ui_when_fullscreen = []
            if self.object_inspector is not None and self.object_inspector.isVisible():
                self._hidden_ui_when_fullscreen.append(
                    ("inspector", self.object_inspector)
                )
                self.object_inspector.hide()
            if self.menuBar().isVisible():
                self.menuBar().setVisible(False)
                self._hidden_ui_when_fullscreen.append(("menubar", self.menuBar()))
            for tb in self.findChildren(QtWidgets.QToolBar):
                if tb.isVisible():
                    tb.setVisible(False)
                    self._hidden_ui_when_fullscreen.append(("toolbar", tb))
            self.showFullScreen()
        else:
            self.showNormal()
            # Restore previously visible chrome
            for kind, w in self._hidden_ui_when_fullscreen:
                w.setVisible(True)
            self._hidden_ui_when_fullscreen.clear()

    def _exit_fullscreen(self):
        if self.isFullScreen():
            self.toggle_fullscreen()

    @property
    def plotter(self):
        return self.active_dock.plotter

    def closeEvent(self, event):
        self.stop_animation()  # remove timer if present
        if self.object_inspector is not None:
            self.object_inspector.stop()
        self._clip_redraw_timer.stop()
        self._plane_widget_enabled = False
        if self._problem_dialog is not None:
            self._problem_dialog.close()
        for dialog in (
            self.node_sets_dialog,
            self.element_sets_dialog,
            self.scalar_expression_dialog,
        ):
            if dialog is not None:
                dialog.close()
        for dock in self.all_docks:
            dock.plotter.close()  # free vtk context
        super().closeEvent(event)


class PlotOptionsDialog(QtWidgets.QDialog):
    # Emitted when user presses Apply; provides a dict with current options
    optionsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot")

        # --- Make it non-modal ---
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        # Scale group
        self.scale_cb = QtWidgets.QCheckBox("Deformed mesh scale factor:")
        self.scale_spin = QtWidgets.QDoubleSpinBox()
        self.scale_spin.setRange(0, 1e12)
        self.scale_spin.setDecimals(6)

        # other options
        self.edges_cb = QtWidgets.QCheckBox("Show edges")
        self.scalarbar_cb = QtWidgets.QCheckBox("Show scalar bar")
        self.show_time_cb = QtWidgets.QCheckBox("Show time")
        self.axes_cb = QtWidgets.QCheckBox("Show axes")
        self.node_labels_cb = QtWidgets.QCheckBox("Show node labels")
        self.element_labels_cb = QtWidgets.QCheckBox("Show element labels")

        # Title group
        self.title_group = QtWidgets.QGroupBox("Title")
        self.show_title_cb = QtWidgets.QCheckBox("Show title")
        self.auto_title_rb = QtWidgets.QRadioButton("Automatic")
        self.custom_title_rb = QtWidgets.QRadioButton("Custom")
        self.title_edit = QtWidgets.QLineEdit()
        self.title_edit.setPlaceholderText("Enter custom title…")

        self.update_values()

        # === Layouts ===
        main = QtWidgets.QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(self.scale_cb)
        row.addWidget(self.scale_spin)
        main.addLayout(row)

        opts_layout = QtWidgets.QFormLayout()
        opts_layout.addRow(self.edges_cb)
        opts_layout.addRow(self.scalarbar_cb)
        opts_layout.addRow(self.show_time_cb)
        opts_layout.addRow(self.axes_cb)
        opts_layout.addRow(self.node_labels_cb)
        opts_layout.addRow(self.element_labels_cb)
        main.addLayout(opts_layout)

        # Title group layout
        title_layout = QtWidgets.QGridLayout()
        title_layout.addWidget(self.show_title_cb, 0, 0, 1, 2)
        title_layout.addWidget(self.auto_title_rb, 1, 0)
        title_layout.addWidget(self.custom_title_rb, 1, 1)
        title_layout.addWidget(QtWidgets.QLabel("Custom title:"), 2, 0)
        title_layout.addWidget(self.title_edit, 2, 1)

        self.title_group.setLayout(title_layout)
        main.addWidget(self.title_group)

        # Buttons
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Apply
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )
        main.addWidget(self.button_box)

        # === Connections ===
        def on_scale_toggled(checked: bool):
            self.scale_spin.setEnabled(checked)

        self.scale_cb.toggled.connect(on_scale_toggled)
        self.button_box.clicked.connect(self._on_button_clicked)
        self.show_title_cb.toggled.connect(self._title_changed)
        self.auto_title_rb.toggled.connect(self._title_changed)
        self.custom_title_rb.toggled.connect(self._title_changed)

        # Initialize enabled state
        self._title_changed()

    def update_values(self):
        # Defaults
        opts = self.parent().opts
        if opts["scale"] == 0:
            self.scale_cb.setChecked(False)
            self.scale_spin.setValue(float(opts["scale_save"]))
        else:
            self.scale_spin.setValue(float(opts["scale"]))
            self.scale_cb.setChecked(True)
        self.scale_spin.setEnabled(self.scale_cb.isChecked())

        self.scalarbar_cb.setChecked(opts["show_scalar_bar"])
        self.show_time_cb.setChecked(opts.get("show_time", False))
        self.edges_cb.setChecked(opts["show_edges"])
        self.axes_cb.setChecked(self.parent().plotter.renderer.axes_enabled)
        self.node_labels_cb.setChecked(opts["node_labels"])
        self.element_labels_cb.setChecked(opts["element_labels"])
        if opts["title_plot"] == "":
            self.show_title_cb.setChecked(False)
            self.auto_title_rb.setChecked(True)
            self.title_edit.setEnabled(False)  # disabled when 'Automatic' is selected
        else:
            self.show_title_cb.setChecked(True)
            if opts["title_plot"] is None:
                self.auto_title_rb.setChecked(True)  # default to automatic title
            else:
                self.custom_title_rb.setChecked(True)
                self.title_edit.setText(opts["title_plot"])

    # --- Helpers ---
    def _on_button_clicked(self, button):
        role = self.button_box.buttonRole(button)
        if role == QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole:
            self.optionsChanged.emit()
            self.close()
        elif role == QtWidgets.QDialogButtonBox.ButtonRole.ApplyRole:
            self.optionsChanged.emit()
        elif role == QtWidgets.QDialogButtonBox.ButtonRole.RejectRole:
            self.close()

    def _title_changed(self, val=None):
        if not (self.show_title_cb.isChecked()) or self.auto_title_rb.isChecked():
            self.title_edit.setEnabled(False)
        else:
            self.title_edit.setEnabled(True)


class ClimOptionsDialog(QtWidgets.QDialog):
    # Signal to tell MainWindow to apply without closing.
    applyRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Scalar bar options")
        self.setModal(False)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        # Radio buttons : mode
        self.rb_current = QtWidgets.QRadioButton("Current frame Min/Max")
        self.rb_all = QtWidgets.QRadioButton("All frames Min/Max")
        self.rb_manual = QtWidgets.QRadioButton("User Min/Max")

        # Saisie des valeurs manuelles
        self.vmax_spin = QtWidgets.QDoubleSpinBox()
        self.vmin_spin = QtWidgets.QDoubleSpinBox()
        for s in (self.vmin_spin, self.vmax_spin):
            s.setRange(-1e12, 1e12)
            s.setDecimals(6)

        # Nombre de couleurs (LUT)
        self.ncolors_spin = QtWidgets.QSpinBox()
        self.ncolors_spin.setRange(2, 2048)
        self.ncolors_spin.setSuffix(" colors")

        # Nombre de labels
        self.nlabels_spin = QtWidgets.QSpinBox()
        self.nlabels_spin.setRange(2, 20)
        self.nlabels_spin.setSuffix(" labels")

        # ----------------------------------------
        # Colormap selector and preview
        # ----------------------------------------
        self.cmap_filter = QtWidgets.QLineEdit()
        self.cmap_filter.setPlaceholderText("Filter colormaps…")

        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.setEditable(False)
        self.cmap_reverse = QtWidgets.QCheckBox("Reverse colormap")

        # Load cmap list
        self._all_cmaps = sorted(
            [c for c in sorted(list(mpl.colormaps)) if not c.endswith("_r")]
        )
        # self._all_cmaps = sorted(list(mpl.colormaps))
        self.cmap_combo.addItems(self._all_cmaps)
        self.cmap_preview = CMapPreview()  # Preview widget

        self.update_values()

        # connections
        # Activate/Deactivated manual clim values
        def refresh_manual_enabled():
            enabled = self.rb_manual.isChecked()
            self.vmin_spin.setEnabled(enabled)
            self.vmax_spin.setEnabled(enabled)
            self._update_clim_values()

        self.rb_current.toggled.connect(refresh_manual_enabled)
        self.rb_all.toggled.connect(refresh_manual_enabled)
        self.rb_manual.toggled.connect(refresh_manual_enabled)
        refresh_manual_enabled()

        # Update preview when needed
        def update_preview(*args):
            name = self.cmap_combo.currentText()
            self.cmap_preview.update_preview(
                name, self.cmap_reverse.isChecked(), int(self.ncolors_spin.value())
            )

        self.ncolors_spin.valueChanged.connect(update_preview)

        self.cmap_combo.currentTextChanged.connect(update_preview)
        self.cmap_reverse.toggled.connect(update_preview)
        self.cmap_filter.textChanged.connect(self._filter_cmaps)
        update_preview()

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow(self.rb_current)
        form.addRow(self.rb_all)
        form.addRow(self.rb_manual)
        form.addRow("Max:", self.vmax_spin)
        form.addRow("Min:", self.vmin_spin)
        form.addRow("Colors:", self.ncolors_spin)
        form.addRow("Labels:", self.nlabels_spin)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Apply
            | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        # --- Colormap groupbox ---
        cmap_group = QtWidgets.QGroupBox("Colormap")
        cmap_layout = QtWidgets.QFormLayout(cmap_group)
        cmap_layout.addRow("Filter:", self.cmap_filter)
        cmap_layout.addRow("Select:", self.cmap_combo)
        cmap_layout.addRow(self.cmap_reverse)
        cmap_layout.addRow("Preview:", self.cmap_preview)

        # Catch and connect Apply button
        apply_btn = btn_box.button(QtWidgets.QDialogButtonBox.Apply)
        apply_btn.clicked.connect(self._on_apply_clicked)

        vlayout = QtWidgets.QVBoxLayout(self)
        vlayout.addLayout(form)
        vlayout.addWidget(cmap_group)
        vlayout.addWidget(btn_box)
        self.resize(420, self.minimumSizeHint().height())

    def update_values(self):
        opts = self.parent().opts
        mode = opts["clim_mode"]
        if mode == "current":
            self.rb_current.setChecked(True)
        elif mode == "all":
            self.rb_all.setChecked(True)
        else:
            self.rb_manual.setChecked(True)
        self.ncolors_spin.setValue(opts["n_colors"])
        self.nlabels_spin.setValue(opts["n_labels"])
        self.cmap_combo.setCurrentText(opts["cmap_name"])

    # ---------------------------------------------------------------------
    # Filtering logic
    # ---------------------------------------------------------------------
    def _filter_cmaps(self, text):
        text = text.lower()
        self.cmap_combo.blockSignals(True)
        self.cmap_combo.clear()

        if not text:
            filtered = self._all_cmaps
        else:
            filtered = [c for c in self._all_cmaps if text in c.lower()]

        self.cmap_combo.addItems(filtered)
        self.cmap_combo.blockSignals(False)

        if filtered:
            self.cmap_preview.update_preview(filtered[0], self.cmap_reverse.isChecked())

    def _update_clim_values(self):
        parent = self.parent()
        if self.rb_manual.isChecked() or parent.current_field is None:
            return
        if self.rb_current.isChecked():
            data = parent.get_current_data()
            data = _global_data_array(data)
            clim = [np.nanmin(data), np.nanmax(data)]
        elif self.rb_all.isChecked():
            if hasattr(parent.active_dock.data, "get_all_frame_lim"):
                clim = parent.active_dock.data.get_all_frame_lim(
                    parent.current_field,
                    parent.current_component,
                    parent.current_data_type,
                )[2]
            else:
                data = parent.get_current_data()
                data = _global_data_array(data)
                clim = [np.nanmin(data), np.nanmax(data)]
        self.vmin_spin.setValue(float(clim[0]))
        self.vmax_spin.setValue(float(clim[1]))

    def _on_apply_clicked(self):
        # Émettre un signal pour que la fenêtre principale applique
        self.applyRequested.emit()

    def closeEvent(self, event):
        self.parent()._clim_dialog = None
        super().closeEvent(event)

    def get_values(self):
        cmap_name = self.cmap_combo.currentText()
        n_colors = int(self.ncolors_spin.value())
        cmap = mpl.colormaps[cmap_name].resampled(n_colors)
        if self.cmap_reverse.isChecked():
            cmap = cmap.reversed()
        if self.rb_current.isChecked():
            mode = "current"
        elif self.rb_all.isChecked():
            mode = "all"
        else:
            mode = "manual"
        return (
            mode,
            [float(self.vmin_spin.value()), float(self.vmax_spin.value())],
            n_colors,
            int(self.nlabels_spin.value()),
            self.cmap_combo.currentText(),
            cmap,
        )


class CMapPreview(QtWidgets.QLabel):
    """
    A QLabel that displays a gradient preview of the selected colormap.
    """

    def __init__(self, height=24, parent=None):
        super().__init__(parent)
        self.setFixedHeight(height)
        self.setMinimumWidth(160)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
        self.setScaledContents(True)
        self._preview_args = None
        self.setFrameShape(QtWidgets.QFrame.Box)
        self.setLineWidth(1)
        self.setAlignment(Qt.AlignCenter)

    def update_preview(self, cmap_name: str, reverse: bool = False, n_colors=256):
        self._preview_args = (cmap_name, reverse, n_colors)
        try:
            # cmap = mpl.colormaps.get(cmap_name)
            cmap = mpl.colormaps[cmap_name].resampled(n_colors)
        except Exception:
            self.setText(f"Colormap '{cmap_name}' introuvable")
            return

        # Create a horizontal gradient [0..1]
        width = max(self.contentsRect().width(), 1)
        height = self.height()
        x = np.linspace(0, 1, width)
        grad = np.tile(x, (height, 1))

        # Apply colormap (RGBA)
        if reverse:
            rgba = cmap(1.0 - grad)
        else:
            rgba = cmap(grad)

        # Convert to QImage/QPixmap (uint8 0..255)
        rgba_uint8 = (rgba * 255).astype(np.uint8)  # shape: (h, w, 4)
        h, w, _ = rgba_uint8.shape
        qimg = QtGui.QImage(rgba_uint8.data, w, h, 4 * w, QtGui.QImage.Format_RGBA8888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.setPixmap(pix)

    def sizeHint(self):
        return QSize(220, self.height())

    def minimumSizeHint(self):
        return QSize(160, self.height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._preview_args is not None:
            self.update_preview(*self._preview_args)


class RedererOptionsDialog(QtWidgets.QDialog):
    """
    Non-modal dialog to set opacity and optional PBR parameters.
    Emits optionsChanged(dict) on Apply.
    """

    optionsChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Renderer Options")
        self.setModal(False)
        self.setSizeGripEnabled(True)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)

        # --- Widgets ---
        self.opacity_spin = QtWidgets.QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setDecimals(2)
        self.opacity_spin.setSingleStep(0.1)
        self.opacity_spin.setToolTip(
            "Overall opacity (0.0 = transparent, 1.0 = opaque)."
        )

        self.color_button = QtWidgets.QPushButton()
        self.color_button.setToolTip("Color used for geometry without field data.")
        self.color_button.clicked.connect(self._choose_color)
        self.color = "lightgray"

        self.missing_fields_cb = QtWidgets.QCheckBox(
            "Apply to objects missing selected field"
        )

        self.pbr_cb = QtWidgets.QCheckBox("Activate Physically Based Rendering (PBR)")
        self.pbr_cb.setToolTip("Enable VTK's PBR shading model for the actor/material.")

        # PBR controls group
        self.pbr_group = QtWidgets.QGroupBox("PBR parameters")
        self.metallic_spin = QtWidgets.QDoubleSpinBox()
        self.metallic_spin.setRange(0.0, 1.0)
        self.metallic_spin.setDecimals(2)
        self.metallic_spin.setSingleStep(0.1)
        self.metallic_spin.setToolTip(
            "Metalness factor (0.0 = dielectric, 1.0 = fully metallic)."
        )

        self.roughness_spin = QtWidgets.QDoubleSpinBox()
        self.roughness_spin.setRange(0.0, 1.0)
        self.roughness_spin.setDecimals(2)
        self.roughness_spin.setSingleStep(0.1)
        self.roughness_spin.setToolTip(
            "Surface roughness (0.0 = smooth, 1.0 = very rough)."
        )

        self.diffuse_spin = QtWidgets.QDoubleSpinBox()
        self.diffuse_spin.setRange(0.0, 1.0)
        self.diffuse_spin.setDecimals(2)
        self.diffuse_spin.setSingleStep(0.1)
        self.diffuse_spin.setToolTip("Diffuse reflectance intensity.")

        self.update_values()

        # --- Layouts ---
        main = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        form.addRow("Opacity:", self.opacity_spin)
        form.addRow("Solid color:", self.color_button)
        form.addRow(self.missing_fields_cb)
        main.addLayout(form)

        # PBR layout
        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("Metallic:"), 0, 0)
        grid.addWidget(self.metallic_spin, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Roughness:"), 1, 0)
        grid.addWidget(self.roughness_spin, 1, 1)
        grid.addWidget(QtWidgets.QLabel("Diffuse:"), 2, 0)
        grid.addWidget(self.diffuse_spin, 2, 1)
        self.pbr_group.setLayout(grid)

        main.addWidget(self.pbr_cb)
        main.addWidget(self.pbr_group)

        # Buttons
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Apply
            | QtWidgets.QDialogButtonBox.Cancel
        )
        main.addWidget(self.button_box)

        # --- Connections ---
        self.button_box.clicked.connect(self._on_button_clicked)
        self.pbr_cb.toggled.connect(self._toggle_pbr_group)

        # Initialize enabled state
        self._toggle_pbr_group(self.pbr_cb.isChecked())

    def update_values(self):
        opts = self.parent().opts
        self.opacity_spin.setValue(float(opts["opacity"]))
        self._set_color(opts.get("color", "lightgray"))
        self.missing_fields_cb.setChecked(opts.get("color_missing_fields", True))
        self.pbr_cb.setChecked(opts["pbr"])
        self.metallic_spin.setValue(float(opts["metallic"]))
        self.roughness_spin.setValue(float(opts["roughness"]))
        self.diffuse_spin.setValue(float(opts["diffuse"]))

    def _set_color(self, color):
        qcolor = QtGui.QColor(color)
        if not qcolor.isValid():
            qcolor = QtGui.QColor("lightgray")
        self.color = qcolor.name()
        self.color_button.setStyleSheet(f"background-color: {self.color};")

    def _choose_color(self):
        color = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.color), self)
        if color.isValid():
            self._set_color(color)

    def _toggle_pbr_group(self, checked: bool):
        self.pbr_group.setEnabled(checked)

    def _on_button_clicked(self, button):
        role = self.button_box.buttonRole(button)
        if role == QtWidgets.QDialogButtonBox.ButtonRole.AcceptRole:
            self.optionsChanged.emit()
            self.close()
        elif role == QtWidgets.QDialogButtonBox.ButtonRole.ApplyRole:
            self.optionsChanged.emit()
        elif role == QtWidgets.QDialogButtonBox.ButtonRole.RejectRole:
            self.close()


# Dialog non modale : ClipPlane
# ----------------------------
class ClipDialog(QtWidgets.QDialog):
    """Non modal window to select origin / normal / invert."""

    clipPlaneChanged = Signal(
        bool, object, object, bool
    )  # (enabled, origin, normal, invert)

    def __init__(
        self,
        parent=None,
        default_origin=(0.0, 0.0, 0.0),
        default_normal=(1.0, 0.0, 0.0),
        invert=False,
    ):
        super().__init__(parent)
        self.setWindowTitle("Clip Plane")
        self.setModal(False)  # non modale
        default_origin = _as_3d_tuple(default_origin)
        default_normal = _as_3d_tuple(default_normal)

        # --- Widgets ---------------------------------------------------------
        self.enableClipChk = QtWidgets.QCheckBox("Activate clipping")
        self.enableClipChk.setChecked(bool(parent.opts["clip_args"]))

        # Number widgets
        self.originX = QtWidgets.QDoubleSpinBox()
        self.originX.setRange(-1e9, 1e9)
        self.originX.setDecimals(6)
        self.originX.setValue(default_origin[0])
        self.originY = QtWidgets.QDoubleSpinBox()
        self.originY.setRange(-1e9, 1e9)
        self.originY.setDecimals(6)
        self.originY.setValue(default_origin[1])
        self.originZ = QtWidgets.QDoubleSpinBox()
        self.originZ.setRange(-1e9, 1e9)
        self.originZ.setDecimals(6)
        self.originZ.setValue(default_origin[2])

        self.normalX = QtWidgets.QDoubleSpinBox()
        self.normalX.setRange(-1e6, 1e6)
        self.normalX.setDecimals(6)
        self.normalX.setValue(default_normal[0])
        self.normalY = QtWidgets.QDoubleSpinBox()
        self.normalY.setRange(-1e6, 1e6)
        self.normalY.setDecimals(6)
        self.normalY.setValue(default_normal[1])
        self.normalZ = QtWidgets.QDoubleSpinBox()
        self.normalZ.setRange(-1e6, 1e6)
        self.normalZ.setDecimals(6)
        self.normalZ.setValue(default_normal[2])

        self.normalizeBtn = QtWidgets.QPushButton("Normalize normal")
        self.invertChk = QtWidgets.QCheckBox("Invert clipping direction")
        self.invertChk.setChecked(bool(invert))

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("Origin X", self.originX)
        form.addRow("Origin Y", self.originY)
        form.addRow("Origin Z", self.originZ)
        form.addRow("Normal X", self.normalX)
        form.addRow("Normal Y", self.normalY)
        form.addRow("Normal Z", self.normalZ)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addWidget(self.normalizeBtn)
        buttons.addStretch(1)

        main = QtWidgets.QVBoxLayout(self)
        main.addWidget(self.enableClipChk)
        main.addLayout(form)
        main.addWidget(self.invertChk)
        main.addLayout(buttons)

        # Connexions
        self.normalizeBtn.clicked.connect(self._normalize_normal)
        self.enableClipChk.toggled.connect(self._emit_clip_params)

        # Mode "live" : émettre à chaque modif
        for w in (
            self.originX,
            self.originY,
            self.originZ,
            self.normalX,
            self.normalY,
            self.normalZ,
        ):
            w.valueChanged.connect(self._emit_clip_params)
        self.invertChk.toggled.connect(self._emit_clip_params)

    def update_values(self):
        opts = self.parent().opts
        self.set_values_from_widget(
            opts["clip_origin"], opts["clip_normal"], opts["clip_invert"]
        )
        blockers = QSignalBlocker(self.enableClipChk)
        self.enableClipChk.setChecked(bool(opts["clip_args"]))

    def _normalize_normal(self):
        nn = normalize(
            (self.normalX.value(), self.normalY.value(), self.normalZ.value())
        )
        blockers = [
            QSignalBlocker(self.normalX),
            QSignalBlocker(self.normalY),
            QSignalBlocker(self.normalZ),
        ]
        self.normalX.setValue(nn[0])
        self.normalY.setValue(nn[1])
        self.normalZ.setValue(nn[2])
        del blockers
        self._emit_clip_params()

    def _emit_clip_params(self):
        origin = (self.originX.value(), self.originY.value(), self.originZ.value())
        normal = (self.normalX.value(), self.normalY.value(), self.normalZ.value())
        invert = self.invertChk.isChecked()
        enabled = self.enableClipChk.isChecked()
        self.clipPlaneChanged.emit(enabled, origin, normalize(normal), invert)

    def set_values_from_widget(self, origin, normal, invert=None):
        """Update widget values without emiting signal."""
        origin = _as_3d_tuple(origin)
        normal = _as_3d_tuple(normal)
        blockers = [
            QSignalBlocker(self.originX),
            QSignalBlocker(self.originY),
            QSignalBlocker(self.originZ),
            QSignalBlocker(self.normalX),
            QSignalBlocker(self.normalY),
            QSignalBlocker(self.normalZ),
        ]
        self.originX.setValue(origin[0])
        self.originY.setValue(origin[1])
        self.originZ.setValue(origin[2])
        self.normalX.setValue(normal[0])
        self.normalY.setValue(normal[1])
        self.normalZ.setValue(normal[2])

        del blockers
        if invert is not None:
            chk_blocker = QSignalBlocker(self.invertChk)
            self.invertChk.setChecked(bool(invert))
            del chk_blocker


# Dialog non modale : Plot over line
# ----------------------------


class PlotOverLineDialog(QtWidgets.QDialog):
    plotRequested = Signal(tuple, tuple, int)
    lineChanged = Signal(object, object)
    requestPick = Signal(int)  # 0 -> pick P1, 1 -> pick P2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Over Line")
        # self.setModal(False)

        self.p1 = [0.0, 0.0, 0.0]
        self.p2 = [1.0, 0.0, 0.0]

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        main_layout = QtWidgets.QVBoxLayout(self)

        # --- Point 1 group ---
        self.p1_edits, p1_group = self._build_point_group(title="Point 1", pick_index=0)

        # --- Point 2 group ---
        self.p2_edits, p2_group = self._build_point_group(title="Point 2", pick_index=1)

        # --- Resolution ---
        self.resSpin = QtWidgets.QSpinBox()
        self.resSpin.setRange(2, 10000)
        self.resSpin.setSingleStep(10)
        self.resSpin.setValue(200)

        res_layout = QtWidgets.QFormLayout()
        res_layout.addRow("Resolution", self.resSpin)

        # --- Plot and invert buttons ---
        self.invertDirBtn = QtWidgets.QPushButton("Invert direction")
        self.plotBtn = QtWidgets.QPushButton("Plot over line")
        self.plotBtn.setDefault(True)

        # --- Assemble layout ---
        main_layout.addWidget(p1_group)
        main_layout.addWidget(p2_group)
        main_layout.addLayout(res_layout)
        main_layout.addSpacing(6)
        main_layout.addWidget(self.invertDirBtn)
        main_layout.addWidget(self.plotBtn)

        # --- Connections ---
        self.invertDirBtn.clicked.connect(self._invert_direction)
        self.plotBtn.clicked.connect(self._emit_plot)

    # ------------------------------------------------------------------
    def _build_point_group(self, title: str, pick_index: int):
        """
        Returns (edits, QGroupBox)
        """
        group = QtWidgets.QGroupBox(title)
        grid = QtWidgets.QGridLayout(group)

        # Spin boxes
        edits = []
        labels = ("X", "Y", "Z")
        for col, lbl in enumerate(labels):
            lab = QtWidgets.QLabel(lbl)
            sb = QtWidgets.QDoubleSpinBox()
            sb.setRange(-1e9, 1e9)
            sb.setDecimals(6)
            sb.valueChanged.connect(self._emit_line_params)

            grid.addWidget(lab, 0, col)
            grid.addWidget(sb, 1, col)
            edits.append(sb)

        # Pick button with icon
        pick_btn = QtWidgets.QPushButton("Pick")
        pick_btn.clicked.connect(lambda: self.requestPick.emit(pick_index))

        grid.addWidget(pick_btn, 1, 3)
        grid.setColumnStretch(4, 1)

        return edits, group

    def _emit_line_params(self):
        self.p1 = tuple(edit.value() for edit in self.p1_edits)
        self.p2 = tuple(edit.value() for edit in self.p2_edits)
        self.lineChanged.emit(self.p1, self.p2)

    def update_line(self, p1, p2):
        """Called from PyVista widget callback"""
        self.p1 = p1
        self.p2 = p2

        all_edits = (*self.p1_edits, *self.p2_edits)
        values = (*p1, *p2)
        for edit, val in zip(all_edits, values):
            edit.blockSignals(True)
            edit.setValue(val)
            edit.blockSignals(False)

    def _invert_direction(self):
        self.parent()._on_pol_dialog_changed(self.p2, self.p1)
        self.update_line(self.p2, self.p1)
        # for i in range(3):
        #     self.p1_edits[i].setValue(self.p2[i])
        #     self.p2_edits[i].setValue(self.p1[i])

    def _emit_plot(self):
        p1 = tuple(edit.value() for edit in self.p1_edits)
        p2 = tuple(edit.value() for edit in self.p2_edits)
        resolution = int(self.resSpin.value())
        self.plotRequested.emit(p1, p2, resolution)


class _SelectionRubberBand(QObject):
    """Qt overlay stays visible with QtInteractor/OpenGL rubber-band picking."""

    def __init__(self, widget, parent):
        super().__init__(parent)
        self.widget = widget
        self.band = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Rectangle, widget)
        self.band.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.origin = None
        widget.installEventFilter(self)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self.origin = event.pos()
            self.band.setGeometry(QRect(self.origin, self.origin))
            self.band.show()
            self.band.raise_()
        elif event.type() == QEvent.MouseMove and self.origin is not None:
            self.band.setGeometry(QRect(self.origin, event.pos()).normalized())
        elif event.type() == QEvent.MouseButtonRelease:
            self.origin = None
            self.band.hide()
        return False

    def stop(self):
        self.widget.removeEventFilter(self)
        self.band.hide()
        self.band.deleteLater()
        self.deleteLater()


class ElementVisibilityDialog(QtWidgets.QDialog):
    """
    Non-modal dialog to manage FEM element visibility and element sets.
    """

    entity = "Element"
    set_attribute = "element_sets"

    def __init__(
        self,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(
            "Element Visibility & Sets" if self.entity == "Element" else "Node Sets"
        )
        self.setModal(False)  # non modale

        # self.setWindowModality(Qt.NonModal)
        # Keep dialog on top if desired
        # self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        # Don’t delete on close so it keeps state if reopened
        # self.setAttribute(Qt.WA_DeleteOnClose, False)

        # External references
        self.mesh = self.parent().active_dock.data.mesh

        # # Track selection and visibility (start fully visible)
        self.current_selection = set()
        self._selection_band = None
        self._surface_pick_dock = None

        # Build UI
        self._build_ui()
        self.layout().activate()
        self.resize(760, self.minimumSizeHint().height())

    def all_elements(self):
        mesh = self.parent().data.mesh
        return set(range(mesh.n_elements if self.entity == "Element" else mesh.n_nodes))

    @property
    def plotter(self):
        return self.parent().plotter

    @property
    def visibility_mode(self):
        if self.entity == "Node":
            return None
        return self.parent().active_dock.visibility_mode

    @visibility_mode.setter
    def visibility_mode(self, value):
        self.parent().active_dock.visibility_mode = value

    # -------------- UI -----------------
    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        splitter = QtWidgets.QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, 1)

        # Left: Element sets management
        sets_box = QtWidgets.QGroupBox(f"{self.entity} Sets")
        sets_layout = QVBoxLayout(sets_box)

        self.sets_list = QtWidgets.QListWidget()
        self._refresh_sets_list()
        sets_layout.addWidget(self.sets_list)

        btn_row = QtWidgets.QHBoxLayout()

        save_selection_btn = QPushButton("New...")
        save_selection_btn.setToolTip("Create a new set from the current selection")
        save_selection_btn.clicked.connect(self._save_selection_as_new_set)
        btn_row.addWidget(save_selection_btn)

        rename_btn = QPushButton("Rename")
        rename_btn.clicked.connect(self._rename_selected_set)
        btn_row.addWidget(rename_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self._remove_selected_set)
        btn_row.addWidget(remove_btn)

        # Add the horizontal row to the left panel
        sets_layout.addLayout(btn_row)

        load_row = QtWidgets.QHBoxLayout()

        load_label = QLabel("Selection:")
        sets_layout.addWidget(load_label)

        # Replace selection
        load_replace_btn = QPushButton("Replace")
        load_replace_btn.setToolTip("Replace current selection with the selected set")
        load_replace_btn.clicked.connect(lambda: self._load_element_set("replace"))
        load_row.addWidget(load_replace_btn)

        # Add to selection
        load_add_btn = QPushButton("Add")
        load_add_btn.setToolTip("Add elements of the selected set to current selection")
        load_add_btn.clicked.connect(lambda: self._load_element_set("add"))
        load_row.addWidget(load_add_btn)

        # Remove from selection
        load_remove_btn = QPushButton("Remove")
        load_remove_btn.setToolTip(
            "Remove elements of the selected set from current selection"
        )
        load_remove_btn.clicked.connect(lambda: self._load_element_set("remove"))
        load_row.addWidget(load_remove_btn)

        # Optional: push the buttons to the left and keep consistent spacing
        load_row.addStretch(1)

        # Add the new row to the left panel
        sets_layout.addLayout(load_row)

        incoming = "node" if self.entity == "Element" else "element"
        conversion_row = QHBoxLayout()
        conversion_label = QLabel("From:")
        conversion_label.setToolTip(f"Create from {incoming} set")
        conversion_row.addWidget(conversion_label)
        self.conversion_sources = QtWidgets.QComboBox()
        self.conversion_sources.setAccessibleName(f"Source {incoming} set")
        self.conversion_sources.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.conversion_sources.setMinimumContentsLength(8)
        self.conversion_sources.currentTextChanged.connect(
            self.conversion_sources.setToolTip
        )
        conversion_row.addWidget(self.conversion_sources, 1)
        conversion = QPushButton("Create")
        conversion.clicked.connect(self._convert_selected_set)
        conversion_row.addWidget(conversion)
        sets_layout.addLayout(conversion_row)
        if self.entity == "Element":
            self.conversion_all_nodes = QCheckBox()
            self.conversion_all_nodes.setChecked(True)
            self.conversion_all_nodes.toggled.connect(self._update_conversion_mode_text)
            self._update_conversion_mode_text()
            sets_layout.addWidget(self.conversion_all_nodes)
        self._refresh_conversion_sources()

        splitter.addWidget(sets_box)

        # Right: Selection & visibility controls
        right_box = QtWidgets.QGroupBox("Selection")
        right_layout = QtWidgets.QVBoxLayout()
        right_layout.setAlignment(Qt.AlignTop)
        right_box.setLayout(right_layout)

        # --- Top row: Clear + Invert + Selection info --------------------------------
        sel_top_row = QtWidgets.QHBoxLayout()

        clear_sel_btn = QPushButton("Clear")
        clear_sel_btn.setToolTip("Clear the current selection")
        clear_sel_btn.clicked.connect(self._clear_selection)
        sel_top_row.addWidget(clear_sel_btn)

        invert_sel_btn = QPushButton("Invert")
        invert_sel_btn.setToolTip("Invert the current selection")
        invert_sel_btn.clicked.connect(self._invert_selection)  # implement this method
        sel_top_row.addWidget(invert_sel_btn)

        sel_top_row.addStretch(1)  # push the label to the right

        self.sel_info_lbl = QLabel(f"0 {self.entity.lower()}s")
        sel_top_row.addWidget(self.sel_info_lbl)

        right_layout.addLayout(sel_top_row)

        # --- Selection by picking ----------------------------------------------------
        pick_group = QtWidgets.QGroupBox("Pick on Screen")
        pick_layout = QtWidgets.QHBoxLayout()
        pick_group.setLayout(pick_layout)

        self.start_pick_btn = QPushButton("Start")
        self.start_pick_btn.setToolTip("Start rectangle picking")
        self.start_pick_btn.clicked.connect(self._start_picking)
        pick_layout.addWidget(self.start_pick_btn)

        self.stop_pick_btn = QPushButton("Stop")
        self.stop_pick_btn.setToolTip("Stop rectangle picking")
        self.stop_pick_btn.clicked.connect(self._stop_picking)
        self.stop_pick_btn.setEnabled(False)
        pick_layout.addWidget(self.stop_pick_btn)

        # --- Add a stretch to push the labels to the far right ---
        pick_layout.addStretch(1)

        # --- Create a vertical column for the two labels ---
        labels_col = QtWidgets.QVBoxLayout()
        labels_col.setSpacing(0)  # optional: tighter stacking
        labels_col.setContentsMargins(0, 0, 0, 0)  # optional: remove extra margins

        modifier_hint = QLabel("Ctrl + / Shift −")
        modifier_hint.setToolTip(
            "Hold Ctrl to add to the selection, or Shift to remove from it."
        )
        labels_col.addWidget(modifier_hint)

        # Add the vertical layout to the horizontal layout
        pick_layout.addLayout(labels_col)

        right_layout.addWidget(pick_group)

        # --- Selection by expression -------------------------------------------------
        expr_group = QtWidgets.QGroupBox("Select by expression")
        expr_form = QtWidgets.QFormLayout()
        expr_group.setLayout(expr_form)

        self.expr_edit = QtWidgets.QLineEdit()
        self.expr_edit.setPlaceholderText("Stress_XX > 100 and X < 0.5")
        self.expr_edit.setToolTip(
            "Coordinates X/Y/Z, ID, field components (Stress_XX, Disp_X), "
            "and NumPy functions. Element fields use element averages; node fields use nodal values."
        )
        expr_form.addRow(QLabel("Expression:"), self.expr_edit)

        apply_expr_btn = QPushButton("Apply expression")
        apply_expr_btn.clicked.connect(self._apply_expression_to_selection)

        expr_form.addRow(apply_expr_btn)

        self.use_def_mesh = QCheckBox("Use deformed mesh")
        self.use_def_mesh.setChecked(True)
        expr_form.addRow(self.use_def_mesh)

        right_layout.addWidget(expr_group)

        if self.entity == "Node":
            self._build_surface_selector(right_layout)

        # --- Show/Hide selected elements ------------------------
        vis_group = QtWidgets.QGroupBox("Show/Hide elements")
        vis_layout = QtWidgets.QHBoxLayout()
        vis_group.setLayout(vis_layout)

        show_all_btn = QPushButton("Show All")
        show_all_btn.clicked.connect(lambda: self._apply_visibility("all"))
        vis_layout.addWidget(show_all_btn)

        add_btn = QPushButton("Show")
        add_btn.setToolTip("Show selected elements")
        add_btn.clicked.connect(lambda: self._apply_visibility("show"))
        vis_layout.addWidget(add_btn)

        remove_btn = QPushButton("Hide")
        remove_btn.setToolTip("Hide selected elements")
        remove_btn.clicked.connect(lambda: self._apply_visibility("hide"))
        vis_layout.addWidget(remove_btn)

        if self.entity == "Element":
            right_layout.addWidget(vis_group)
        else:
            vis_group.deleteLater()

        splitter.addWidget(right_box)
        splitter.setSizes([300, 420])

        self.status_lbl = QLabel("Ready.")
        self.status_lbl.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed
        )
        main_layout.addWidget(self.status_lbl)

    def _build_surface_selector(self, right_layout):
        surface_group = QtWidgets.QGroupBox("Select exterior surface nodes")
        self.surface_group = surface_group
        surface_form = QtWidgets.QFormLayout(surface_group)
        self.surface_mode = QtWidgets.QComboBox()
        self.surface_mode.setSizeAdjustPolicy(
            QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.surface_mode.setMinimumContentsLength(10)
        self.surface_mode.currentTextChanged.connect(self.surface_mode.setToolTip)
        surface_form.addRow("Surface:", self.surface_mode)
        self.surface_seed_id = QSpinBox()
        self.surface_seed_id.setRange(0, 2147483647)
        self.surface_seed_id.setMinimumWidth(70)
        self.surface_seed_id.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed
        )
        seed_row = QHBoxLayout()
        seed_row.addWidget(self.surface_seed_id)
        self.surface_select_button = QPushButton("Select")
        self.surface_select_button.clicked.connect(self._surface_seed_from_selection)
        seed_row.addWidget(self.surface_select_button)
        surface_form.addRow("Seed node:", seed_row)
        self.surface_angle = QDoubleSpinBox()
        self.surface_angle.setRange(0, 180)
        self.surface_angle.setDecimals(1)
        self.surface_angle.setValue(30)
        self.surface_angle.setSuffix("°")
        self.surface_angle.setToolTip(
            "Maximum angle between neighboring face normals. 0° is flat; "
            "30° stops at a 90° corner. Uses the deformed-mesh option above."
        )
        self.surface_angle_label = QLabel("Angle limit:")
        surface_form.addRow(self.surface_angle_label, self.surface_angle)
        self.surface_feature_angle = QDoubleSpinBox()
        self.surface_feature_angle.setRange(0, 180)
        self.surface_feature_angle.setDecimals(1)
        self.surface_feature_angle.setValue(30)
        self.surface_feature_angle.setSuffix("°")
        self.surface_feature_angle.setToolTip(
            "Minimum angle between exterior face normals to identify a crease. Open shell boundaries always qualify."
        )
        self.surface_feature_label = QLabel("Sharpness:")
        surface_form.addRow(self.surface_feature_label, self.surface_feature_angle)
        surface_button = QPushButton("Select surface nodes")
        self.surface_apply_button = surface_button
        surface_button.clicked.connect(self._select_surface_nodes)
        surface_form.addRow(surface_button)
        right_layout.addWidget(surface_group)
        self.surface_mode.currentIndexChanged.connect(self._update_surface_mode)
        self._refresh_surface_options()

    def update_plot(self):
        """Rebuild displayed geometry from visible_elements."""
        self.parent().update_plot()

    # -------------- Element sets UI helpers -------------
    def _default_element_sets(self):
        return self.parent()._default_element_sets_for_dock(self.parent().active_dock)

    def _refresh_sets_list(self):
        if self.element_sets is None:
            element_sets, unsaved = self._default_element_sets()
            self.element_sets = element_sets
            self.unsaved_set_names = unsaved
        self.parent()._sync_viewer_sets(self.parent().active_dock)
        self.sets_list.clear()
        for name in sorted(self.element_sets.keys()):
            values = self.element_sets[name]
            if values is None:
                n_elements = len(self.all_elements())
            else:
                n_elements = len(values)
            item = QtWidgets.QListWidgetItem(
                f"{name} ({n_elements} {self.entity.lower()}s)"
            )
            item.setData(Qt.UserRole, name)
            item.setToolTip(name)
            self.sets_list.addItem(item)
        self._refresh_conversion_sources()
        for dialog in (
            self.parent().node_sets_dialog,
            self.parent().element_sets_dialog,
        ):
            if dialog is not None and dialog is not self:
                dialog._refresh_conversion_sources()
        if self.entity == "Node" and hasattr(self, "surface_mode"):
            self._refresh_surface_options()

    def _refresh_conversion_sources(self):
        if not hasattr(self, "conversion_sources"):
            return
        dock = self.parent().active_dock
        self.parent()._ensure_dock_node_sets(dock)
        self.parent()._ensure_dock_element_sets(dock)
        sources = dock.node_sets if self.entity == "Element" else dock.element_sets
        selected = self.conversion_sources.currentText()
        self.conversion_sources.clear()
        self.conversion_sources.addItems(sorted(sources))
        if selected in sources:
            self.conversion_sources.setCurrentText(selected)

    def _update_conversion_mode_text(self, *args):
        self.conversion_all_nodes.setText(
            "All nodes required"
            if self.conversion_all_nodes.isChecked()
            else "At least one node"
        )
        self.conversion_all_nodes.setToolTip(
            "All element nodes must belong to the source set"
            if self.conversion_all_nodes.isChecked()
            else "At least one element node must belong to the source set"
        )

    def _refresh_surface_options(self):
        dock = self.parent().active_dock
        if getattr(self, "_surface_options_dock", None) is dock:
            return
        self._cancel_surface_pick()
        self._surface_options_dock = dock
        mesh = dock.data.mesh
        blocks = getattr(mesh, "submeshes", (mesh,))
        is_surface = all(block.elm_type.startswith(("tri", "quad")) for block in blocks)
        blocker = QSignalBlocker(self.surface_mode)
        self.surface_mode.clear()
        if is_surface:
            self.surface_mode.addItem("Boundary edges (2D mesh)", "boundary")
            if mesh.nodes.shape[1] == 3:
                self.surface_mode.addItem("Surface faces (shell mesh)", "faces")
                points = mesh.nodes
                planar = (
                    np.linalg.matrix_rank(points - points[0]) <= 2
                    if len(points)
                    else True
                )
                if not planar or "_ShellSection" in dock.data.field_metadata:
                    self.surface_mode.setCurrentIndex(1)
        else:
            self.surface_mode.addItem("Exterior faces (solid mesh)", "solid")
        if mesh.nodes.shape[1] == 3:
            self.surface_mode.addItem("Exterior edges (3D)", "edges")
        del blocker
        self._update_surface_mode()

    def _update_surface_mode(self, *args):
        if self._surface_pick_dock is not None:
            self._cancel_surface_pick()
        edges = self.surface_mode.currentData() == "edges"
        self.surface_feature_label.setVisible(edges)
        self.surface_feature_angle.setVisible(edges)
        self.surface_angle_label.setText("Turn limit:" if edges else "Angle limit:")
        self.surface_angle.setToolTip(
            "Maximum change of direction while following an edge. 0° follows straight edges; traversal stops at branches."
            if edges
            else "Maximum angle between neighboring face normals. 0° is flat; 30° stops at a 90° corner."
        )
        self.surface_apply_button.setText(
            "Select edge nodes" if edges else "Select surface nodes"
        )
        self.surface_group.setTitle(
            "Select exterior edge nodes"
            if edges
            else "Select boundary line nodes"
            if self.surface_mode.currentData() == "boundary"
            else "Select exterior surface nodes"
        )

    def _selected_set_name(self):
        item = self.sets_list.currentItem()
        return item.data(Qt.UserRole) if item else None

    def update_selection(self, selection, check_modifiers=False):
        if check_modifiers:
            # Check modifier keys
            mods = QtWidgets.QApplication.keyboardModifiers()
            ctrl_down = bool(mods & Qt.ControlModifier)
            shift_down = bool(mods & Qt.ShiftModifier)

            if shift_down:
                self.current_selection -= selection
            elif ctrl_down:
                self.current_selection |= selection
            else:
                self.current_selection = selection
        else:
            self.current_selection = set(selection)

        if self.entity == "Node":
            self.plotter.remove_actor("_picked_node_set")
            if self.current_selection:
                points = self._node_positions()[sorted(self.current_selection)]
                self.plotter.add_points(
                    points,
                    color="pink",
                    point_size=12,
                    name="_picked_node_set",
                    render_points_as_spheres=True,
                    reset_camera=False,
                )
            self._update_selection_info()
            return
        self.plotter.remove_actor("_picked_cell")
        if self.visibility_mode:
            if self.visibility_mode == "show":
                # show selection
                self.visible_elements = list(self.current_selection)
            elif self.visibility_mode == "hide":
                # hide selection
                self.visible_elements = list(
                    self.all_elements() - self.current_selection
                )
            self.update_plot()
        else:
            pvmesh = self.parent().data.mesh.to_pyvista()
            pvmesh.points = self._node_positions()
            if self.current_selection:
                cell = pvmesh.extract_cells(list(self.current_selection))
                self.plotter.remove_actor("_picked_cell")
                self.plotter.add_mesh(
                    cell,
                    style="wireframe",
                    color="pink",
                    name="_picked_cell",
                    reset_camera=False,
                )
        self._update_selection_info()

        # self.status_lbl.setText(f"Selection updated -> {len(self.current_selection)} elements)")

    # -------------- Picking integration -----------------
    def _start_picking(self):
        self._cancel_surface_pick()
        for dialog in (
            self.parent().node_sets_dialog,
            self.parent().element_sets_dialog,
        ):
            if dialog is not None and dialog is not self:
                dialog._cancel_surface_pick()
                dialog._stop_picking()

        def _on_rectangle(rect_selection):
            # rect_selection.frustum is a vtkPlanes; rect_selection.viewport is the 2D rectangle.
            planes = rect_selection.frustum  # vtkPlanes (camera frustum through the drawn rectangle)  [1](https://docs.pyvista.org/api/plotting/_autosummary/pyvista.plotter.enable_rectangle_picking)

            # Compute cell-centers once; cache if your mesh is large
            pvmesh = self.parent().active_dock.pv_mesh
            centers = (
                self._node_positions()
                if self.entity == "Node"
                else pvmesh.cell_centers().points
            )

            f = planes.EvaluateFunction  # C++ method: <=0 means "inside" the convex volume. [2](https://vtk.org/doc/nightly/html/classvtkExtractSelectedFrustum.html)
            inside_mask = np.fromiter(
                (f(x, y, z) <= 0.0 for x, y, z in centers),
                count=len(centers),
                dtype=bool,
            )
            picked_ids = np.nonzero(inside_mask)[0].astype(int)
            if self.entity == "Node":
                pass  # Physical nodes already use the mesh's original numbering.
            elif "_fedoo_global_cell_ids" in pvmesh.cell_data:
                picked_ids = pvmesh.cell_data["_fedoo_global_cell_ids"][picked_ids]
            elif "vtkOriginalCellIds" in pvmesh.cell_data:
                picked_ids = pvmesh.cell_data["vtkOriginalCellIds"][picked_ids]

            picked_ids = set(picked_ids)

            self.update_selection(picked_ids, True)
            self.status_lbl.setText(
                f"Selected {len(self.current_selection)} {self.entity.lower()}s from rectangle."
            )

        self.plotter.disable_picking()
        self.plotter.enable_rectangle_picking(
            callback=_on_rectangle,
            start=True,  # start immediately
            show_message=False,
        )
        widget = self.plotter.interactor
        if not isinstance(widget, QtWidgets.QWidget):
            widget = self.plotter
        if self._selection_band is not None:
            self._selection_band.stop()
        self._selection_band = _SelectionRubberBand(widget, self)

        self.start_pick_btn.setEnabled(False)
        self.stop_pick_btn.setEnabled(True)
        self.status_lbl.setText(
            f"Picking enabled: drag a rectangle to select {self.entity.lower()}s."
        )
        # except Exception as e:
        #     QtWidgets.QMessageBox.warning(self, "Picking error", f"Failed to enable picking:\n{e}")

    def _stop_picking(self):
        if self._selection_band is not None:
            self._selection_band.stop()
            self._selection_band = None
        if not self.stop_pick_btn.isEnabled():
            return
        try:
            self.plotter.enable_trackball_style()
            self.plotter.disable_picking()
            self.plotter.enable_trackball_style()
            self.start_pick_btn.setEnabled(True)
            self.stop_pick_btn.setEnabled(False)
            self.status_lbl.setText("Picking disabled.")
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Picking error", f"Failed to disable picking:\n{e}"
            )

    def _clear_selection(self):
        self.current_selection.clear()
        self.update_selection(set())

    def _invert_selection(self):
        self.current_selection = self.all_elements() - self.current_selection
        self.update_selection(self.current_selection)

    def _update_selection_info(self):
        self.sel_info_lbl.setText(
            f"{len(self.current_selection)} {self.entity.lower()}s"
        )

    def _node_positions(self):
        coordinates = as_3d_coordinates(self.parent().data.mesh.nodes).astype(
            float, copy=True
        )
        if self.use_def_mesh.isChecked() and "Disp" in self.parent().data.node_data:
            coordinates += self.parent().opts["scale"] * as_3d_coordinates(
                self.parent().data.node_data["Disp"].T
            )
        return coordinates

    def _surface_seed_from_selection(self):
        if self._surface_pick_dock is not None:
            self._cancel_surface_pick()
            return
        main = self.parent()
        for dialog in (main.node_sets_dialog, main.element_sets_dialog):
            if dialog is not None:
                dialog._stop_picking()
        main.select_node_action.setChecked(False)
        main.select_elm_action.setChecked(False)
        self._surface_pick_dock = main.active_dock
        self.surface_select_button.setText("Cancel")
        self.status_lbl.setText("Click one node in the view to set the seed.")
        main.statusBar().setVisible(True)
        picked_dock = main.active_dock

        def selected(entity_id):
            if (
                self._surface_pick_dock is not picked_dock
                or main.active_dock is not picked_dock
            ):
                return
            self.surface_seed_id.setValue(entity_id)
            self.status_lbl.setText(f"Seed node ID {entity_id} selected.")
            # Remove picker observers after the native click callback returns.
            QTimer.singleShot(0, self._cancel_surface_pick)

        main._pick_node(on_pick=selected)

    def _cancel_surface_pick(self, *args):
        dock = self._surface_pick_dock
        if dock is None:
            return
        main = self.parent()
        dock.plotter.disable_picking()
        if getattr(main, "_lbp_tag", None) is not None and main.active_dock is dock:
            dock.plotter.interactor.RemoveObserver(main._lbp_tag)
            main._lbp_tag = None
        self._surface_pick_dock = None
        self.surface_select_button.setText("Select")

    def _select_surface_nodes(self):
        self._cancel_surface_pick()
        mesh = self.parent().data.mesh
        if self.use_def_mesh.isChecked():
            mesh = mesh.copy()
            mesh.nodes = self._node_positions()[:, : mesh.nodes.shape[1]]
            for block in getattr(mesh, "submeshes", ()):
                block.nodes = mesh.nodes
        seed = self.surface_seed_id.value()
        angle = self.surface_angle.value()
        edge_mode = self.surface_mode.currentData() == "edges"
        try:
            try:
                nodes = (
                    find_edge_nodes(
                        mesh, seed, angle, self.surface_feature_angle.value()
                    )
                    if edge_mode
                    else find_surface_nodes(
                        mesh,
                        seed,
                        angle=angle,
                        boundary_edges=self.surface_mode.currentData() == "boundary",
                    )
                )
            except AmbiguousEdgeSeed as exc:
                choices = [
                    f"Edge {index}: nodes {a}–{b}; direction ({direction[0]:.3g}, {direction[1]:.3g}, {direction[2]:.3g})"
                    for index, (a, b), direction in zip(
                        exc.edge_ids, exc.endpoints, exc.directions
                    )
                ]
                choice, accepted = QtWidgets.QInputDialog.getItem(
                    self,
                    "Choose the starting edge",
                    "The seed touches several edge chains. Choose a starting edge:",
                    choices,
                    0,
                    False,
                )
                if not accepted:
                    return
                nodes = find_edge_nodes(
                    mesh,
                    seed,
                    angle,
                    self.surface_feature_angle.value(),
                    seed_edge=int(exc.edge_ids[choices.index(choice)]),
                )
            except AmbiguousSurfaceSeed as exc:
                choices = [
                    f"Face {face_id}: normal ({normal[0]:.3g}, {normal[1]:.3g}, {normal[2]:.3g})"
                    for face_id, normal in zip(exc.face_ids, exc.normals)
                ]
                choice, accepted = QtWidgets.QInputDialog.getItem(
                    self,
                    "Choose the starting surface",
                    "The seed touches several patches. Choose a starting face:",
                    choices,
                    0,
                    False,
                )
                if not accepted:
                    return
                face_id = int(exc.face_ids[choices.index(choice)])
                nodes = find_surface_nodes(
                    mesh,
                    seed,
                    angle=angle,
                    seed_face=face_id,
                    boundary_edges=self.surface_mode.currentData() == "boundary",
                )
        except (ValueError, NotImplementedError) as exc:
            message = str(exc)
            if "does not belong to an exterior" in message:
                message = "The seed is inside the mesh. Select a node on the exterior boundary."
            QtWidgets.QMessageBox.warning(self, "Surface selection", message)
            return
        self.parent().open_node_sets_dialog()
        dialog = self.parent().node_sets_dialog
        dialog.update_selection(set(nodes))
        message = f"Selected {len(nodes)} {'edge' if edge_mode else 'surface'} nodes. Save the selection as a node set."
        self.status_lbl.setText(message)
        dialog.status_lbl.setText(message)

    # -------------- Expression selection ----------------
    def _apply_expression_to_selection(self):
        expr = self.expr_edit.text().strip()
        if not expr:
            QtWidgets.QMessageBox.information(
                self, "No expression", "Please enter an expression."
            )
            return
        data = self.parent().data
        mesh = data.mesh.copy()
        mesh.nodes = self._node_positions()[:, : data.mesh.nodes.shape[1]]
        if _is_multimesh(mesh):
            for submesh in mesh.submeshes:
                submesh.nodes = mesh.nodes
        try:
            ids = set(
                np.flatnonzero(
                    selection_mask(expr, mesh=mesh, data=data, data_type=self.entity)
                )
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Expression error", str(exc))
            return
        # Clamp to valid ids
        self.current_selection = ids
        self.update_selection(ids)
        self.status_lbl.setText(
            f"Selection set from expression ({len(ids)} {self.entity.lower()}s)."
        )
        # except Exception as e:
        #     QtWidgets.QMessageBox.warning(self, "Expression error", f"Failed to parse:\n{e}")

    @property
    def visible_elements(self):
        return self.parent().opts["element_set"]

    @visible_elements.setter
    def visible_elements(self, value):
        self.parent().opts["element_set"] = value

    @property
    def element_sets(self):
        return getattr(self.parent().active_dock, self.set_attribute)

    @element_sets.setter
    def element_sets(self, value):
        setattr(self.parent().active_dock, self.set_attribute, value)

    @property
    def unsaved_set_names(self):
        return getattr(
            self.parent().active_dock, "unsaved_" + self.set_attribute[:-1] + "_names"
        )

    @unsaved_set_names.setter
    def unsaved_set_names(self, value):
        setattr(
            self.parent().active_dock,
            "unsaved_" + self.set_attribute[:-1] + "_names",
            value,
        )

    # -------------- Visibility operations ----------------

    def _load_element_set(self, mode: str):
        if mode not in {"add", "remove", "replace"}:
            return
        name = self._selected_set_name()
        if not name:
            QtWidgets.QMessageBox.information(
                self, "No set selected", "Please select a set to rename."
            )
            return

        element_set = self.element_sets[name]
        if element_set is None:
            element_set = self.all_elements()
        else:
            element_set = set(element_set)

        if mode == "add":
            self.current_selection |= element_set
        elif mode == "remove":
            self.current_selection -= element_set
        elif mode == "replace":
            self.current_selection = element_set

        self.update_selection(self.current_selection)
        self.status_lbl.setText(
            f"{self.entity} set loaded with {len(self.current_selection)} {self.entity.lower()}s"
        )

    def _apply_visibility(self, mode: str):
        if mode not in {"all", "show", "hide"}:
            return
        self.visibility_mode = mode
        if mode == "all":
            self.visible_elements = None
            self.visibility_mode = None
            self.status_lbl.setText("All elements visible.")
        elif mode == "show":
            self.visible_elements = list(self.current_selection)
            self.status_lbl.setText("Show selected elements.")
        elif mode == "hide":
            self.visible_elements = list(self.all_elements() - self.current_selection)
            self.status_lbl.setText("Hide selected elements.")

        self.update_plot()

    def _inverse_visible(self):
        self.visible_elements = list(self.all_elements() - set(self.visible_elements))
        self.update_plot()
        self.status_lbl.setText(
            f"Visibility inverted. Visible={len(self.visible_elements)}"
        )

    # -------------- Element set management ----------------
    def _rename_selected_set(self):
        name = self._selected_set_name()
        if not name:
            QtWidgets.QMessageBox.information(
                self, "No set selected", "Please select a set to rename."
            )
            return
        if name in self.unsaved_set_names:
            QtWidgets.QMessageBox.information(
                self,
                "Built-in set",
                f"Built-in {self.entity.lower()} sets cannot be renamed.",
            )
            return
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Rename Set", f"New name for '{name}':", text=name
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name in self.element_sets and new_name != name:
            QtWidgets.QMessageBox.warning(
                self, "Name exists", f"A set named '{new_name}' already exists."
            )
            return
        self.element_sets[new_name] = self.element_sets.pop(name)
        self._refresh_sets_list()
        self.status_lbl.setText(f"Renamed set '{name}' → '{new_name}'")

    def _remove_selected_set(self):
        name = self._selected_set_name()
        if not name:
            QtWidgets.QMessageBox.information(
                self, "No set selected", "Please select a set to rename."
            )
            return
        if name in self.unsaved_set_names:
            QtWidgets.QMessageBox.information(
                self,
                "Built-in set",
                f"Built-in {self.entity.lower()} sets cannot be removed.",
            )
            return
        del self.element_sets[name]
        self._refresh_sets_list()
        self.status_lbl.setText(f"Set '{name}' removed")

    def _save_selection_as_new_set(self):
        new_name, ok = QtWidgets.QInputDialog.getText(
            self, "Save Selection as Set", "Set name:"
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name in self.element_sets:
            QtWidgets.QMessageBox.warning(
                self, "Name exists", f"A set named '{new_name}' already exists."
            )
            return
        if new_name in self.unsaved_set_names:
            QtWidgets.QMessageBox.warning(
                self, "Reserved name", f"'{new_name}' is a built-in set name."
            )
            return
        if self.current_selection:
            self.element_sets[new_name] = list(self.current_selection)
        else:
            QtWidgets.QMessageBox.warning(
                self, "Empty selection", "Select at least one item."
            )
            return
        self._refresh_sets_list()
        # self._refresh_sets_combo()
        self.status_lbl.setText(f"Saved selection as set '{new_name}'")

    def closeEvent(self, event):
        if self.parent().active_dock is not None:
            self._cancel_surface_pick()
            self._stop_picking()
            self.plotter.remove_actor(
                "_picked_node_set" if self.entity == "Node" else "_picked_cell"
            )
        super().closeEvent(event)

    def _convert_selected_set(self):
        name = self.conversion_sources.currentText()
        if not name:
            QtWidgets.QMessageBox.information(
                self, "No set selected", "Choose a source set from the dropdown."
            )
            return
        dock = self.parent().active_dock
        mesh = dock.data.mesh
        sources = dock.node_sets if self.entity == "Element" else dock.element_sets
        count = mesh.n_nodes if self.entity == "Element" else mesh.n_elements
        values = (
            np.arange(count)
            if sources[name] is None
            else np.asarray(sources[name], dtype=int)
        )
        indices = (
            nodes_to_elements(mesh, values, self.conversion_all_nodes.isChecked())
            if self.entity == "Element"
            else elements_to_nodes(mesh, values)
        )
        new_name, accepted = QtWidgets.QInputDialog.getText(
            self,
            f"Create {self.entity.lower()} set",
            "New set name:",
            text=name,
        )
        if not accepted or not new_name.strip():
            return
        new_name = new_name.strip()
        self.parent()._sync_viewer_sets(self.parent().active_dock)
        sets = self.element_sets
        if new_name in sets:
            QtWidgets.QMessageBox.warning(
                self, "Name exists", f"Set '{new_name}' already exists."
            )
            return
        sets[new_name] = indices
        self.parent()._sync_viewer_sets(self.parent().active_dock)
        self._refresh_sets_list()
        self.status_lbl.setText(
            f"Created {self.entity.lower()} set '{new_name}' ({len(indices)} IDs)."
        )


class NodeSetsDialog(ElementVisibilityDialog):
    """Node selection and named sets, without element visibility controls."""

    entity = "Node"
    set_attribute = "node_sets"

    def _default_element_sets(self):
        self.parent()._ensure_dock_node_sets(self.parent().active_dock)
        return (
            self.parent().active_dock.node_sets,
            self.parent().active_dock.unsaved_node_set_names,
        )


def _workspace_problems():
    """Discover live Problems in this process without evaluating user code."""
    found = {}

    def add(label, problem):
        if _is_problem(problem):
            entry = found.setdefault(id(problem), [problem, set()])
            entry[1].add(str(label))

    main = sys.modules.get("__main__")
    namespaces = [vars(main)] if main is not None else []
    ipython = sys.modules.get("IPython")
    shell = ipython.get_ipython() if ipython is not None else None
    if shell is not None:
        namespaces.append(shell.user_ns)
    for namespace in namespaces:
        for name, value in list(namespace.items()):
            if not name.startswith("_"):
                add(name, value)
    for name, problem in list(ProblemBase.get_all().items()):
        add(name, problem)
    active = ProblemBase.get_active()
    if _is_problem(active) and id(active) not in found:
        add(active.name or "Active problem", active)
    return sorted(
        [(" / ".join(sorted(labels)), problem) for problem, labels in found.values()],
        key=lambda item: item[0].casefold(),
    )


class OpenProblemDialog(QtWidgets.QDialog):
    """Choose a live Problem from the current interpreter workspace."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Open existing problem")
        self.resize(650, 320)
        layout = QVBoxLayout(self)
        description = QLabel(
            "Choose a Problem from this Python session (workspace variables, "
            "named Fedoo problems, or the active problem). Opening shows its "
            "current state without solving."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Variable / name", "Type", "Nodes", "Elements"]
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )
        self.table.itemDoubleClicked.connect(self.open_selected)
        self.table.itemSelectionChanged.connect(self._update_open_button)
        layout.addWidget(self.table)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.refresh)
        buttons.addWidget(refresh)
        buttons.addStretch()
        self.open_button = QPushButton("Open")
        self.open_button.setDefault(True)
        self.open_button.clicked.connect(self.open_selected)
        buttons.addWidget(self.open_button)
        close = QPushButton("Cancel")
        close.clicked.connect(self.reject)
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.problems = []
        self.open_button.setEnabled(False)

    def refresh(self):
        previous = self.selected_problem()
        self.problems = _workspace_problems()
        # Keep unnamed problems already open here discoverable too.
        for dock in self.parent().all_docks:
            if isinstance(dock.data, ProblemDataSet) and not any(
                problem is dock.data.problem for _, problem in self.problems
            ):
                self.problems.append((dock.title, dock.data.problem))
        blocker = QSignalBlocker(self.table)
        self.table.setRowCount(len(self.problems))
        selected = 0
        for row, (label, problem) in enumerate(self.problems):
            mesh = getattr(problem, "mesh", None)
            for column, value in enumerate(
                (
                    label,
                    type(problem).__name__,
                    getattr(mesh, "n_nodes", "—"),
                    getattr(mesh, "n_elements", "—"),
                )
            ):
                self.table.setItem(row, column, QtWidgets.QTableWidgetItem(str(value)))
            if problem is previous or (
                previous is None and problem is ProblemBase.get_active()
            ):
                selected = row
        if self.problems:
            self.table.selectRow(selected)
        del blocker
        self._update_open_button()
        self.status.setText(
            ""
            if self.problems
            else "No Problem objects found. Create a Problem in this Python session, then click Refresh."
        )

    def selected_problem(self):
        row = self.table.currentRow()
        return self.problems[row][1] if 0 <= row < len(self.problems) else None

    def _update_open_button(self):
        self.open_button.setEnabled(self.selected_problem() is not None)

    def open_selected(self, *args):
        row = self.table.currentRow()
        if not 0 <= row < len(self.problems):
            return
        title, problem = self.problems[row]
        try:
            self.parent().open_workspace_problem(problem, title)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Cannot open problem", str(exc))
            return
        self.accept()


class BoundaryConditionsDialog(QtWidgets.QDialog):
    """Inspect all BCs and edit standard conditions on a live Problem."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Boundary conditions")
        self.resize(850, 580)
        self.records = []
        self.editing = None
        self._building_form = False
        layout = QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(
            ["Name", "Type", "Variable", "Set / IDs", "Value"]
        )
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.itemSelectionChanged.connect(self._selected)
        layout.addWidget(self.table)
        self.details = QtWidgets.QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumHeight(100)
        layout.addWidget(self.details)
        form = QtWidgets.QFormLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.type_combo = QtWidgets.QComboBox()
        self.type_combo.addItems(["Dirichlet", "Neumann"])
        self.variable_combo = QtWidgets.QComboBox()
        self.variable_combo.currentTextChanged.connect(self._refresh_targets)
        self.target_combo = QtWidgets.QComboBox()
        self.value_edit = QtWidgets.QLineEdit("0")
        self.value_edit.setPlaceholderText(
            "Constant or array expression; blank preserves an existing array"
        )
        for label, widget in (
            ("Name:", self.name_edit),
            ("Type:", self.type_combo),
            ("Variable / component:", self.variable_combo),
            ("Apply to:", self.target_combo),
            ("Value:", self.value_edit),
        ):
            form.addRow(label, widget)
        layout.addLayout(form)
        explanation = QLabel(
            "Element sets select all of their nodes. Neumann values are nodal forces, "
            "not surface pressures. Existing time functions and start values are preserved "
            "when editing. Custom constraints are inspectable; the editor handles "
            "Dirichlet/Neumann conditions. Changes affect the live problem; solve it "
            "from Python to recompute the solution."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        buttons = QHBoxLayout()
        self.new_button = QPushButton("New")
        self.new_button.clicked.connect(self.new_condition)
        self.apply_button = QPushButton("Add condition")
        self.apply_button.clicked.connect(self.apply_condition)
        self.highlight_button = QPushButton("Show associated nodes")
        self.highlight_button.clicked.connect(self.highlight_selected)
        self.remove_button = QPushButton("Remove selected")
        self.remove_button.clicked.connect(self.remove_selected)
        for button in (
            self.new_button,
            self.apply_button,
            self.highlight_button,
            self.remove_button,
        ):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.status = QLabel()
        layout.addWidget(self.status)
        self.refresh()

    @property
    def problem(self):
        data = self.parent().data if self.parent().active_dock else None
        return data.problem if isinstance(data, ProblemDataSet) else None

    def refresh(self):
        problem = self.problem
        self.records = []
        self.editing = None
        blocker = QSignalBlocker(self.table)
        self.table.setRowCount(0)
        self._building_form = True
        self.variable_combo.clear()
        if problem is not None:
            self.parent()._sync_viewer_sets(self.parent().active_dock)
            self.variable_combo.addItems(
                list(problem.space.list_variables())
                + list(problem.global_dof._variable)
            )

            def collect(boundaries):
                for container in (boundaries.data, boundaries.data_end):
                    for index, condition in enumerate(container):
                        if isinstance(condition, ListBC):
                            collect(condition)
                        else:
                            self.records.append((container, index, condition))

            collect(problem.bc)
            for _, _, condition in self.records:
                row = self.table.rowCount()
                self.table.insertRow(row)
                target = getattr(condition, "_viewer_set_reference", None)
                source = getattr(
                    condition, "node_set_name", getattr(condition, "node_set", "—")
                )
                source = f"{target[0]}: {target[1]}" if target else str(source)
                values = (
                    condition.name,
                    getattr(condition, "bc_type", type(condition).__name__),
                    str(getattr(condition, "variable_name", "—")),
                    source,
                    str(getattr(condition, "value", "—")),
                )
                for column, text in enumerate(values):
                    self.table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        self._building_form = False
        del blocker
        self._refresh_targets()
        self.new_condition()
        self.setEnabled(problem is not None)

    def _refresh_targets(self, *args):
        if self._building_form:
            return
        self.target_combo.clear()
        problem = self.problem
        if problem is None:
            return
        variable = self.variable_combo.currentText()
        if variable in problem.global_dof._variable:
            count = len(problem.global_dof[variable])
            self.target_combo.addItem("All global DOF IDs", ("IDs", list(range(count))))
            for index in range(count):
                self.target_combo.addItem(f"Global DOF ID {index}", ("IDs", [index]))
            return
        dock = self.parent().active_dock
        for label, attribute in (("Node", "node_sets"), ("Element", "element_sets")):
            for name in sorted(getattr(dock, attribute)):
                self.target_combo.addItem(f"{label} set: {name}", (label, name))

    def new_condition(self):
        self.editing = None
        self.table.clearSelection()
        self.name_edit.clear()
        self.value_edit.setText("0")
        self.apply_button.setText("Add condition")
        self.apply_button.setEnabled(self.problem is not None)
        self.highlight_button.setEnabled(False)
        self.remove_button.setEnabled(False)
        self.details.clear()

    def _selected(self):
        selected = self.table.selectionModel().selectedRows()
        if len(selected) != 1:
            return
        record = self.records[selected[0].row()]
        condition = record[2]
        self.editing = record
        self.details.setPlainText(str(condition))
        self.name_edit.setText(condition.name)
        self.apply_button.setText("Update condition")
        editable = isinstance(condition, BoundaryCondition)
        self.apply_button.setEnabled(editable)
        self.highlight_button.setEnabled(True)
        self.remove_button.setEnabled(editable)
        if not editable:
            self.status.setText(
                "Custom constraint: inspect its details and associated nodes."
            )
            return
        self.type_combo.setCurrentText(condition.bc_type)
        self.variable_combo.setCurrentText(condition.variable_name)
        self._refresh_targets()
        reference = getattr(condition, "_viewer_set_reference", None)
        if reference is None:
            source = condition.node_set_name
            reference = (
                ("Node", source)
                if isinstance(source, str)
                else ("IDs", list(np.asarray(source, dtype=int)))
            )
        found = -1
        for index in range(self.target_combo.count()):
            candidate = self.target_combo.itemData(index)
            if candidate[0] == reference[0] and np.array_equal(
                candidate[1], reference[1]
            ):
                found = index
                break
        if found < 0:
            self.target_combo.addItem("Current condition IDs", reference)
            found = self.target_combo.count() - 1
        self.target_combo.setCurrentIndex(found)
        self.value_edit.setText(
            repr(float(condition.value)) if np.ndim(condition.value) == 0 else ""
        )
        self.status.setText("")

    def _target_nodes(self, reference):
        kind, target = reference
        dock = self.parent().active_dock
        if kind == "IDs":
            return np.asarray(target, dtype=int)
        sets = dock.node_sets if kind == "Node" else dock.element_sets
        values = sets[target]
        count = dock.data.mesh.n_nodes if kind == "Node" else dock.data.mesh.n_elements
        ids = np.arange(count) if values is None else np.asarray(values, dtype=int)
        return ids if kind == "Node" else elements_to_nodes(dock.data.mesh, ids)

    def apply_condition(self):
        problem = self.problem
        if problem is None:
            return
        try:
            reference = self.target_combo.currentData()
            if reference is None:
                raise ValueError("Select a node set, element set or global DOF ID")
            indices = self._target_nodes(reference)
            if not len(indices):
                raise ValueError("The target set is empty")
            variable = self.variable_combo.currentText()
            if (
                variable not in problem.space.list_variables()
                and variable not in problem.global_dof._variable
            ):
                raise ValueError("Select a problem variable")
            old = self.editing[2] if self.editing else None
            expression = self.value_edit.text().strip()
            if not expression and old is not None:
                value = old.value
            else:
                value = np.asarray(
                    evaluate_expression(expression, data=self.parent().data),
                    dtype=float,
                )
                if (
                    value.ndim > 1
                    or value.size not in (1, len(indices))
                    or not np.all(np.isfinite(value))
                ):
                    raise ValueError(
                        "Value must be finite: one number or one value per target ID"
                    )
                value = value.item() if value.ndim == 0 else value
            node_set = (
                reference[1]
                if reference[0] == "Node" and reference[1] in problem.mesh.node_sets
                else indices
            )
            condition = BoundaryCondition(
                self.type_combo.currentText(),
                node_set,
                variable,
                value,
                time_func=old.time_func if old is not None else None,
                start_value=old.start_value if old is not None else None,
                name=self.name_edit.text().strip(),
            )
            condition._viewer_set_reference = reference
            if old is None:
                problem.bc.add(condition)
            else:
                condition.initialize(problem)
                condition._start_value_default = old._start_value_default
                self.editing[0][self.editing[1]] = condition
            problem.bc._checked_signature = None
            problem.invalidate_factorization()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Boundary condition error", str(exc))
            return
        self.refresh()
        self.status.setText(
            "Problem boundary conditions updated. Solve from Python to update the solution."
        )

    def highlight_selected(self):
        if self.editing is None:
            return
        condition = self.editing[2]
        variable = getattr(condition, "variable_name", None)
        if variable in self.problem.global_dof._variable:
            self.status.setText("Global DOF IDs have no mesh-node association.")
            return
        nodes = getattr(condition, "node_set", None)
        if nodes is None:
            assembly = getattr(condition, "assembly", None)
            nodes = np.unique(assembly.mesh.elements) if assembly is not None else None
        if nodes is None:
            self.status.setText("This custom constraint does not expose a node set.")
            return
        nodes = np.unique(np.asarray(nodes, dtype=int))
        nodes = nodes[(nodes >= 0) & (nodes < self.problem.mesh.n_nodes)]
        self.parent().open_node_sets_dialog()
        self.parent().node_sets_dialog.update_selection(set(nodes))

    def remove_selected(self):
        if self.editing is None:
            return
        del self.editing[0][self.editing[1]]
        self.problem.bc._checked_signature = None
        self.problem.invalidate_factorization()
        self.refresh()


class ScalarExpressionDialog(QtWidgets.QDialog):
    """Define scalar results evaluated at each frame from a shared expression."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Compute scalar")
        self.resize(520, 360)
        self._help_dialog = None
        layout = QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        self.name_edit = QtWidgets.QLineEdit()
        self.expression_edit = QtWidgets.QLineEdit()
        self.expression_edit.setPlaceholderText('sum(Disp_X[nodes("Top")])')
        self.data_type_combo = QtWidgets.QComboBox()
        self.data_type_combo.addItems(["Node", "Element"])
        form.addRow("Scalar name:", self.name_edit)
        expression_row = QHBoxLayout()
        expression_row.addWidget(self.expression_edit, 1)
        self.help_button = QtWidgets.QToolButton()
        self.help_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogHelpButton)
        )
        self.help_button.setToolTip("Expression help and examples")
        self.help_button.setAccessibleName("Expression help")
        self.help_button.setAutoRaise(True)
        self.help_button.clicked.connect(self.open_help)
        expression_row.addWidget(self.help_button)
        form.addRow("Expression:", expression_row)
        form.addRow("Field values at:", self.data_type_combo)
        layout.addLayout(form)
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Scalar", "Current value", "Expression"])
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.itemSelectionChanged.connect(self._load_selected)
        self.table.itemDoubleClicked.connect(self._insert_scalar)
        self.table.setToolTip("Double-click a scalar to insert it into the expression.")
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        apply_button = QPushButton("Compute / update scalar")
        apply_button.clicked.connect(self.compute)
        layout.addWidget(apply_button)
        self.status = QLabel()
        layout.addWidget(self.status)
        self.refresh()

    def open_help(self):
        if self._help_dialog is None:
            self._help_dialog = QtWidgets.QDialog(self)
            self._help_dialog.setWindowTitle("Compute scalar — Expression help")
            self._help_dialog.resize(520, 390)
            layout = QVBoxLayout(self._help_dialog)
            text = QtWidgets.QTextBrowser()
            text.setPlainText(
                "Examples\n\n"
                "CompressionForce * PlateShortening\n"
                'sum(Disp_X[nodes("Top")])\n'
                'mean(Stress_XX[elements("Wall")])\n'
                'sum(scalar("Fext(RigidDisp)")[2])\n\n'
                "Available values\n\n"
                "Use X, Y, Z, ID, field/component names (Stress_XX, Disp_X), "
                'node_sets["name"], element_sets["name"], and NumPy functions '
                "(sum, mean, min, max, sqrt, ...). Functions can also use the np. prefix.\n\n"
                "Field values at selects Node or Element values. Use Element for "
                'expressions indexed by elements("Wall"); use Node for nodes("Top").\n'
                'Use scalar("name") for scalar names containing punctuation.\n\n'
                "Expressions use reference positions and must return one finite number.\n\n"
                "Editing and saving\n\n"
                "Select a computed scalar to edit its definition. Double-click a scalar "
                "to insert it into the expression.\n\n"
                "Expressions update on frame changes and are exported as scalar histories "
                "by Save As. Their definitions are saved too, so they remain editable "
                "after reopening the FDH5 file."
            )
            layout.addWidget(text)
            close = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
            close.rejected.connect(self._help_dialog.close)
            layout.addWidget(close)
        self._help_dialog.show()
        self._help_dialog.raise_()
        self._help_dialog.activateWindow()

    def closeEvent(self, event):
        if self._help_dialog is not None:
            self._help_dialog.close()
        super().closeEvent(event)

    def refresh(self):
        data = self.parent().data
        blocker = QSignalBlocker(self.table)
        self.table.setRowCount(0)
        for name, value in sorted(data.scalar_data.items()):
            row = self.table.rowCount()
            self.table.insertRow(row)
            definition = data.scalar_expressions.get(name, {})
            for column, text in enumerate(
                (name, str(value), definition.get("expression", ""))
            ):
                self.table.setItem(row, column, QtWidgets.QTableWidgetItem(text))
        del blocker

    def _load_selected(self):
        row = self.table.currentRow()
        if row < 0:
            return
        name = self.table.item(row, 0).text()
        definition = self.parent().data.scalar_expressions.get(name)
        if definition is not None:
            self.name_edit.setText(name)
            self.expression_edit.setText(definition["expression"])
            self.data_type_combo.setCurrentText(definition["data_type"])

    def _insert_scalar(self, item):
        name = self.table.item(item.row(), 0).text()
        self.expression_edit.insert(f"scalar({name!r})")

    def compute(self):
        main = self.parent()
        main._sync_viewer_sets(main.active_dock)
        try:
            value = main.data.add_scalar_expression(
                self.name_edit.text(),
                self.expression_edit.text(),
                self.data_type_combo.currentText(),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Scalar expression error", str(exc))
            return
        self.status.setText(f"{self.name_edit.text().strip()} = {value}")
        self.refresh()
        if main._history_plot_dialog is not None:
            main._history_plot_dialog.update_fields()


class MplLinePlotDialog(QtWidgets.QDialog):
    """Modeless dialog embedding a Matplotlib figure to show plot_over_line results."""

    requestSave = Signal()  # optional external hook

    def __init__(self, parent=None, title="Plot over line"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(False)

        # --- Matplotlib figure/canvas ---
        self.figure = Figure(constrained_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        # --- Toolbar ---
        self.toolbar = NavigationToolbar(self.canvas, self)

        # --- Header with line info + save ---
        self.lblP1 = QtWidgets.QLabel("P1: (—, —, —)")
        self.lblP2 = QtWidgets.QLabel("P2: (—, —, —)")
        # self.saveBtn = QtWidgets.QPushButton("Save as…")
        # self.saveBtn.clicked.connect(self._save_as)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(self.lblP1)
        header.addSpacing(10)
        header.addWidget(self.lblP2)
        header.addStretch(1)
        # header.addWidget(self.saveBtn)

        # --- Main layout ---
        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        # Internal state
        self._line_plot = None

    def update_curve(self, x, y, p1=None, p2=None, ylabel="Scalar", xlabel="Distance"):
        """
        Update the embedded plot with new data.
        x: 1D sequence (distance)
        y: 1D sequence (scalar values)
        p1, p2: optional tuples for UI display
        """
        self.ax.clear()
        (self._line_plot,) = self.ax.plot(x, y, lw=1.5)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_title("Plot over line")

        if p1 is not None:
            self.lblP1.setText(f"P1: ({p1[0]:.3f}, {p1[1]:.3f}, {p1[2]:.3f})")
        if p2 is not None:
            self.lblP2.setText(f"P2: ({p2[0]:.3f}, {p2[1]:.3f}, {p2[2]:.3f})")

        self.canvas.draw_idle()

    # def _save_as(self):
    #     """Save the current figure to file."""
    #     dlg = QtWidgets.QFileDialog(
    #         self, "Save plot", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
    #     )
    #     dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
    #     if dlg.exec_():  # .exec() if PySide6
    #         path = dlg.selectedFiles()[0]
    #         # Choose DPI depending on format (optional)
    #         dpi = 150 if path.lower().endswith(".png") else None
    #         self.figure.savefig(path, dpi=dpi, bbox_inches="tight")


# --- New Dialog for History Plot ---


class HistoryPlotDialog(QtWidgets.QDialog):
    """
    Dialog for selecting and configuring history plot parameters.
    Opens the plot in a new modeless Matplotlib window (like MplLinePlotDialog) when plotting.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("History Plot")
        self.setModal(False)
        self.resize(600, 500)

        layout = QtWidgets.QVBoxLayout(self)

        # --- QLineEdit for field_component filter ---
        self.filter_edit = QtWidgets.QLineEdit()
        self.filter_edit.setPlaceholderText(
            "Type field_component (e.g. Stress_XY) or field for all components..."
        )
        layout.addWidget(self.filter_edit)

        # Controls for field, component, data type, id
        form_layout = QtWidgets.QFormLayout()
        self.field_combo = QtWidgets.QComboBox()
        self.comp_combo = QtWidgets.QComboBox()
        self.data_type_combo = QtWidgets.QComboBox()
        self.data_type_combo.addItems(["Node", "GaussPoint", "Element"])
        self.id_spin = QtWidgets.QSpinBox()
        self.id_spin.setMinimum(0)
        self.id_spin.setMaximum(parent.active_dock.data.mesh.n_nodes)

        # Mesh pick button
        self.pick_mesh_btn = QtWidgets.QPushButton("Pick from mesh")

        form_layout.addRow("Field:", self.field_combo)
        form_layout.addRow("Component:", self.comp_combo)
        form_layout.addRow("Data type:", self.data_type_combo)
        # Put id_spin and pick_mesh_btn on the same row
        id_row = QtWidgets.QHBoxLayout()
        id_row.addWidget(self.id_spin)
        id_row.addWidget(self.pick_mesh_btn)
        self.id_label = QtWidgets.QLabel("Cell/Node ID:")
        form_layout.addRow(self.id_label, id_row)

        layout.addLayout(form_layout)

        # Axis selection

        axis_layout = QtWidgets.QHBoxLayout()
        self.x_axis_btn = QtWidgets.QPushButton("Set as X axis")
        axis_layout.addWidget(self.x_axis_btn)
        layout.addLayout(axis_layout)

        # Add Y data button below X axis
        self.add_y_btn = QtWidgets.QPushButton("Add Y data")
        layout.addWidget(self.add_y_btn)

        # --- X info zone ---
        self.x_info = QtWidgets.QTextEdit()
        self.x_info.setReadOnly(True)
        self.x_info.setMaximumHeight(40)
        layout.addWidget(QtWidgets.QLabel("Current X data:"))
        layout.addWidget(self.x_info)

        # --- Y data table ---
        y_table_layout = QtWidgets.QHBoxLayout()
        self.y_table = QtWidgets.QTableWidget(0, 5)
        self.y_table.setHorizontalHeaderLabels(
            ["Field", "Component", "Data type", "ID", "Label"]
        )
        self.y_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.y_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.y_table.setMaximumHeight(120)
        y_table_layout.addWidget(self.y_table)
        # Add/Remove buttons
        y_btns = QtWidgets.QVBoxLayout()
        self.remove_y_btn = QtWidgets.QPushButton("Remove selected")
        self.modify_y_btn = QtWidgets.QPushButton("Modify selected")
        self.modify_y_btn.setEnabled(False)
        y_btns.addWidget(self.remove_y_btn)
        y_btns.addWidget(self.modify_y_btn)
        y_btns.addStretch(1)
        y_table_layout.addLayout(y_btns)
        layout.addWidget(QtWidgets.QLabel("Y data to plot (multiple allowed):"))
        layout.addLayout(y_table_layout)

        # Button to plot
        self.plot_btn = QtWidgets.QPushButton("Plot History")
        layout.addWidget(self.plot_btn)

        # Internal state for axis selection
        self.x_data_info = None  # tuple: (field, comp, data_type, id)
        self.y_data_list = []  # list of dicts: {field, comp, data_type, id, label, data}

        # Autocomplete setup
        self.completion_model = QtGui.QStandardItemModel()
        self.completer = QtWidgets.QCompleter(self.completion_model, self)
        self.completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.filter_edit.setCompleter(self.completer)

        # Connect signals
        self.plot_btn.clicked.connect(self.plot_history)
        self.x_axis_btn.clicked.connect(self.set_x_axis)
        self.add_y_btn.clicked.connect(self.add_y_data)
        self.remove_y_btn.clicked.connect(self.remove_selected_y_data)
        self.field_combo.currentTextChanged.connect(self.on_field_combo_changed)
        self.data_type_combo.currentTextChanged.connect(self.update_data_type)

        self.filter_edit.textChanged.connect(self.on_filter_edit)
        self.comp_combo.currentTextChanged.connect(self.on_comp_combo_changed)

        self.pick_mesh_btn.clicked.connect(self.on_pick_mesh)

        self.y_table.itemSelectionChanged.connect(self.on_y_table_row_selected)
        self.modify_y_btn.clicked.connect(self.on_modify_y_data)

        # Populate combos from parent data
        self.parent = parent
        self.update_fields()

        # Set 'Time' as default X axis if it exists
        data = getattr(self.parent, "data", None)
        if (
            data is not None
            and hasattr(data, "scalar_data")
            and "Time" in data.scalar_data
        ):
            self.field_combo.setCurrentText("Time")
            self.comp_combo.setCurrentText("0")
            self.data_type_combo.setCurrentText("Node")
            self.id_spin.setValue(0)
            self.update_x_info("Time", "0", "Node", 0)

        # Track plot dialog
        self._plot_dialog = None

    def on_field_combo_changed(self, field):
        data = getattr(self.parent, "data", None)
        if data is not None and field in data.scalar_data:
            self.data_type_combo.setEnabled(False)
            self.data_type_combo.setToolTip(
                "Global values have no mesh association; data type is ignored."
            )
            self.pick_mesh_btn.setEnabled(False)
        else:
            self.data_type_combo.setEnabled(True)
            self.data_type_combo.setToolTip("")
            self.id_spin.setEnabled(True)
            self.pick_mesh_btn.setEnabled(True)

        self.update_components(field)
        self.update_data_type(self.data_type_combo.currentText())

    def on_y_table_row_selected(self):
        selected = self.y_table.selectionModel().selectedRows()
        if len(selected) == 1:
            idx = selected[0].row()
            entry = self.y_data_list[idx]
            # Update controls
            self.field_combo.setCurrentText(
                _viewer_field_name(self.parent.data, entry["field"])
            )
            self.comp_combo.setCurrentText(str(entry["comp"]))
            self.data_type_combo.setCurrentText(entry["data_type"])
            self.id_spin.setValue(entry["id"])
            self.modify_y_btn.setEnabled(True)
        else:
            self.modify_y_btn.setEnabled(False)

    def on_modify_y_data(self):
        selected = self.y_table.selectionModel().selectedRows()
        if len(selected) != 1:
            return
        idx = selected[0].row()
        # Update the entry with current controls
        field = self.field_combo.currentText()
        comp = self.comp_combo.currentText()
        dtype = self.data_type_combo.currentText()
        id_val = self.id_spin.value()
        label = f"{field}_{comp} ({dtype}, ID={id_val})"
        # data = self.extract_history()
        # if data is None:
        #     QtWidgets.QMessageBox.warning(self, "No data", "Could not extract Y data.")
        #     return
        entry = {
            "field": field,
            "comp": comp,
            "data_type": dtype,
            "id": id_val,
            "label": label,
        }
        self.y_data_list[idx] = entry
        self.y_table.setItem(idx, 0, QtWidgets.QTableWidgetItem(field))
        self.y_table.setItem(idx, 1, QtWidgets.QTableWidgetItem(str(comp)))
        self.y_table.setItem(idx, 2, QtWidgets.QTableWidgetItem(dtype))
        self.y_table.setItem(idx, 3, QtWidgets.QTableWidgetItem(str(id_val)))
        self.y_table.setItem(idx, 4, QtWidgets.QTableWidgetItem(label))

    def on_pick_mesh(self):
        # Determine pick type from data_type_combo
        dtype = self.data_type_combo.currentText()
        mainwin = self.parent
        if dtype == "Node":
            self._start_pick_node(mainwin)
        else:
            self._start_pick_element(mainwin)

    def _start_pick_node(self, mainwin):
        def picked(node_id):
            self.id_spin.setValue(node_id)
            mainwin.plotter.disable_picking()

        mainwin._pick_node(on_pick=picked)

    def _start_pick_element(self, mainwin):
        def picked(element_id):
            self.id_spin.setValue(element_id)
            mainwin._stop_element_picking()

        mainwin._pick_element(on_pick=picked)

    def update_fields(self):
        data = getattr(self.parent, "data", None)
        if data is not None:
            self.field_combo.clear()
            self.field_combo.addItems(_viewer_field_names(data))
            self.field_combo.addItems(data.scalar_data.keys())
            self.update_components(self.field_combo.currentText())
            self.update_completer()

    def update_components(self, field):
        data = getattr(self.parent, "data", None)
        self.comp_combo.blockSignals(True)
        self.comp_combo.clear()
        if not field or data is None:
            self.comp_combo.blockSignals(False)
            return
        # Use get_components from parent
        comps = []
        if hasattr(self.parent, "get_components"):
            comps = self.parent.get_components(field)
        if comps:
            self.comp_combo.addItems(comps)
            self.comp_combo.addItem("All components")
        else:
            self.comp_combo.addItem("0")
            self.comp_combo.addItem("All components")
        self.comp_combo.blockSignals(False)

    def update_data_type(self, data_type):
        data = self.parent.data
        field = self.field_combo.currentText()
        if field in data.scalar_data:
            values = np.asarray(data.scalar_data[field])
            count = values.shape[-1] if values.ndim else 1
            self.id_label.setText("ID:")
            self.id_spin.setToolTip(
                "Index in the global values; unrelated to mesh nodes or elements."
            )
            self.id_spin.setEnabled(count > 1)
        elif data_type == "Node":
            count = data.mesh.n_nodes
            self.id_label.setText("Cell/Node ID:")
            self.id_spin.setToolTip("")
            self.id_spin.setEnabled(True)
        else:
            count = data.mesh.n_elements
            self.id_label.setText("Cell/Node ID:")
            self.id_spin.setToolTip("")
            self.id_spin.setEnabled(True)
        self.id_spin.setMaximum(max(0, count - 1))

    def update_completer(self):
        # Build list of field_component strings for autocompletion
        data = getattr(self.parent, "data", None)
        items = []
        if data is not None:
            for field in _viewer_field_names(data):
                items.append(field)  # for 'All components'
                comps = []
                if hasattr(self.parent, "get_components"):
                    comps = self.parent.get_components(field)
                if comps:
                    for comp in comps:
                        items.append(f"{field}_{comp}")
            for scalar in data.scalar_data:
                items.append(scalar)

        self.completion_model.clear()
        for item in items:
            self.completion_model.appendRow(QtGui.QStandardItem(item))

    def on_filter_edit(self, text):
        # Match whole names first, then the longest field prefix. Global DOF
        # names (including Fext(...)) can themselves contain underscores.
        names = [self.field_combo.itemText(i) for i in range(self.field_combo.count())]
        field, comp = text, "All components"
        for name in sorted(names, key=len, reverse=True):
            if text.casefold() == name.casefold():
                field = name
                break
        else:
            for name in sorted(names, key=len, reverse=True):
                if text.casefold().startswith(name.casefold() + "_"):
                    field, comp = name, text[len(name) + 1 :]
                    break
        idx_field = self.field_combo.findText(field, Qt.MatchFixedString)
        if idx_field >= 0:
            self.field_combo.setCurrentIndex(idx_field)
            field = self.field_combo.currentText()  # update to match str case
            self.update_components(field)
            idx_comp = self.comp_combo.findText(comp, Qt.MatchFixedString)
            if idx_comp >= 0:
                self.comp_combo.setCurrentIndex(idx_comp)

    def on_comp_combo_changed(self, comp):
        if comp == "All components":
            self.filter_edit.setText(self.field_combo.currentText())
        else:
            self.filter_edit.setText(f"{self.field_combo.currentText()}_{comp}")

    def set_x_axis(self):
        self.update_x_info(
            self.field_combo.currentText(),
            self.comp_combo.currentText(),
            self.data_type_combo.currentText(),
            self.id_spin.value(),
        )

    def update_x_info(self, field, comp, data_type, indice):
        data = getattr(self.parent, "data", None)
        if data is None:  # should never been triggered
            self.x_info.setText("Not set")
            return

        n_comps = len(self.parent.get_components(field))
        if n_comps > 1 and self.comp_combo.currentText() == "All components":
            QtWidgets.QMessageBox.warning(
                self,
                "No components defined",
                "A component must be selected for x axis data.",
            )
            return

        if field in data.scalar_data:
            label = field if n_comps <= 1 else f"{field}_{comp}"
            if np.ndim(data.scalar_data[field]):
                label += f" (ID={indice})"
            self.x_info.setText(label)
        else:
            if n_comps <= 1:
                self.x_info.setText(f"{field} ({data_type}, ID={indice})")
            else:
                self.x_info.setText(f"{field}_{comp} ({data_type}, ID={indice})")
        resolved = _viewer_tensor_field(
            data, field, getattr(self.parent.active_dock, "beam_csys", "local")
        )
        self.x_data_info = (resolved, comp, data_type, indice)

    def check_valide(self, field, comp, data_type, indice):
        data = getattr(self.parent, "data", None)
        if data is None:
            return False
        if field in data.scalar_data:
            values = np.asarray(data.scalar_data[field])
            return 0 <= indice < (values.shape[-1] if values.ndim else 1)
        else:
            if field not in data.field_names():
                return False
            if comp not in self.parent.get_components(field):
                return False
            if data_type == "Node":
                if indice >= data.mesh.n_nodes:
                    return False
            elif data_type == "Element":
                if indice >= data.mesh.n_elements:
                    return False
            else:  # GaussPoint
                gp_data = data.get_data(field, comp, "GaussPoint")
                gp_data = _global_data_array(gp_data)
                if indice >= gp_data.shape[-1]:
                    return False
        return True

    def _on_change_active(self):
        # update the maximal value
        self.update_data_type(self.data_type_combo.currentText())
        if self.x_data_info is not None:
            if not self.check_valide(*self.x_data_info):
                self.x_data_info = None
                self.x_info.setText("Not set")
        for i, ydat in enumerate(self.y_data_list):
            if not self.check_valide(
                ydat["field"], ydat["comp"], ydat["data_type"], ydat["id"]
            ):
                self.y_table.removeRow(i)
                del self.y_data_list[i]
        self.update_fields()

    def add_y_data(self, **kargs):
        indice = kargs.get("indice", None)
        # Get current selection
        field = self.field_combo.currentText()
        comp = self.comp_combo.currentText()
        dtype = self.data_type_combo.currentText()
        if indice is None:
            idx = self.id_spin.value()
        else:
            idx = indice
        data = getattr(self.parent, "data", None)
        if data is None:
            return
        resolved = _viewer_tensor_field(
            data, field, getattr(self.parent.active_dock, "beam_csys", "local")
        )
        if field in data.scalar_data:
            label = f"{field}_{comp}"
            if np.ndim(data.scalar_data[field]):
                label += f" (ID={idx})"
        else:
            if indice is None and dtype == "GaussPoint":
                # add all gauss points values
                for gp_id in _gausspoint_global_indices_for_element(
                    data, resolved, comp, idx
                ):
                    self.add_y_data(indice=gp_id)
                return
            label = f"{field}_{comp} ({dtype}, ID={idx})"
        # Add to list and table
        entry = {
            "field": resolved,
            "comp": comp,
            "data_type": dtype,
            "id": idx,
            "label": label,
        }
        self.y_data_list.append(entry)
        row = self.y_table.rowCount()
        self.y_table.insertRow(row)
        self.y_table.setItem(row, 0, QtWidgets.QTableWidgetItem(field))
        self.y_table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(comp)))
        self.y_table.setItem(row, 2, QtWidgets.QTableWidgetItem(dtype))
        self.y_table.setItem(row, 3, QtWidgets.QTableWidgetItem(str(idx)))
        self.y_table.setItem(row, 4, QtWidgets.QTableWidgetItem(label))

    def remove_selected_y_data(self):
        selected = self.y_table.selectionModel().selectedRows()
        for idx in sorted([s.row() for s in selected], reverse=True):
            self.y_table.removeRow(idx)
            del self.y_data_list[idx]

    def plot_history(self):
        if not self.y_data_list:
            QtWidgets.QMessageBox.warning(
                self, "No Y data", "Add at least one Y data to plot."
            )
            return
        if self.x_data_info is None:
            QtWidgets.QMessageBox.warning(self, "No X data", "Set X axis data first.")
            return
        data = getattr(self.parent, "data", None)
        if data is None or not hasattr(data, "loaded_iter"):
            QtWidgets.QMessageBox.warning(
                self, "No history data", "The active data have no time evolution."
            )
            return

        if self._plot_dialog is not None:
            self._plot_dialog.close()
        self._plot_dialog = MplLinePlotDialog(self, title="History Plot")
        ax = self._plot_dialog.ax
        ax.clear()

        list_comps = [self.x_data_info[1]] + [
            entry["comp"] for entry in self.y_data_list
        ]
        list_comps = [
            int(comp) if comp.isdigit() else None if comp == "All components" else comp
            for comp in list_comps
        ]

        list_data = data.get_history(
            [self.x_data_info[0]] + [entry["field"] for entry in self.y_data_list],
            [self.x_data_info[3]] + [entry["id"] for entry in self.y_data_list],
            list_comps,
            [self.x_data_info[2]] + [entry["data_type"] for entry in self.y_data_list],
            return_list=True,
        )

        # Only show label info for non-scalar_data
        x_data = list_data[0]
        for i, entry in enumerate(self.y_data_list):
            ydata = list_data[i + 1]
            if (
                self.y_data_list[i]["comp"] == "All components"
                and hasattr(ydata, "ndim")
                and ydata.ndim == 2
            ):
                # Plot each component separately
                n_comp = ydata.shape[1]
                for j in range(n_comp):
                    ax.plot(
                        x_data,
                        ydata[:, j],
                        marker="o",
                        label=f"{self.y_data_list[i]['field']}_{j}",
                    )
            else:
                label = entry["label"]
                ax.plot(x_data, ydata, marker="o", label=label)
        ax.legend()
        # X label: only show field if scalar_data
        if self.x_data_info is not None:
            xlabel = self.x_info.toPlainText()  # .replace("\n", ", ")
        else:
            xlabel = "X"
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Y")
        ax.set_title("History Evolution")
        self._plot_dialog.canvas.draw()
        self._plot_dialog.show()


def normalize(v):
    x, y, z = v
    n = np.sqrt(x * x + y * y + z * z)
    if n == 0:
        return (1.0, 0.0, 0.0)
    return (x / n, y / n, z / n)


def viewer(res=None):
    if not (USE_PYVISTA_QT):
        raise ImportError(
            "pyvistaqt is required to launch the viewer. "
            "Install it with: pip install pyvistaqt pyqt5"
        )

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    if res is None:
        window = MainWindow()
    else:
        window = MainWindow(res)
    window.resize(800, 600)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    viewer()
