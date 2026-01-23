try:
    import fedoo as fd
    import numpy as np
    import sys
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
    from qtpy.QtCore import Qt, Signal, QSignalBlocker, QTimer, QEvent, QSize
    import matplotlib as mpl  # only for colormap
    import pyvista as pv
    from pyvistaqt import QtInteractor
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
    import os
    import re

    USE_PYVISTA_QT = True
except ImportError:
    USE_PYVISTA_QT = False


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
            base = pal.color(QtGui.QPalette.AlternateBase)
            text = pal.color(QtGui.QPalette.WindowText)
            self.setStyleSheet(f"""
                QWidget {{ background-color: {base.name()}; }}
                QLabel {{ color: {text.name()}; font-weight: normal; }}
            """)


class PlotDock(QDockWidget):
    _n_created_dock = 0  # total dock created

    def __init__(self, data, title, parent=None, opts=None):
        PlotDock._n_created_dock += 1
        self._dock_index = PlotDock._n_created_dock
        self.title = title
        title = f"{self._dock_index }: " + str(title)
        super().__init__(title, parent)
        self.data = data

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
                # Clip plane parameters
                "clip_args": None,
                "clip_origin": None,
                "clip_normal": None,
                "clip_invert": False,
            }

        if hasattr(data, "loaded_iter"):
            if data.loaded_iter is None:
                data.load(-1)
            self.current_iter = data.loaded_iter
        else:
            self.current_iter = 0

        field_names = data.field_names()
        if "Stress" in field_names:
            self.current_field = "Stress"
            self.current_comp = "vm"
        elif "Disp" in field_names:
            self.current_field = "disp"
            self.current_comp = "X"
        else:
            try:
                self.current_field = field_names[0]
            except:
                self.current_field = None
            self.current_comp = None
        self.current_data_type = "Node"

        titlebar = DockTitleBar(self)
        titlebar.clicked.connect(lambda d=self: parent._set_active(d))
        self.setTitleBarWidget(titlebar)
        self._titlebar = titlebar  # keep reference

        parent.all_docks.append(self)
        if parent.active_dock is None:
            for action in parent.actions_requiring_data:
                action.setEnabled(True)
        parent._set_active(self)

    def update_plot(self, val=None, iteration=None, lock_view=True, plotter=None):
        if self.data.mesh is None:
            # don't plot anything without a mesh
            return

        if plotter is None:
            plotter = self.plotter

        if iteration is not None and hasattr(self.data, "loaded_iter"):
            self.data.load(iteration)

        if self.opts["clim_mode"] == "current":
            clim = None
        else:
            clim = self.opts["clim"]

        sargs = {
            # "n_colors": self.opts['n_colors'],
            "n_labels": self.opts["n_labels"],
        }
        # plotter.clear()  # not compatible with pbr ???
        plotter.renderer.clear_actors()

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
            opacity=self.opts["opacity"],
            pbr=self.opts["pbr"],
            metallic=self.opts["metallic"],
            roughness=self.opts["roughness"],
            diffuse=self.opts["diffuse"],
            clip_args=self.opts["clip_args"],
            lock_view=lock_view,
            title=self.opts["title_plot"],
            cmap=self.opts["cmap"],
        )

        if self.parent()._plane_widget_enabled:
            self.parent().enable_plane_widget()

        if self.parent()._line_widget_enabled:
            self.parent()._rebuild_line_widget()

    def closeEvent(self, event):
        self.plotter.close()
        if self is self.parent().active_dock:
            self.parent().active_dock = None
        self.parent().all_docks.remove(self)
        self.parent()._update_dock_selector()
        if self.parent().active_dock is None:
            for action in self.parent().actions_requiring_data:
                action.setEnabled(False)
        super().closeEvent(event)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, data=None):
        super().__init__()

        self.all_docks = []
        self.active_dock = None
        self._clim_dialog = None  # clim windows if open
        self._clip_dialog = None  # clip windows if open
        self._plot_over_line_dialog = None
        self._pol_results_dialog = None
        self._renderer_dialog = None
        self._plot_dialog = None
        self._window_index = 1
        self._picking_target = -1  # internal arg for picking tools. -1 = No target

        # if plane_widget or line wiget should be shown in the active dock
        self._plane_widget_enabled = False
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

        toolbar_fields.addSeparator()  # <-- adds a small gap

        toolbar_fields.addWidget(QLabel("Data type: "))
        self.avg_combo = QtWidgets.QComboBox()
        self.avg_combo.addItems(["Node", "GaussPoint", "Element"])
        toolbar_fields.addWidget(self.avg_combo)

        self.addToolBar(Qt.TopToolBarArea, toolbar_fields)

        # ------------------------------------------------
        # Toolbar 2: Iteration controls
        # ------------------------------------------------
        toolbar_iter = QToolBar("Iteration")
        toolbar_iter.setMovable(True)

        toolbar_iter.addWidget(QLabel("Iteration:"))
        self.iter_slider = QSlider(Qt.Horizontal)
        self.iter_spin = QSpinBox()
        self.iter_slider.setTracking(False)
        self.iter_slider.setTickPosition(QSlider.TicksBelow)

        toolbar_iter.addWidget(self.iter_slider)
        toolbar_iter.addWidget(self.iter_spin)

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
        # Toolbar 3 View
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

        # add view actions to toolbars
        view_toolbar.addAction(view_top_action)
        view_toolbar.addAction(view_bottom_action)
        view_toolbar.addAction(view_front_action)
        view_toolbar.addAction(view_back_action)
        view_toolbar.addAction(view_left_action)
        view_toolbar.addAction(view_right_action)
        view_toolbar.addAction(view_isometric_action)

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

        # Save Image
        save_image_action = QtWidgets.QAction("Save Image...", self)
        save_image_action.setShortcut("Ctrl+S")
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
        # add same actions as view toolbar
        link_views_action = QtWidgets.QAction("Link views", self)
        link_views_action.setCheckable(True)
        link_views_action.triggered.connect(self._toggle_link_views)
        view_menu.addAction(link_views_action)

        view_menu.addAction(view_top_action)
        view_menu.addAction(view_bottom_action)
        view_menu.addAction(view_front_action)
        view_menu.addAction(view_back_action)
        view_menu.addAction(view_left_action)
        view_menu.addAction(view_right_action)
        view_menu.addAction(view_isometric_action)

        # --- Menu Tools ---
        tools_menu = menubar.addMenu("Tools")

        clipAction = QtWidgets.QAction("Clip...", self)
        clipAction.setShortcut("Ctrl+K")
        clipAction.triggered.connect(self.open_clip_dialog)
        tools_menu.addAction(clipAction)

        plotOverLineAction = QtWidgets.QAction("Plot over line...", self)
        plotOverLineAction.triggered.connect(self.open_plot_over_line_dialog)
        tools_menu.addAction(plotOverLineAction)

        # --- Menu Windows ---
        windows_menu = menubar.addMenu("Windows")

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
            copy_action,
            clipAction,
            plotOverLineAction,
            view_top_action,
            view_bottom_action,
            view_left_action,
            view_right_action,
            view_front_action,
            view_back_action,
            view_isometric_action,
        ]
        # -------------------------
        # Dockable PyVista Widget
        # -------------------------
        if data is not None:
            if isinstance(data, str):
                title = os.path.basename(data)
                data = fd.read_data(data)
            else:
                title = "Data"

            self.add_dataset_dock(data, title=title)
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

    def add_dataset_dock(self, data, title, opts=None):
        dock = PlotDock(data, title, self, opts)

        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        dock.visibilityChanged.connect(
            lambda v, d=dock: self._set_active(d) if v else None  # self._hide_dock
        )
        dock.widget().installEventFilter(self)
        self._update_dock_selector()
        self.update_plot(lock_view=False)

    # def _hide_dock(self, dock):
    #     dock.setFloating(True)
    #     dock.move(-10000, -10000)

    def copy_active_dock(self):
        dock = self.active_dock
        if dock:
            self.add_dataset_dock(dock.data.copy(), dock.title, opts=dict(dock.opts))
            self.active_dock.plotter.camera_position = dock.plotter.camera_position

    def eventFilter(self, obj, event):
        et = event.type()
        if et in (QEvent.FocusIn, QEvent.MouseButtonPress):
            dock = self._dock_for_widget(obj)
            if dock:
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

    def _distribute_vertical(self):
        if not self.all_docks:
            return

        self.addDockWidget(Qt.RightDockWidgetArea, self.all_docks[0])

        for dock in self.all_docks[1:]:
            self.splitDockWidget(self.all_docks[0], dock, Qt.Vertical)

    def _distribute_horizontal(self):
        if not self.all_docks:
            return

        # Add the first dock
        self.addDockWidget(Qt.RightDockWidgetArea, self.all_docks[0])

        # Split horizontally for the rest
        for dock in self.all_docks[1:]:
            self.splitDockWidget(self.all_docks[0], dock, Qt.Horizontal)

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

    def _toggle_link_views(self, checked):
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

    def _set_active(self, dock):
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
            # except Exception:
            #     self.plotter.clear_widgets()
        if self._line_widget_enabled and self.active_dock:
            self.plotter.clear_line_widgets()

        self.active_dock = dock

        for d in self.all_docks:
            d._titlebar._set_active(d is dock)

        if hasattr(dock.data, "n_iter"):
            max_iter = dock.data.n_iter - 1
        else:
            max_iter = 0
        current_iter = dock.current_iter  # may be modified by setRange
        self.setRange(0, max_iter)
        self.iter_slider.setValue(current_iter)
        self.iter_spin.setValue(current_iter)

        # old_state = self.field_combo.blockSignals(True)

        # update_field_combo
        self.field_combo.clear()
        if self.data is not None:
            self.field_combo.addItems(self.data.field_names())
        dock = self.active_dock
        if dock.current_field is not None:
            self.field_combo.setCurrentText(dock.current_field)
        else:
            self.field_combo.setCurrentIndex(0)
        # self.field_combo.blockSignals(old_state)

        # update data_type value
        # old_state = self.avg_combo.blockSignals(True)
        self.avg_combo.setCurrentText(dock.current_data_type)

        # update component combo
        self.update_components(dock.current_field)

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
        if self.active_dock in self.all_docks:
            self.dock_selector_combo.setCurrentIndex(
                self.all_docks.index(self.active_dock)
            )

    def _on_anim_tick(self):
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
        # sync iter spin_box without emit signal
        if self.iter_spin.value() != val:
            old_state = self.iter_spin.blockSignals(True)
            self.iter_spin.setValue(val)
            self.active_dock.current_iter = val
            self.iter_spin.blockSignals(old_state)

        self.update_plot(iteration=self.iteration)

    def _on_spin_changed(self, val: int):
        # sync slider without emit signal
        if self.iter_slider.value() != val:
            old_state = self.iter_slider.blockSignals(True)
            self.iter_slider.setValue(val)
            self.active_dock.current_iter = val
            self.iter_slider.blockSignals(old_state)

        self.update_plot(iteration=self.iteration)

    # set slider and iteration spinbox min/max values
    def setRange(self, min_iter: int, max_iter: int):
        self.iter_slider.setRange(min_iter, max_iter)
        self.iter_spin.setRange(min_iter, max_iter)
        # Ajuster les ticks si besoin
        self.iter_slider.setTickInterval(max(1, (max_iter - min_iter) // 10))

    # --- Animation ---
    def toggle_animation(self, checked: bool):
        if checked:
            self.start_animation()
        else:
            self.stop_animation()

    def start_animation(self):
        if self.anim_timer.isActive():
            return
        # Si la plage est vide, ne rien faire
        if self.iter_slider.maximum() <= self.iter_slider.minimum():
            self.animate_btn.setChecked(False)
            return
        self.animate_btn.setText("⏸ Pause")
        self.anim_timer.start()  # interval déjà réglé par _on_anim_fps_changed

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
        self.update_components(field)
        self.update_plot_with_clim(lock_view=True)

    def update_components(self, field):
        if field == "" or not self.active_dock or not field:
            return [""]

        data = self.data
        self.active_dock.current_field = field

        if data[field].ndim == 1:
            comps = ["0"]
        else:
            if field == "Stress":
                comps = ["XX", "YY", "ZZ", "XY", "XZ", "YZ", "vm", "pressure"]
            elif field == "Strain":
                comps = ["XX", "YY", "ZZ", "XY", "XZ", "YZ"]
            elif field == "Disp":
                if len(data["Disp"]) == 2:
                    comps = ["X", "Y", "norm"]
                else:
                    comps = ["X", "Y", "Z", "norm"]
            else:
                comps = [str(i) for i in range(data[field].shape[0])]

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
        if field == "":
            return None
        else:
            return self.field_combo.currentText()

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
            "Fedoo files (*.fdz) ;;VTK Files (*.vtk);;CSV Files (*.csv) ;; All Files (*)",
        )
        if fname:
            data = fd.read_data(fname)
            title = os.path.basename(fname)
            self.add_dataset_dock(data, title=title)

    def open_clim_dialog(self):
        if self._clim_dialog is None:
            self._clim_dialog = ClimOptionsDialog(parent=self)
            self._clim_dialog.applyRequested.connect(self.apply_clim_from_dialog)
            self._clim_dialog.accepted.connect(self.apply_clim_and_close)
            self._clim_dialog.rejected.connect(self._clim_dialog.close)

        self._clim_dialog.setWindowModality(Qt.NonModal)
        self._clim_dialog.show()
        self._clim_dialog.raise_()

    def _update_dialog_state(self, dlg, mode, vmin, vmax, n_colors, n_labels):
        # Mettre à jour les valeurs du popup sans le recréer
        if mode == "current":
            dlg.rb_current.setChecked(True)
        elif mode == "all":
            dlg.rb_all.setChecked(True)
        else:
            dlg.rb_manual.setChecked(True)
        dlg.vmin_spin.setValue(float(vmin))
        dlg.vmax_spin.setValue(float(vmax))
        dlg.ncolors_spin.setValue(int(n_colors))
        dlg.nlabels_spin.setValue(int(n_labels))

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
            dock.opts["pbr"] = self._renderer_dialog.pbr_cb.isChecked()
            dock.opts["metallic"] = float(self._renderer_dialog.metallic_spin.value())
            dock.opts["roughness"] = float(self._renderer_dialog.roughness_spin.value())
            dock.opts["diffuse"] = float(self._renderer_dialog.diffuse_spin.value())
            dock.update_plot()

    def update_plot_with_clim(self, val=None, iteration=None, lock_view=True):
        self.active_dock.current_comp = self.current_component
        self.active_dock.current_data_type = self.current_data_type
        if self.opts["clim_mode"] == "all":
            if hasattr(self.data, "get_all_frame_lim"):
                self.opts["clim"] = self.data.get_all_frame_lim(
                    field=self.current_field,
                    component=self.current_component,
                    data_type=self.current_data_type,
                )[2]
            else:
                self.opts["clim"] = None
        self.update_plot(val=val, iteration=iteration, lock_view=lock_view)

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

        for iteration in range(1, self.data.n_iter):
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

    # Clip dialog : Open and close
    # ----------------------------
    def open_clip_dialog(self):
        # Create clip dialog only if not already exist
        if self._clip_dialog is None:
            origin = (
                tuple(self.data.mesh.bounding_box.center)
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

    def _on_dialog_clip_changed(self, enabled, origin, normal, invert):
        """Dialog → Apply clip and update clip plane widget."""
        self.opts["clip_invert"] = bool(invert)
        self.opts["clip_normal"] = normal
        self.opts["clip_origin"] = origin
        if enabled:
            self.opts["clip_args"] = {
                "normal": normal,
                "origin": origin,
                "invert": self.opts["clip_invert"],
            }
        else:
            self.opts["clip_args"] = None

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
            self.opts["clip_origin"] = tuple(self.data.mesh.bounding_box.center)
            self.opts["clip_normal"] = (1.0, 0.0, 0.0)

        origin = self.opts["clip_origin"]
        normal = self.opts["clip_normal"]
        # bounds = self._bounds

        def _cb(normal_cb, origin_cb):
            """Widget → Dialog + Clip (live)."""
            if self._clip_dialog is not None:
                self._clip_dialog.set_values_from_widget(
                    origin_cb, normal_cb, invert=self.opts["clip_invert"]
                )
                self._clip_dialog._emit_clip_params()  # emit signal
                self.update_plot()

        self._plane_widget = self.plotter.add_plane_widget(
            callback=_cb,
            bounds=bounds,
            origin=origin,
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

        # Init dialog si ouverte
        # if self._clip_dialog is not None:
        #     self._clip_dialog.set_values_from_widget(origin, normal, invert=self.opts['clip_invert'])

    def disable_plane_widget(self):
        if not self._plane_widget_enabled:
            return
        # try:
        self.plotter.clear_plane_widgets()
        # except Exception:
        # self.plotter.clear_widgets()
        self._plane_widget = None
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

        self._line_widget_enabled = True
        # Line widget
        self._line_widget = self.plotter.add_line_widget(
            callback=self.on_line_changed, use_vertices=True
        )
        # Show dialog
        self._plot_over_line_dialog.show()
        self._plot_over_line_dialog.raise_()

    def _on_pol_dialog_closed(self, *args, **kwargs):
        # remove line widget
        self._line_widget_enabled = False
        self.plotter.clear_line_widgets()

    def _on_pol_dialog_changed(self, p1, p2):
        # update widget (emit no signal)
        self._line_widget.SetPoint1(p1)
        self._line_widget.SetPoint2(p2)

    def on_line_changed(self, p1, p2):
        """Called interactively while moving the widget"""
        self._plot_over_line_dialog.update_line(p1, p2)

    def _rebuild_line_widget(self):
        if self._plot_over_line_dialog:
            p1 = self._plot_over_line_dialog.p1
            p2 = self._plot_over_line_dialog.p2
            self.plotter.clear_line_widgets()
            self._line_widget = self.plotter.add_line_widget(
                callback=self.on_line_changed, use_vertices=True
            )
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

            # Sync UI and scene
            self._on_pol_dialog_changed(p1, p2)  # update widget
            self.on_line_changed(p1, p2)  # update dialog values

            # Stop picking
            self.plotter.disable_picking()
            self._picking_target = -1

        # Enable one-shot point picking (left click)
        self.plotter.disable_picking()
        self.plotter.enable_point_picking(
            callback=_picked,
            left_clicking=True,
            show_message=True,  # shows hint in the render window
            # picker='point'        # snap to mesh points
        )

    def plot_over_line(self, p1, p2, resolution):
        """
        Compute plot_over_line and push data to the Matplotlib dialog.
        Set live=True if you call this frequently (dragging) and want to avoid titles/reflows.
        """
        # Compute line result
        if "data1" not in self.plotter.actors:
            QtWidgets.QMessageBox.information(self, "No compatible data found.")
            return

        pv_mesh = pv.wrap(self.plotter.actors["data1"].GetMapper().GetInput())
        res = pv_mesh.sample_over_line(p1, p2, resolution=resolution)
        x = res["Distance"]
        y = res["Data"]  # or y = res.active_scalars
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

    def toggle_fullscreen(self):
        # Hide chrome (optional): menu + toolbars when in fullscreen
        if not (self.isFullScreen()):
            # Remember current visible toolbars/menubar to restore later
            self._hidden_ui_when_fullscreen = []
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
        for dock in self.all_docks:
            dock.plotter.close()  # libère le contexte VTK
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
        form.addRow("Number of colors:", self.ncolors_spin)
        form.addRow("Number of labels:", self.nlabels_spin)

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
        if self.rb_manual.isChecked():
            return
        parent = self.parent()
        if self.rb_current.isChecked():
            data = parent.get_current_data()
            clim = [data.min(), data.max()]
        elif self.rb_all.isChecked():
            if hasattr(parent.active_dock.data, "get_all_frame_lim"):
                clim = parent.active_dock.data.get_all_frame_lim(
                    parent.current_field,
                    parent.current_component,
                    parent.current_data_type,
                )[2]
            else:
                data = parent.get_current_data()
                clim = [data.min(), data.max()]
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
        cmap = mpl.cm.get_cmap(cmap_name, n_colors)
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
        self.setMinimumWidth(300)
        self.setFrameShape(QtWidgets.QFrame.Box)
        self.setLineWidth(1)
        self.setAlignment(Qt.AlignCenter)

    def update_preview(self, cmap_name: str, reverse: bool = False, n_colors=256):
        try:
            # cmap = mpl.colormaps.get(cmap_name)
            cmap = mpl.cm.get_cmap(cmap_name, n_colors)
        except Exception:
            self.setText(f"Colormap '{cmap_name}' introuvable")
            return

        # Create a horizontal gradient [0..1]
        width = max(self.width(), 300)
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
        self.pbr_cb.setChecked(opts["pbr"])
        self.metallic_spin.setValue(float(opts["metallic"]))
        self.roughness_spin.setValue(float(opts["roughness"]))
        self.diffuse_spin.setValue(float(opts["diffuse"]))

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

    # def _build_ui(self):
    #     self.edits = []
    #     form = QtWidgets.QFormLayout()

    #     for label in ("P1 X", "P1 Y", "P1 Z", "P2 X", "P2 Y", "P2 Z"):
    #         sb = QtWidgets.QDoubleSpinBox()
    #         sb.setRange(-1e9, 1e9)
    #         sb.setDecimals(6)
    #         self.edits.append(sb)
    #         form.addRow(label, sb)
    #         sb.valueChanged.connect(self._emit_line_params)

    #     # Resolution spinbox
    #     self.resSpin = QtWidgets.QSpinBox()
    #     self.resSpin.setRange(2, 10000)
    #     self.resSpin.setSingleStep(10)
    #     self.resSpin.setValue(200)  # default
    #     form.addRow("Resolution", self.resSpin)

    #     # Pick buttons
    #     self.btnPickP1 = QtWidgets.QPushButton("Pick P1")
    #     self.btnPickP2 = QtWidgets.QPushButton("Pick P2")

    #
    #     self.plotBtn = QtWidgets.QPushButton("Plot over line")

    #     layout = QtWidgets.QVBoxLayout(self)
    #     layout.addLayout(form)
    #     layout.addWidget(self.invertDirBtn)
    #     layout.addWidget(self.btnPickP1)
    #     layout.addWidget(self.btnPickP2)
    #     layout.addWidget(self.plotBtn)

    #     # Connections
    #     self.btnPickP1.clicked.connect(lambda: self.requestPick.emit(0))
    #     self.btnPickP2.clicked.connect(lambda: self.requestPick.emit(1))

    #     self.plotBtn.clicked.connect(self._emit_plot)

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
        self.saveBtn = QtWidgets.QPushButton("Save as…")
        self.saveBtn.clicked.connect(self._save_as)

        header = QtWidgets.QHBoxLayout()
        header.addWidget(self.lblP1)
        header.addSpacing(10)
        header.addWidget(self.lblP2)
        header.addStretch(1)
        header.addWidget(self.saveBtn)

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

    def _save_as(self):
        """Save the current figure to file."""
        dlg = QtWidgets.QFileDialog(
            self, "Save plot", "", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        if dlg.exec_():  # .exec() if PySide6
            path = dlg.selectedFiles()[0]
            # Choose DPI depending on format (optional)
            dpi = 150 if path.lower().endswith(".png") else None
            self.figure.savefig(path, dpi=dpi, bbox_inches="tight")


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
            "Install it with: pip install pyvistaqt"
        )

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
