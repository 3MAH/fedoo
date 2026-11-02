"""Optional Fedoo object browser for the active viewer pane."""

import numpy as np
from qtpy import QtWidgets
from qtpy.QtCore import Qt, QEvent, QSignalBlocker, QTimer

from fedoo.core.boundary_conditions import ListBC
from fedoo.core.dataset import as_3d_coordinates
from fedoo.core.multimeshdata import MultiMeshData
from fedoo.core.problem_dataset import ProblemDataSet
from fedoo.util.field_components import get_field_components


class ObjectInspector(QtWidgets.QDockWidget):
    """Browse meaningful objects and delegate edits to the existing managers."""

    def __init__(self, window):
        super().__init__("Object inspector", window)
        self.setObjectName("fedoo_object_inspector")
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setMinimumWidth(250)
        self.window = window
        self._objects = {}
        self._source_dock = None
        self._highlight_dock = None
        self._initial_width_applied = False
        self._width_timer = QTimer(self)
        self._width_timer.setSingleShot(True)
        self._width_timer.timeout.connect(self._restore_minimum_width)
        window.installEventFilter(self)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.timeout.connect(self.refresh)
        body = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(body)
        self.context_label = QtWidgets.QLabel("No active dataset")
        self.context_label.setWordWrap(True)
        layout.addWidget(self.context_label)
        splitter = QtWidgets.QSplitter(Qt.Vertical)
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.currentItemChanged.connect(self._selection_changed)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.properties = QtWidgets.QTableWidget(0, 2)
        self.properties.setHorizontalHeaderLabels(["Property", "Value"])
        self.properties.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.properties.verticalHeader().hide()
        self.properties.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        self.properties.horizontalHeader().setStretchLastSection(True)
        self.properties.setWordWrap(True)
        splitter.addWidget(self.tree)
        splitter.addWidget(self.properties)
        splitter.setSizes([260, 200])
        layout.addWidget(splitter, 1)
        buttons = QtWidgets.QHBoxLayout()
        self.edit_button = QtWidgets.QPushButton("Manage…")
        self.edit_button.setToolTip("Open the existing manager for this object")
        self.edit_button.clicked.connect(self.open_manager)
        self.plot_button = QtWidgets.QPushButton("Plot")
        self.plot_button.setToolTip("Plot this field in the active view")
        self.plot_button.clicked.connect(self.plot_field)
        refresh = QtWidgets.QToolButton()
        refresh.setIcon(self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload))
        refresh.setToolTip("Refresh inspector")
        refresh.setAccessibleName("Refresh inspector")
        refresh.clicked.connect(self.refresh)
        buttons.addWidget(self.edit_button)
        buttons.addWidget(self.plot_button)
        buttons.addWidget(refresh)
        layout.addLayout(buttons)
        self.setWidget(body)
        self.visibilityChanged.connect(self._visibility_changed)
        self.edit_button.setEnabled(False)
        self.plot_button.setEnabled(False)

    def eventFilter(self, watched, event):
        if watched is self.window and event.type() == QEvent.Resize:
            self._width_timer.start(0)
        return super().eventFilter(watched, event)

    def _restore_minimum_width(self):
        dock = self.window.active_dock
        if (
            dock is not None
            and self.isVisible()
            and not self.isFloating()
            and not self.window.isFullScreen()
        ):
            width = max(self.minimumWidth(), self.minimumSizeHint().width())
            self.window.resizeDocks(
                [self, dock],
                [width, max(1, self.window.width() - width)],
                Qt.Horizontal,
            )
            self._initial_width_applied = True

    def _visibility_changed(self, visible):
        if visible and self.window.isFullScreen():
            self.hide()
        elif visible:
            self.refresh()
        else:
            self.stop()

    def schedule_refresh(self):
        if (
            self._source_dock is not None
            and self._source_dock not in self.window.all_docks
        ):
            self._source_dock = None
            self._objects = {}
            blocker = QSignalBlocker(self.tree)
            self.tree.clear()
            del blocker
            self.properties.setRowCount(0)
            self._clear_highlight()
        if self.isVisible() and not self.window.isFullScreen():
            self._refresh_timer.start(0)

    def stop(self):
        self._refresh_timer.stop()
        self._width_timer.stop()
        self._clear_highlight()

    def _key(self):
        item = self.tree.currentItem()
        return (
            tuple(item.data(0, Qt.UserRole))
            if item and item.data(0, Qt.UserRole)
            else None
        )

    def _item(self, parent, label, kind, key="", value=None):
        item = QtWidgets.QTreeWidgetItem(parent, [label])
        token = (kind, key)
        item.setData(0, Qt.UserRole, token)
        self._objects[token] = value
        return item

    @staticmethod
    def _conditions(boundaries):
        for container in (boundaries.data, boundaries.data_end):
            for condition in container:
                if isinstance(condition, ListBC):
                    yield from ObjectInspector._conditions(condition)
                else:
                    yield condition

    def refresh(self):
        dock = self.window.active_dock
        source_changed = dock is not self._source_dock
        previous = self._key() if dock is self._source_dock else None
        expanded = set()
        iterator = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            if item.isExpanded() and item.data(0, Qt.UserRole):
                expanded.add(tuple(item.data(0, Qt.UserRole)))
            iterator += 1
        self._source_dock = dock
        self._clear_highlight()
        blocker = QSignalBlocker(self.tree)
        self.tree.clear()
        self._objects = {}
        if dock is None:
            self._initial_width_applied = False
            self.context_label.setText("No active dataset")
            del blocker
            self._selection_changed()
            return
        if (
            not self._initial_width_applied
            and self.isVisible()
            and not self.isFloating()
        ):
            self._restore_minimum_width()
        if source_changed:
            # Restore compact width after Qt redistributes space for changed views.
            self._width_timer.start(0)
        data = dock.data
        self.window._ensure_dock_node_sets(dock)
        self.window._ensure_dock_element_sets(dock)
        live = isinstance(data, ProblemDataSet)
        self.context_label.setText(
            f"{dock.title} · Current state"
            if live
            else f"{dock.title} · Frame {dock.current_iter}"
        )
        root = self._item(
            self.tree,
            f"Problem: {data.problem.name or 'unnamed'}" if live else dock.title,
            "dataset",
            value=data,
        )
        mesh_item = self._item(root, "Mesh", "mesh", value=data.mesh)
        for index, block in enumerate(getattr(data.mesh, "submeshes", ())):
            self._item(
                mesh_item,
                f"{index}: {block.name or block.elm_type} ({block.n_elements})",
                "submesh",
                str(index),
                block,
            )
        for kind, label, sets in (
            ("nodes", "Node sets", dock.node_sets),
            ("elements", "Element sets", dock.element_sets),
        ):
            group = self._item(root, label, "group", kind)
            for name, ids in sorted(sets.items()):
                if (
                    kind == "elements"
                    and name.startswith("submesh_")
                    and name in dock.unsaved_element_set_names
                    and getattr(data.mesh, "submeshes", ())
                ):
                    # Generated block sets duplicate the entries below Mesh.
                    continue
                count = (
                    (data.mesh.n_nodes if kind == "nodes" else data.mesh.n_elements)
                    if ids is None
                    else len(ids)
                )
                self._item(group, f"{name} ({count})", kind, name, ids)
        fields = self._item(root, "Fields", "group", "fields")
        for association, values in data.dict_data.items():
            if not values:
                continue
            group = self._item(fields, association, "group", association)
            for name in sorted(values):
                if not name.startswith("_"):
                    self._item(group, name, "field", (association, name), name)
        if live:
            boundaries = self._item(root, "Boundary conditions", "group", "bc")
            for index, condition in enumerate(self._conditions(data.problem.bc)):
                label = condition.name or f"Condition {index + 1}"
                self._item(boundaries, label, "bc", str(id(condition)), condition)
        selected = root
        iterator = QtWidgets.QTreeWidgetItemIterator(self.tree)
        while iterator.value():
            item = iterator.value()
            token = tuple(item.data(0, Qt.UserRole))
            # Field keys contain a tuple inside the token.
            if token == previous:
                selected = item
            if token in expanded or item.parent() is None or item.parent() is root:
                item.setExpanded(True)
            iterator += 1
        self.tree.setCurrentItem(selected)
        del blocker
        self._selection_changed()

    @staticmethod
    def _preview(value):
        if isinstance(value, str):
            return value
        array = np.asarray(value)
        return np.array2string(array, threshold=12, precision=5, max_line_width=60)

    def _selection_changed(self, *args):
        self._clear_highlight()
        key = self._key()
        dock = self.window.active_dock
        if dock is not self._source_dock:
            self.schedule_refresh()
            return
        self.edit_button.setEnabled(False)
        self.plot_button.setEnabled(False)
        rows = []
        if key in self._objects and dock is not None:
            kind, name = key
            value = self._objects[key]
            data = dock.data
            if kind == "dataset":
                rows = [("Type", type(data).__name__), ("View", dock.title)]
                if isinstance(data, ProblemDataSet):
                    rows += [
                        ("Problem type", type(data.problem).__name__),
                        ("Name", data.problem.name or "unnamed"),
                        ("State", "Live; Reload reads the current solution"),
                    ]
                else:
                    rows += [("Frame", dock.current_iter)]
                if "Time" in data.scalar_data:
                    rows.append(("Time", self._preview(data.scalar_data["Time"])))
            elif kind in ("mesh", "submesh"):
                blocks = getattr(value, "submeshes", (value,))
                rows = [
                    ("Nodes", value.n_nodes),
                    ("Elements", value.n_elements),
                    ("Coordinate dimensions", value.nodes.shape[1]),
                    ("Element types", ", ".join(block.elm_type for block in blocks)),
                    ("Submeshes", len(blocks)),
                    ("Visibility", self._visibility_state(key)),
                ]
                if kind == "submesh":
                    self._highlight("elements", self._element_ids(key))
            elif kind in ("nodes", "elements"):
                count = data.mesh.n_nodes if kind == "nodes" else data.mesh.n_elements
                ids = (
                    np.arange(count) if value is None else np.asarray(value, dtype=int)
                )
                rows = [
                    ("Name", name),
                    ("Type", "Node set" if kind == "nodes" else "Element set"),
                    ("Count", len(ids)),
                    ("IDs", self._preview(ids)),
                ]
                if kind == "elements":
                    rows.append(("Visibility", self._visibility_state(key)))
                self._highlight(kind, ids)
                self.edit_button.setEnabled(True)
            elif kind == "field":
                association, field = name
                raw = data.dict_data[association][field]
                chunks = (
                    [np.asarray(block) for _, block in raw.items()]
                    if isinstance(raw, MultiMeshData)
                    else [np.asarray(raw)]
                )
                rows = [
                    ("Name", field),
                    ("Association", association),
                    ("Shape", ", ".join(str(chunk.shape) for chunk in chunks)),
                ]
                if chunks:
                    components = chunks[0].shape[0] if chunks[0].ndim > 1 else 1
                    if association != "Scalar":
                        rows.append(
                            (
                                "Components",
                                ", ".join(get_field_components(field, components)),
                            )
                        )
                    numeric = [
                        chunk[np.isfinite(chunk)]
                        for chunk in chunks
                        if chunk.dtype.kind in "biuf"
                    ]
                    numeric = [chunk for chunk in numeric if chunk.size]
                    if numeric:
                        rows.append(
                            (
                                "Stored range",
                                f"{min(chunk.min() for chunk in numeric):.6g} … {max(chunk.max() for chunk in numeric):.6g}",
                            )
                        )
                    if association == "Scalar":
                        rows.append(("Value", self._preview(raw)))
                definition = data.scalar_expressions.get(field)
                if definition:
                    rows += [
                        ("Expression", definition["expression"]),
                        ("Values at", definition.get("data_type", "Node")),
                    ]
                    self.edit_button.setEnabled(True)
                self.plot_button.setEnabled(
                    association != "Scalar"
                    and self.window.field_combo.findText(field) >= 0
                )
            elif kind == "bc":
                rows = [
                    ("Name", value.name or "unnamed"),
                    ("Type", getattr(value, "bc_type", type(value).__name__)),
                    ("Variable", getattr(value, "variable_name", "—")),
                ]
                if hasattr(value, "value"):
                    rows.append(("Value", self._preview(value.value)))
                target = getattr(value, "node_set_name", None)
                if target is not None:
                    rows.append(("Target", self._preview(target)))
                variable = getattr(value, "variable_name", None)
                if (
                    isinstance(variable, str)
                    and variable in data.problem.global_dof._variable
                ):
                    rows.append(
                        ("Association", "Global DOFs; no mesh-node association")
                    )
                else:
                    ids = getattr(value, "node_set", None)
                    if ids is None:
                        assembly = getattr(value, "assembly", None)
                        ids = (
                            np.unique(assembly.mesh.elements)
                            if assembly is not None
                            else None
                        )
                    if ids is not None:
                        self._highlight("nodes", np.asarray(ids, dtype=int))
                self.edit_button.setEnabled(True)
        self.properties.setRowCount(len(rows))
        for row, (label, value) in enumerate(rows):
            self.properties.setItem(row, 0, QtWidgets.QTableWidgetItem(str(label)))
            item = QtWidgets.QTableWidgetItem(str(value))
            item.setToolTip(str(value))
            self.properties.setItem(row, 1, item)
        self.properties.resizeRowsToContents()

    def _clear_highlight(self):
        dock = self._highlight_dock
        self._highlight_dock = None
        if dock is not None and not getattr(dock.plotter, "_closed", False):
            dock.plotter.remove_actor("_inspector_highlight")

    def _element_ids(self, key):
        dock = self.window.active_dock
        if dock is None or key is None:
            return None
        mesh = dock.data.mesh
        kind, name = key
        if kind == "mesh":
            return np.arange(mesh.n_elements)
        if kind == "submesh":
            blocks = getattr(mesh, "submeshes", ())
            index = int(name)
            if not 0 <= index < len(blocks):
                return None
            offset = sum(block.n_elements for block in blocks[:index])
            return np.arange(offset, offset + blocks[index].n_elements)
        if kind == "elements" and name in dock.element_sets:
            ids = dock.element_sets[name]
            if ids is None:
                return np.arange(mesh.n_elements)
            ids = np.unique(np.asarray(ids, dtype=int))
            return ids[(ids >= 0) & (ids < mesh.n_elements)]
        return None

    def _visible_element_ids(self):
        dock = self.window.active_dock
        ids = dock.opts.get("element_set")
        if isinstance(ids, str):
            ids = dock.element_sets[ids]
        return (
            np.arange(dock.data.mesh.n_elements)
            if ids is None
            else np.asarray(ids, dtype=int)
        )

    def _visibility_state(self, key):
        ids = self._element_ids(key)
        if ids is None or not len(ids):
            return "Empty"
        shown = np.count_nonzero(np.isin(ids, self._visible_element_ids()))
        return (
            "Masked"
            if shown == 0
            else "Visible"
            if shown == len(ids)
            else f"Partially masked ({shown}/{len(ids)} visible)"
        )

    def _visibility_menu(self):
        dock = self.window.active_dock
        key = self._key()
        if dock is not self._source_dock:
            return None
        ids = self._element_ids(key)
        if ids is None:
            return None
        menu = QtWidgets.QMenu(self)
        show_only = menu.addAction("Show only")
        show_only.setEnabled(bool(len(ids)))
        show_only.triggered.connect(
            lambda: self._apply_visibility(key, "only", dock=dock)
        )
        mask = menu.addAction("Mask")
        mask.setCheckable(True)
        mask.setEnabled(bool(len(ids)))
        mask.setChecked(self._visibility_state(key) == "Masked")
        mask.triggered.connect(
            lambda checked: self._apply_visibility(key, "mask", checked, dock=dock)
        )
        return menu

    def _show_context_menu(self, position):
        item = self.tree.itemAt(position)
        if item is None:
            return
        self.tree.setCurrentItem(item)
        menu = self._visibility_menu()
        if menu is not None:
            menu.exec_(self.tree.viewport().mapToGlobal(position))
            menu.deleteLater()

    def _apply_visibility(self, key, operation, checked=False, *, dock):
        if dock is not self.window.active_dock:
            return
        ids = self._element_ids(key)
        if ids is None:
            return
        current = self._visible_element_ids()
        if operation == "only":
            visible = ids
        elif checked:
            visible = np.setdiff1d(current, ids)
        else:
            visible = np.union1d(current, ids)
        full = len(visible) == dock.data.mesh.n_elements
        dock.opts["element_set"] = None if full else visible.tolist()
        dock.visibility_mode = None if full else "show"
        manager = self.window.element_sets_dialog
        if manager is not None:
            manager.current_selection = set() if full else set(visible)
            manager._update_selection_info()
        dock.update_plot()

    def _highlight(self, kind, ids):
        if ids is None:
            return
        dock = self.window.active_dock
        mesh = dock.data.mesh
        limit = mesh.n_nodes if kind == "nodes" else mesh.n_elements
        ids = ids[(ids >= 0) & (ids < limit)]
        if kind == "elements":
            ids = ids[np.isin(ids, self._visible_element_ids())]
        if not len(ids):
            return
        points = as_3d_coordinates(mesh.nodes).astype(float, copy=True)
        displacement = dock.data.node_data.get("Disp")
        if displacement is not None and np.ndim(displacement) == 2:
            points += dock.opts.get("scale", 1.0) * as_3d_coordinates(
                np.asarray(displacement).T
            )
        options = dict(
            color="magenta",
            name="_inspector_highlight",
            reset_camera=False,
            pickable=False,
        )
        if kind == "nodes":
            dock.plotter.add_points(
                points[ids], point_size=10, render_points_as_spheres=True, **options
            )
        else:
            grid = mesh.to_pyvista()
            grid.points = points
            dock.plotter.add_mesh(
                grid.extract_cells(ids), style="wireframe", line_width=3, **options
            )
        self._highlight_dock = dock

    def open_manager(self):
        if self.window.active_dock is not self._source_dock:
            self.refresh()
            return
        key = self._key()
        if key not in self._objects:
            return
        kind, name = key
        if kind in ("nodes", "elements"):
            getattr(
                self.window,
                "open_node_sets_dialog"
                if kind == "nodes"
                else "open_element_sets_dialog",
            )()
            manager = (
                self.window.node_sets_dialog
                if kind == "nodes"
                else self.window.element_sets_dialog
            )
            for index in range(manager.sets_list.count()):
                if manager.sets_list.item(index).data(Qt.UserRole) == name:
                    manager.sets_list.setCurrentRow(index)
                    break
        elif kind == "bc":
            condition = self._objects[key]
            self.window.open_boundary_conditions_dialog()
            manager = self.window.boundary_conditions_dialog
            for index, record in enumerate(manager.records):
                if record[2] is condition:
                    manager.table.selectRow(index)
                    break
        elif kind == "field":
            self.window.open_scalar_expression_dialog()
            manager = self.window.scalar_expression_dialog
            for index in range(manager.table.rowCount()):
                if manager.table.item(index, 0).text() == name[1]:
                    manager.table.selectRow(index)
                    break

    def plot_field(self):
        if self.window.active_dock is not self._source_dock:
            self.refresh()
            return
        key = self._key()
        if key and key[0] == "field" and self.plot_button.isEnabled():
            association, field = key[1]
            self.window.field_combo.setCurrentText(field)
            self.window.avg_combo.setCurrentText(association)
