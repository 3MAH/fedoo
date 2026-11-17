"""Independent local-frame display options for the result viewer."""

from qtpy import QtWidgets
from qtpy.QtCore import Qt, Signal

from fedoo.util.frame_fields import frame_field_names


class LocalFrameOptionsDialog(QtWidgets.QDialog):
    optionsChanged = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("Local frames")
        self.setModal(False)
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.local_frames_cb = QtWidgets.QCheckBox("Show local frames")
        self.frame_legend_cb = QtWidgets.QCheckBox("Show local frame legend")
        self.frame_legend_cb.setObjectName("local_frame_legend")
        self.frame_field = QtWidgets.QComboBox()
        self.frame_field.setObjectName("local_frame_field")
        self.frame_axes = [
            QtWidgets.QCheckBox(label)
            for label in ("1 (red)", "2 (green)", "3 (blue / normal)")
        ]
        self.frame_placement = QtWidgets.QComboBox()
        self.frame_placement.addItems(["Element centres", "Gauss points"])
        self.frame_size = QtWidgets.QDoubleSpinBox()
        self.frame_size.setRange(0.01, 100)
        self.frame_size.setSuffix(" %")
        self.frame_max = QtWidgets.QSpinBox()
        self.frame_max.setRange(1, 100000)

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        form.addRow(self.local_frames_cb)
        form.addRow(self.frame_legend_cb)
        form.addRow("Frame field:", self.frame_field)
        axes = QtWidgets.QHBoxLayout()
        for control in self.frame_axes:
            axes.addWidget(control)
        form.addRow("Axes:", axes)
        form.addRow("Placement:", self.frame_placement)
        form.addRow("Length / mesh size:", self.frame_size)
        form.addRow("Maximum frames:", self.frame_max)
        layout.addLayout(form)
        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Apply
            | QtWidgets.QDialogButtonBox.Cancel
        )
        layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self._accept_options)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QtWidgets.QDialogButtonBox.Apply).clicked.connect(
            lambda: self.optionsChanged.emit()
        )
        self.update_values()

    def update_values(self):
        dock = self.parent().active_dock
        if dock is None:
            return
        opts = dock.opts
        self.frame_field.clear()
        self.frame_field.addItem("Automatic", None)
        for name in frame_field_names(dock.data):
            self.frame_field.addItem(name, name)
        self.frame_field.setCurrentIndex(
            max(
                0,
                self.frame_field.findData(
                    opts.get("frame_field", dock.data.current_local_frame)
                ),
            )
        )
        self.local_frames_cb.setChecked(opts.get("show_local_frames", False))
        self.frame_legend_cb.setChecked(opts.get("frame_legend", False))
        for control, checked in zip(
            self.frame_axes, opts.get("frame_axes", (True, True, True))
        ):
            control.setChecked(
                checked
                if control is not self.frame_axes[2] or dock.data.mesh.ndim == 3
                else False
            )
        self.frame_axes[2].setEnabled(dock.data.mesh.ndim == 3)
        self.frame_placement.setCurrentIndex(int(opts.get("frame_gauss_points", False)))
        self.frame_size.setValue(opts.get("frame_size", 2.0))
        self.frame_max.setValue(opts.get("frame_max", 1000))

    def get_values(self):
        return {
            "show_local_frames": self.local_frames_cb.isChecked(),
            "frame_legend": self.frame_legend_cb.isChecked(),
            "frame_field": self.frame_field.currentData(),
            "frame_axes": tuple(control.isChecked() for control in self.frame_axes),
            "frame_gauss_points": bool(self.frame_placement.currentIndex()),
            "frame_size": self.frame_size.value(),
            "frame_max": self.frame_max.value(),
        }

    def _accept_options(self):
        self.optionsChanged.emit()
        self.accept()
