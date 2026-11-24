"""Qt platform of the viewer tests."""

import os
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def native_qt_platform_on_macos():
    """Create the application on the native platform on macOS.

    The fixtures ask for Qt's offscreen platform, whose window ids are not
    Cocoa views: VTK's render window crashes on them. An application that
    already exists keeps its platform.
    """
    if sys.platform != "darwin":
        yield
        return
    qt_widgets = pytest.importorskip("qtpy.QtWidgets")
    requested = os.environ.pop("QT_QPA_PLATFORM", None)
    try:
        app = qt_widgets.QApplication.instance() or qt_widgets.QApplication([])
    finally:
        if requested is not None:
            os.environ["QT_QPA_PLATFORM"] = requested
    yield app
