from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QLabel, QMainWindow, QStatusBar, QToolBar

from lkmap.map_view.view import MapView
from lkmap.models import LocationResult


class MainWindow(QMainWindow):
    select_region_requested = Signal()
    tracking_toggled = Signal(bool)
    reset_requested = Signal()
    resources_toggled = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LKMapTools")
        self.resize(1100, 820)
        self.map_view = MapView()
        self.setCentralWidget(self.map_view)

        self._toolbar = QToolBar("Controls", self)
        self.addToolBar(self._toolbar)

        self._select_action = QAction("框选小地图", self)
        self._select_action.triggered.connect(self.select_region_requested)
        self._toolbar.addAction(self._select_action)

        self._start_action = QAction("开始跟踪", self)
        self._start_action.setCheckable(True)
        self._start_action.toggled.connect(self._emit_tracking_toggle)
        self._toolbar.addAction(self._start_action)

        self._reset_action = QAction("重置定位", self)
        self._reset_action.triggered.connect(self.reset_requested)
        self._toolbar.addAction(self._reset_action)

        self._resources_action = QAction("显示资源点", self)
        self._resources_action.setCheckable(True)
        self._resources_action.setChecked(True)
        self._resources_action.toggled.connect(self.resources_toggled)
        self._toolbar.addAction(self._resources_action)

        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)
        self._state_label = QLabel("State: idle")
        self._coord_label = QLabel("Position: -")
        self._locate_label = QLabel("定位: 未开始")
        self._zoom_label = QLabel("Zoom: 1.00x")
        self._status_bar.addPermanentWidget(self._state_label)
        self._status_bar.addPermanentWidget(self._coord_label)
        self._status_bar.addPermanentWidget(self._locate_label)
        self._status_bar.addPermanentWidget(self._zoom_label)
        self.map_view.zoom_changed.connect(self.set_zoom)

    def set_state(self, state: str, message: str = "") -> None:
        self._state_label.setText(f"State: {state}")
        if message:
            self._status_bar.showMessage(message, 5000)

    def set_location(self, result: LocationResult) -> None:
        locate_text = (
            f"定位: 成功({result.mode}) 置信度 {result.confidence:.2f}"
            if result.found
            else f"定位: 失败({result.message})"
        )
        self._locate_label.setText(locate_text)
        if result.x is None or result.y is None:
            self._coord_label.setText("Position: -")
            return
        self._coord_label.setText(f"Position: ({int(result.x)}, {int(result.y)})")

    def set_zoom(self, zoom: float) -> None:
        self._zoom_label.setText(f"Zoom: {zoom:.2f}x")

    def set_tracking_running(self, running: bool) -> None:
        self._start_action.blockSignals(True)
        self._start_action.setChecked(running)
        self._start_action.setText("暂停跟踪" if running else "开始跟踪")
        self._start_action.blockSignals(False)

    def set_resources_visible(self, visible: bool) -> None:
        self._resources_action.blockSignals(True)
        self._resources_action.setChecked(visible)
        self._resources_action.blockSignals(False)

    @property
    def is_tracking(self) -> bool:
        return self._start_action.isChecked()

    def _emit_tracking_toggle(self, checked: bool) -> None:
        self._start_action.setText("暂停跟踪" if checked else "开始跟踪")
        self.tracking_toggled.emit(checked)
