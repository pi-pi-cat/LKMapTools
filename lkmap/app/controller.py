from __future__ import annotations

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import QMessageBox

from lkmap.capture.manual_region import ManualRegionCaptureStrategy
from lkmap.locator.orb_locator import OrbLocator
from lkmap.models import AppSettings, CaptureRegion, LocationResult
from lkmap.services.assets import AssetService
from lkmap.services.config import ConfigService


class TrackingWorker(QObject):
    location_ready = Signal(object)
    state_changed = Signal(str, str)
    error_raised = Signal(str)

    def __init__(
        self,
        settings: AppSettings,
        capture_strategy: ManualRegionCaptureStrategy,
        locator: OrbLocator,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._capture_strategy = capture_strategy
        self._locator = locator
        self._running = False
        self._timer: QTimer | None = None

    @Slot()
    def start_tracking(self) -> None:
        if self._running:
            return
        try:
            self._locator.initialize()
        except Exception as exc:
            self.error_raised.emit(str(exc))
            return

        self._running = True
        if self._timer is None:
            self._timer = QTimer(self)
            self._timer.timeout.connect(self._tick)
        self._timer.start(self._settings.orb.refresh_interval_ms)
        self.state_changed.emit("Tracking", "开始实时定位")

    @Slot()
    def stop_tracking(self) -> None:
        self._running = False
        if self._timer is not None:
            self._timer.stop()
        self.state_changed.emit("Ready", "已暂停跟踪")

    @Slot()
    def reset_locator(self) -> None:
        self._locator.reset()
        self.state_changed.emit("Ready", "已重置定位器")

    @Slot()
    def _tick(self) -> None:
        if not self._running:
            return

        frame = self._capture_strategy.current_frame()
        if frame is None:
            self.state_changed.emit("Error", "截图区域尚未配置")
            return

        try:
            result = self._locator.locate(frame)
        except Exception as exc:
            self.error_raised.emit(str(exc))
            return

        if result.found:
            self.state_changed.emit("Tracking", result.message)
        else:
            self.state_changed.emit("Lost", result.message)
        self.location_ready.emit(result)


class AppController(QObject):
    start_worker = Signal()
    stop_worker = Signal()
    reset_worker = Signal()

    def __init__(self, window) -> None:
        super().__init__()
        self.window = window
        self.config_service = ConfigService()
        self.settings = self.config_service.load()
        self.capture_strategy = ManualRegionCaptureStrategy(self.settings.capture_region)
        self.asset_service = AssetService(self.settings)
        self.locator = OrbLocator(self.settings, self.asset_service)

        self.worker_thread = QThread()
        self.worker = TrackingWorker(self.settings, self.capture_strategy, self.locator)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.start()

        self.start_worker.connect(self.worker.start_tracking, Qt.ConnectionType.QueuedConnection)
        self.stop_worker.connect(self.worker.stop_tracking, Qt.ConnectionType.QueuedConnection)
        self.reset_worker.connect(self.worker.reset_locator, Qt.ConnectionType.QueuedConnection)

        self.worker.location_ready.connect(self._handle_location)
        self.worker.state_changed.connect(self.window.set_state)
        self.worker.error_raised.connect(self._handle_error)

        self.window.select_region_requested.connect(self.select_region)
        self.window.tracking_toggled.connect(self.toggle_tracking)
        self.window.reset_requested.connect(self.reset_locator)
        self.window.resources_toggled.connect(self.toggle_resources)

    def bootstrap(self) -> None:
        try:
            display_map = self.asset_service.load_display_map()
            resources = self.asset_service.load_resource_points()
        except Exception as exc:
            self._handle_error(str(exc))
            return

        self.window.map_view.set_map_image(display_map)
        self.window.map_view.set_resource_points(resources)
        self.window.map_view.apply_zoom(self.settings.view.zoom)
        self.window.map_view.set_show_resources(self.settings.view.show_resources)
        self.window.set_resources_visible(self.settings.view.show_resources)

        if self.settings.capture_region and self.settings.capture_region.is_valid:
            self.window.set_state("Ready", "已加载截图区域，点击开始跟踪即可")
        else:
            self.window.set_state("Unconfigured", "请先框选小地图区域")
            self.select_region()

    def select_region(self) -> None:
        was_running = self.window.is_tracking
        if was_running:
            self.stop_worker.emit()
            self.window.set_tracking_running(False)

        region = self.capture_strategy.select_region(self.window, self.settings.capture_region)
        if region is None:
            self.window.set_state("Unconfigured", "未完成区域框选")
            return

        self.settings.capture_region = region
        self.capture_strategy.region = region
        self.config_service.save_capture_region(self.settings, region)
        self.window.set_state("Ready", "小地图区域已保存")

    def toggle_tracking(self, tracking: bool) -> None:
        if tracking and not (self.settings.capture_region and self.settings.capture_region.is_valid):
            self.window.set_tracking_running(False)
            self.select_region()
            return

        if tracking:
            self.start_worker.emit()
        else:
            self.stop_worker.emit()
        self.window.set_tracking_running(tracking)

    def reset_locator(self) -> None:
        self.reset_worker.emit()

    def toggle_resources(self, visible: bool) -> None:
        self.window.map_view.set_show_resources(visible)
        self.settings.view.show_resources = visible
        self.config_service.save_view_settings(
            self.settings,
            zoom=self.window.map_view.zoom_value,
            show_resources=visible,
        )

    @Slot(object)
    def _handle_location(self, result: LocationResult) -> None:
        self.window.set_location(result)
        self.window.map_view.set_location(result)
        self.settings.view.zoom = self.window.map_view.zoom_value

    @Slot(str)
    def _handle_error(self, message: str) -> None:
        self.window.set_tracking_running(False)
        self.window.set_state("Error", message)
        QMessageBox.critical(self.window, "运行错误", message)

    def shutdown(self) -> None:
        self.stop_worker.emit()
        self.worker_thread.quit()
        self.worker_thread.wait(2000)
        self.config_service.save_view_settings(
            self.settings,
            zoom=self.window.map_view.zoom_value,
            show_resources=self.settings.view.show_resources,
        )
