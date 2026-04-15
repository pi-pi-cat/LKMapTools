from __future__ import annotations

from PySide6.QtCore import QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import QMessageBox

from lkmap.capture.manual_region import ManualRegionCaptureStrategy
from lkmap.locator.pyramid_locator import PyramidLocator
from lkmap.models import AppSettings, CaptureRegion, LocationResult
from lkmap.services.assets import AssetService
from lkmap.services.config import ConfigService
from lkmap.services.icon_detector import detect_player_icon


class TrackingWorker(QObject):
    location_ready = Signal(object)
    state_changed = Signal(str, str)
    error_raised = Signal(str)

    def __init__(
        self,
        settings: AppSettings,
        capture_strategy: ManualRegionCaptureStrategy,
        locator: PyramidLocator,
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
        self._timer.start(self._settings.template.refresh_interval_ms)
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
        self.locator = PyramidLocator(self.settings, self.asset_service)

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

        # 区域确认后立即截一帧，检测玩家图标位置并缓存到 settings
        self._refresh_icon_mask()

        self.config_service.save(self.settings)
        # 新区域/新图标遮罩 → 强制重建 mask 缓存
        self.locator.invalidate_mask()
        self.window.set_state("Ready", "小地图区域已保存")

    def _refresh_icon_mask(self) -> None:
        """抓取当前小地图一帧，检测玩家图标位置，结果写入 settings.icon_mask。"""
        import os
        frame_data = self.capture_strategy.current_frame()
        if frame_data is None:
            return
        frame = frame_data.image
        icon_path = self.settings.assets.map_path.replace("raw.png", "me.png")
        candidates = [icon_path, "assest/me.png"]
        resolved = next((p for p in candidates if os.path.exists(p)), "")
        self.settings.icon_mask = detect_player_icon(frame, resolved)
        self._save_icon_mask_debug(frame)

    def _save_icon_mask_debug(self, frame_bgr) -> None:
        """Save a debug visualisation of the icon mask to debug_icon_mask.png.

        Legend:
          black dot   -- detected icon centre (player position)
          green dot   -- geometric centre of the capture frame
          red circle  -- mask area (pixels inside are hidden from the algorithm)
          yellow dash -- search boundary used during detection
          white dash  -- minimap valid-area boundary
          cyan line   -- offset vector (frame centre -> icon centre)
        """
        import cv2
        import numpy as np

        icon = self.settings.icon_mask
        h, w = frame_bgr.shape[:2]
        cx, cy = w // 2, h // 2
        offset_x = icon.x - cx
        offset_y = icon.y - cy

        # Scale up so the small minimap is easy to inspect
        scale = max(1, 400 // max(w, h))
        vis = cv2.resize(frame_bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

        def sp(x, y):
            return (int(x * scale), int(y * scale))

        ix, iy   = sp(icon.x, icon.y)
        fcx, fcy = sp(cx, cy)
        r_mask   = icon.radius * scale

        # Dashed search boundary (yellow) — 与 icon_detector._SEARCH_RATIO 一致
        search_r = int(min(w, h) // 2 * 0.90) * scale
        for a in range(0, 360, 12):
            cv2.ellipse(vis, (fcx, fcy), (search_r, search_r),
                        0, a, a + 6, (0, 220, 255), max(1, scale))

        # Dashed minimap boundary (white)
        outer_r = max(8, w // 2 - 8) * scale
        for a in range(0, 360, 12):
            cv2.ellipse(vis, (fcx, fcy), (outer_r, outer_r),
                        0, a, a + 6, (210, 210, 210), max(1, scale))

        # Mask circle -- semi-transparent red fill + solid outline
        overlay = vis.copy()
        cv2.circle(overlay, (ix, iy), r_mask, (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.18, vis, 0.82, 0, vis)
        cv2.circle(vis, (ix, iy), r_mask, (0, 0, 255), max(2, scale))

        # Offset vector (cyan line)
        if (ix, iy) != (fcx, fcy):
            cv2.line(vis, (fcx, fcy), (ix, iy), (255, 220, 0), max(1, scale))

        # Green dot: frame geometric centre
        dot_r = max(4, scale * 3)
        cv2.circle(vis, (fcx, fcy), dot_r, (30, 220, 30), -1)
        cv2.circle(vis, (fcx, fcy), dot_r, (255, 255, 255), max(1, scale))

        # Black dot: detected icon centre
        cv2.circle(vis, (ix, iy), dot_r, (0, 0, 0), -1)
        cv2.circle(vis, (ix, iy), dot_r, (255, 255, 255), max(1, scale))

        # Legend panel
        font = cv2.FONT_HERSHEY_SIMPLEX
        fs   = max(0.38, scale * 0.30)
        pad  = 4
        lines = [
            ("● black  = icon centre",   (220, 220, 220)),
            ("● green  = frame centre",  (30,  200,  30)),
            ("○ red    = mask area",     (60,   60, 255)),
            ("○ yellow = search bound",  (0,   200, 255)),
            ("",                         (0,     0,   0)),
            (f"Icon   ({icon.x}, {icon.y})",           (255, 255, 100)),
            (f"Centre ({cx}, {cy})",                    (255, 255, 100)),
            (f"Offset ({offset_x:+d}, {offset_y:+d}) px", (255, 255, 100)),
            (f"Radius  {icon.radius} px",               (255, 255, 100)),
        ]
        lh      = int(fs * 36) + 2
        label_h = len(lines) * lh + pad * 2
        label_w = 220
        sub = vis[pad: pad + label_h, pad: pad + label_w]
        dark = np.zeros_like(sub)
        cv2.addWeighted(dark, 0.55, sub, 0.45, 0, sub)
        for i, (txt, color) in enumerate(lines):
            if txt:
                cv2.putText(vis, txt, (pad + 4, pad + lh * i + lh - 4),
                            font, fs, color, 1, cv2.LINE_AA)

        out_path = "debug_icon_mask.png"
        cv2.imwrite(out_path, vis)

        tag = "valid" if icon.is_valid else "fallback (centre)"
        print(
            f"[icon-mask debug] {tag}\n"
            f"  icon   = ({icon.x}, {icon.y})  frame centre = ({cx}, {cy})\n"
            f"  offset = ({offset_x:+d}, {offset_y:+d}) px   radius = {icon.radius} px\n"
            f"  saved  -> {out_path}"
        )

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
        if result.found:
            status_message = (
                f"定位成功：({int(result.x)}, {int(result.y)})，"
                f"置信度 {result.confidence:.2f}"
            )
            self.window.set_state("Tracking", status_message)
        else:
            self.window.set_state("Lost", f"定位失败：{result.message}")
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
