from __future__ import annotations

import time

import cv2
import mss
import numpy as np
from PySide6.QtCore import QEventLoop, QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QGuiApplication, QImage, QKeyEvent, QMouseEvent, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QWidget

from lkmap.capture.base import CaptureStrategy
from lkmap.models import CaptureRegion, FrameData


class RegionSelectionOverlay(QWidget):
    region_selected = Signal(object)
    selection_canceled = Signal()

    def __init__(self, background: QPixmap, monitor: dict[str, int]) -> None:
        super().__init__()
        self._background = background
        self._monitor = monitor
        self._dragging = False
        self._start = QPoint()
        self._end = QPoint()

        self.setGeometry(
            monitor["left"],
            monitor["top"],
            monitor["width"],
            monitor["height"],
        )
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._hint = QLabel("拖拽框选小地图区域，按 ESC 取消", self)
        self._hint.setStyleSheet(
            "background: rgba(20, 20, 20, 180); color: white; padding: 8px 12px; border-radius: 8px;"
        )
        self._hint.move(24, 24)
        self._hint.adjustSize()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._background)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 80))

        selection = QRect(self._start, self._end).normalized()
        if selection.isValid() and not selection.isNull():
            painter.drawPixmap(selection, self._background, selection)
            painter.setPen(QPen(QColor(255, 196, 68), 2))
            painter.drawRect(selection)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._dragging = True
        self._start = event.position().toPoint()
        self._end = self._start
        self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if not self._dragging:
            return
        self._end = event.position().toPoint()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or not self._dragging:
            return
        self._dragging = False
        self._end = event.position().toPoint()
        selection = QRect(self._start, self._end).normalized()
        if selection.width() < 40 or selection.height() < 40:
            self.selection_canceled.emit()
            self.close()
            return

        region = CaptureRegion(
            left=self._monitor["left"] + selection.left(),
            top=self._monitor["top"] + selection.top(),
            width=selection.width(),
            height=selection.height(),
        )
        self.region_selected.emit(region)
        self.close()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.selection_canceled.emit()
            self.close()


class ManualRegionCaptureStrategy(CaptureStrategy):
    def __init__(self, region: CaptureRegion | None = None) -> None:
        self.region = region

    def select_region(self, parent=None, initial: CaptureRegion | None = None) -> CaptureRegion | None:
        if initial is not None:
            self.region = initial

        with mss.mss() as screen_capture:
            monitor = screen_capture.monitors[0]
            screenshot = np.array(screen_capture.grab(monitor))
            rgba = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2RGBA)
            image = QImage(
                rgba.data,
                rgba.shape[1],
                rgba.shape[0],
                rgba.shape[1] * 4,
                QImage.Format.Format_RGBA8888,
            ).copy()
            pixmap = QPixmap.fromImage(image)

        overlay = RegionSelectionOverlay(pixmap, monitor)
        loop = QEventLoop()
        selected_region: CaptureRegion | None = None

        def handle_selected(region: CaptureRegion) -> None:
            nonlocal selected_region
            selected_region = region
            loop.quit()

        def handle_canceled() -> None:
            loop.quit()

        overlay.region_selected.connect(handle_selected)
        overlay.selection_canceled.connect(handle_canceled)
        overlay.show()
        overlay.raise_()
        overlay.activateWindow()
        loop.exec()
        overlay.deleteLater()

        self.region = selected_region
        return selected_region

    def current_frame(self) -> FrameData | None:
        if not self.region or not self.region.is_valid:
            return None

        with mss.mss() as screen_capture:
            screenshot = np.array(screen_capture.grab(self.region.to_dict()))
        frame = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)
        return FrameData(image=frame, timestamp=time.time())

