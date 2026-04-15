from __future__ import annotations

import time

import cv2
import mss
import numpy as np
from PySide6.QtCore import QEventLoop, QPoint, QRect, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QLabel, QPushButton, QWidget

from lkmap.capture.base import CaptureStrategy
from lkmap.models import CaptureRegion, FrameData


class RegionSelectionOverlay(QWidget):
    """全屏覆盖层：预置一个可移动/缩放的圆形选框，按 Enter 或双击圆内确认。"""

    region_selected = Signal(object)
    selection_canceled = Signal()

    _HANDLE_SIZE = 16
    _MIN_SIZE = 60
    _DEFAULT_SIZE = 200

    def __init__(
        self,
        background: QPixmap,
        monitor: dict[str, int],
        initial: CaptureRegion | None = None,
    ) -> None:
        super().__init__()
        self._background = background
        self._monitor = monitor

        mon_w = monitor["width"]
        mon_h = monitor["height"]

        if initial and initial.is_valid:
            x = initial.left - monitor["left"]
            y = initial.top - monitor["top"]
            size = min(initial.width, initial.height)
            self._sq = QRect(x, y, size, size)
        else:
            size = min(self._DEFAULT_SIZE, mon_w // 4, mon_h // 4)
            self._sq = QRect((mon_w - size) // 2, (mon_h - size) // 2, size, size)

        self._drag_mode: str | None = None  # 'move' | 'tl' | 'tr' | 'bl' | 'br'
        self._drag_origin = QPoint()
        self._sq_at_drag = QRect()

        self.setGeometry(monitor["left"], monitor["top"], mon_w, mon_h)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)

        self._hint = QLabel("拖动圆形移位 · 拖动角点缩放 · 双击或 Enter 确认 · ESC 取消", self)
        self._hint.setStyleSheet(
            "background: rgba(20,20,20,180); color: white; padding: 8px 12px; border-radius: 8px;"
        )
        self._hint.move(24, 24)
        self._hint.adjustSize()

        self._confirm_btn = QPushButton("✓ 确认", self)
        self._confirm_btn.setStyleSheet(
            "background: #ffc444; color: #1a1a1a; font-weight: bold;"
            "padding: 6px 18px; border-radius: 6px; border: none;"
        )
        self._confirm_btn.adjustSize()
        self._confirm_btn.clicked.connect(self._confirm)
        self._reposition_btn()

    # ── 内部辅助 ──────────────────────────────────────────────────────────────

    def _reposition_btn(self) -> None:
        """将确认按钮放在圆形正下方。"""
        btn_x = self._sq.center().x() - self._confirm_btn.width() // 2
        btn_y = self._sq.bottom() + 12
        mon_h = self._monitor["height"]
        if btn_y + self._confirm_btn.height() > mon_h - 8:
            btn_y = self._sq.top() - self._confirm_btn.height() - 12
        self._confirm_btn.move(btn_x, btn_y)

    def _corner_rects(self) -> dict[str, QRect]:
        h = self._HANDLE_SIZE
        half = h // 2
        r = self._sq
        return {
            "tl": QRect(r.left() - half, r.top() - half, h, h),
            "tr": QRect(r.right() - half, r.top() - half, h, h),
            "bl": QRect(r.left() - half, r.bottom() - half, h, h),
            "br": QRect(r.right() - half, r.bottom() - half, h, h),
        }

    def _hit_test(self, pos: QPoint) -> str | None:
        for name, rect in self._corner_rects().items():
            if rect.contains(pos):
                return name
        if self._sq.contains(pos):
            return "move"
        return None

    def _cursor_for_hit(self, hit: str | None) -> Qt.CursorShape:
        if hit == "move":
            return Qt.CursorShape.SizeAllCursor
        if hit in ("tl", "br"):
            return Qt.CursorShape.SizeFDiagCursor
        if hit in ("tr", "bl"):
            return Qt.CursorShape.SizeBDiagCursor
        return Qt.CursorShape.ArrowCursor

    # ── 绘制 ──────────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 背景截图
        painter.drawPixmap(0, 0, self._background)
        # 暗色蒙层
        painter.fillRect(self.rect(), QColor(0, 0, 0, 110))

        sq = self._sq

        # 圆形区域透出原始画面
        path = QPainterPath()
        path.addEllipse(sq)
        painter.setClipPath(path)
        painter.drawPixmap(sq, self._background, sq)
        painter.setClipping(False)

        # 黄色圆形边框
        painter.setPen(QPen(QColor(255, 196, 68), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(sq)

        # 虚线十字准星
        pen = QPen(QColor(255, 196, 68, 140), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        cx, cy = sq.center().x(), sq.center().y()
        painter.drawLine(cx, sq.top(), cx, sq.bottom())
        painter.drawLine(sq.left(), cy, sq.right(), cy)

        # 四角拖拽手柄（实心圆点）
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setBrush(QColor(255, 196, 68))
        for rect in self._corner_rects().values():
            painter.drawEllipse(rect)

    # ── 鼠标事件 ──────────────────────────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton:
            return
        hit = self._hit_test(event.position().toPoint())
        if hit is None:
            return
        self._drag_mode = hit
        self._drag_origin = event.position().toPoint()
        self._sq_at_drag = QRect(self._sq)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        pos = event.position().toPoint()

        if self._drag_mode is None:
            self.setCursor(self._cursor_for_hit(self._hit_test(pos)))
            return

        delta = pos - self._drag_origin
        sq = QRect(self._sq_at_drag)

        if self._drag_mode == "move":
            sq.translate(delta)
        else:
            # 保持对角固定，强制正方形
            fixed_map = {"tl": sq.bottomRight(), "tr": sq.bottomLeft(),
                         "bl": sq.topRight(),    "br": sq.topLeft()}
            moving_map = {"tl": sq.topLeft(),    "tr": sq.topRight(),
                          "bl": sq.bottomLeft(), "br": sq.bottomRight()}
            fixed = fixed_map[self._drag_mode]
            moving = moving_map[self._drag_mode] + delta

            dx = abs(moving.x() - fixed.x())
            dy = abs(moving.y() - fixed.y())
            side = max(dx, dy, self._MIN_SIZE)

            sign_x = 1 if moving.x() >= fixed.x() else -1
            sign_y = 1 if moving.y() >= fixed.y() else -1
            new_corner = QPoint(fixed.x() + sign_x * side, fixed.y() + sign_y * side)
            sq = QRect(fixed, new_corner).normalized()

        self._sq = sq
        self._reposition_btn()
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_mode = None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._sq.contains(event.position().toPoint())
        ):
            self._confirm()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.selection_canceled.emit()
            self.close()
        elif event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._confirm()

    # ── 确认 ──────────────────────────────────────────────────────────────────

    def _confirm(self) -> None:
        if self._sq.width() < self._MIN_SIZE:
            self.selection_canceled.emit()
            self.close()
            return
        region = CaptureRegion(
            left=self._monitor["left"] + self._sq.left(),
            top=self._monitor["top"] + self._sq.top(),
            width=self._sq.width(),
            height=self._sq.height(),
        )
        self.region_selected.emit(region)
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

        overlay = RegionSelectionOverlay(pixmap, monitor, initial=self.region)
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
