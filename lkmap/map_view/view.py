from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from lkmap.layers.resources import ResourceLayer
from lkmap.models import LocationResult


def _numpy_to_pixmap(image: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb.shape
    qimage = QImage(
        rgb.data,
        width,
        height,
        width * 3,
        QImage.Format.Format_RGB888,
    ).copy()
    return QPixmap.fromImage(qimage)


class MapView(QGraphicsView):
    zoom_changed = Signal(float)

    def __init__(self) -> None:
        super().__init__()
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QBrush(QColor("#1c1f24")))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self._map_item: QGraphicsPixmapItem | None = None
        self._player_item = QGraphicsEllipseItem(-8, -8, 16, 16)
        self._player_item.setPen(QPen(QColor("#ffffff"), 2))
        self._player_item.setBrush(QBrush(QColor("#e24a33")))
        self._player_item.setZValue(10)
        self._player_item.setVisible(False)
        self._scene.addItem(self._player_item)
        self._resource_layer = ResourceLayer()
        self._resource_layer.attach(self._scene)
        self._zoom = 1.0
        self._player_point: QPointF | None = None

    @property
    def zoom_value(self) -> float:
        return self._zoom

    def set_map_image(self, image: np.ndarray) -> None:
        pixmap = _numpy_to_pixmap(image)
        if self._map_item is None:
            self._map_item = self._scene.addPixmap(pixmap)
            self._map_item.setZValue(0)
        else:
            self._map_item.setPixmap(pixmap)
        self._scene.setSceneRect(QRectF(pixmap.rect()))
        self._player_item.setVisible(True)

    def set_resource_points(self, points) -> None:
        self._resource_layer.set_points(points)
        self._resource_layer.update_viewport(self.mapToScene(self.viewport().rect()).boundingRect())

    def set_show_resources(self, visible: bool) -> None:
        self._resource_layer.set_visible(visible)
        self._resource_layer.update_viewport(self.mapToScene(self.viewport().rect()).boundingRect())

    def apply_zoom(self, zoom: float) -> None:
        zoom = max(0.4, min(zoom, 3.0))
        factor = zoom / self._zoom
        self._zoom = zoom
        self.scale(factor, factor)
        self._recenter()
        self.zoom_changed.emit(self._zoom)

    def set_location(self, result: LocationResult) -> None:
        if result.x is None or result.y is None:
            return
        self._player_point = QPointF(result.x, result.y)
        self._player_item.setPos(self._player_point)
        self._recenter()

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.apply_zoom(self._zoom * delta)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._recenter()

    def _recenter(self) -> None:
        if self._player_point is not None:
            self.centerOn(self._player_point)
        view_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        self._resource_layer.update_viewport(view_rect)
