from __future__ import annotations

import hashlib

from PySide6.QtCore import QPointF, QRectF
from PySide6.QtGui import QColor, QBrush, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsScene

from lkmap.models import ResourcePoint


class ResourceLayer:
    def __init__(self) -> None:
        self._scene: QGraphicsScene | None = None
        self._items: list[tuple[ResourcePoint, QGraphicsEllipseItem]] = []
        self._visible = True

    def attach(self, scene: QGraphicsScene) -> None:
        self._scene = scene

    def set_points(self, points: list[ResourcePoint]) -> None:
        if self._scene is None:
            return
        for _, item in self._items:
            self._scene.removeItem(item)
        self._items.clear()

        for point in points:
            color = self._color_for_type(point.resource_type)
            item = QGraphicsEllipseItem(-5, -5, 10, 10)
            item.setPen(QPen(QColor("white"), 1))
            item.setBrush(QBrush(color))
            item.setPos(QPointF(point.x, point.y))
            item.setZValue(5)
            item.setVisible(self._visible)
            self._scene.addItem(item)
            self._items.append((point, item))

    def set_visible(self, visible: bool) -> None:
        self._visible = visible
        for _, item in self._items:
            item.setVisible(visible)

    def update_viewport(self, view_rect: QRectF) -> None:
        margin = 100.0
        expanded = view_rect.adjusted(-margin, -margin, margin, margin)
        for point, item in self._items:
            item.setVisible(self._visible and expanded.contains(QPointF(point.x, point.y)))

    def _color_for_type(self, resource_type: str) -> QColor:
        digest = hashlib.sha1(resource_type.encode("utf-8")).digest()
        hue = int.from_bytes(digest[:2], byteorder="big") % 360
        return QColor.fromHsv(hue, 180, 220)
