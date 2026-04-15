from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from lkmap.models import AppSettings, ResourcePoint, ensure_parent_dir

X_MIN = -12
Y_MIN = -11
TILE_SIZE = 256
SCALE = 1


class AssetService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def load_display_map(self) -> np.ndarray:
        image = cv2.imread(self.settings.assets.map_path)
        if image is None:
            raise FileNotFoundError(f"找不到地图文件: {self.settings.assets.map_path}")
        return image

    def load_feature_map(self) -> np.ndarray:
        feature_map = cv2.imread(self.settings.assets.feature_map_path)
        if feature_map is not None:
            return feature_map
        return self.load_display_map()

    def load_resource_points(self) -> list[ResourcePoint]:
        json_path = Path(self.settings.assets.points_path)
        if not json_path.exists():
            return []

        with json_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        raw_points = data if isinstance(data, list) else data.get("points", [])
        resource_points: list[ResourcePoint] = []
        for item in raw_points:
            lat = item["point"]["lat"]
            lng = item["point"]["lng"]
            px = int((lng / TILE_SIZE - X_MIN) * TILE_SIZE * SCALE)
            py = int((lat / TILE_SIZE - Y_MIN) * TILE_SIZE * SCALE)
            resource_points.append(
                ResourcePoint(
                    point_id=str(item.get("id", "")),
                    resource_type=str(item.get("markType", "unknown")),
                    x=px,
                    y=py,
                )
            )
        return resource_points

    def feature_cache_path(self) -> Path:
        return ensure_parent_dir(self.settings.assets.feature_cache_path)

    def legacy_feature_cache_path(self) -> Path:
        return Path(self.settings.assets.legacy_feature_cache_path)

