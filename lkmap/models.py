from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class CaptureRegion:
    left: int = 0
    top: int = 0
    width: int = 0
    height: int = 0

    @property
    def is_valid(self) -> bool:
        return self.width > 0 and self.height > 0

    def to_dict(self) -> dict[str, int]:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "width": int(self.width),
            "height": int(self.height),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "CaptureRegion | None":
        if not data:
            return None
        region = cls(
            left=int(data.get("left", 0)),
            top=int(data.get("top", 0)),
            width=int(data.get("width", 0)),
            height=int(data.get("height", 0)),
        )
        return region if region.is_valid else None


@dataclass(slots=True)
class AssetPaths:
    map_path: str = "assest/raw.png"
    feature_map_path: str = "assest/raw_noedge.png"
    points_path: str = "assest/points.json"
    feature_cache_path: str = "cache/orb_features.npz"
    legacy_feature_cache_path: str = "assest/ORB_features.npz"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "AssetPaths":
        base = cls()
        if not data:
            return base
        return cls(
            map_path=str(data.get("map_path", base.map_path)),
            feature_map_path=str(data.get("feature_map_path", base.feature_map_path)),
            points_path=str(data.get("points_path", base.points_path)),
            feature_cache_path=str(data.get("feature_cache_path", base.feature_cache_path)),
            legacy_feature_cache_path=str(
                data.get("legacy_feature_cache_path", base.legacy_feature_cache_path)
            ),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "map_path": self.map_path,
            "feature_map_path": self.feature_map_path,
            "points_path": self.points_path,
            "feature_cache_path": self.feature_cache_path,
            "legacy_feature_cache_path": self.legacy_feature_cache_path,
        }


@dataclass(slots=True)
class ViewSettings:
    zoom: float = 1.0
    show_resources: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ViewSettings":
        base = cls()
        if not data:
            return base
        return cls(
            zoom=float(data.get("zoom", base.zoom)),
            show_resources=bool(data.get("show_resources", base.show_resources)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"zoom": self.zoom, "show_resources": self.show_resources}


@dataclass(slots=True)
class OrbSettings:
    map_nfeatures: int = 300_000
    minimap_nfeatures: int = 2_500
    min_match_count: int = 5
    scale_factor: float = 1.2
    nlevels: int = 8
    fast_threshold: int = 5
    edge_threshold: int = 25
    grid_rows: int = 120
    grid_cols: int = 120
    max_kp_per_layer: int = 100_000
    refresh_interval_ms: int = 80
    local_search_radius: int = 800
    local_search_expand_step: int = 200
    global_search_after_failures: int = 10
    match_distance_threshold: int = 50
    max_good_matches: int = 80
    ransac_threshold: float = 3.0
    jump_limit: int = 220

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "OrbSettings":
        base = cls()
        if not data:
            return base
        values = {field_name: data.get(field_name, getattr(base, field_name)) for field_name in cls.__dataclass_fields__}
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in self.__dataclass_fields__}


@dataclass(slots=True)
class AppSettings:
    config_path: str = "config/settings.json"
    capture_region: CaptureRegion | None = None
    assets: AssetPaths = field(default_factory=AssetPaths)
    view: ViewSettings = field(default_factory=ViewSettings)
    orb: OrbSettings = field(default_factory=OrbSettings)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppSettings":
        return cls(
            config_path=str(data.get("config_path", "config/settings.json")),
            capture_region=CaptureRegion.from_dict(data.get("capture_region")),
            assets=AssetPaths.from_dict(data.get("assets")),
            view=ViewSettings.from_dict(data.get("view")),
            orb=OrbSettings.from_dict(data.get("orb")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "capture_region": self.capture_region.to_dict() if self.capture_region else None,
            "assets": self.assets.to_dict(),
            "view": self.view.to_dict(),
            "orb": self.orb.to_dict(),
        }


@dataclass(slots=True)
class FrameData:
    image: np.ndarray
    timestamp: float


@dataclass(slots=True)
class ResourcePoint:
    point_id: str
    resource_type: str
    x: int
    y: int


@dataclass(slots=True)
class LocationResult:
    found: bool
    x: float | None = None
    y: float | None = None
    confidence: float = 0.0
    mode: str = "idle"
    message: str = ""
    timestamp: float = 0.0


def ensure_parent_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

