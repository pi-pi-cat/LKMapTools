from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lkmap.models import AppSettings, CaptureRegion, ensure_parent_dir


class ConfigService:
    def __init__(self, path: str | Path = "config/settings.json") -> None:
        self.path = Path(path)
        self.legacy_path = Path("config.json")

    def load(self) -> AppSettings:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            settings = AppSettings.from_dict(data)
        elif self.legacy_path.exists():
            settings = self._migrate_legacy_config()
            self.save(settings)
        else:
            settings = AppSettings(config_path=str(self.path))
            self.save(settings)

        settings.config_path = str(self.path)
        return settings

    def save(self, settings: AppSettings) -> None:
        target = ensure_parent_dir(self.path)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(settings.to_dict(), handle, ensure_ascii=False, indent=2)

    def save_capture_region(self, settings: AppSettings, region: CaptureRegion) -> None:
        settings.capture_region = region
        self.save(settings)

    def save_view_settings(self, settings: AppSettings, *, zoom: float, show_resources: bool) -> None:
        settings.view.zoom = zoom
        settings.view.show_resources = show_resources
        self.save(settings)

    def _migrate_legacy_config(self) -> AppSettings:
        with self.legacy_path.open("r", encoding="utf-8") as handle:
            legacy_data: dict[str, Any] = json.load(handle)

        settings = AppSettings()
        region = legacy_data.get("MINIMAP")
        settings.capture_region = CaptureRegion.from_dict(region)
        settings.assets.map_path = str(legacy_data.get("ORB_MAP_PATH", settings.assets.map_path))
        settings.assets.feature_map_path = str(
            legacy_data.get("ORB_MAP_NOEDGE_PATH", settings.assets.feature_map_path)
        )
        settings.assets.points_path = str(legacy_data.get("POINTS_PATH", settings.assets.points_path))
        settings.assets.feature_cache_path = "cache/orb_features.npz"
        settings.view.zoom = 1.0

        settings.orb.map_nfeatures = int(legacy_data.get("ORB_NFEATURES", settings.orb.map_nfeatures))
        settings.orb.minimap_nfeatures = int(
            legacy_data.get("ORB_MINI_NFEATURES", settings.orb.minimap_nfeatures)
        )
        settings.orb.min_match_count = int(
            legacy_data.get("ORB_MIN_MATCH_COUNT", settings.orb.min_match_count)
        )
        grid = legacy_data.get("ORB_GRID", [settings.orb.grid_rows, settings.orb.grid_cols])
        if isinstance(grid, list | tuple) and len(grid) == 2:
            settings.orb.grid_rows = int(grid[0])
            settings.orb.grid_cols = int(grid[1])
        settings.orb.max_kp_per_layer = int(
            legacy_data.get("MAX_KP_PER_LAYER", settings.orb.max_kp_per_layer)
        )
        return settings

