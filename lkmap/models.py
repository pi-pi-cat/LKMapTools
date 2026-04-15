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
    min_valid_scale: float = 0.35
    max_valid_scale: float = 2.20

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
class TemplateSettings:
    """多尺度金字塔模板匹配定位器参数。"""

    # 全局搜索 —— 遍历的缩放比例（minimap像素 : 大地图像素）
    global_scales: list = field(
        default_factory=lambda: [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8]
    )
    # 粗筛金字塔降采样倍数（越大越快，但精度越低）
    pyramid_factor: int = 4
    # 粗筛阶段最低得分（低于此值的候选不进入精化）
    coarse_threshold: float = 0.15
    # 精化/本地匹配最低置信度（低于此值视为失败）
    match_threshold: float = 0.35
    # Canny 边缘检测低/高阈值（用于预处理，抑制图标干扰）
    canny_low: int = 50
    canny_high: int = 150
    # 本地搜索半径（大地图像素）
    local_search_radius: int = 500
    # 连续失败时每次扩大的搜索半径增量
    local_expand_step: int = 150
    # 连续失败多少次后切换到全局搜索
    global_search_after_failures: int = 8
    # 跳变检测阈值（大地图像素）
    jump_limit: int = 250
    # 定位刷新间隔（毫秒）
    refresh_interval_ms: int = 80
    # 是否启用光流追踪（在草地等无纹理区域代替模板匹配）
    use_optical_flow: bool = True
    # 光流模式下每隔多少帧用模板匹配校正一次漂移
    flow_validation_interval: int = 15
    # 光流一致性低于此值时认为帧间变化过大，降级到模板匹配
    flow_consistency_threshold: float = 3.0
    # 模板校正死区：漂移小于此值（大地图像素）时忽略校正，避免微小跳动
    correction_min_drift: int = 12
    # 将偏移量分摊到多少帧里平滑施加（越大越平滑，但修正越慢）
    correction_smooth_frames: int = 8
    # 传送检测：小地图平均亮度低于此值时认为是黑屏/传送画面（0~255）
    transition_dark_threshold: int = 15
    # 传送检测：可选的传送画面模板图路径（截小地图区域保存）
    # 留空则只用黑屏检测；填路径后同时做模板匹配，更可靠
    transition_template_path: str = ""
    # 全局搜索连续失败超过此次数 → 判定为不在大地图范围内（副本/地下城等）
    offmap_max_failures: int = 10
    # SEARCHING 状态下每隔多少帧才执行一次全局搜索（限流，避免CPU浪费）
    searching_interval: int = 4
    # 跟踪失败后先进入"本地恢复窗口"，尝试本地搜索最多这么多帧，再切全局
    local_recovery_frames: int = 6
    # 直方图相关度阈值：TRACKING 中连续两帧低于此值 → 场景突变，停止光流切全局搜索
    # 范围 0~1；正常移动约 0.80~0.95，传送/菜单切换通常 < 0.65
    hist_scene_break_threshold: float = 0.70
    # 光流每帧位移（小地图像素）小于此值时认为"画面静止"
    static_flow_threshold: float = 0.5
    # 连续静止帧数达到此值时，强制做一次模板位置验证（防止传送后粘在错误坐标）
    static_verify_interval: int = 30
    # ── 性能 & 精度配置 ───────────────────────────────────────────────────────
    # 是否启用 CUDA 加速（需要 opencv-contrib-python + NVIDIA GPU）
    # 影响：模板匹配 + 光流均走 GPU，FPS 可提升 2~5 倍
    use_cuda: bool = True
    # Farneback 光流参数（越小越快，越大越平滑）
    flow_win_size: int = 11      # 搜索窗口大小（原硬编码 15）
    flow_levels: int = 2         # 图像金字塔层数（原硬编码 3）
    flow_iterations: int = 2     # 每层迭代次数（原硬编码 3）
    # 输出坐标偏移（大地图像素）——用于修正系统性定位误差
    # 例如：箭头图标本身相对角色中心有偏移时在此处补偿
    position_offset_x: int = 0
    position_offset_y: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "TemplateSettings":
        base = cls()
        if not data:
            return base
        scales_raw = data.get("global_scales", base.global_scales)
        scales = [float(s) for s in scales_raw] if isinstance(scales_raw, list) else base.global_scales
        return cls(
            global_scales=scales,
            pyramid_factor=int(data.get("pyramid_factor", base.pyramid_factor)),
            coarse_threshold=float(data.get("coarse_threshold", base.coarse_threshold)),
            match_threshold=float(data.get("match_threshold", base.match_threshold)),
            canny_low=int(data.get("canny_low", base.canny_low)),
            canny_high=int(data.get("canny_high", base.canny_high)),
            local_search_radius=int(data.get("local_search_radius", base.local_search_radius)),
            local_expand_step=int(data.get("local_expand_step", base.local_expand_step)),
            global_search_after_failures=int(
                data.get("global_search_after_failures", base.global_search_after_failures)
            ),
            jump_limit=int(data.get("jump_limit", base.jump_limit)),
            refresh_interval_ms=int(data.get("refresh_interval_ms", base.refresh_interval_ms)),
            use_optical_flow=bool(data.get("use_optical_flow", base.use_optical_flow)),
            flow_validation_interval=int(
                data.get("flow_validation_interval", base.flow_validation_interval)
            ),
            flow_consistency_threshold=float(
                data.get("flow_consistency_threshold", base.flow_consistency_threshold)
            ),
            correction_min_drift=int(data.get("correction_min_drift", base.correction_min_drift)),
            correction_smooth_frames=int(
                data.get("correction_smooth_frames", base.correction_smooth_frames)
            ),
            transition_dark_threshold=int(
                data.get("transition_dark_threshold", base.transition_dark_threshold)
            ),
            transition_template_path=str(
                data.get("transition_template_path", base.transition_template_path)
            ),
            offmap_max_failures=int(data.get("offmap_max_failures", base.offmap_max_failures)),
            searching_interval=int(data.get("searching_interval", base.searching_interval)),
            local_recovery_frames=int(
                data.get("local_recovery_frames", base.local_recovery_frames)
            ),
            hist_scene_break_threshold=float(
                data.get("hist_scene_break_threshold", base.hist_scene_break_threshold)
            ),
            static_flow_threshold=float(
                data.get("static_flow_threshold", base.static_flow_threshold)
            ),
            static_verify_interval=int(
                data.get("static_verify_interval", base.static_verify_interval)
            ),
            use_cuda=bool(data.get("use_cuda", base.use_cuda)),
            flow_win_size=int(data.get("flow_win_size", base.flow_win_size)),
            flow_levels=int(data.get("flow_levels", base.flow_levels)),
            flow_iterations=int(data.get("flow_iterations", base.flow_iterations)),
            position_offset_x=int(data.get("position_offset_x", base.position_offset_x)),
            position_offset_y=int(data.get("position_offset_y", base.position_offset_y)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "global_scales": self.global_scales,
            "pyramid_factor": self.pyramid_factor,
            "coarse_threshold": self.coarse_threshold,
            "match_threshold": self.match_threshold,
            "canny_low": self.canny_low,
            "canny_high": self.canny_high,
            "local_search_radius": self.local_search_radius,
            "local_expand_step": self.local_expand_step,
            "global_search_after_failures": self.global_search_after_failures,
            "jump_limit": self.jump_limit,
            "refresh_interval_ms": self.refresh_interval_ms,
            "use_optical_flow": self.use_optical_flow,
            "flow_validation_interval": self.flow_validation_interval,
            "flow_consistency_threshold": self.flow_consistency_threshold,
            "correction_min_drift": self.correction_min_drift,
            "correction_smooth_frames": self.correction_smooth_frames,
            "transition_dark_threshold": self.transition_dark_threshold,
            "transition_template_path": self.transition_template_path,
            "offmap_max_failures": self.offmap_max_failures,
            "searching_interval": self.searching_interval,
            "local_recovery_frames": self.local_recovery_frames,
            "hist_scene_break_threshold": self.hist_scene_break_threshold,
            "static_flow_threshold": self.static_flow_threshold,
            "static_verify_interval": self.static_verify_interval,
            "use_cuda": self.use_cuda,
            "flow_win_size": self.flow_win_size,
            "flow_levels": self.flow_levels,
            "flow_iterations": self.flow_iterations,
            "position_offset_x": self.position_offset_x,
            "position_offset_y": self.position_offset_y,
        }


@dataclass(slots=True)
class IconMask:
    """玩家图标遮罩——在框选小地图时自动检测并缓存，定位时挖去该区域减少干扰。"""

    # 图标中心相对于捕获区域左上角的坐标（-1 表示未检测到，使用中心兜底）
    x: int = -1
    y: int = -1
    # 遮掩半径（像素，基于捕获区域分辨率）
    radius: int = 20

    @property
    def is_valid(self) -> bool:
        return self.x >= 0 and self.y >= 0

    def to_dict(self) -> dict[str, int]:
        return {"x": self.x, "y": self.y, "radius": self.radius}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "IconMask":
        if not data:
            return cls()
        return cls(
            x=int(data.get("x", -1)),
            y=int(data.get("y", -1)),
            radius=int(data.get("radius", 20)),
        )


@dataclass(slots=True)
class AppSettings:
    config_path: str = "config/settings.toml"
    capture_region: CaptureRegion | None = None
    assets: AssetPaths = field(default_factory=AssetPaths)
    view: ViewSettings = field(default_factory=ViewSettings)
    orb: OrbSettings = field(default_factory=OrbSettings)
    template: TemplateSettings = field(default_factory=TemplateSettings)
    icon_mask: IconMask = field(default_factory=IconMask)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppSettings":
        return cls(
            config_path=str(data.get("config_path", "config/settings.toml")),
            capture_region=CaptureRegion.from_dict(data.get("capture_region")),
            assets=AssetPaths.from_dict(data.get("assets")),
            view=ViewSettings.from_dict(data.get("view")),
            orb=OrbSettings.from_dict(data.get("orb")),
            template=TemplateSettings.from_dict(data.get("template")),
            icon_mask=IconMask.from_dict(data.get("icon_mask")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "capture_region": self.capture_region.to_dict() if self.capture_region else None,
            "assets": self.assets.to_dict(),
            "view": self.view.to_dict(),
            "orb": self.orb.to_dict(),
            "template": self.template.to_dict(),
            "icon_mask": self.icon_mask.to_dict(),
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

