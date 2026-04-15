"""
多尺度金字塔模板匹配 + 光流追踪 + 场景状态机 定位器。

状态机（_state）：
  tracking   —— 正常追踪（光流 + 定期模板校正）
  searching  —— 全局搜索中（首次启动 / 传送结束后 / 重置后）
  transition —— 传送/切换场景画面中（黑屏或匹配到传送模板）
  off_map    —— 持续全局搜索失败，判定为不在大地图范围内（副本等）

状态转移：
  任意       → transition  :  检测到黑屏/传送模板
  transition → searching   :  传送画面消失，立即全局搜索
  searching  → tracking    :  全局搜索成功
  searching  → off_map     :  连续 offmap_max_failures 次全局搜索失败
  tracking   → searching   :  连续失败 >= global_search_after_failures
  off_map    → transition  :  检测到传送（然后再进入 searching）
  off_map    → searching   :  每隔 offmap_retry_interval 帧主动重试
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from lkmap.locator.base import LocatorStrategy
from lkmap.models import AppSettings, FrameData, LocationResult
from lkmap.services.assets import AssetService

_USE_CUDA: bool = True

# off_map 状态下每隔多少帧主动尝试一次全局搜索
_OFFMAP_RETRY_INTERVAL = 60


class PyramidLocator(LocatorStrategy):
    _TOP_K_COARSE = 5
    _MIN_TMPL_PX = 16
    _REFINE_MARGIN = 1.0

    def __init__(self, settings: AppSettings, assets: AssetService) -> None:
        self.settings = settings
        self.assets = assets
        self._initialized = False

        self._map_edge: np.ndarray | None = None
        self._map_edge_coarse: np.ndarray | None = None
        self._map_h = 0
        self._map_w = 0
        self._pyramid_factor = 1

        self._display_map: np.ndarray | None = None
        self._mask: np.ndarray | None = None

        # 场景状态机
        self._state: str = "searching"      # searching / tracking / transition / off_map
        self._global_failures: int = 0      # 全局搜索连续失败次数（用于 off_map 判定）
        self._offmap_retry_counter: int = 0
        self._searching_frame_counter: int = 0  # SEARCHING 状态的帧计数（用于限流）
        self._local_recovery_counter: int = 0   # 本地恢复窗口帧计数

        # 追踪状态
        self._last_point: tuple[int, int] | None = None
        self._cached_scale: float | None = None
        self._consecutive_failures: int = 0

        # 光流状态
        self._prev_mini_masked: np.ndarray | None = None
        self._flow_validation_counter: int = 0
        # 光流最近一帧的小地图像素位移（供静止检测使用）
        self._last_flow_dx: float = 0.0
        self._last_flow_dy: float = 0.0
        # 连续静止帧计数（位移 < static_flow_threshold 的帧数）
        self._static_frames_count: int = 0

        # 平滑校正
        self._pending_corr_x: float = 0.0
        self._pending_corr_y: float = 0.0
        self._corr_frames_left: int = 0

        # 传送检测模板（可选）
        self._transition_tmpl: np.ndarray | None = None

        self._cuda_matcher = None

    @property
    def display_map(self) -> np.ndarray:
        if self._display_map is None:
            raise RuntimeError("定位器尚未初始化")
        return self._display_map

    # ── 初始化 ────────────────────────────────────────────────────────────────

    def initialize(self) -> None:
        if self._initialized:
            return

        t = self.settings.template
        self._display_map = self.assets.load_display_map()
        feature_map = self.assets.load_feature_map()

        gray = cv2.cvtColor(feature_map, cv2.COLOR_BGR2GRAY)
        self._map_h, self._map_w = gray.shape
        self._map_edge = cv2.Canny(gray, t.canny_low, t.canny_high)

        pf = t.pyramid_factor
        self._pyramid_factor = pf
        self._map_edge_coarse = cv2.resize(
            self._map_edge,
            (max(1, self._map_w // pf), max(1, self._map_h // pf)),
            interpolation=cv2.INTER_AREA,
        )

        # 加载传送画面模板（可选）
        if t.transition_template_path:
            path = Path(t.transition_template_path)
            if path.exists():
                tmpl = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if tmpl is not None:
                    self._transition_tmpl = tmpl

        if _USE_CUDA:
            self._try_init_cuda()

        self._initialized = True

    def _try_init_cuda(self) -> None:
        try:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:  # type: ignore[attr-defined]
                self._cuda_matcher = cv2.cuda.createTemplateMatching(  # type: ignore[attr-defined]
                    cv2.CV_8U, cv2.TM_CCOEFF_NORMED
                )
        except Exception:
            pass

    # ── 主定位入口（状态机驱动）──────────────────────────────────────────────

    def locate(self, frame: FrameData) -> LocationResult:
        self.initialize()
        t = self.settings.template

        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        mask = self._ensure_mask(gray.shape)
        mini_masked = cv2.bitwise_and(gray, gray, mask=mask)

        # ══ 传送/场景切换检测（最高优先级，任何状态均响应）══════════════════
        in_transition = self._is_transition_frame(mini_masked)
        if in_transition:
            if self._state != "transition":
                self._enter_transition()
            # 注意：transition 帧内不更新 _prev_mini_masked，
            # 避免传送帧污染后续光流计算
            return LocationResult(
                found=False, mode="transition", message="传送/切换场景中",
                timestamp=frame.timestamp,
                x=self._last_point[0] if self._last_point else None,
                y=self._last_point[1] if self._last_point else None,
            )

        # 刚从传送画面恢复 → 立即全局搜索
        if self._state == "transition":
            self._state = "searching"
            self._global_failures = 0

        # ══ OFF_MAP 状态：冻结位置，定期重试 ═════════════════════════════════
        if self._state == "off_map":
            self._offmap_retry_counter += 1
            if self._offmap_retry_counter >= _OFFMAP_RETRY_INTERVAL:
                self._offmap_retry_counter = 0
                mini_edge = self._make_edge(mini_masked, mask, t)
                hit = self._global_search(mini_edge)
                if hit and hit[0] >= t.match_threshold:
                    self._state = "tracking"
                    self._global_failures = 0
                    self._prev_mini_masked = mini_masked
                    return self._commit(hit, frame.timestamp, "global")
            return LocationResult(
                found=False, mode="off_map", message="不在大地图范围内",
                timestamp=frame.timestamp,
                x=self._last_point[0] if self._last_point else None,
                y=self._last_point[1] if self._last_point else None,
            )

        # ══ SEARCHING 状态：限流全局搜索，避免每帧都跑 ══════════════════════
        if self._state == "searching":
            self._searching_frame_counter += 1
            # 限流：每隔 searching_interval 帧才执行一次全局搜索
            # 跳过的帧保持上次已知坐标，不浪费 CPU
            if self._searching_frame_counter % t.searching_interval != 0:
                return LocationResult(
                    found=False, mode="searching",
                    message=f"全局搜索等待({self._global_failures}/{t.offmap_max_failures})",
                    timestamp=frame.timestamp,
                    x=self._last_point[0] if self._last_point else None,
                    y=self._last_point[1] if self._last_point else None,
                )

            mini_edge = self._make_edge(mini_masked, mask, t)
            hit = self._global_search(mini_edge)
            self._prev_mini_masked = mini_masked
            if hit and hit[0] >= t.match_threshold:
                self._state = "tracking"
                self._global_failures = 0
                self._searching_frame_counter = 0
                self._local_recovery_counter = 0
                self._flow_validation_counter = 0
                return self._commit(hit, frame.timestamp, "global")
            # 全局搜索失败
            self._global_failures += 1
            if self._global_failures >= t.offmap_max_failures:
                self._state = "off_map"
                self._offmap_retry_counter = 0
                return LocationResult(
                    found=False, mode="off_map", message="判定为不在大地图范围内",
                    timestamp=frame.timestamp,
                    x=self._last_point[0] if self._last_point else None,
                    y=self._last_point[1] if self._last_point else None,
                )
            return LocationResult(
                found=False, mode="searching",
                message=f"全局搜索中({self._global_failures}/{t.offmap_max_failures})",
                timestamp=frame.timestamp,
                x=self._last_point[0] if self._last_point else None,
                y=self._last_point[1] if self._last_point else None,
            )

        # ══ TRACKING 状态：光流 + 定期模板校正 ═══════════════════════════════

        # ① 层级1：直方图相关度检查（场景突变预警）
        #    在黑屏/模板检测之后再做一道：捕捉那些不是纯黑屏但画面剧变的情况
        #    （传送结束时新场景加载完成、突然切到地图界面等）
        if self._prev_mini_masked is not None:
            corr = self._hist_correlation(self._prev_mini_masked, mini_masked)
            if corr < t.hist_scene_break_threshold:
                self._state = "searching"
                self._global_failures = 0
                self._searching_frame_counter = 0
                self._static_frames_count = 0
                self._prev_mini_masked = None
                return self._fail_result(
                    frame.timestamp,
                    f"画面突变(corr={corr:.2f})，切换全局重搜",
                )

        # ② 光流模式（含无纹理区域）
        if t.use_optical_flow and self._prev_mini_masked is not None:
            flow_result = self._locate_by_flow(mini_masked, mask, frame.timestamp)

            # 层级2：静止帧验证（防止传送后粘在错误坐标）
            #   传送结束后新场景静止加载：光流位移≈0 → 置信度高但坐标是错的
            #   解决方案：连续静止超过阈值时强制做一次模板验证
            if flow_result.found:
                if (abs(self._last_flow_dx) < t.static_flow_threshold
                        and abs(self._last_flow_dy) < t.static_flow_threshold):
                    self._static_frames_count += 1
                    if self._static_frames_count % t.static_verify_interval == 0:
                        mini_edge = self._make_edge(mini_masked, mask, t)
                        hit = self._local_search(mini_edge)
                        if hit is None or hit[0] < t.match_threshold:
                            # 当前坐标在大地图上对不上，强制全局重搜
                            self._state = "searching"
                            self._global_failures = 0
                            self._searching_frame_counter = 0
                            self._static_frames_count = 0
                            self._prev_mini_masked = None
                            return self._fail_result(
                                frame.timestamp,
                                f"静止{self._static_frames_count}帧验证失败，切换全局重搜",
                            )
                else:
                    self._static_frames_count = 0

            # 定期模板校正（修正光流累积漂移）
            self._flow_validation_counter += 1
            if self._flow_validation_counter >= t.flow_validation_interval:
                mini_edge = self._make_edge(mini_masked, mask, t)
                hit = self._local_search(mini_edge)
                if hit is not None and hit[0] >= t.match_threshold:
                    self._flow_validation_counter = 0
                    assert self._last_point is not None
                    drift_x = hit[1] - self._last_point[0]
                    drift_y = hit[2] - self._last_point[1]
                    if float(np.hypot(drift_x, drift_y)) >= t.correction_min_drift:
                        self._pending_corr_x += drift_x
                        self._pending_corr_y += drift_y
                        self._corr_frames_left = t.correction_smooth_frames
                else:
                    self._flow_validation_counter = t.flow_validation_interval // 2

            # 检查是否需要降级（连续失败）
            if flow_result.found:
                self._consecutive_failures = 0
                self._local_recovery_counter = 0
            else:
                self._consecutive_failures += 1
                # 先进"本地恢复窗口"：用本地模板搜索几帧（覆盖战斗/设置短暂遮挡）
                if self._consecutive_failures >= t.global_search_after_failures:
                    self._local_recovery_counter += 1
                    mini_edge = self._make_edge(mini_masked, mask, t)
                    hit = self._local_search(mini_edge)
                    if hit and hit[0] >= t.match_threshold:
                        # 本地恢复成功，不需要全局搜索
                        self._consecutive_failures = 0
                        self._local_recovery_counter = 0
                        self._prev_mini_masked = mini_masked
                        return self._commit(hit, frame.timestamp, "local")
                    if self._local_recovery_counter >= t.local_recovery_frames:
                        # 本地恢复窗口耗尽，真正需要全局搜索
                        self._state = "searching"
                        self._global_failures = 0
                        self._searching_frame_counter = 0
                        self._last_point = None
                        self._cached_scale = None
                        self._local_recovery_counter = 0

            self._prev_mini_masked = mini_masked
            return flow_result

        # ③ 本地模板备用（光流不可用时）
        mini_edge = self._make_edge(mini_masked, mask, t)
        hit = self._local_search(mini_edge)
        self._prev_mini_masked = mini_masked

        if hit is None or hit[0] < t.match_threshold:
            self._state = "searching"
            self._global_failures = 0
            return self._fail_result(frame.timestamp, "本地匹配失败，切换全局搜索")

        return self._commit(hit, frame.timestamp, "local")

    def reset(self) -> None:
        self._state = "searching"
        self._global_failures = 0
        self._offmap_retry_counter = 0
        self._searching_frame_counter = 0
        self._local_recovery_counter = 0
        self._last_point = None
        self._cached_scale = None
        self._consecutive_failures = 0
        self._prev_mini_masked = None
        self._flow_validation_counter = 0
        self._pending_corr_x = 0.0
        self._pending_corr_y = 0.0
        self._corr_frames_left = 0
        self._static_frames_count = 0
        self._last_flow_dx = 0.0
        self._last_flow_dy = 0.0

    # ── 传送检测 ──────────────────────────────────────────────────────────────

    def _is_transition_frame(self, mini_masked: np.ndarray) -> bool:
        """
        检测是否处于传送/场景切换画面。
        策略1（自动）：小地图平均亮度极低 → 黑屏
        策略2（可选）：与传送画面模板做匹配（需用户提供截图）
        """
        t = self.settings.template
        mean_val = float(np.mean(mini_masked))
        if mean_val < t.transition_dark_threshold:
            return True

        if self._transition_tmpl is not None:
            tmpl = self._transition_tmpl
            mh, mw = mini_masked.shape
            th, tw = tmpl.shape
            if th <= mh and tw <= mw:
                try:
                    res = cv2.matchTemplate(mini_masked, tmpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > 0.75:
                        return True
                except cv2.error:
                    pass
        return False

    def _enter_transition(self) -> None:
        """进入传送状态：清空所有追踪状态，防止光流累积错误。"""
        self._state = "transition"
        self._last_point = None
        self._cached_scale = None
        self._prev_mini_masked = None
        self._pending_corr_x = 0.0
        self._pending_corr_y = 0.0
        self._corr_frames_left = 0
        self._consecutive_failures = 0
        self._flow_validation_counter = 0
        self._global_failures = 0
        self._static_frames_count = 0
        self._last_flow_dx = 0.0
        self._last_flow_dy = 0.0

    # ── 光流追踪 ──────────────────────────────────────────────────────────────

    def _locate_by_flow(
        self, curr_masked: np.ndarray, mask: np.ndarray, timestamp: float
    ) -> LocationResult:
        """
        Farneback 稠密光流：估计两帧间平移，转换为大地图坐标增量。
        方向约定：小地图内容向左漂移(dx<0) ↔ 玩家向右 ↔ 大地图 x 增大。
        """
        assert self._prev_mini_masked is not None
        assert self._last_point is not None
        assert self._cached_scale is not None

        if self._prev_mini_masked.shape != curr_masked.shape:
            return self._fail_result(timestamp, "光流帧尺寸不匹配")

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_mini_masked, curr_masked, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )

        mask_bool = mask > 0
        flow_u = flow[mask_bool, 0]
        flow_v = flow[mask_bool, 1]
        dx_px = float(np.median(flow_u))
        dy_px = float(np.median(flow_v))
        # 记录本帧位移，供 TRACKING 状态静止检测使用
        self._last_flow_dx = dx_px
        self._last_flow_dy = dy_px

        consistency = float(np.std(np.hypot(flow_u, flow_v)))
        t = self.settings.template

        if consistency > t.flow_consistency_threshold:
            mini_edge = self._make_edge(curr_masked, mask, t)
            hit = self._local_search(mini_edge)
            if hit and hit[0] >= t.match_threshold:
                return self._commit(hit, timestamp, "local")
            return self._fail_result(timestamp, f"光流不一致({consistency:.1f})")

        scale = self._cached_scale
        lx, ly = self._last_point
        new_x = lx - dx_px * scale
        new_y = ly - dy_px * scale

        # 平滑校正：每帧取出 1/remaining 份，线性分摊
        if self._corr_frames_left > 0:
            step_x = self._pending_corr_x / self._corr_frames_left
            step_y = self._pending_corr_y / self._corr_frames_left
            new_x += step_x
            new_y += step_y
            self._pending_corr_x -= step_x
            self._pending_corr_y -= step_y
            self._corr_frames_left -= 1

        new_x = int(round(max(0.0, min(float(self._map_w), new_x))))
        new_y = int(round(max(0.0, min(float(self._map_h), new_y))))

        self._last_point = (new_x, new_y)
        self._consecutive_failures = 0
        conf = max(0.0, 1.0 - consistency / t.flow_consistency_threshold)

        return LocationResult(
            found=True, x=new_x, y=new_y, confidence=conf,
            mode="flow", message="光流追踪", timestamp=timestamp,
        )

    # ── 本地模板搜索 ──────────────────────────────────────────────────────────

    def _local_search(
        self, mini_edge: np.ndarray
    ) -> tuple[float, int, int, float] | None:
        assert self._last_point is not None and self._cached_scale is not None
        t = self.settings.template
        cx, cy = self._last_point
        scale = self._cached_scale
        radius = t.local_search_radius + self._consecutive_failures * t.local_expand_step
        roi = self._clamp_roi(cx - radius, cy - radius, cx + radius, cy + radius)
        hit = self._match_in_roi(mini_edge, scale, roi)
        return None if hit is None else (*hit, scale)

    # ── 全局搜索 ──────────────────────────────────────────────────────────────

    def _global_search(
        self, mini_edge: np.ndarray
    ) -> tuple[float, int, int, float] | None:
        t = self.settings.template
        best: tuple[float, int, int, float] | None = None
        mh, mw = mini_edge.shape
        pf = self._pyramid_factor
        assert self._map_edge_coarse is not None

        for scale in t.global_scales:
            th = max(1, int(mh * scale))
            tw = max(1, int(mw * scale))
            ctw, cth = max(1, tw // pf), max(1, th // pf)
            if ctw < self._MIN_TMPL_PX or cth < self._MIN_TMPL_PX:
                continue
            if ctw >= self._map_edge_coarse.shape[1] or cth >= self._map_edge_coarse.shape[0]:
                continue

            tmpl_scaled = cv2.resize(mini_edge, (tw, th), interpolation=cv2.INTER_LINEAR)
            coarse_tmpl = cv2.resize(tmpl_scaled, (ctw, cth), interpolation=cv2.INTER_AREA)
            res = self._run_match(self._map_edge_coarse, coarse_tmpl)
            if res is None:
                continue

            flat = res.ravel()
            k = min(self._TOP_K_COARSE, flat.size)
            top_idx = np.argpartition(flat, -k)[-k:]
            top_idx = top_idx[np.argsort(flat[top_idx])[::-1]]

            for idx in top_idx:
                if float(flat[idx]) < t.coarse_threshold:
                    break
                cx_full = int(idx % res.shape[1]) * pf
                cy_full = int(idx // res.shape[1]) * pf
                margin = int(max(tw, th) * self._REFINE_MARGIN)
                roi = self._clamp_roi(
                    cx_full - margin, cy_full - margin,
                    cx_full + tw + margin, cy_full + th + margin,
                )
                hit = self._match_in_roi(mini_edge, scale, roi)
                if hit and (best is None or hit[0] > best[0]):
                    best = (*hit, scale)

        return best

    # ── 单次 ROI 模板匹配 ─────────────────────────────────────────────────────

    def _match_in_roi(
        self, mini_edge: np.ndarray, scale: float, roi: tuple[int, int, int, int]
    ) -> tuple[float, int, int] | None:
        assert self._map_edge is not None
        x1, y1, x2, y2 = roi
        roi_img = self._map_edge[y1:y2, x1:x2]
        mh, mw = mini_edge.shape
        th = max(1, int(mh * scale))
        tw = max(1, int(mw * scale))
        if th > roi_img.shape[0] or tw > roi_img.shape[1]:
            return None
        template = cv2.resize(mini_edge, (tw, th), interpolation=cv2.INTER_LINEAR)
        res = self._run_match(roi_img, template)
        if res is None:
            return None
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        return float(max_val), int(x1 + max_loc[0] + tw // 2), int(y1 + max_loc[1] + th // 2)

    def _run_match(self, image: np.ndarray, template: np.ndarray) -> np.ndarray | None:
        if _USE_CUDA and self._cuda_matcher is not None:
            try:
                gpu_img = cv2.cuda_GpuMat(image)      # type: ignore[attr-defined]
                gpu_tmpl = cv2.cuda_GpuMat(template)  # type: ignore[attr-defined]
                return self._cuda_matcher.match(gpu_img, gpu_tmpl).download()
            except Exception:
                pass
        try:
            return cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            return None

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_edge(masked_gray: np.ndarray, mask: np.ndarray, t) -> np.ndarray:
        edge = cv2.Canny(masked_gray, t.canny_low, t.canny_high)
        return cv2.bitwise_and(edge, edge, mask=mask)

    @staticmethod
    def _hist_correlation(prev: np.ndarray, curr: np.ndarray) -> float:
        """
        计算两帧小地图的直方图相关度（0~1，越高越相似）。
        使用 64 bin 直方图，性能极高，每帧开销可忽略不计。
        """
        hist_p = cv2.calcHist([prev], [0], None, [64], [0, 256])
        hist_c = cv2.calcHist([curr], [0], None, [64], [0, 256])
        return float(cv2.compareHist(hist_p, hist_c, cv2.HISTCMP_CORREL))

    def invalidate_mask(self) -> None:
        """外部通知图标遮罩设置已变更，强制下次重建 mask 缓存。"""
        self._mask = None
        self._mask_cache_key = ()

    def _ensure_mask(self, shape: tuple[int, int]) -> np.ndarray:
        icon = self.settings.icon_mask
        cache_key = (shape, icon.x, icon.y, icon.radius)
        if self._mask is not None and getattr(self, "_mask_cache_key", ()) == cache_key:
            return self._mask

        mask = np.zeros(shape, dtype=np.uint8)
        h, w = shape
        # 基础圆形：只保留小地图圆形区域
        cv2.circle(mask, (w // 2, h // 2), max(8, w // 2 - 8), 255, -1)
        # 图标挖空：在玩家图标位置涂黑，消除箭头干扰
        if icon.is_valid:
            ix = max(0, min(w - 1, icon.x))
            iy = max(0, min(h - 1, icon.y))
            cv2.circle(mask, (ix, iy), icon.radius, 0, -1)
        else:
            # 未检测到图标时，挖去中心固定区域作为保守兜底
            default_r = max(10, w // 8)
            cv2.circle(mask, (w // 2, h // 2), default_r, 0, -1)

        self._mask = mask
        self._mask_cache_key = cache_key
        return mask

    def _clamp_roi(self, x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
        return (
            max(0, int(x1)), max(0, int(y1)),
            min(self._map_w, int(x2)), min(self._map_h, int(y2)),
        )

    def _validate_result(self, raw_x: int, raw_y: int, is_global: bool) -> bool:
        if not (0 <= raw_x <= self._map_w and 0 <= raw_y <= self._map_h):
            return False
        if self._last_point is None or is_global:
            return True
        return float(np.hypot(raw_x - self._last_point[0], raw_y - self._last_point[1])) \
            <= self.settings.template.jump_limit

    def _commit(
        self, hit: tuple[float, int, int, float], timestamp: float, mode: str
    ) -> LocationResult:
        conf, raw_x, raw_y, scale = hit
        if not self._validate_result(raw_x, raw_y, mode == "global"):
            return self._fail_result(timestamp, "定位结果跳变异常")
        self._last_point = (raw_x, raw_y)
        self._cached_scale = scale
        self._consecutive_failures = 0
        return LocationResult(
            found=True, x=raw_x, y=raw_y, confidence=conf,
            mode=mode, message="定位成功", timestamp=timestamp,
        )

    def _fail_result(self, timestamp: float, message: str) -> LocationResult:
        self._consecutive_failures += 1
        return LocationResult(
            found=False, confidence=0.0, mode="lost", message=message,
            timestamp=timestamp,
            x=self._last_point[0] if self._last_point else None,
            y=self._last_point[1] if self._last_point else None,
        )
