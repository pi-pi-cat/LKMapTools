from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from lkmap.locator.base import LocatorStrategy
from lkmap.models import AppSettings, FrameData, LocationResult
from lkmap.services.assets import AssetService


class OrbLocator(LocatorStrategy):
    def __init__(self, settings: AppSettings, assets: AssetService) -> None:
        self.settings = settings
        self.assets = assets
        self._initialized = False
        self._last_point: tuple[int, int] | None = None
        self._consecutive_failures = 0
        self._mask: np.ndarray | None = None
        self._map_width = 0
        self._map_height = 0
        self._feature_map_gray: np.ndarray | None = None
        self._display_map: np.ndarray | None = None
        self._kp_big: list[cv2.KeyPoint] = []
        self._des_big: np.ndarray | None = None
        self._pts_big_np: np.ndarray | None = None

    @property
    def display_map(self) -> np.ndarray:
        if self._display_map is None:
            raise RuntimeError("定位器尚未初始化")
        return self._display_map

    def initialize(self) -> None:
        if self._initialized:
            return

        self._display_map = self.assets.load_display_map()
        feature_map = self.assets.load_feature_map()
        self._feature_map_gray = cv2.cvtColor(feature_map, cv2.COLOR_BGR2GRAY)
        self._map_height, self._map_width = self._feature_map_gray.shape[:2]
        self._orb_mini = cv2.ORB_create(
            nfeatures=self.settings.orb.minimap_nfeatures,
            scaleFactor=self.settings.orb.scale_factor,
            nlevels=self.settings.orb.nlevels,
            fastThreshold=self.settings.orb.fast_threshold,
            edgeThreshold=self.settings.orb.edge_threshold,
        )
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._load_or_build_feature_cache()
        self._pts_big_np = np.array([kp.pt for kp in self._kp_big], dtype=np.float32)
        self._initialized = True

    def locate(self, frame: FrameData) -> LocationResult:
        self.initialize()
        gray = cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY)
        mask = self._ensure_mask(gray.shape)
        kp_mini, des_mini = self._orb_mini.detectAndCompute(gray, mask)

        if des_mini is None or len(kp_mini) < self.settings.orb.min_match_count:
            return self._fail_result(frame.timestamp, "小地图特征不足")

        is_global = self._last_point is None or (
            self._consecutive_failures >= self.settings.orb.global_search_after_failures
        )
        current_kp_big, current_des_big, mode = self._select_search_scope(is_global)
        if current_des_big is None or len(current_des_big) == 0:
            return self._fail_result(frame.timestamp, "搜索范围内无特征点")

        matches = self._matcher.match(des_mini, current_des_big)
        good_matches = [
            match for match in matches if match.distance < self.settings.orb.match_distance_threshold
        ]
        good_matches = sorted(good_matches, key=lambda item: item.distance)[
            : self.settings.orb.max_good_matches
        ]
        if len(good_matches) < self.settings.orb.min_match_count:
            return self._fail_result(frame.timestamp, "匹配点不足")

        src_pts = np.float32([kp_mini[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([current_kp_big[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        matrix, inliers = cv2.estimateAffinePartial2D(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.settings.orb.ransac_threshold,
        )
        if matrix is None:
            return self._fail_result(frame.timestamp, "仿射估计失败")

        scale = float(np.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2))
        if not 0.6 <= scale <= 1.4:
            return self._fail_result(frame.timestamp, "缩放异常")

        center = np.array([[[gray.shape[1] / 2, gray.shape[0] / 2]]], dtype=np.float32)
        transformed = cv2.transform(center, matrix)
        raw_x, raw_y = int(transformed[0][0][0]), int(transformed[0][0][1])

        if not self._validate_result(raw_x, raw_y, is_global):
            return self._fail_result(frame.timestamp, "定位结果跳变异常")

        inlier_ratio = 0.0
        if inliers is not None and len(inliers) > 0:
            inlier_ratio = float(np.count_nonzero(inliers)) / float(len(inliers))

        self._last_point = (raw_x, raw_y)
        self._consecutive_failures = 0
        return LocationResult(
            found=True,
            x=raw_x,
            y=raw_y,
            confidence=max(inlier_ratio, len(good_matches) / self.settings.orb.max_good_matches),
            mode=mode,
            message="定位成功",
            timestamp=frame.timestamp,
        )

    def reset(self) -> None:
        self._last_point = None
        self._consecutive_failures = self.settings.orb.global_search_after_failures

    def _ensure_mask(self, frame_shape: tuple[int, int]) -> np.ndarray:
        if self._mask is not None and self._mask.shape == frame_shape:
            return self._mask
        mask = np.zeros(frame_shape, dtype=np.uint8)
        mask_h, mask_w = frame_shape
        cv2.circle(mask, (mask_w // 2, mask_h // 2), max(8, mask_w // 2 - 5), 255, -1)
        self._mask = mask
        return mask

    def _select_search_scope(self, is_global: bool) -> tuple[list[cv2.KeyPoint], np.ndarray | None, str]:
        if is_global or self._last_point is None or self._pts_big_np is None or self._des_big is None:
            return self._kp_big, self._des_big, "global"

        last_x, last_y = self._last_point
        dist_sq = (self._pts_big_np[:, 0] - last_x) ** 2 + (self._pts_big_np[:, 1] - last_y) ** 2
        search_radius = self.settings.orb.local_search_radius + (
            self._consecutive_failures * self.settings.orb.local_search_expand_step
        )
        near_indices = np.where(dist_sq < search_radius**2)[0]
        if len(near_indices) <= 20:
            self._consecutive_failures = self.settings.orb.global_search_after_failures
            return self._kp_big, self._des_big, "global"

        current_kp = [self._kp_big[index] for index in near_indices]
        current_des = self._des_big[near_indices]
        return current_kp, current_des, "local"

    def _validate_result(self, raw_x: int, raw_y: int, is_global: bool) -> bool:
        if not (0 <= raw_x <= self._map_width and 0 <= raw_y <= self._map_height):
            return False
        if self._last_point is None or is_global:
            return True
        last_x, last_y = self._last_point
        distance = np.sqrt((raw_x - last_x) ** 2 + (raw_y - last_y) ** 2)
        return bool(distance <= self.settings.orb.jump_limit)

    def _fail_result(self, timestamp: float, message: str) -> LocationResult:
        self._consecutive_failures += 1
        return LocationResult(
            found=False,
            confidence=0.0,
            mode="lost",
            message=message,
            timestamp=timestamp,
            x=self._last_point[0] if self._last_point else None,
            y=self._last_point[1] if self._last_point else None,
        )

    def _load_or_build_feature_cache(self) -> None:
        cache_path = self.assets.feature_cache_path()
        legacy_cache_path = self.assets.legacy_feature_cache_path()

        for candidate in (cache_path, legacy_cache_path):
            if candidate.exists():
                try:
                    self._kp_big, self._des_big = self._load_features(candidate)
                    return
                except Exception:
                    continue

        self._kp_big, self._des_big = self._build_multi_scale_feature_pool()
        self._save_features(cache_path, self._kp_big, self._des_big)

    def _build_multi_scale_feature_pool(self) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        if self._feature_map_gray is None:
            raise RuntimeError("特征地图未加载")

        scales = [0.6, 0.8, 1.0, 1.2, 1.4]
        all_kp: list[cv2.KeyPoint] = []
        all_des: list[np.ndarray] = []

        for scale in scales:
            if scale == 1.0:
                layer_gray = self._feature_map_gray
            else:
                width = int(self._map_width * scale)
                height = int(self._map_height * scale)
                layer_gray = cv2.resize(
                    self._feature_map_gray, (width, height), interpolation=cv2.INTER_LINEAR
                )

            kp, des = self._extract_grid_features(
                layer_gray,
                total_features=min(
                    self.settings.orb.map_nfeatures,
                    self.settings.orb.max_kp_per_layer,
                ),
                grid_rows=max(1, int(self.settings.orb.grid_rows * scale)),
                grid_cols=max(1, int(self.settings.orb.grid_cols * scale)),
            )
            if des is None:
                continue

            for keypoint in kp:
                keypoint.pt = (keypoint.pt[0] / scale, keypoint.pt[1] / scale)
            all_kp.extend(kp)
            all_des.append(des)

        if not all_des:
            raise RuntimeError("无法为大地图构建 ORB 特征缓存")
        return all_kp, np.vstack(all_des)

    def _extract_grid_features(
        self,
        image_gray: np.ndarray,
        *,
        total_features: int,
        grid_rows: int,
        grid_cols: int,
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None]:
        height, width = image_gray.shape
        cell_h = max(1, height // grid_rows)
        cell_w = max(1, width // grid_cols)
        features_per_grid = max(8, total_features // max(1, grid_rows * grid_cols))
        grid_orb = cv2.ORB_create(
            nfeatures=features_per_grid,
            scaleFactor=self.settings.orb.scale_factor,
            nlevels=self.settings.orb.nlevels,
            fastThreshold=self.settings.orb.fast_threshold,
            edgeThreshold=self.settings.orb.edge_threshold,
        )

        all_kp: list[cv2.KeyPoint] = []
        all_des: list[np.ndarray] = []
        for row in range(grid_rows):
            for col in range(grid_cols):
                y1 = row * cell_h
                y2 = height if row == grid_rows - 1 else (row + 1) * cell_h
                x1 = col * cell_w
                x2 = width if col == grid_cols - 1 else (col + 1) * cell_w
                roi = image_gray[y1:y2, x1:x2]
                kp, des = grid_orb.detectAndCompute(roi, None)
                if des is None:
                    continue
                for keypoint in kp:
                    keypoint.pt = (keypoint.pt[0] + x1, keypoint.pt[1] + y1)
                all_kp.extend(kp)
                all_des.append(des)

        if not all_des:
            return [], None
        return all_kp, np.vstack(all_des)

    def _save_features(self, path: Path, keypoints: list[cv2.KeyPoint], descriptors: np.ndarray) -> None:
        kp_array = np.array(
            [
                (
                    kp.pt[0],
                    kp.pt[1],
                    kp.size,
                    kp.angle,
                    kp.response,
                    kp.octave,
                    kp.class_id,
                )
                for kp in keypoints
            ],
            dtype=[
                ("pt_x", "f4"),
                ("pt_y", "f4"),
                ("size", "f4"),
                ("angle", "f4"),
                ("response", "f4"),
                ("octave", "i4"),
                ("class_id", "i4"),
            ],
        )
        np.savez_compressed(path, keypoints=kp_array, descriptors=descriptors)

    def _load_features(self, path: Path) -> tuple[list[cv2.KeyPoint], np.ndarray]:
        data = np.load(path)
        kp_array = data["keypoints"]
        descriptors = data["descriptors"]
        keypoints = [
            cv2.KeyPoint(
                x=row["pt_x"],
                y=row["pt_y"],
                size=row["size"],
                angle=row["angle"],
                response=row["response"],
                octave=row["octave"],
                class_id=row["class_id"],
            )
            for row in kp_array
        ]
        return keypoints, descriptors
