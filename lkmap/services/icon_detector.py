"""
Player icon detector.

Strategy: colour-first, then template verification.
  1. Find all bright orange/yellow blobs in the minimap (HSV filter)
  2. For each candidate blob, score it with multi-angle template matching
  3. Return the blob with the highest combined score
     (template score × proximity-to-centre weight)
  4. If template unavailable, pick the closest blob to the frame centre
  5. Last resort: geometric centre of the frame

Why colour-first?
  The player arrow (me.png) is a distinctive bright orange/yellow that terrain
  rarely reproduces.  Pure template matching on the full frame finds false
  positives on brown/orange buildings.  Colour pre-filtering shrinks the
  candidate set to 1-3 blobs and makes template scoring much more reliable.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from lkmap.models import IconMask

# ── Colour filter (bright orange-yellow, tight range) ───────────────────────
_HSV_LOWER = np.array([16, 160, 170], dtype=np.uint8)
_HSV_UPPER = np.array([38, 255, 255], dtype=np.uint8)

# Valid blob area range (px²) to exclude dust noise and large terrain patches
_BLOB_AREA_MIN = 30
_BLOB_AREA_MAX = 2500

# 搜索半径 = 小地图半径 × 此比例（尽量覆盖大部分小地图圆形区域）
_SEARCH_RATIO = 0.90

# Template-matching parameters for candidate verification
_ROT_STEP  = 15        # rotation step (degrees)
_SCALES    = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]
_TMPL_CONF = 0.20      # minimum score to count as a valid template match


def detect_player_icon(
    frame_bgr: np.ndarray,
    icon_path: str = "",
) -> IconMask:
    """
    Detect the player arrow icon and return an IconMask.

    Parameters
    ----------
    frame_bgr : BGR minimap crop (already clipped to the capture region)
    icon_path : path to assest/me.png; empty → skip template scoring
    """
    h, w = frame_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    search_r = int(min(w, h) // 2 * _SEARCH_RATIO)

    blobs = _find_orange_blobs(frame_bgr, cx, cy, search_r)

    if blobs:
        if icon_path:
            result = _score_blobs_with_template(frame_bgr, blobs, icon_path)
        else:
            # No template: pick the blob closest to frame centre
            result = min(blobs, key=lambda b: math.hypot(b[0] - cx, b[1] - cy))

        if result is not None:
            bx, by, bradius = result
            return IconMask(x=bx, y=by, radius=bradius)

    # Last resort: geometric centre
    default_r = max(14, min(w, h) // 8)
    return IconMask(x=cx, y=cy, radius=default_r)


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 – colour blob detection
# ──────────────────────────────────────────────────────────────────────────────

def _find_orange_blobs(
    frame_bgr: np.ndarray,
    cx: int,
    cy: int,
    search_r: int,
) -> list[tuple[int, int, int]]:
    """
    Return a list of (centre_x, centre_y, radius) for each orange blob
    found within `search_r` of (cx, cy).
    """
    hsv    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask   = cv2.inRange(hsv, _HSV_LOWER, _HSV_UPPER)

    # Restrict to the search circle
    roi = np.zeros_like(mask)
    cv2.circle(roi, (cx, cy), search_r, 255, -1)
    mask = cv2.bitwise_and(mask, roi)

    # Small morphological close to merge nearby pixels into one blob
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: list[tuple[int, int, int]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if not (_BLOB_AREA_MIN <= area <= _BLOB_AREA_MAX):
            continue
        M = cv2.moments(cnt)
        if M["m00"] <= 0:
            continue
        bx = int(M["m10"] / M["m00"])
        by = int(M["m01"] / M["m00"])
        # Use bounding-circle radius + 4 px margin
        _, enc_r = cv2.minEnclosingCircle(cnt)
        radius = int(math.ceil(enc_r)) + 4
        blobs.append((bx, by, radius))
    return blobs


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 – template scoring for each candidate blob
# ──────────────────────────────────────────────────────────────────────────────

def _score_blobs_with_template(
    frame_bgr: np.ndarray,
    blobs: list[tuple[int, int, int]],
    icon_path: str,
) -> tuple[int, int, int] | None:
    """
    对每个候选色块做多角度模板匹配，结合距离权重选出最终结果。

    综合得分 = template_score × proximity_weight
    proximity_weight 随离中心距离线性衰减，确保中心附近的候选有显著优势。
    """
    path = Path(icon_path)
    if not path.exists():
        return None

    icon_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if icon_bgr is None:
        return None

    ih, iw = icon_bgr.shape[:2]
    fh, fw = frame_bgr.shape[:2]
    cx, cy = fw // 2, fh // 2
    max_dist = math.hypot(cx, cy) or 1.0
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    scored: list[tuple[float, tuple[int, int, int]]] = []

    for blob in blobs:
        bx, by, bradius = blob
        tmpl_score = _best_template_score_at(frame_gray, bx, by, bradius,
                                              icon_bgr, ih, iw, fh, fw)
        dist = math.hypot(bx - cx, by - cy)
        # 距离权重：中心=1.0，最远处=0.3，避免远处候选因模板碰巧高分而胜出
        proximity_w = 1.0 - 0.7 * (dist / max_dist)
        combined = tmpl_score * proximity_w
        scored.append((combined, blob))

    if not scored:
        return None

    best_score, best_blob = max(scored, key=lambda t: t[0])
    if best_score >= _TMPL_CONF:
        return best_blob

    return min(blobs, key=lambda b: math.hypot(b[0] - cx, b[1] - cy))


def _best_template_score_at(
    frame_gray: np.ndarray,
    bx: int, by: int, blob_r: int,
    icon_bgr: np.ndarray,
    ih: int, iw: int,
    fh: int, fw: int,
) -> float:
    """
    Run multi-angle × multi-scale template matching in a small window centred
    on (bx, by).  Return the highest TM_CCOEFF_NORMED score found.
    """
    best = 0.0
    # Search window = blob radius × 3 (gives room for rotation bbox growth)
    pad  = max(blob_r * 3, max(iw, ih) * 2)
    x1   = max(0, bx - pad)
    y1   = max(0, by - pad)
    x2   = min(fw, bx + pad)
    y2   = min(fh, by + pad)
    window = frame_gray[y1:y2, x1:x2]

    for sc in _SCALES:
        sw = max(4, int(iw * sc))
        sh = max(4, int(ih * sc))
        if sw >= window.shape[1] or sh >= window.shape[0]:
            continue
        icon_scaled = cv2.resize(icon_bgr, (sw, sh), interpolation=cv2.INTER_LINEAR)
        icon_gray   = cv2.cvtColor(icon_scaled, cv2.COLOR_BGR2GRAY)
        for angle in range(0, 360, _ROT_STEP):
            M   = cv2.getRotationMatrix2D((sw / 2, sh / 2), float(angle), 1.0)
            rot = cv2.warpAffine(icon_gray, M, (sw, sh))
            try:
                res = cv2.matchTemplate(window, rot, cv2.TM_CCOEFF_NORMED)
            except cv2.error:
                continue
            _, val, _, _ = cv2.minMaxLoc(res)
            if val > best:
                best = val
    return best
