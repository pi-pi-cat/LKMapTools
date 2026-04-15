"""
定位诊断工具 —— 不调阈值，先看问题出在哪里。

运行方式：
    python debug_locator.py

输出：
  控制台：每个 scale 的最佳置信度，找出最匹配的 scale
  窗口 1  mini_edge      —— 小地图 Canny 边缘（是否检测到了有效边缘？）
  窗口 2  map_edge_patch —— 大地图匹配位置的边缘（内容是否对应？）
  窗口 3  map_overview   —— 全局地图，红框标出匹配位置
  窗口 4  heatmap        —— 最佳 scale 下的 matchTemplate 热图（峰值越明显越好）
"""
from __future__ import annotations

import sys
import time

import cv2
import mss
import numpy as np

# 确保从项目根目录运行
sys.path.insert(0, ".")

from lkmap.services.assets import AssetService
from lkmap.services.config import ConfigService


# ── 参数（和 settings.toml 一致，可手动覆盖调试） ─────────────────────────────
CANNY_LOW = 50
CANNY_HIGH = 150
SCALES = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8]
TOP_DISPLAY_W = 600   # 可视化窗口宽度（像素）

# ── 加载配置和素材 ─────────────────────────────────────────────────────────────
cfg = ConfigService()
settings = cfg.load()
assets = AssetService(settings)

region = settings.capture_region
if region is None or not region.is_valid:
    print("❌ 未配置截图区域，请先在主程序中框选小地图。")
    sys.exit(1)

print(f"截图区域：left={region.left} top={region.top} w={region.width} h={region.height}")

# ── 截取当前小地图 ─────────────────────────────────────────────────────────────
with mss.mss() as sct:
    raw = np.array(sct.grab(region.to_dict()))
mini_bgr = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

# ── 加载大地图特征图 ──────────────────────────────────────────────────────────
feat_map = assets.load_feature_map()
map_gray = cv2.cvtColor(feat_map, cv2.COLOR_BGR2GRAY)
map_edge = cv2.Canny(map_gray, CANNY_LOW, CANNY_HIGH)
map_h, map_w = map_gray.shape

# ── 处理小地图边缘 ────────────────────────────────────────────────────────────
mini_gray = cv2.cvtColor(mini_bgr, cv2.COLOR_BGR2GRAY)
mh, mw = mini_gray.shape

# 圆形 mask（略微内缩避免边界假边缘）
mask = np.zeros((mh, mw), dtype=np.uint8)
cv2.circle(mask, (mw // 2, mh // 2), max(8, mw // 2 - 8), 255, -1)

mini_masked = cv2.bitwise_and(mini_gray, mini_gray, mask=mask)
mini_edge = cv2.Canny(mini_masked, CANNY_LOW, CANNY_HIGH)
mini_edge = cv2.bitwise_and(mini_edge, mini_edge, mask=mask)

edge_pixels = int(np.count_nonzero(mini_edge))
print(f"\n小地图边缘像素数：{edge_pixels}（建议 >50，过少说明 Canny 阈值偏高）")
if edge_pixels < 20:
    print("⚠️  边缘极少！可能原因：")
    print("   1. CANNY_LOW/HIGH 过高 → 尝试调低，如 CANNY_LOW=20, CANNY_HIGH=80")
    print("   2. 小地图内容本身纹理极少（例如进入无特征区域）")

# ── 遍历所有 scale，记录最佳置信度 ───────────────────────────────────────────
print("\n{:<8} {:<12} {:<14} {:<14}".format("scale", "confidence", "map_x(center)", "map_y(center)"))
print("-" * 50)

best_conf = -1.0
best_scale = 1.0
best_loc_map: tuple[int, int] = (0, 0)
best_heatmap: np.ndarray | None = None
best_tw = mw
best_th = mh

for scale in SCALES:
    th = max(1, int(mh * scale))
    tw = max(1, int(mw * scale))

    if tw > map_w or th > map_h:
        print(f"{scale:<8.1f} 模板超出地图尺寸，跳过")
        continue

    template = cv2.resize(mini_edge, (tw, th), interpolation=cv2.INTER_LINEAR)

    try:
        res = cv2.matchTemplate(map_edge, template, cv2.TM_CCOEFF_NORMED)
    except cv2.error as e:
        print(f"{scale:<8.1f} matchTemplate 错误: {e}")
        continue

    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    cx = max_loc[0] + tw // 2
    cy = max_loc[1] + th // 2

    marker = " ← 最佳" if max_val > best_conf else ""
    print(f"{scale:<8.1f} {max_val:<12.4f} {cx:<14} {cy:<14}{marker}")

    if max_val > best_conf:
        best_conf = max_val
        best_scale = scale
        best_loc_map = (cx, cy)
        best_heatmap = res.copy()
        best_tw = tw
        best_th = th

print(f"\n全局最佳置信度：{best_conf:.4f}  scale={best_scale}  位置=({best_loc_map[0]}, {best_loc_map[1]})")
print(f"当前 match_threshold = {settings.template.match_threshold}")
gap = settings.template.match_threshold - best_conf
if gap > 0:
    print(f"🔴 差距：{gap:.4f}，需要将阈值降至 {best_conf - 0.02:.4f} 以下，或改善匹配质量")
else:
    print("✅ 置信度已超过阈值，定位应当成功（若仍失败请检查跳变限制或坐标范围）")

# ── 可视化 ────────────────────────────────────────────────────────────────────

def fit(img: np.ndarray, max_w: int, max_h: int = 9999) -> np.ndarray:
    """等比缩放到最大宽/高。"""
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# 1. 小地图边缘
cv2.namedWindow("1_mini_edge", cv2.WINDOW_NORMAL)
mini_disp = np.hstack([to_bgr(mini_gray), to_bgr(mask), to_bgr(mini_edge)])
cv2.imshow("1_mini_edge", fit(mini_disp, TOP_DISPLAY_W * 3, 400))
cv2.setWindowTitle("1_mini_edge", "小地图：原图 | 遮罩 | 边缘（看是否有有效边缘）")

# 2. 匹配区域对比
bx = max(0, best_loc_map[0] - best_tw // 2)
by = max(0, best_loc_map[1] - best_th // 2)
ex = min(map_w, bx + best_tw)
ey = min(map_h, by + best_th)
patch_edge = map_edge[by:ey, bx:ex]
template_show = cv2.resize(mini_edge, (best_tw, best_th))

cv2.namedWindow("2_edge_compare", cv2.WINDOW_NORMAL)
patch_disp = np.hstack([
    fit(to_bgr(template_show), TOP_DISPLAY_W),
    fit(to_bgr(patch_edge), TOP_DISPLAY_W),
])
cv2.imshow("2_edge_compare", patch_disp)
cv2.setWindowTitle("2_edge_compare",
    f"左=小地图边缘(scale={best_scale})  右=大地图匹配位置边缘  conf={best_conf:.4f}")

# 3. 大地图全景 + 红框标出匹配位置
overview_scale = min(1.0, 1000 / max(map_w, map_h))
overview = cv2.resize(feat_map, (int(map_w * overview_scale), int(map_h * overview_scale)))
rx1 = int(bx * overview_scale)
ry1 = int(by * overview_scale)
rx2 = int(ex * overview_scale)
ry2 = int(ey * overview_scale)
cv2.rectangle(overview, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
cv2.circle(overview,
    (int(best_loc_map[0] * overview_scale), int(best_loc_map[1] * overview_scale)),
    6, (0, 255, 0), -1)
cv2.namedWindow("3_map_overview", cv2.WINDOW_NORMAL)
cv2.imshow("3_map_overview", overview)
cv2.setWindowTitle("3_map_overview", "大地图全景（红框=最佳匹配位置，绿点=中心）")

# 4. 热图（峰值越尖锐说明匹配越有辨识度）
if best_heatmap is not None:
    hm_norm = cv2.normalize(best_heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_norm, cv2.COLORMAP_JET)
    cv2.namedWindow("4_heatmap", cv2.WINDOW_NORMAL)
    cv2.imshow("4_heatmap", fit(hm_color, 800))
    cv2.setWindowTitle("4_heatmap",
        f"matchTemplate 热图（scale={best_scale}）—— 峰值越孤立越好")

print("\n按任意键关闭所有窗口...")
cv2.waitKey(0)
cv2.destroyAllWindows()
