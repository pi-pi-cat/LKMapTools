# PySide6 地图导航重写 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 PySide6 重写一个可运行的地图导航底座，支持手动框选小地图、ORB 实时定位、可缩放北向固定跟随视图和资源点显示。

**Architecture:** 项目采用模块化单进程架构。`capture`、`locator`、`map_view`、`layers`、`services` 与 `app` 分层，通过控制器连接截图、定位和视图刷新；UI 不直接承载 ORB 算法逻辑，定位器不依赖 Qt 视图控件。

**Tech Stack:** Python 3.13, PySide6, OpenCV, NumPy, MSS

---

## 文件结构

- Create: `lkmap/__init__.py`
- Create: `lkmap/app/__init__.py`
- Create: `lkmap/app/controller.py`
- Create: `lkmap/app/window.py`
- Create: `lkmap/capture/__init__.py`
- Create: `lkmap/capture/base.py`
- Create: `lkmap/capture/manual_region.py`
- Create: `lkmap/layers/__init__.py`
- Create: `lkmap/layers/resources.py`
- Create: `lkmap/locator/__init__.py`
- Create: `lkmap/locator/base.py`
- Create: `lkmap/locator/orb_locator.py`
- Create: `lkmap/map_view/__init__.py`
- Create: `lkmap/map_view/view.py`
- Create: `lkmap/models.py`
- Create: `lkmap/services/__init__.py`
- Create: `lkmap/services/assets.py`
- Create: `lkmap/services/config.py`
- Create: `main.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `.gitignore`
- Delete: `config.py`
- Delete: `main_orb.py`
- Delete: `requirements.txt`
- Delete: `point.json`
- Delete: `log.txt`
- Delete: `crash_log.txt`
- Delete: `runtime_checks.py`
- Delete: `tests/test_runtime_checks.py`

### Task 1: 建立新项目骨架

**Files:**
- Create: `lkmap/**`
- Modify: `pyproject.toml`, `.gitignore`, `main.py`
- Delete: `config.py`, `main_orb.py`, `requirements.txt`, `point.json`, `runtime_checks.py`, `tests/test_runtime_checks.py`, `log.txt`, `crash_log.txt`

- [ ] 创建 `lkmap` 包结构与空模块，保留清晰分层。
- [ ] 将 `pyproject.toml` 改为 PySide6 项目依赖，移除与首版无关依赖。
- [ ] 重写 `.gitignore`，忽略 `.venv/`、`.superpowers/`、`cache/`、运行日志等。
- [ ] 删除旧 Tk 主线和无关调试残留，使仓库只保留重写所需资源和文档。

### Task 2: 配置与资源加载

**Files:**
- Create: `lkmap/services/config.py`, `lkmap/services/assets.py`, `lkmap/models.py`

- [ ] 设计应用配置结构，至少包含地图路径、资源点路径、截图区域、缓存路径和视图缩放倍率。
- [ ] 实现配置加载/保存，缺省时自动生成默认配置。
- [ ] 实现地图和资源点加载。
- [ ] 复用旧项目资源点坐标换算公式，输出统一的 `ResourcePoint` 模型。

### Task 3: 手动框选截图策略

**Files:**
- Create: `lkmap/capture/base.py`, `lkmap/capture/manual_region.py`
- Modify: `lkmap/models.py`

- [ ] 定义 `CaptureStrategy` 抽象接口。
- [ ] 实现手动框选小地图区域的全屏覆盖选择器。
- [ ] 实现基于 MSS 的区域截图逻辑。
- [ ] 让截图策略支持读取和保存区域配置。

### Task 4: ORB 定位器

**Files:**
- Create: `lkmap/locator/base.py`, `lkmap/locator/orb_locator.py`
- Modify: `lkmap/models.py`, `lkmap/services/assets.py`

- [ ] 定义 `LocatorStrategy` 接口与 `LocationResult` 模型。
- [ ] 实现 ORB 定位器的地图加载、特征缓存加载/构建。
- [ ] 实现小地图特征提取、局部搜索、全图回退与仿射估计。
- [ ] 实现结果校验、平滑和重置逻辑。

### Task 5: 跟随视图与资源点图层

**Files:**
- Create: `lkmap/map_view/view.py`, `lkmap/layers/resources.py`

- [ ] 基于 Qt 图形视图实现可缩放、北向固定的跟随视图。
- [ ] 实现玩家位置图元。
- [ ] 实现资源点图层与可见范围裁剪。
- [ ] 保证缩放围绕玩家位置工作。

### Task 6: 主窗口与控制器

**Files:**
- Create: `lkmap/app/window.py`, `lkmap/app/controller.py`
- Modify: `main.py`

- [ ] 构建主窗口、状态栏和基础工具栏。
- [ ] 使用 `QThread + worker` 串起截图与定位流程。
- [ ] 将定位结果同步到地图视图。
- [ ] 实现开始/暂停跟踪、重置定位、资源点开关与状态显示。

### Task 7: 最小必要验证与文档

**Files:**
- Modify: `README.md`

- [ ] 更新 README，说明运行方式、资源依赖和首版功能边界。
- [ ] 用“大地图裁小图”方式做最小必要定位自测。
- [ ] 运行一次模块导入/基础启动验证，确认项目可以启动到 UI 主流程。
- [ ] 记录仍未实现的后续能力，但不为其扩展当前实现范围。
