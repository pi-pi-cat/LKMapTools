# LKMapTools

一个基于 `PySide6 + OpenCV ORB` 的地图定位与导航底座。

当前阶段目标：

- 手动框选屏幕上的小地图区域
- 持续截图小地图并定位到大地图
- 在独立窗口中显示可缩放、北向固定的跟随视图
- 显示当前位置与资源点图层

## 运行环境

- Python 3.13+
- `uv`

## 安装依赖

```bash
uv sync
```

## 启动

```bash
uv run python main.py
```

## 当前资源要求

默认配置会读取以下资源：

- `assest/raw.png`
- `assest/raw_noedge.png`
- `assest/points.json`

如果 `cache/orb_features.npz` 不存在，程序会尝试加载旧缓存 `assest/ORB_features.npz`；如果两者都不存在，会根据地图重新构建缓存。

运行时会自动生成本地配置 `config/settings.json`，用于保存小地图截图区域和视图缩放状态。

## 第一阶段范围

- `PySide6` 主窗口
- 手动框选截图策略
- ORB 定位器
- 可缩放跟随视图
- 资源点图层显示

后续阶段再加入路线图层与导航能力。
