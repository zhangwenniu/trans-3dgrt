# 无界面3D高斯渲染

本工具允许在没有图形界面（无显示）的环境下渲染3D高斯混合模型，直接使用Tracer类而不依赖Polyscope库进行可视化。

## 功能特点

- 无需图形界面即可渲染3D高斯模型
- 支持自定义相机参数（位置、方向、视场角）
- 快速渲染并导出高质量图像
- 兼容现有的3DGRUT模型检查点

## 使用前提

- 已安装CUDA和PyTorch
- 已配置好3DGRUT的相关库和依赖

## 使用方法

### 基本用法

```bash
python headless_render.py --config CONFIG_PATH --checkpoint CHECKPOINT_PATH --output OUTPUT_IMAGE_PATH
```

### 参数说明

- `--config`: 配置文件路径（必需）
- `--checkpoint`: 模型检查点路径（必需）
- `--height`: 输出图像高度（默认: 800）
- `--width`: 输出图像宽度（默认: 800）
- `--fov`: 相机视场角，单位为度（默认: 45.0）
- `--output`: 输出图像路径（默认: output.png）

### 示例

```bash
python headless_render.py --config configs/base.yaml --checkpoint runs/experiment1/model_30000.pt --height 1080 --width 1920 --fov 35.0 --output renders/view1.png
```

## 批量渲染

您可以使用`batch_render.py`脚本从多个视角渲染场景：

```bash
python batch_render.py --config CONFIG_PATH --checkpoint CHECKPOINT_PATH --output_dir RENDERS_DIR
```

### 批量渲染参数

- `--config`: 配置文件路径（必需）
- `--checkpoint`: 模型检查点路径（必需）
- `--output_dir`: 输出目录（默认: "renders"）
- `--mode`: 相机模式, 可选 "orbit"(环绕) 或 "path"(路径)（默认: "orbit"）
- `--height`: 输出图像高度（默认: 720）
- `--width`: 输出图像宽度（默认: 1280）
- `--fov`: 视场角（默认: 45.0度）
- `--num_views`: 视角数量（默认: 36）
- `--radius`: 相机到原点的距离（默认: 3.0）
- `--camera_height`: 相机高度（默认: 0.5）
- `--center_x/y/z`: 中心点坐标（默认: 0.0）
- `--tilt`: 相机倾斜角度（默认: 0.0度）
- `--step`: 渲染间隔，每n帧渲染一次（默认: 1）

### 批量渲染示例

```bash
# 环绕渲染
python batch_render.py --config configs/base.yaml --checkpoint runs/model_30000.pt --height 1080 --width 1920 --num_views 60 --radius 5.0 --camera_height 1.0 --output_dir orbit_renders

# 沿路径渲染
python batch_render.py --config configs/base.yaml --checkpoint runs/model_30000.pt --mode path --num_views 120 --output_dir path_renders
```

## 自定义相机位置

默认情况下，相机位于Z轴负方向3个单位处，朝向原点。如需自定义相机位置和方向，可以修改脚本中的`camera_pose`变量：

```python
camera_pose = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],  # 右向量 + x偏移
    [0.0, 1.0, 0.0, 0.0],  # 上向量 + y偏移
    [0.0, 0.0, 1.0, -3.0]  # 前向量 + z偏移
], device=renderer.device)
```

这个3x4矩阵表示相机的位置和方向：
- 前3列是相机的坐标系（右向量、上向量、前向量）
- 最后一列是相机在世界坐标系中的位置

## 进阶用法：作为库使用

您也可以将HeadlessRenderer类作为库在自己的Python代码中导入使用：

```python
from headless_render import HeadlessRenderer
import torch

# 初始化渲染器
renderer = HeadlessRenderer(
    conf_path="configs/base.yaml",
    checkpoint_path="runs/experiment1/model_30000.pt"
)

# 定义相机姿态
camera_pose = torch.tensor([
    [0.866, 0.0, -0.5, 1.0],   # 右向量 + x偏移
    [0.0, 1.0, 0.0, 0.5],      # 上向量 + y偏移
    [0.5, 0.0, 0.866, -3.0]    # 前向量 + z偏移
], device="cuda")

# 渲染图像
image = renderer.render_image(
    height=1080, 
    width=1920, 
    fov=40.0, 
    camera_pose=camera_pose
)

# 保存图像
renderer.save_image(image, "output_view.png")
```

## 渲染实现说明

本工具直接使用Tracer类的trace函数，而不是通过高级渲染接口，从而绕过了对Polyscope显示环境的依赖。渲染过程的主要步骤：

1. 生成相机射线
2. 提取高斯混合模型的参数（位置、密度、旋转、缩放）
3. 直接调用tracer_wrapper.trace函数进行渲染
4. 应用背景模型
5. 将渲染结果重塑为图像并输出

注意，如果您遇到以下错误：

```
TypeError: trace(): incompatible function arguments
```

这是因为直接调用trace函数需要传递正确的参数。修复后的版本已正确处理了参数格式，确保它们匹配底层C++实现的预期。

## 故障排除

如果您在运行时遇到模型加载错误，请尝试检查以下几点：

1. 确保checkpoint文件格式正确，包含必要的模型参数
2. 确保配置文件与checkpoint匹配
3. 如果模型在加载后无法渲染，可能是模型格式或版本不兼容

对于其他常见问题:

- **内存不足错误**: 尝试减小渲染分辨率或使用更小的模型
- **渲染效果不佳**: 调整相机位置和FOV参数
- **CUDA错误**: 确保安装了正确版本的CUDA驱动与PyTorch

## 注意事项

1. 本脚本适用于在无界面环境下渲染，例如服务器或集群环境
2. 请确保配置文件和检查点路径正确
3. 要获得最佳性能，请使用支持CUDA的GPU
4. 图像尺寸越大，渲染时间越长，内存消耗也越多 