# 3DGRUT 透明物体折射渲染和微调

本目录包含用于透明物体折射渲染和微调的代码。该代码实现了一个联合训练框架，可以同时训练NeuS（用于透明物体的几何形状）和3DGRT/3DGUT（用于背景场景）。

## 主要组件

- `neus_loader.py`: 加载预训练的NeuS模型，用于表示透明物体的SDF
- `gaussians_loader.py`: 加载预训练的3DGRT/3DGUT模型，用于表示背景场景
- `refractive_renderer.py`: 实现折射渲染器，结合NeuS和3DGRT/3DGUT模型
- `joint_trainer.py`: 实现联合训练器，用于微调NeuS模型
- `train_refractive.py`: 启动联合训练的脚本
- `test_refractive_renderer.py`: 测试折射渲染器的脚本

## 环境准备

确保已经按照主README中的说明设置好环境。另外，您需要准备以下数据：

1. 预训练的NeuS模型（SDF模型）
2. 预训练的3DGRT/3DGUT模型（背景场景）
3. 数据集（包含RGB图像和相机参数）
4. 透明物体掩码（可选）

## 使用说明

### 测试折射渲染器

首先，您可以使用以下命令测试折射渲染器：

```bash
python test_refractive_renderer.py \
    --gaussians-ckpt /path/to/3dgrt_checkpoint.pt \
    --data-path /path/to/dataset \
    --mask-path /path/to/masks \
    --output-dir renderer_test \
    --frame-idx 0
```

参数说明：
- `--neus-conf`: NeuS配置文件路径（默认：`../sdf/NeTO/Use3DGRUT/confs/silhouette.conf`）
- `--neus-case`: NeuS数据集名称（默认：`eiko_ball_masked`）
- `--neus-ckpt`: NeuS模型checkpoint路径（默认：使用最新的checkpoint）
- `--gaussians-ckpt`: 3D高斯模型checkpoint路径（必需）
- `--gaussians-conf`: 3D高斯模型配置文件路径（默认：从checkpoint中加载）
- `--data-path`: 数据集路径（必需）
- `--downsample`: 图像下采样因子（默认：1）
- `--mask-path`: 掩码图像目录路径（可选）
- `--frame-idx`: 要渲染的帧索引（默认：0）
- `--n1`: 空气的折射率（默认：1.0003）
- `--n2`: 玻璃的折射率（默认：1.51）
- `--output-dir`: 输出目录（默认：`renderer_test`）
- `--device`: 运行设备（默认：`cuda`）

### 开始联合训练

使用以下命令启动联合训练：

```bash
python train_refractive.py \
    --gaussians-ckpt /path/to/3dgrt_checkpoint.pt \
    --data-path /path/to/dataset \
    --output-dir finetune_output \
    --num-epochs 30 \
    --lr 5e-5
```

参数说明：
- `--neus-conf`: NeuS配置文件路径（默认：`../sdf/NeTO/Use3DGRUT/confs/silhouette.conf`）
- `--neus-case`: NeuS数据集名称（默认：`eiko_ball_masked`）
- `--neus-ckpt`: NeuS模型checkpoint路径（默认：使用最新的checkpoint）
- `--gaussians-ckpt`: 3D高斯模型checkpoint路径（必需）
- `--gaussians-conf`: 3D高斯模型配置文件路径（默认：从checkpoint中加载）
- `--data-path`: 数据集路径（必需）
- `--downsample`: 图像下采样因子（默认：1）
- `--num-epochs`: 训练的epoch数（默认：30）
- `--batch-size`: 批量大小（默认：1）
- `--lr`: 学习率（默认：5e-5）
- `--output-dir`: 输出目录（默认：`finetune_output`）
- `--n1`: 空气的折射率（默认：1.0003）
- `--n2`: 玻璃的折射率（默认：1.51）
- `--resume`: 恢复训练的检查点路径（可选）
- `--seed`: 随机种子（默认：42）
- `--device`: 运行设备（默认：`cuda`）

## 训练流程

1. 加载预训练的NeuS模型和3DGRT/3DGUT模型
2. 使用掩码分离透明物体区域和背景区域
3. 计算折射光线并追踪
4. 使用RGB损失、Eikonal损失和法向量一致性损失优化NeuS模型

## 结果可视化

训练过程中，渲染结果将保存在输出目录的`visualizations`子目录中。训练完成后，测试结果将保存在`test_results`子目录中。

## 注意事项

- 目前仅支持优化NeuS模型，3DGRT/3DGUT模型保持固定不变
- 对于大型场景，建议使用较小的学习率
- 确保掩码文件与数据集图像一一对应

## 调试注意事项

在实现透明物体折射渲染时，以下是一些常见问题及解决方案：

### 1. 光线与SDF模型相交

- **问题**: 射线可能无法与SDF模型相交，导致无法计算折射
- **解决方案**: 
  - 确保正确设置了`object_bounding_sphere`参数（对应物体的边界球半径）
  - 使用`near_far_from_sphere`函数计算合适的近远平面
  - 确保射线方向是正确的，指向模型中心

### 2. 坐标转换问题

- **问题**: 世界坐标系与模型局部坐标系不一致
- **解决方案**:
  - 使用`to_local_coords`和`to_world_coords`函数进行坐标转换
  - 正确设置`scale_mat`矩阵，包含缩放和平移信息

### 3. 折射计算问题

- **问题**: 折射方向计算错误或张量维度不匹配
- **解决方案**:
  - 确保正确实现斯涅尔定律，特别是公式的符号
  - 注意张量广播时的维度匹配
  - 折射公式: η₁(I) - (η₁cosθᵢ - η₂cosθₜ)N
  - 其中 cosθₜ = sqrt(1 - (η₁/η₂)²(1 - cosθᵢ²))

### 4. 全反射处理

- **问题**: 当入射角过大时，可能发生全反射，需要特殊处理
- **解决方案**:
  - 检测判别式k < 0的情况，这表示发生了全反射
  - 在这种情况下，计算反射方向而不是折射方向

### 5. 数值稳定性

- **问题**: 由于浮点精度问题，光线可能被困在表面
- **解决方案**:
  - 使用小的epsilon值将折射光线起点偏移到表面内部或外部
  - 通常使用法向量的方向进行偏移 