# SDF求交和折射计算批量优化

本文档描述了对NeuS模型中SDF求交和折射计算的批量优化，旨在提高透明物体渲染的性能。

## 优化内容

我们对以下关键函数进行了批量优化：

1. **SDF求交算法**：
   - 实现了`compute_batch_intersection`函数，采用批量处理方式计算光线与SDF表面的交点
   - 使用二分法细化求交结果，提高精度
   - 避免了原始实现中的循环处理

2. **SDF梯度计算**：
   - 实现了`get_batch_sdf_gradients`函数，支持批量计算SDF梯度（法向量）
   - 添加了分块处理机制，避免GPU内存溢出

3. **折射光线计算**：
   - 实现了`compute_batch_refractive_rays`函数，批量计算折射光线
   - 优化了逻辑，减少了不必要的维度调整和扩展操作

4. **完整折射光路计算**：
   - 实现了`compute_batch_full_refractive_path`函数，批量计算完整的折射光路
   - 包括计算入射点、折射光线和出射点，处理全反射等情况

## 性能提升

通过批量处理，我们可以获得以下性能提升：

1. **减少GPU调用次数**：将单个光线的多次计算合并为一次批量计算，减少了GPU的调用开销
2. **提高并行度**：充分利用GPU的并行计算能力，同时处理多条光线
3. **减少循环开销**：避免Python循环带来的性能损失
4. **优化内存访问**：通过批量访问内存，提高了内存访问效率

## 测试脚本

为了验证优化效果，我们提供了两个测试脚本：

1. **test_batch_intersection.py**：
   - 测试批量SDF求交算法的性能和正确性
   - 比较原始方法和批量方法的执行时间
   - 验证求交结果的一致性

2. **test_batch_refraction.py**：
   - 测试批量折射计算的性能
   - 评估单次求交、折射计算和完整光路计算的时间
   - 验证折射和出射光线的正确性

## 使用方法

### 测试批量SDF求交

```bash
python test_batch_intersection.py \
    --neus-conf "../sdf/NeTO/Use3DGRUT/confs/silhouette.conf" \
    --neus-case "eiko_ball_masked" \
    --batch-size 1024 \
    --n-tests 10 \
    --vis-output "batch_intersection_test" \
    --device "cuda"
```

### 测试批量折射计算

```bash
python test_batch_refraction.py \
    --neus-conf "../sdf/NeTO/Use3DGRUT/confs/silhouette.conf" \
    --neus-case "eiko_ball_masked" \
    --batch-size 1024 \
    --n-tests 5 \
    --n1 1.0003 \
    --n2 1.51 \
    --vis-output "batch_refraction_test" \
    --device "cuda"
```

## 注意事项

1. 批量计算可能会占用更多的GPU内存，对于大型批次，可能需要减小批次大小或启用分块处理
2. 在某些边缘情况下，批量处理结果可能与原始逐个处理结果有微小差异，但这通常不会影响渲染质量
3. 为避免全反射光线的计算，建议使用适当的折射率差异
4. 在联合训练过程中，这些批量优化可以显著提高训练速度

## 未来优化方向

1. 使用CUDA实现关键算法，进一步提高性能
2. 优化内存使用，减少临时张量分配
3. 实现更高效的AABB树或八叉树数据结构加速求交
4. 支持更复杂的材质模型，如粗糙表面的散射 