import os
import sys
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(".."))

from neus_loader import NeuSModelLoader

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试批量SDF求交算法的性能和正确性")
    
    # NeuS模型相关参数
    parser.add_argument("--neus-conf", type=str, default="../sdf/NeTO/Use3DGRUT/confs/silhouette.conf",
                        help="NeuS配置文件路径")
    parser.add_argument("--neus-case", type=str, default="eiko_ball_masked",
                        help="NeuS数据集名称")
    parser.add_argument("--neus-ckpt", type=str, default=None,
                        help="NeuS模型checkpoint路径，如果为None则使用最新的checkpoint")
    
    # 测试相关参数
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="测试的光线批量大小")
    parser.add_argument("--n-tests", type=int, default=10,
                        help="重复测试的次数")
    parser.add_argument("--vis-output", type=str, default="batch_intersection_test",
                        help="可视化结果输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                        help="运行设备")
    
    return parser.parse_args()

def test_intersection(neus_loader, args):
    """测试SDF求交算法的性能和正确性"""
    device = torch.device(args.device)
    batch_size = args.batch_size
    
    # 创建输出目录
    os.makedirs(args.vis_output, exist_ok=True)
    
    # 生成随机光线（在单位球附近）
    ray_origins = torch.randn(batch_size, 3, device=device) * 2.0
    ray_directions = torch.randn(batch_size, 3, device=device)
    ray_directions = torch.nn.functional.normalize(ray_directions, p=2, dim=-1)
    
    # 执行多次测试以获得稳定的时间测量
    original_times = []
    batch_times = []
    
    print(f"开始测试，批量大小: {batch_size}，测试次数: {args.n_tests}")
    
    for i in range(args.n_tests):
        print(f"测试 {i+1}/{args.n_tests}")
        
        # 测试原始方法
        torch.cuda.synchronize()
        start_time = time.time()
        intersection_points_orig, intersection_normals_orig, valid_mask_orig = neus_loader.compute_intersection(
            ray_origins, ray_directions
        )
        torch.cuda.synchronize()
        original_time = time.time() - start_time
        original_times.append(original_time)
        
        # 测试批量方法
        torch.cuda.synchronize()
        start_time = time.time()
        intersection_points_batch, intersection_normals_batch, valid_mask_batch = neus_loader.compute_batch_intersection(
            ray_origins, ray_directions
        )
        torch.cuda.synchronize()
        batch_time = time.time() - start_time
        batch_times.append(batch_time)
        
        # 验证结果是否一致
        if i == 0:  # 只在第一次测试时进行验证
            verify_results(
                intersection_points_orig, intersection_normals_orig, valid_mask_orig,
                intersection_points_batch, intersection_normals_batch, valid_mask_batch,
                args
            )
    
    # 计算平均时间
    avg_original_time = np.mean(original_times)
    avg_batch_time = np.mean(batch_times)
    speedup = avg_original_time / avg_batch_time
    
    print("\n性能比较结果:")
    print(f"原始方法平均时间: {avg_original_time:.6f} 秒")
    print(f"批量方法平均时间: {avg_batch_time:.6f} 秒")
    print(f"加速比: {speedup:.2f}x")
    
    # 生成性能比较图表
    plt.figure(figsize=(10, 6))
    plt.bar(['Original Method', 'Batch Method'], [avg_original_time, avg_batch_time])
    plt.ylabel('Execution Time (seconds)')
    plt.title('SDF Intersection Method Performance Comparison')
    plt.savefig(os.path.join(args.vis_output, 'performance_comparison.png'))
    
    return {
        'original_time': avg_original_time,
        'batch_time': avg_batch_time,
        'speedup': speedup
    }

def verify_results(points_orig, normals_orig, mask_orig, points_batch, normals_batch, mask_batch, args):
    """验证两种方法的结果是否一致"""
    # 检查掩码是否一致
    mask_equal = torch.all(mask_orig == mask_batch)
    mask_diff_count = (mask_orig != mask_batch).sum().item()
    
    # 检查交点坐标是否接近
    # 只比较两种方法都认为有效的交点
    valid_both = mask_orig & mask_batch
    points_diff = torch.norm(points_orig[valid_both] - points_batch[valid_both], dim=1)
    max_point_diff = points_diff.max().item() if points_diff.numel() > 0 else 0
    avg_point_diff = points_diff.mean().item() if points_diff.numel() > 0 else 0
    
    # 检查法向量是否接近
    normals_diff = torch.norm(normals_orig[valid_both] - normals_batch[valid_both], dim=1)
    max_normal_diff = normals_diff.max().item() if normals_diff.numel() > 0 else 0
    avg_normal_diff = normals_diff.mean().item() if normals_diff.numel() > 0 else 0
    
    print("\n结果验证:")
    print(f"掩码一致性: {'通过' if mask_equal else '失败'}")
    if not mask_equal:
        print(f"  掩码差异数量: {mask_diff_count}")
    
    print(f"交点坐标最大差异: {max_point_diff:.6f}")
    print(f"交点坐标平均差异: {avg_point_diff:.6f}")
    print(f"法向量最大差异: {max_normal_diff:.6f}")
    print(f"法向量平均差异: {avg_normal_diff:.6f}")
    
    # 可视化部分结果
    # 选择一些有效的光线进行可视化
    if valid_both.sum() > 0:
        vis_indices = torch.where(valid_both)[0][:5]  # 选择前5个有效交点
        
        # 创建可视化结果
        vis_results = []
        for idx in vis_indices:
            result = {
                'orig_point': points_orig[idx].cpu().numpy(),
                'batch_point': points_batch[idx].cpu().numpy(),
                'orig_normal': normals_orig[idx].cpu().numpy(),
                'batch_normal': normals_batch[idx].cpu().numpy(),
                'point_diff': torch.norm(points_orig[idx] - points_batch[idx]).item(),
                'normal_diff': torch.norm(normals_orig[idx] - normals_batch[idx]).item()
            }
            vis_results.append(result)
        
        # 保存结果到文本文件
        with open(os.path.join(args.vis_output, 'result_comparison.txt'), 'w') as f:
            for i, result in enumerate(vis_results):
                f.write(f"样本 {i+1}:\n")
                f.write(f"  原始方法交点: {result['orig_point']}\n")
                f.write(f"  批量方法交点: {result['batch_point']}\n")
                f.write(f"  交点差异: {result['point_diff']:.6f}\n")
                f.write(f"  原始方法法向量: {result['orig_normal']}\n")
                f.write(f"  批量方法法向量: {result['batch_normal']}\n")
                f.write(f"  法向量差异: {result['normal_diff']:.6f}\n\n")
    
    return {
        'mask_equal': mask_equal,
        'mask_diff_count': mask_diff_count,
        'max_point_diff': max_point_diff,
        'avg_point_diff': avg_point_diff,
        'max_normal_diff': max_normal_diff,
        'avg_normal_diff': avg_normal_diff
    }

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 加载NeuS模型
    print(f"加载NeuS模型...")
    neus_loader = NeuSModelLoader(
        conf_path=args.neus_conf,
        case_name=args.neus_case,
        checkpoint_path=args.neus_ckpt
    )
    
    # 测试SDF求交算法
    print(f"开始测试SDF求交算法...")
    results = test_intersection(neus_loader, args)
    
    print("\n测试完成!")
    print(f"结果已保存到目录: {args.vis_output}")

if __name__ == "__main__":
    main() 