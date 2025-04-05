import os
import sys
import time
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(".."))

from neus_loader import NeuSModelLoader
from gaussians_loader import GaussiansModelLoader

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试批量折射光路计算的性能和正确性")
    
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
    parser.add_argument("--n-tests", type=int, default=5,
                        help="重复测试的次数")
    parser.add_argument("--n1", type=float, default=1.0003,
                        help="空气的折射率")
    parser.add_argument("--n2", type=float, default=1.51,
                        help="玻璃的折射率")                        
    parser.add_argument("--vis-output", type=str, default="batch_refraction_test",
                        help="可视化结果输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                        help="运行设备")
    
    return parser.parse_args()

def generate_test_rays(batch_size, device):
    """生成用于测试的随机光线"""
    # 生成随机光线起点（在单位球体外围）
    radius = 2.0
    theta = torch.rand(batch_size, device=device) * 2 * np.pi
    phi = torch.rand(batch_size, device=device) * np.pi
    
    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi)
    
    ray_origins = torch.stack([x, y, z], dim=-1)
    
    # 生成随机光线方向（指向原点附近）
    center = torch.zeros(3, device=device)
    center += torch.randn(3, device=device) * 0.1  # 向原点添加少量噪声
    
    # 计算指向中心的方向
    ray_directions = center - ray_origins
    ray_directions = torch.nn.functional.normalize(ray_directions, p=2, dim=-1)
    
    return ray_origins, ray_directions

def test_refraction(neus_loader, args):
    """测试折射光路计算的性能和正确性"""
    device = torch.device(args.device)
    batch_size = args.batch_size
    
    # 创建输出目录
    os.makedirs(args.vis_output, exist_ok=True)
    
    # 生成测试光线
    ray_origins, ray_directions = generate_test_rays(batch_size, device)
    
    # 执行多次测试以获得稳定的时间测量
    batch_compute_times = []
    batch_refractive_times = []
    batch_full_times = []
    
    print(f"开始测试，批量大小: {batch_size}，测试次数: {args.n_tests}")
    
    for i in range(args.n_tests):
        print(f"测试 {i+1}/{args.n_tests}")
        
        # 测试批量求交性能
        torch.cuda.synchronize()
        start_time = time.time()
        intersection_points, intersection_normals, valid_mask = neus_loader.compute_batch_intersection(
            ray_origins, ray_directions
        )
        torch.cuda.synchronize()
        batch_compute_time = time.time() - start_time
        batch_compute_times.append(batch_compute_time)
        
        # 测试批量折射计算性能
        torch.cuda.synchronize()
        start_time = time.time()
        refracted_origins, refracted_directions, valid_refraction = neus_loader.compute_batch_refractive_rays(
            ray_origins, ray_directions, args.n1, args.n2
        )
        torch.cuda.synchronize()
        batch_refractive_time = time.time() - start_time
        batch_refractive_times.append(batch_refractive_time)
        
        # 测试完整折射光路计算性能
        torch.cuda.synchronize()
        start_time = time.time()
        exit_origins, exit_directions, valid_exit = neus_loader.compute_batch_full_refractive_path(
            ray_origins, ray_directions, args.n1, args.n2
        )
        torch.cuda.synchronize()
        batch_full_time = time.time() - start_time
        batch_full_times.append(batch_full_time)
        
        # 验证结果有效性
        if i == 0:
            verify_results(
                intersection_points, intersection_normals, valid_mask,
                refracted_origins, refracted_directions, valid_refraction,
                exit_origins, exit_directions, valid_exit,
                args
            )
    
    # 计算平均时间
    avg_batch_compute_time = np.mean(batch_compute_times)
    avg_batch_refractive_time = np.mean(batch_refractive_times)
    avg_batch_full_time = np.mean(batch_full_times)
    
    print("\n性能测试结果:")
    print(f"批量求交平均时间: {avg_batch_compute_time:.6f} 秒")
    print(f"批量折射计算平均时间: {avg_batch_refractive_time:.6f} 秒")
    print(f"完整折射光路计算平均时间: {avg_batch_full_time:.6f} 秒")
    
    # 生成性能比较图表
    plt.figure(figsize=(10, 6))
    plt.bar(['Intersection', 'Refraction', 'Complete Path'], [avg_batch_compute_time, avg_batch_refractive_time, avg_batch_full_time])
    plt.ylabel('Execution Time (seconds)')
    plt.title('Refraction Path Calculation Performance Test')
    plt.savefig(os.path.join(args.vis_output, 'refraction_performance.png'))
    
    return {
        'compute_time': avg_batch_compute_time,
        'refractive_time': avg_batch_refractive_time,
        'full_time': avg_batch_full_time
    }

def verify_results(
    intersection_points, intersection_normals, valid_mask,
    refracted_origins, refracted_directions, valid_refraction,
    exit_origins, exit_directions, valid_exit,
    args
):
    """验证折射计算结果的正确性"""
    # 打印统计信息
    total_rays = valid_mask.shape[0]
    valid_intersection_count = valid_mask.sum().item()
    valid_refraction_count = valid_refraction.sum().item()
    valid_exit_count = valid_exit.sum().item()
    
    print("\n结果验证:")
    print(f"总光线数: {total_rays}")
    print(f"有效交点数: {valid_intersection_count} ({100*valid_intersection_count/total_rays:.2f}%)")
    print(f"有效折射数: {valid_refraction_count} ({100*valid_refraction_count/total_rays:.2f}%)")
    print(f"有效出射数: {valid_exit_count} ({100*valid_exit_count/total_rays:.2f}%)")
    
    # 验证折射光线的归一化
    if valid_refraction_count > 0:
        refracted_dir_norms = torch.norm(refracted_directions[valid_refraction], dim=1)
        refracted_dir_min = refracted_dir_norms.min().item()
        refracted_dir_max = refracted_dir_norms.max().item()
        refracted_dir_mean = refracted_dir_norms.mean().item()
        
        print(f"折射光线方向归一化: 最小={refracted_dir_min:.6f}, 最大={refracted_dir_max:.6f}, 平均={refracted_dir_mean:.6f}")
    
    # 验证出射光线的归一化
    if valid_exit_count > 0:
        exit_dir_norms = torch.norm(exit_directions[valid_exit], dim=1)
        exit_dir_min = exit_dir_norms.min().item()
        exit_dir_max = exit_dir_norms.max().item()
        exit_dir_mean = exit_dir_norms.mean().item()
        
        print(f"出射光线方向归一化: 最小={exit_dir_min:.6f}, 最大={exit_dir_max:.6f}, 平均={exit_dir_mean:.6f}")
    
    # 保存部分结果到文件
    with open(os.path.join(args.vis_output, 'refraction_stats.txt'), 'w') as f:
        f.write(f"折射率: n1={args.n1}, n2={args.n2}\n")
        f.write(f"总光线数: {total_rays}\n")
        f.write(f"有效交点数: {valid_intersection_count} ({100*valid_intersection_count/total_rays:.2f}%)\n")
        f.write(f"有效折射数: {valid_refraction_count} ({100*valid_refraction_count/total_rays:.2f}%)\n")
        f.write(f"有效出射数: {valid_exit_count} ({100*valid_exit_count/total_rays:.2f}%)\n")
        
        if valid_exit_count > 0:
            # 提取一些成功的折射-出射示例
            valid_exit_indices = torch.where(valid_exit)[0][:5]  # 最多5个示例
            
            for idx, ray_idx in enumerate(valid_exit_indices):
                f.write(f"\n示例 {idx+1}:\n")
                f.write(f"  光线起点: {refracted_origins[ray_idx].cpu().numpy()}\n")
                f.write(f"  光线方向: {refracted_directions[ray_idx].cpu().numpy()}\n")
                f.write(f"  折射光线起点: {refracted_origins[ray_idx].cpu().numpy()}\n")
                f.write(f"  折射光线方向: {refracted_directions[ray_idx].cpu().numpy()}\n")
                f.write(f"  出射光线起点: {exit_origins[ray_idx].cpu().numpy()}\n")
                f.write(f"  出射光线方向: {exit_directions[ray_idx].cpu().numpy()}\n")

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
    
    # 测试折射计算性能
    print(f"开始测试折射计算性能...")
    results = test_refraction(neus_loader, args)
    
    print("\n测试完成!")
    print(f"结果已保存到目录: {args.vis_output}")

# 测试批量折射计算与逐条光线处理的性能差异
def test_batch_vs_loop_refraction(neus_model_path, num_rays=10000, batch_size=10000):
    """
    测试批量折射计算与逐条光线处理的性能差异
    
    Args:
        neus_model_path: NeuS模型路径
        num_rays: 测试光线总数
        batch_size: 批处理大小
    """
    print(f"测试批量折射与循环处理性能 (光线数: {num_rays})")
    
    # 创建NeuS模型加载器
    neus_loader = NeuSModelLoader(checkpoint_path=neus_model_path)
    
    # 创建随机光线
    torch.manual_seed(42)  # 固定随机种子以便比较
    rays_o = torch.rand(num_rays, 3, device='cuda') * 2 - 1  # 随机光线起点
    rays_d = torch.rand(num_rays, 3, device='cuda')
    rays_d = F.normalize(rays_d, p=2, dim=-1)  # 归一化方向
    
    # 计时：逐条光线处理
    print("测试逐条光线处理...")
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    loop_results = []
    for i in range(num_rays):
        ray_o = rays_o[i:i+1]
        ray_d = rays_d[i:i+1]
        
        results = neus_loader.compute_full_refractive_path(ray_o, ray_d)
        loop_results.append(results)
        
        # 每处理1000条光线清理一次内存
        if i % 1000 == 0 and i > 0:
            torch.cuda.empty_cache()
            print(f"已处理 {i}/{num_rays} 条光线")
    
    torch.cuda.synchronize()
    loop_time = time.time() - start_time
    print(f"逐条处理时间: {loop_time:.4f} 秒")
    
    # 计时：批量处理
    print("测试批量处理...")
    torch.cuda.empty_cache()  # 清理内存
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    # 分批处理以避免内存问题
    batch_results = []
    for i in range(0, num_rays, batch_size):
        end_idx = min(i + batch_size, num_rays)
        batch_rays_o = rays_o[i:end_idx]
        batch_rays_d = rays_d[i:end_idx]
        
        # 使用批量计算
        results = neus_loader.compute_batch_full_refractive_path(batch_rays_o, batch_rays_d)
        batch_results.append(results)
        
        torch.cuda.empty_cache()  # 清理内存
        print(f"已批量处理 {end_idx}/{num_rays} 条光线")
    
    torch.cuda.synchronize()
    batch_time = time.time() - start_time
    print(f"批量处理时间: {batch_time:.4f} 秒")
    
    # 计算加速比
    speedup = loop_time / batch_time
    print(f"批量处理相比逐条处理加速了 {speedup:.2f}x")
    
    return loop_time, batch_time, speedup


# 测试不同批量大小对性能和内存的影响
def test_batch_size_impact(neus_model_path, total_rays=50000, batch_sizes=[5000, 10000, 20000, 50000]):
    """
    测试不同批量大小对性能和内存的影响
    
    Args:
        neus_model_path: NeuS模型路径
        total_rays: 测试光线总数
        batch_sizes: 测试的不同批量大小
    """
    print(f"测试不同批量大小对折射计算的影响 (总光线数: {total_rays})")
    
    # 创建NeuS模型加载器
    neus_loader = NeuSModelLoader(checkpoint_path=neus_model_path)
    
    # 创建随机光线
    torch.manual_seed(42)  # 固定随机种子以便比较
    rays_o = torch.rand(total_rays, 3, device='cuda') * 2 - 1  # 随机光线起点
    rays_d = torch.rand(total_rays, 3, device='cuda')
    rays_d = F.normalize(rays_d, p=2, dim=-1)  # 归一化方向
    
    # 存储结果
    timings = []
    
    # 测试每种批量大小
    for batch_size in batch_sizes:
        print(f"\n测试批量大小: {batch_size}")
        torch.cuda.empty_cache()  # 清理内存
        
        try:
            torch.cuda.synchronize()
            start_time = time.time()
            
            # 分批处理
            for i in range(0, total_rays, batch_size):
                end_idx = min(i + batch_size, total_rays)
                current_size = end_idx - i
                print(f"处理批次 {i//batch_size + 1}: {current_size} 条光线")
                
                batch_rays_o = rays_o[i:end_idx]
                batch_rays_d = rays_d[i:end_idx]
                
                # 使用内存优化的批量计算
                results = neus_loader.compute_batch_full_refractive_path(
                    batch_rays_o, batch_rays_d, 
                    max_batch_size=min(5000, batch_size)  # 内部批处理大小限制
                )
                
                torch.cuda.empty_cache()  # 主动释放内存
            
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            print(f"批量大小 {batch_size} 处理时间: {elapsed:.4f} 秒")
            timings.append((batch_size, elapsed))
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"批量大小 {batch_size} 导致内存溢出")
                timings.append((batch_size, float('inf')))
            else:
                raise e
    
    # 分析结果
    valid_timings = [(bs, t) for bs, t in timings if t != float('inf')]
    if valid_timings:
        best_batch_size, best_time = min(valid_timings, key=lambda x: x[1])
        print(f"\n最佳批量大小: {best_batch_size}，处理时间: {best_time:.4f} 秒")
        
        # 绘制性能图表
        batch_sizes = [bs for bs, _ in valid_timings]
        times = [t for _, t in valid_timings]
        
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, times, 'o-')
        plt.xlabel('批量大小')
        plt.ylabel('处理时间 (秒)')
        plt.title('批量大小对折射计算性能的影响')
        plt.grid(True)
        plt.savefig('batch_size_performance.png')
        plt.close()
    else:
        print("所有测试的批量大小都导致内存溢出")
    
    return timings

# 内存优化版：低分辨率渲染测试
def test_memory_optimized_rendering(neus_model_path, gaussians_model_path, scene_path, resolution=400):
    """
    测试内存优化的渲染（使用低分辨率）
    
    Args:
        neus_model_path: NeuS模型路径
        gaussians_model_path: 3D高斯模型路径
        scene_path: 场景数据路径
        resolution: 渲染分辨率
    """
    print(f"测试内存优化渲染 (分辨率: {resolution}x{resolution})")
    
    # 导入必要的类
    from refractive_renderer import RefractiveRenderer
    
    try:
        # 设置较低的分辨率以减少内存使用
        H, W = resolution, resolution
        
        # 限制GPU内存使用
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)  # 使用最多80%的GPU内存
        except:
            print("无法设置CUDA内存限制")
        
        # 初始化折射渲染器
        print("初始化折射渲染器...")
        renderer = RefractiveRenderer(
            neus_model_path=neus_model_path,
            gaussians_model_path=gaussians_model_path,
            scene_path=scene_path,
            auto_scale=True
        )
        
        # 创建输出目录
        output_dir = "memory_optimized_test"
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载相机数据并渲染
        view_index = 0
        print(f"渲染视图 {view_index}...")
        outputs = renderer.render_view(
            view_index=view_index,
            H=H,
            W=W,
            render_refractive=True,
            output_dir=output_dir
        )
        
        return True
    
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 设置模型路径
    neus_model_path = "/workspace/sdf/NeTO/Use3DGRUT/exp/eiko_ball_masked/silhouette/checkpoints/ckpt_300000.pth"
    gaussians_model_path = "/workspace/runs/eiko_ball_masked_expanded_3dgrt/eiko_ball_masked_expanded-2703_050608/ckpt_last.pt"
    scene_path = "/workspace/sdf/NeTO/Use3DGRUT/data/eiko_ball_masked"
    
    # 执行内存优化的测试
    print("开始内存优化测试...")
    
    # 测试1: 测试批量大小的影响
    # test_batch_size_impact(neus_model_path, total_rays=50000, batch_sizes=[1000, 2000, 5000, 10000])
    
    # 测试2: 内存优化的渲染测试（低分辨率）
    test_memory_optimized_rendering(neus_model_path, gaussians_model_path, scene_path, resolution=400) 