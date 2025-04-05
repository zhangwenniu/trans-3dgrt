import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(".."))

from refractive_renderer import RefractiveRenderer
from threedgrut.datasets import ColmapDataset

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="测试折射渲染器")
    
    # NeuS模型相关参数
    parser.add_argument("--neus-conf", type=str, default="../sdf/NeTO/Use3DGRUT/confs/silhouette.conf",
                        help="NeuS配置文件路径")
    parser.add_argument("--neus-case", type=str, default="eiko_ball_masked",
                        help="NeuS数据集名称")
    parser.add_argument("--neus-ckpt", type=str, default=None,
                        help="NeuS模型checkpoint路径，如果为None则使用最新的checkpoint")
    
    # 3D高斯模型相关参数
    parser.add_argument("--gaussians-ckpt", type=str, required=True,
                        help="3D高斯模型checkpoint路径")
    parser.add_argument("--gaussians-conf", type=str, default=None,
                        help="3D高斯模型配置文件路径，如果为None则从checkpoint中加载")
    
    # 数据集相关参数
    parser.add_argument("--data-path", type=str, required=True,
                        help="数据集路径")
    parser.add_argument("--downsample", type=int, default=1,
                        help="图像下采样因子")
    parser.add_argument("--mask-path", type=str, default=None,
                        help="掩码图像目录路径，如果提供则加载透明物体掩码")
    parser.add_argument("--frame-idx", type=int, default=0,
                        help="要渲染的帧索引")
    
    # 物理参数
    parser.add_argument("--n1", type=float, default=1.0003,
                        help="空气的折射率")
    parser.add_argument("--n2", type=float, default=1.51,
                        help="玻璃的折射率")
    
    # 输出参数
    parser.add_argument("--output-dir", type=str, default="renderer_test",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                        help="运行设备")
    
    return parser.parse_args()

def load_mask(mask_path, idx, shape):
    """
    加载透明物体掩码
    
    Args:
        mask_path: 掩码目录路径
        idx: 帧索引
        shape: 图像形状 (H, W)
        
    Returns:
        mask: 透明物体掩码 [1, H, W, 1]
    """
    import cv2
    
    if mask_path is None:
        return None
    
    # 查找掩码文件
    mask_files = sorted([f for f in os.listdir(mask_path) if f.endswith('.png') or f.endswith('.jpg')])
    
    if not mask_files:
        print(f"警告: 在{mask_path}中未找到掩码文件")
        return None
    
    if idx >= len(mask_files):
        print(f"警告: 帧索引{idx}超出掩码文件数量{len(mask_files)}，使用第一个掩码")
        idx = 0
    
    # 加载掩码
    mask_file = os.path.join(mask_path, mask_files[idx])
    print(f"加载掩码: {mask_file}")
    
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"错误: 无法加载掩码{mask_file}")
        return None
    
    # 调整掩码大小
    H, W = shape
    mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    
    # 二值化掩码
    _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    
    # 转换为PyTorch张量
    mask = torch.from_numpy(mask).float().to('cuda')
    mask = mask.unsqueeze(0).unsqueeze(-1)  # [1, H, W, 1]
    
    return mask

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 创建设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据集
    dataset = ColmapDataset(
        args.data_path,
        split="val",
        downsample_factor=args.downsample
    )
    
    # 创建折射渲染器
    renderer = RefractiveRenderer(
        neus_conf_path=args.neus_conf,
        neus_case_name=args.neus_case,
        neus_checkpoint_path=args.neus_ckpt,
        gaussians_checkpoint_path=args.gaussians_ckpt,
        gaussians_config_path=args.gaussians_conf,
        n1=args.n1,
        n2=args.n2,
        device=device
    )
    
    # 设置模型为评估模式
    renderer.eval()
    
    # 获取指定帧
    if args.frame_idx >= len(dataset):
        print(f"警告: 帧索引{args.frame_idx}超出数据集大小{len(dataset)}，使用第一帧")
        args.frame_idx = 0
    
    # 创建批次
    batch = dataset[args.frame_idx]
    
    # 将batch中的每个张量转换为batch形式 (添加批次维度)
    for key, value in batch.__dict__.items():
        if isinstance(value, torch.Tensor):
            setattr(batch, key, value.unsqueeze(0).to(device))
    
    # 获取图像形状
    H, W = batch.rgb_gt.shape[1:3]
    
    # 加载掩码
    mask = load_mask(args.mask_path, args.frame_idx, (H, W))
    
    # 渲染图像
    with torch.no_grad():
        print("渲染中...")
        outputs = renderer(batch, mask)
    
    # 保存结果
    pred_rgb = outputs['pred_rgb'][0].cpu()
    gt_rgb = batch.rgb_gt[0].cpu()
    
    # 创建比较图像
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(gt_rgb.numpy())
    plt.title('Ground Truth')
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.imshow(pred_rgb.numpy())
    plt.title('Predicted')
    plt.axis('off')
    
    # 如果有折射结果，也显示
    if 'refracted_rgb' in outputs:
        refracted_rgb = outputs['refracted_rgb'][0].cpu()
        plt.subplot(2, 2, 3)
        plt.imshow(refracted_rgb.numpy())
        plt.title('Refracted')
        plt.axis('off')
    
    # 如果有掩码，也显示
    if mask is not None:
        mask_vis = mask[0, ..., 0].cpu().numpy()
        plt.subplot(2, 2, 4)
        plt.imshow(mask_vis, cmap='gray')
        plt.title('Mask')
        plt.axis('off')
    
    # 保存图像
    plt.savefig(os.path.join(args.output_dir, 'comparison.png'))
    print(f"比较图像已保存到 {os.path.join(args.output_dir, 'comparison.png')}")
    
    # 保存各个单独的图像
    vutils.save_image(gt_rgb.permute(2, 0, 1), os.path.join(args.output_dir, 'gt.png'))
    vutils.save_image(pred_rgb.permute(2, 0, 1), os.path.join(args.output_dir, 'pred.png'))
    
    if 'refracted_rgb' in outputs:
        vutils.save_image(refracted_rgb.permute(2, 0, 1), os.path.join(args.output_dir, 'refracted.png'))
    
    # 如果有掩码，也保存
    if mask is not None:
        mask_vis = mask[0, ..., 0].cpu().unsqueeze(0)  # [1, H, W]
        vutils.save_image(mask_vis, os.path.join(args.output_dir, 'mask.png'))
    
    # 计算PSNR
    mse = torch.mean((pred_rgb - gt_rgb) ** 2)
    psnr = -10 * torch.log10(mse)
    print(f"PSNR: {psnr.item():.2f} dB")
    
    with open(os.path.join(args.output_dir, 'metrics.txt'), 'w') as f:
        f.write(f"PSNR: {psnr.item():.2f} dB\n")
    
    print("测试完成！")

if __name__ == "__main__":
    main() 