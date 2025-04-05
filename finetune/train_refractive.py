import os
import sys
import argparse
import torch
import torch.utils.data
import numpy as np

# 将项目根目录添加到Python路径
sys.path.append(os.path.abspath(".."))

from joint_trainer import JointTrainer
from threedgrut.datasets import ColmapDataset

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="联合训练透明物体的几何形状和背景场景")
    
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
    
    # 训练相关参数
    parser.add_argument("--num-epochs", type=int, default=30,
                        help="训练的epoch数")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="批量大小")
    parser.add_argument("--lr", type=float, default=5e-5,
                        help="学习率")
    parser.add_argument("--output-dir", type=str, default="finetune_output",
                        help="输出目录")
    parser.add_argument("--device", type=str, default="cuda",
                        help="运行设备")
    
    # 物理参数
    parser.add_argument("--n1", type=float, default=1.0003,
                        help="空气的折射率")
    parser.add_argument("--n2", type=float, default=1.51,
                        help="玻璃的折射率")
    
    # 其他参数
    parser.add_argument("--resume", type=str, default=None,
                        help="恢复训练的检查点路径")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # 加载数据集
    train_dataset = ColmapDataset(
        args.data_path,
        split="train",
        downsample_factor=args.downsample
    )
    
    val_dataset = ColmapDataset(
        args.data_path,
        split="val",
        downsample_factor=args.downsample
    )
    
    # 创建数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # 创建联合训练器
    trainer = JointTrainer(
        neus_conf_path=args.neus_conf,
        neus_case_name=args.neus_case,
        neus_checkpoint_path=args.neus_ckpt,
        gaussians_checkpoint_path=args.gaussians_ckpt,
        gaussians_config_path=args.gaussians_conf,
        output_dir=args.output_dir,
        n1=args.n1,
        n2=args.n2,
        device=device
    )
    
    # 如果提供了恢复训练的检查点，则加载
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 设置优化器参数
    optimizer_params = {
        'lr': args.lr
    }
    
    # 训练模型
    print(f"开始训练，共{args.num_epochs}个epoch")
    trainer.train(train_dataloader, args.num_epochs, optimizer_params)
    
    # 测试模型
    print("开始测试")
    trainer.test(val_dataloader)
    
    print("训练和测试完成！")

if __name__ == "__main__":
    main() 