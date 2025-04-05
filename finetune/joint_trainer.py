    import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from tqdm import tqdm

from refractive_renderer import RefractiveRenderer

class JointTrainer:
    """
    联合训练器，用于同时训练NeuS模型和3D高斯模型
    """
    
    def __init__(self,
                 neus_conf_path=None,
                 neus_case_name=None,
                 neus_checkpoint_path=None,
                 gaussians_checkpoint_path=None,
                 gaussians_config_path=None,
                 output_dir="finetune_output",
                 n1=1.0003,  # 空气的折射率
                 n2=1.51,    # 玻璃的折射率
                 device=None):
        """
        初始化联合训练器
        
        Args:
            neus_conf_path: NeuS配置文件路径
            neus_case_name: NeuS数据集名称
            neus_checkpoint_path: NeuS模型checkpoint路径
            gaussians_checkpoint_path: 3D高斯模型checkpoint路径
            gaussians_config_path: 3D高斯模型配置路径
            output_dir: 输出目录
            n1: 介质1的折射率(空气)
            n2: 介质2的折射率(玻璃)
            device: 运行设备
        """
        self.device = device if device is not None else torch.device("cuda")
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建折射渲染器
        self.renderer = RefractiveRenderer(
            neus_conf_path=neus_conf_path,
            neus_case_name=neus_case_name,
            neus_checkpoint_path=neus_checkpoint_path,
            gaussians_checkpoint_path=gaussians_checkpoint_path,
            gaussians_config_path=gaussians_config_path,
            n1=n1,
            n2=n2,
            device=device
        )
        
        # 设置NeuS模型的优化器
        self.setup_neus_optimizer()
        
        # 初始化训练状态
        self.global_step = 0
        self.epoch = 0
        
        print("联合训练器初始化完成")
    
    def setup_neus_optimizer(self, lr=1e-4):
        """设置NeuS模型的优化器"""
        # 获取SDF网络的参数
        sdf_params = list(self.renderer.neus_loader.sdf_network.parameters())
        
        # 创建优化器
        self.neus_optimizer = optim.Adam(sdf_params, lr=lr)
    
    def train_step(self, batch, mask, lambda_rgb=1.0, lambda_eikonal=0.1, lambda_normal=0.05):
        """
        执行一步训练
        
        Args:
            batch: 包含rays_ori, rays_dir等信息的batch数据
            mask: 透明物体的掩码
            lambda_rgb: RGB损失权重
            lambda_eikonal: Eikonal损失权重
            lambda_normal: 法向量一致性损失权重
            
        Returns:
            loss: 总损失
            metrics: 训练指标字典
        """
        # 确保NeuS模型处于训练模式
        self.renderer.neus_loader.sdf_network.train()
        
        # 清空梯度
        self.neus_optimizer.zero_grad()
        
        # 前向传播
        outputs = self.renderer(batch, mask, train=True)
        
        # 计算RGB损失
        rgb_loss = torch.nn.functional.mse_loss(outputs['pred_rgb'], batch.rgb_gt)
        
        # 获取一些点用于计算Eikonal损失和法向量一致性损失
        # 在计算折射时找到的表面点是很好的候选点
        surface_points = []
        
        # 如果有掩码，提取透明物体区域
        if mask is not None and mask.any():
            # 获取射线
            rays_o_flat = batch.rays_ori.reshape(-1, 3)
            rays_d_flat = batch.rays_dir.reshape(-1, 3)
            mask_flat = mask.reshape(-1, 1).squeeze(-1)
            
            # 获取透明区域的光线
            transparent_rays_o = rays_o_flat[mask_flat]
            transparent_rays_d = rays_d_flat[mask_flat]
            
            # 计算交点
            entry_points, _, valid_entry = self.renderer.neus_loader.compute_intersection(
                transparent_rays_o, transparent_rays_d
            )
            
            if valid_entry.any():
                # 添加有效的交点
                surface_points.append(entry_points[valid_entry])
        
        # 如果没有找到足够的表面点，随机生成一些点
        if not surface_points or sum(p.shape[0] for p in surface_points) < 100:
            # 生成随机点
            num_points = 1000
            random_points = torch.rand((num_points, 3), device=self.device) * 2 - 1  # 范围[-1, 1]
            surface_points.append(random_points)
        
        # 合并所有表面点
        all_points = torch.cat(surface_points, dim=0)
        
        # 计算Eikonal损失
        eikonal_loss = self.renderer.compute_eikonal_loss(all_points)
        
        # 计算法向量一致性损失
        normal_loss = self.renderer.compute_normal_consistency_loss(all_points)
        
        # 计算总损失
        loss = lambda_rgb * rgb_loss + lambda_eikonal * eikonal_loss + lambda_normal * normal_loss
        
        # 反向传播
        loss.backward()
        
        # 更新参数
        self.neus_optimizer.step()
        
        # 返回损失和指标
        metrics = {
            'loss': loss.item(),
            'rgb_loss': rgb_loss.item(),
            'eikonal_loss': eikonal_loss.item(),
            'normal_loss': normal_loss.item(),
            'psnr': -10 * torch.log10(rgb_loss).item()
        }
        
        return loss, metrics
    
    def train_epoch(self, dataloader, optimizer_params=None):
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            optimizer_params: 优化器参数
        
        Returns:
            metrics: 平均训练指标
        """
        # 设置优化器参数
        if optimizer_params is not None:
            self.setup_neus_optimizer(lr=optimizer_params.get('lr', 1e-4))
        
        # 记录训练指标
        metrics = {
            'loss': [],
            'rgb_loss': [],
            'eikonal_loss': [],
            'normal_loss': [],
            'psnr': []
        }
        
        # 训练循环
        self.renderer.train()
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {self.epoch}")):
            # 获取掩码
            mask = batch.mask if hasattr(batch, 'mask') else None
            
            # 训练步骤
            _, batch_metrics = self.train_step(batch, mask)
            
            # 记录指标
            for k, v in batch_metrics.items():
                metrics[k].append(v)
            
            self.global_step += 1
            
            # 每N步保存一次检查点
            if self.global_step % 1000 == 0:
                self.save_checkpoint()
                
            # 每N步可视化一次结果
            if self.global_step % 100 == 0:
                self.visualize(batch, mask)
        
        # 计算平均指标
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        
        # 增加epoch计数
        self.epoch += 1
        
        return avg_metrics
    
    def train(self, dataloader, num_epochs=10, optimizer_params=None):
        """
        训练多个epoch
        
        Args:
            dataloader: 数据加载器
            num_epochs: 训练的epoch数
            optimizer_params: 优化器参数
        
        Returns:
            metrics_history: 训练历史指标
        """
        # 记录训练历史指标
        metrics_history = []
        
        # 训练循环
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # 训练一个epoch
            avg_metrics = self.train_epoch(dataloader, optimizer_params)
            
            # 记录训练时间
            avg_metrics['time'] = time.time() - start_time
            
            # 保存指标历史
            metrics_history.append(avg_metrics)
            
            # 打印训练信息
            print(f"Epoch {self.epoch-1}/{num_epochs}, "
                  f"Loss: {avg_metrics['loss']:.6f}, "
                  f"PSNR: {avg_metrics['psnr']:.2f}, "
                  f"Time: {avg_metrics['time']:.2f}s")
            
            # 保存检查点
            self.save_checkpoint()
        
        return metrics_history
    
    def visualize(self, batch, mask=None):
        """
        可视化当前结果
        
        Args:
            batch: 包含rays_ori, rays_dir等信息的batch数据
            mask: 透明物体的掩码
        """
        # 确保模型处于评估模式
        self.renderer.eval()
        
        with torch.no_grad():
            # 渲染结果
            outputs = self.renderer(batch, mask)
            
            # 获取预测RGB和真实RGB
            pred_rgb = outputs['pred_rgb'][0].detach().cpu()
            gt_rgb = batch.rgb_gt[0].detach().cpu()
            
            # 如果有折射结果，也保存
            if 'refracted_rgb' in outputs:
                refracted_rgb = outputs['refracted_rgb'][0].detach().cpu()
                combined = torch.cat([gt_rgb, pred_rgb, refracted_rgb], dim=1)
            else:
                combined = torch.cat([gt_rgb, pred_rgb], dim=1)
            
            # 保存图像
            os.makedirs(os.path.join(self.output_dir, 'visualizations'), exist_ok=True)
            torchvision.utils.save_image(
                combined.permute(2, 0, 1),
                os.path.join(self.output_dir, 'visualizations', f'step_{self.global_step:06d}.png')
            )
        
        # 恢复训练模式
        self.renderer.train()
    
    def save_checkpoint(self):
        """保存检查点"""
        # 创建检查点目录
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        
        # 保存NeuS模型
        neus_state = {
            'sdf_network': self.renderer.neus_loader.sdf_network.state_dict(),
            'deviation_network': self.renderer.neus_loader.deviation_network.state_dict(),
            'neus_optimizer': self.neus_optimizer.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch
        }
        
        # 保存检查点
        torch.save(
            neus_state,
            os.path.join(self.output_dir, 'checkpoints', f'neus_step_{self.global_step:06d}.pt')
        )
        
        # 保存最新检查点
        torch.save(
            neus_state,
            os.path.join(self.output_dir, 'checkpoints', 'neus_latest.pt')
        )
        
        print(f"保存检查点: step {self.global_step}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载NeuS模型
        self.renderer.neus_loader.sdf_network.load_state_dict(checkpoint['sdf_network'])
        self.renderer.neus_loader.deviation_network.load_state_dict(checkpoint['deviation_network'])
        
        # 加载优化器
        self.neus_optimizer.load_state_dict(checkpoint['neus_optimizer'])
        
        # 加载训练状态
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        
        print(f"加载检查点: global_step {self.global_step}, epoch {self.epoch}")
        
    def test(self, dataloader):
        """
        测试模型
        
        Args:
            dataloader: 测试数据加载器
        
        Returns:
            avg_metrics: 平均测试指标
        """
        # 确保模型处于评估模式
        self.renderer.eval()
        
        # 记录测试指标
        metrics = {
            'loss': [],
            'psnr': []
        }
        
        # 测试循环
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Testing")):
                # 获取掩码
                mask = batch.mask if hasattr(batch, 'mask') else None
                
                # 前向传播
                outputs = self.renderer(batch, mask)
                
                # 计算RGB损失
                rgb_loss = torch.nn.functional.mse_loss(outputs['pred_rgb'], batch.rgb_gt)
                psnr = -10 * torch.log10(rgb_loss).item()
                
                # 记录指标
                metrics['loss'].append(rgb_loss.item())
                metrics['psnr'].append(psnr)
                
                # 保存渲染结果
                os.makedirs(os.path.join(self.output_dir, 'test_results'), exist_ok=True)
                # 获取预测RGB和真实RGB
                pred_rgb = outputs['pred_rgb'][0].detach().cpu()
                gt_rgb = batch.rgb_gt[0].detach().cpu()
                
                # 如果有折射结果，也保存
                if 'refracted_rgb' in outputs:
                    refracted_rgb = outputs['refracted_rgb'][0].detach().cpu()
                    combined = torch.cat([gt_rgb, pred_rgb, refracted_rgb], dim=1)
                else:
                    combined = torch.cat([gt_rgb, pred_rgb], dim=1)
                
                torchvision.utils.save_image(
                    combined.permute(2, 0, 1),
                    os.path.join(self.output_dir, 'test_results', f'test_{batch_idx:04d}.png')
                )
        
        # 计算平均指标
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        
        print(f"Test Results - Loss: {avg_metrics['loss']:.6f}, PSNR: {avg_metrics['psnr']:.2f}")
        
        return avg_metrics 