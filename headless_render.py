#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
import numpy as np
import torch
import cv2
import argparse
from pathlib import Path
from omegaconf import OmegaConf
from typing import Dict, Tuple, Optional

# 导入必要的模块
from threedgrut.model.model import MixtureOfGaussians
from playground.tracer import Tracer

class HeadlessRenderer:
    """无界面渲染器，直接使用Tracer类进行渲染"""
    
    def __init__(self, conf_path: str, checkpoint_path: Optional[str] = None, device: str = "cuda"):
        """
        初始化渲染器
        
        Args:
            conf_path: 配置文件路径
            checkpoint_path: 模型检查点路径（可选）
            device: 渲染设备
        """
        self.device = device
        
        # 加载配置
        self.conf = OmegaConf.load(conf_path)
        
        # 确保渲染相关配置存在
        if 'render' not in self.conf:
            raise ValueError("配置文件中缺少render部分")
            
        # 创建Tracer对象
        self.tracer = Tracer(self.conf)
        
        # 创建MixtureOfGaussians模型
        self.model = MixtureOfGaussians(self.conf)
        
        # 如果提供了检查点，加载模型
        if checkpoint_path:
            self._load_model(checkpoint_path)
            
    def _load_model(self, checkpoint_path: str):
        """
        从检查点加载模型
        
        Args:
            checkpoint_path: 检查点文件路径
        """
        print(f"正在加载模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if "model" in checkpoint:
            model_params = checkpoint["model"]
            # 设置模型参数
            for param_name, param_value in model_params.items():
                if hasattr(self.model, param_name) and isinstance(getattr(self.model, param_name), torch.nn.Parameter):
                    getattr(self.model, param_name).data = param_value.to(self.device)
            # 加载背景模型
            if "background" in model_params:
                self.model.background.load_state_dict(model_params["background"])
        else:
            # 尝试直接加载整个模型
            try:
                self.model.load_state_dict(checkpoint)
            except Exception as e:
                print(f"加载模型失败: {e}")
                raise
            
        # 构建加速结构
        self.tracer.build_gs_acc(self.model, rebuild=True)
        print(f"模型加载完成，高斯数量: {self.model.num_gaussians}")
        
    def _generate_rays(self, height: int, width: int, fov: float, camera_pose: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        生成相机射线
        
        Args:
            height: 图像高度
            width: 图像宽度
            fov: 垂直视场角（度）
            camera_pose: 相机姿态矩阵 [3, 4]
            
        Returns:
            包含射线原点和方向的字典
        """
        device = self.device
        focal = height / (2 * np.tan(np.radians(fov) / 2))
        
        # 生成像素坐标
        i, j = torch.meshgrid(
            torch.linspace(0, width - 1, width, device=device),
            torch.linspace(0, height - 1, height, device=device),
            indexing='ij'
        )
        
        # 转换为NDC坐标
        i = (i - width / 2) / focal
        j = (j - height / 2) / focal
        
        # 创建射线方向向量
        directions = torch.stack([i, -j, -torch.ones_like(i)], dim=-1)  # [width, height, 3]
        
        # 应用相机旋转
        rotation = camera_pose[:3, :3]  # [3, 3]
        directions = directions.reshape(-1, 3)  # [width*height, 3]
        rays_d = directions @ rotation.T  # [width*height, 3]
        
        # 射线归一化
        rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
        
        # 射线原点（相机位置）
        rays_o = camera_pose[:3, 3].expand(rays_d.shape)  # [width*height, 3]
        
        # 构建渲染所需的批次数据
        batch = {
            'rays_o_cam': rays_o.contiguous(),
            'rays_d_cam': rays_d.contiguous(),
            'poses': camera_pose[None].contiguous()  # [1, 3, 4]
        }
        
        return batch

    def render_image(self, height: int, width: int, fov: float, camera_pose: torch.Tensor) -> np.ndarray:
        """
        渲染图像
        
        Args:
            height: 图像高度
            width: 图像宽度
            fov: 垂直视场角（度）
            camera_pose: 相机姿态矩阵 [3, 4]
            
        Returns:
            渲染的RGB图像 [height, width, 3]
        """
        device = self.device
        
        # 确保相机姿态是正确的形状
        if camera_pose.shape != (3, 4):
            raise ValueError(f"相机姿态应为[3, 4]形状，但得到了{camera_pose.shape}")
            
        # 生成射线
        batch = self._generate_rays(height, width, fov, camera_pose)
        
        # 从模型中获取必要的数据
        mog_pos = self.model.positions.contiguous()
        mog_dns = self.model.get_density().contiguous()
        mog_rot = self.model.get_rotation().contiguous()
        mog_scl = self.model.get_scale().contiguous()
        
        # 创建包含高斯参数的密度张量
        particle_density = torch.concat([
            mog_pos, 
            mog_dns, 
            mog_rot, 
            mog_scl, 
            torch.zeros_like(mog_dns)
        ], dim=1).contiguous()
        
        # 获取特征
        features = self.model.get_features().contiguous()
        
        # 使用tracer直接调用trace函数
        with torch.cuda.nvtx.range("render") if hasattr(torch.cuda, "nvtx") else nullcontext():
            with torch.no_grad():
                (
                    pred_rgb,
                    pred_opacity,
                    pred_dist,
                    pred_normals,
                    hits_count
                ) = self.tracer.tracer_wrapper.trace(
                    0,  # frame_id
                    batch['poses'].contiguous(),
                    batch['rays_o_cam'].contiguous(),
                    batch['rays_d_cam'].contiguous(),
                    particle_density,
                    features,
                    self.model.n_active_features,
                    self.conf.render.min_transmittance,
                )
                
                # 应用背景
                pred_rgb, pred_opacity = self.model.background(
                    batch['poses'].contiguous(),
                    batch['rays_d_cam'].contiguous(),
                    pred_rgb,
                    pred_opacity,
                    False  # train mode
                )
            
        # 重塑为图像尺寸
        rgb_image = pred_rgb.reshape(width, height, 3).permute(1, 0, 2)  # [height, width, 3]
        
        # 转换为numpy并应用色调映射
        rgb_np = rgb_image.clamp(0, 1).cpu().numpy()
        
        return rgb_np
        
    def save_image(self, image: np.ndarray, output_path: str):
        """
        保存图像
        
        Args:
            image: RGB图像 [height, width, 3]
            output_path: 输出路径
        """
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        # 转换为BGR（OpenCV格式）并保存
        bgr_image = (image[:, :, [2, 1, 0]] * 255).astype(np.uint8)
        cv2.imwrite(output_path, bgr_image)
        print(f"图像已保存至: {output_path}")


# 定义一个空的上下文管理器，用于替代torch.cuda.nvtx.range当它不可用时
class nullcontext:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return None
    def __exit__(self, *exc):
        return False


def main():
    parser = argparse.ArgumentParser(description='无显示3D高斯渲染')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--height', type=int, default=800, help='输出图像高度')
    parser.add_argument('--width', type=int, default=800, help='输出图像宽度')
    parser.add_argument('--fov', type=float, default=45.0, help='相机视场角（度）')
    parser.add_argument('--output', type=str, default='output.png', help='输出图像路径')
    args = parser.parse_args()
    
    # 创建渲染器
    renderer = HeadlessRenderer(args.config, args.checkpoint)
    
    # 定义相机姿态（示例姿态）
    # 格式为 [3, 4] 矩阵，包含旋转和平移
    # 这里只是一个示例，实际应用中应根据需要调整
    camera_pose = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # 右向量 + x偏移
        [0.0, 1.0, 0.0, 0.0],  # 上向量 + y偏移
        [0.0, 0.0, 1.0, -3.0]  # 前向量 + z偏移（相机后移3个单位）
    ], device=renderer.device)
    
    # 渲染图像
    image = renderer.render_image(args.height, args.width, args.fov, camera_pose)
    
    # 保存图像
    renderer.save_image(image, args.output)


if __name__ == "__main__":
    main() 