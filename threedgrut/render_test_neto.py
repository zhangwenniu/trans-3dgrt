# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

import numpy as np
import torch
import torchvision
from torchmetrics import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from threedgrut.datasets import NeRFDataset, ColmapDataset, ScannetppDataset
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.logger import logger
from threedgrut.utils.misc import create_summary_writer

import sys
sys.path.append(os.path.abspath("./sdf/NeTO/Use3DGRUT"))
import torch.nn.functional as F
from models_silhouette.fields import SDFNetwork, SingleVarianceNetwork
from models_silhouette.renderer import NeuSRenderer
from pyhocon import ConfigFactory

# 使用新的推荐API设置默认张量类型和设备
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')



class Renderer:
    def __init__(
        self, model, conf, global_step, out_dir, path="", save_gt=True, writer=None, compute_extra_metrics=True
    ) -> None:

        if path:  # Replace the path to the test data
            conf.path = path

        self.model = model
        self.out_dir = out_dir
        self.save_gt = save_gt
        self.path = path
        self.conf = conf
        self.global_step = global_step
        self.dataset, self.dataloader = self.create_test_dataloader(conf)
        self.writer = writer
        self.compute_extra_metrics = compute_extra_metrics

        if conf.model.background.color == "black":
            self.bg_color = torch.zeros((3,), dtype=torch.float32, device="cuda")
        elif conf.model.background.color == "white":
            self.bg_color = torch.ones((3,), dtype=torch.float32, device="cuda")
        else:
            assert False, f"{conf.model.background.color} is not a supported background color."

    def create_test_dataloader(self, conf):
        """Create the test dataloader for the given configuration."""

        match conf.dataset.type:
            case "nerf":
                dataset = NeRFDataset(
                    conf.path, split="test", return_alphas=False, bg_color=conf.model.background.color
                )
            case "colmap":
                dataset = ColmapDataset(conf.path, split="val", downsample_factor=conf.dataset.downsample_factor)
            case "scannetpp":
                dataset = ScannetppDataset(conf.path, split="val")
            case _:
                raise ValueError(
                    f'Unsupported dataset type: {conf.dataset.type}. Choose between: ["colmap", "nerf", "scannetpp"].'
                )

        dataloader = torch.utils.data.DataLoader(
            dataset, 
            num_workers=4,  # 减少worker数量
            batch_size=1, 
            shuffle=False, 
            collate_fn=None
        )
        return dataset, dataloader

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path, out_dir, path="", save_gt=True, writer=None, model=None, computes_extra_metrics=True
    ):
        """Loads checkpoint for test path.
        If path is stated, it will override the test path in checkpoint.
        If model is None, it will be loaded base on the
        """

        checkpoint = torch.load(checkpoint_path)
        global_step = checkpoint["global_step"]

        conf = checkpoint["config"]
        # overrides
        if conf["render"]["method"] == "3dgrt":
            conf["render"]["particle_kernel_density_clamping"] = True
            conf["render"]["min_transmittance"] = 0.03
        conf["render"]["enable_kernel_timings"] = True

        object_name = Path(conf.path).stem
        experiment_name = conf["experiment_name"]
        writer, out_dir, run_name = create_summary_writer(conf, object_name, out_dir, experiment_name, use_wandb=False)

        if model is None:
            # Initialize the model and the optix context
            model = MixtureOfGaussians(conf)
            # Initialize the parameters from checkpoint
            model.init_from_checkpoint(checkpoint)
        model.build_acc()

        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=computes_extra_metrics,
        )

    @classmethod
    def from_preloaded_model(
        cls, model, out_dir, path="", save_gt=True, writer=None, global_step=None, compute_extra_metrics=False
    ):
        """Loads checkpoint for test path."""

        conf = model.conf
        if global_step is None:
            global_step = ""
        model.build_acc()
        return Renderer(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
            path=path,
            save_gt=save_gt,
            writer=writer,
            compute_extra_metrics=compute_extra_metrics,
        )

    def rays_to_world(self, gpu_batch):
        # 获取参数形状
        batch_size, H, W, _ = gpu_batch.rays_ori.shape
        
        # 提取T_to_world矩阵
        T = gpu_batch.T_to_world[:, :3, :]  # [batch_size, 3, 4]
        
        # 处理射线原点(rays_ori)转换
        # 首先重塑rays_ori以便于矩阵乘法
        rays_ori_flat = gpu_batch.rays_ori.reshape(batch_size, H*W, 3)  # [batch_size, H*W, 3]
        
        # 应用旋转(前3x3部分)和平移(最后一列)
        # 对于位置向量，我们需要完整的变换(旋转+平移)
        rays_ori_rotated = torch.bmm(rays_ori_flat, T[:, :, :3].transpose(1, 2))  # [batch_size, H*W, 3]
        
        # 添加平移部分
        translation = T[:, :, 3].unsqueeze(1).expand(-1, H*W, -1)  # [batch_size, H*W, 3]
        rays_ori_world_flat = rays_ori_rotated + translation  # [batch_size, H*W, 3]
        
        # 处理射线方向(rays_dir)转换
        # 首先重塑rays_dir以便于矩阵乘法
        rays_dir_flat = gpu_batch.rays_dir.reshape(batch_size, H*W, 3)  # [batch_size, H*W, 3]
        
        # 对于方向向量，我们只需要应用旋转部分(前3x3部分)，不需要平移
        rays_dir_world_flat = torch.bmm(rays_dir_flat, T[:, :, :3].transpose(1, 2))  # [batch_size, H*W, 3]
        
        # 确保射线方向归一化
        rays_dir_world_flat = torch.nn.functional.normalize(rays_dir_world_flat, dim=2)

        # 重塑回原始形状
        rays_ori_world = rays_ori_world_flat.reshape(batch_size, H, W, 3)  # [batch_size, H, W, 3]
        rays_dir_world = rays_dir_world_flat.reshape(batch_size, H, W, 3)  # [batch_size, H, W, 3]
        
        # 将转换后的rays_dir和rays_ori赋值给gpu_batch
        gpu_batch.rays_dir = rays_dir_world
        gpu_batch.rays_ori = rays_ori_world
        
        # 将T_to_world设置为单位矩阵，因为射线已经在世界坐标系中
        identity = torch.eye(4, dtype=T.dtype, device=T.device)
        identity = identity[:3, :].unsqueeze(0).expand(batch_size, -1, -1)
        gpu_batch.T_to_world = identity
        
        return gpu_batch
    
    def test_render_reflective_ball_param(self, gpu_batch):
        # 获取参数形状
        batch_size, H, W, _ = gpu_batch.rays_ori.shape
        
        rays_ori_world_flat = gpu_batch.rays_ori.reshape(batch_size, H*W, 3)
        rays_dir_world_flat = gpu_batch.rays_dir.reshape(batch_size, H*W, 3)
        
        # 计算射线与球体的交点
        # 射线方程: p(t) = o + t*d，其中o是原点，d是方向
        # 球体方程: ||p - c||^2 = r^2，其中c是球心，r是半径
        # 代入得: ||o + t*d - c||^2 = r^2
        # 展开: (o - c + t*d)^2 = r^2
        # 展开: (o-c)^2 + 2t*(o-c)·d + t^2*d^2 = r^2
        # 令a=d^2, b=2*(o-c)·d, c=(o-c)^2-r^2
        # 求解一元二次方程: at^2 + bt + c = 0
        
        # 计算系数
        oc = rays_ori_world_flat - sphere_center  # [batch_size, H*W, 3]
        a = torch.ones_like(rays_ori_world_flat[..., 0])  # 因为d是单位向量，所以d^2=1
        b = 2.0 * torch.sum(rays_dir_world_flat * oc, dim=2)  # 2*(o-c)·d
        c = torch.sum(oc * oc, dim=2) - sphere_radius**2  # (o-c)^2 - r^2
        
        # 计算判别式
        discriminant = b*b - 4*a*c  # [batch_size, H*W]
        
        # 创建掩码，表示射线与球体相交
        intersect_mask = discriminant >= 0  # [batch_size, H*W]
        
        # 初始化新的rays_ori和rays_dir，默认与原始值相同
        new_rays_ori_flat = rays_ori_world_flat.clone()
        new_rays_dir_flat = rays_dir_world_flat.clone()
        
        # 只处理相交的射线
        if intersect_mask.any():
            # 计算交点参数t (选择较小的t值，即第一个交点)
            # t = (-b - sqrt(discriminant)) / (2*a)
            sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0.0))
            t = (-b - sqrt_discriminant) / (2.0 * a)  # [batch_size, H*W]
            
            # 只考虑正t值（在射线前方的交点）
            valid_intersect = (t > 0) & intersect_mask
            
            if valid_intersect.any():
                # 扩展t的维度以便于计算
                t_expanded = t.unsqueeze(-1)  # [batch_size, H*W, 1]
                
                # 对于有效的交点，计算交点位置
                # 创建掩码用于索引
                valid_mask_expanded = valid_intersect.unsqueeze(-1).expand(-1, -1, 3)
                
                # 计算所有射线的交点（无效的会被后面的掩码过滤）
                intersect_points = rays_ori_world_flat + t_expanded * rays_dir_world_flat
                
                # 计算球面法线（从球心指向交点的单位向量）
                normals = torch.nn.functional.normalize(intersect_points - sphere_center, dim=2)
                
                # 计算反射方向: r = d - 2(d·n)n，其中d是入射方向，n是法线
                dot_product = torch.sum(rays_dir_world_flat * normals, dim=2, keepdim=True)
                reflected_dirs = rays_dir_world_flat - 2.0 * dot_product * normals
                
                # 更新有效交点的射线原点和方向
                new_rays_ori_flat = torch.where(valid_mask_expanded, intersect_points, new_rays_ori_flat)
                new_rays_dir_flat = torch.where(valid_mask_expanded, reflected_dirs, new_rays_dir_flat)
        
        # 重塑回原始形状
        rays_ori_world = new_rays_ori_flat.reshape(batch_size, H, W, 3)  # [batch_size, H, W, 3]
        rays_dir_world = new_rays_dir_flat.reshape(batch_size, H, W, 3)  # [batch_size, H, W, 3]
        
        # 将转换后的rays_dir和rays_ori赋值给gpu_batch
        gpu_batch.rays_dir = rays_dir_world
        gpu_batch.rays_ori = rays_ori_world
        
        # 将T_to_world设置为单位矩阵，因为射线已经在世界坐标系中
        identity = torch.eye(4, dtype=T.dtype, device=T.device)
        identity = identity[:3, :].unsqueeze(0).expand(batch_size, -1, -1)
        gpu_batch.T_to_world = identity
        
        return gpu_batch

    def test_render_from_neto(self, gpu_batch, renderer):

        # 获取射线原点和方向
        rays_o = gpu_batch.rays_ori
        rays_d = gpu_batch.rays_dir
        rgb_gt = gpu_batch.rgb_gt
        mask = gpu_batch.mask
        
        downsample_factor = 1
        rays_o = rays_o[:, ::downsample_factor, ::downsample_factor, :]
        rays_d = rays_d[:, ::downsample_factor, ::downsample_factor, :]
        rgb_gt = rgb_gt[:, ::downsample_factor, ::downsample_factor, :]
        mask = mask[:, ::downsample_factor, ::downsample_factor, :]
        gpu_batch.rgb_gt = rgb_gt
        gpu_batch.mask = mask

        # 获取批次大小和图像尺寸
        batch_size, H, W, _ = rays_o.shape
        
        # 扁平化射线以便处理
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        
        # 确保射线方向是单位向量
        rays_d_flat = F.normalize(rays_d_flat, p=2, dim=-1)
        
        # 引入坐标系变换，缩放到单位球空间里
        scale_mat = self.dataset.scale_mat
        
        rays_o_flat = (rays_o_flat - scale_mat[:3, 3]) / scale_mat[0, 0]
        
        # 每次处理的批次大小
        iter_size = 512
        
        # 创建存储新的射线原点和方向的张量
        new_rays_o_flat = rays_o_flat.clone()
        new_rays_d_flat = rays_d_flat.clone()
        
        # 创建有效交点掩码（默认全False）
        valid_intersection_mask = torch.zeros(rays_o_flat.shape[0], 1, dtype=torch.bool, device=rays_o_flat.device)
        
        # 按批次处理射线
        from tqdm import tqdm
        for i in tqdm(range(0, rays_o_flat.shape[0], iter_size), desc="Ray Reflecting", unit="batch"):
            end_idx = min(i + iter_size, rays_o_flat.shape[0])
            rays_o_batch = rays_o_flat[i:end_idx]
            rays_d_batch = rays_d_flat[i:end_idx]
            
            # 获取近远平面
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            
            # 在每条射线上均匀采样128个点
            n_samples = 128
            z_vals = torch.linspace(0.0, 1.0, n_samples, device=rays_o_batch.device)
            z_vals = near + (far - near) * z_vals[None, :]  # [batch_size, n_samples]
            
            # 计算采样点的3D坐标
            pts = rays_o_batch.unsqueeze(1) + z_vals.unsqueeze(-1) * rays_d_batch.unsqueeze(1)
            pts = pts.reshape(-1, 3)  # [batch_size * n_samples, 3]
            
            # 评估SDF值
            with torch.no_grad():
                sdf_values = renderer.sdf_network.sdf(pts)
            sdf_values = sdf_values.reshape(-1, n_samples)  # [batch_size, n_samples]
            
            # 计算相邻采样点之间的SDF差值
            sdf_diff = sdf_values[:, 1:] - sdf_values[:, :-1]  # [batch_size, n_samples-1]
            sdf_sign_change = (sdf_values[:, 1:] * sdf_values[:, :-1]) < 0  # [batch_size, n_samples-1]
            
            # 找到第一个符号变化的点
            first_sign_change = torch.argmax(sdf_sign_change.float(), dim=1)  # [batch_size]
            
            # 使用线性插值计算精确的交点位置
            batch_indices = torch.arange(sdf_values.shape[0], device=sdf_values.device)
            sdf_before = sdf_values[batch_indices, first_sign_change]
            sdf_after = sdf_values[batch_indices, first_sign_change + 1]
            z_before = z_vals[batch_indices, first_sign_change]
            z_after = z_vals[batch_indices, first_sign_change + 1]
            
            
            # 线性插值计算交点z值
            intersection_z = z_before - sdf_before * (z_after - z_before) / (sdf_after - sdf_before)
            
            # 计算交点位置
            intersection_points = rays_o_batch + intersection_z.unsqueeze(-1) * rays_d_batch
                        
            # 判断哪些射线有有效交点
            intersection_mask = sdf_sign_change.any(dim=1)
            
            # 对于所有有效交点的射线
            if intersection_mask.any():
                # 获取交点位置作为新的射线原点
                valid_points = intersection_points[intersection_mask]
                
                # 计算交点处的梯度作为法向量
                valid_pts = intersection_points[intersection_mask].clone()
                valid_pts.requires_grad_(True)
                
                # 使用enable_grad上下文进行前向传播和反向传播
                with torch.enable_grad():
                    sdf_at_intersection = renderer.sdf_network.sdf(valid_pts)
                    gradients = torch.autograd.grad(
                        sdf_at_intersection,
                        valid_pts,
                        grad_outputs=torch.ones_like(sdf_at_intersection),
                        create_graph=False,
                        retain_graph=False,
                    )[0]
                
                # 归一化法向量
                normals = F.normalize(gradients, p=2, dim=-1)
                    
                # 获取入射方向
                incident_dirs = rays_d_batch[intersection_mask]
                
                # 计算反射方向: r = d - 2(d·n)n
                dot_product = torch.sum(incident_dirs * normals, dim=-1, keepdim=True)
                reflected_dirs = incident_dirs - 2.0 * dot_product * normals
                
                # 确保反射方向是单位向量
                reflected_dirs = F.normalize(reflected_dirs, p=2, dim=-1)
                
                # 更新射线原点和方向
                new_rays_o_flat[i:end_idx][intersection_mask] = valid_points
                new_rays_d_flat[i:end_idx][intersection_mask] = reflected_dirs
                
                # 更新有效交点掩码
                valid_intersection_mask[i:end_idx][intersection_mask] = True
                
            # 清空CUDA缓存
            torch.cuda.empty_cache()
        
        # 将rays_o_flat放到没有缩放的世界坐标系中
        new_rays_o_flat = new_rays_o_flat * scale_mat[0, 0] + scale_mat[:3, 3]
        
        # 重塑回原始形状
        new_rays_o = new_rays_o_flat.reshape(batch_size, H, W, 3)
        new_rays_d = new_rays_d_flat.reshape(batch_size, H, W, 3)
        
        # 输出找到的反射点数量
        num_reflected = valid_intersection_mask.sum().item()
        total_rays = valid_intersection_mask.numel()
        logger.info(f"找到 {num_reflected}/{total_rays} ({100.0*num_reflected/total_rays:.2f}%) 个SDF表面反射点")
        
        # 关键修改：确保返回的射线是不带梯度的新副本
        with torch.no_grad():
            # 创建完全分离的拷贝，不带任何梯度历史
            detached_rays_o = new_rays_o.detach().clone()
            detached_rays_d = new_rays_d.detach().clone()
            
            # 更新gpu_batch中的射线为无梯度版本
            gpu_batch.rays_ori = detached_rays_o
            gpu_batch.rays_dir = detached_rays_d
        
        # 清理不再需要的变量
        del rays_o_flat, rays_d_flat, new_rays_o_flat, new_rays_d_flat, valid_intersection_mask, new_rays_o, new_rays_d
        
        # 强制进行垃圾回收和清理CUDA缓存
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        return gpu_batch
    
    def test_render_refractive_from_neto(self, gpu_batch, renderer):

        # 获取射线原点和方向
        rays_o = gpu_batch.rays_ori
        rays_d = gpu_batch.rays_dir
        rgb_gt = gpu_batch.rgb_gt
        mask = gpu_batch.mask
        
        downsample_factor = 1
        rays_o = rays_o[:, ::downsample_factor, ::downsample_factor, :]
        rays_d = rays_d[:, ::downsample_factor, ::downsample_factor, :]
        rgb_gt = rgb_gt[:, ::downsample_factor, ::downsample_factor, :]
        mask = mask[:, ::downsample_factor, ::downsample_factor, :]
        gpu_batch.rgb_gt = rgb_gt
        gpu_batch.mask = mask

        # 获取批次大小和图像尺寸
        batch_size, H, W, _ = rays_o.shape
        
        # 设置折射率
        n1 = 1.0003  # 空气的折射率
        n2 = 1.51  # 玻璃的折射率
        
        # 扁平化射线以便处理
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        
        # 确保射线方向是单位向量
        rays_d_flat = F.normalize(rays_d_flat, p=2, dim=-1)
        
        # 引入坐标系变换，缩放到单位球空间里
        scale_mat = self.dataset.scale_mat
        
        rays_o_flat = (rays_o_flat - scale_mat[:3, 3]) / scale_mat[0, 0]
        
        # 每次处理的批次大小
        iter_size = 512
        
        # 创建存储新的射线原点和方向的张量
        new_rays_o_flat = rays_o_flat.clone()
        new_rays_d_flat = rays_d_flat.clone()
        
        # 创建有效交点掩码（默认全False）
        valid_intersection_mask = torch.zeros(rays_o_flat.shape[0], 1, dtype=torch.bool, device=rays_o_flat.device)
        
        # 按批次处理射线
        from tqdm import tqdm
        for i in tqdm(range(0, rays_o_flat.shape[0], iter_size), desc="Ray Refracting (First)", unit="batch"):
            end_idx = min(i + iter_size, rays_o_flat.shape[0])
            rays_o_batch = rays_o_flat[i:end_idx]
            rays_d_batch = rays_d_flat[i:end_idx]
            
            # 获取近远平面
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            
            # 在每条射线上均匀采样128个点
            n_samples = 128
            z_vals = torch.linspace(0.0, 1.0, n_samples, device=rays_o_batch.device)
            z_vals = near + (far - near) * z_vals[None, :]  # [batch_size, n_samples]
            
            # 计算采样点的3D坐标
            pts = rays_o_batch.unsqueeze(1) + z_vals.unsqueeze(-1) * rays_d_batch.unsqueeze(1)
            pts = pts.reshape(-1, 3)  # [batch_size * n_samples, 3]
            
            # 评估SDF值
            with torch.no_grad():
                sdf_values = renderer.sdf_network.sdf(pts)
            sdf_values = sdf_values.reshape(-1, n_samples)  # [batch_size, n_samples]
            
            # 计算相邻采样点之间的SDF差值
            sdf_sign_change = (sdf_values[:, 1:] * sdf_values[:, :-1]) < 0  # [batch_size, n_samples-1]
            
            # 找到第一个符号变化的点
            first_sign_change = torch.argmax(sdf_sign_change.float(), dim=1)  # [batch_size]
            
            # 使用线性插值计算精确的交点位置
            batch_indices = torch.arange(sdf_values.shape[0], device=sdf_values.device)
            sdf_before = sdf_values[batch_indices, first_sign_change]
            sdf_after = sdf_values[batch_indices, first_sign_change + 1]
            z_before = z_vals[batch_indices, first_sign_change]
            z_after = z_vals[batch_indices, first_sign_change + 1]
            
            # 线性插值计算交点z值
            intersection_z = z_before - sdf_before * (z_after - z_before) / (sdf_after - sdf_before)
            
            # 计算交点位置
            intersection_points = rays_o_batch + intersection_z.unsqueeze(-1) * rays_d_batch
                        
            # 判断哪些射线有有效交点
            intersection_mask = sdf_sign_change.any(dim=1)
            
            # 对于所有有效交点的射线
            if intersection_mask.any():
                # 获取交点位置作为新的射线原点
                valid_points = intersection_points[intersection_mask]
                
                # 计算交点处的梯度作为法向量
                valid_pts = intersection_points[intersection_mask].clone()
                valid_pts.requires_grad_(True)
                
                # 使用enable_grad上下文进行前向传播和反向传播
                with torch.enable_grad():
                    sdf_at_intersection = renderer.sdf_network.sdf(valid_pts)
                    gradients = torch.autograd.grad(
                        sdf_at_intersection,
                        valid_pts,
                        grad_outputs=torch.ones_like(sdf_at_intersection),
                        create_graph=False,
                        retain_graph=False,
                    )[0]
                
                # 归一化法向量（指向物体外部）
                normals = F.normalize(gradients, p=2, dim=-1)
                    
                # 获取入射方向
                incident_dirs = rays_d_batch[intersection_mask]
                
                # 计算入射角的余弦值（入射方向与法线的点积的负值）
                # 注意：入射方向与法线的夹角通常用入射方向与表面法线的负值来计算
                cos_i = -torch.sum(incident_dirs * normals, dim=-1, keepdim=True)
                
                # 处理射线从外部进入物体的情况（cos_i > 0）
                # 如果cos_i < 0，说明射线从内部射出，这不是我们期望的第一次折射情形
                valid_entry = cos_i > 0
                
                if valid_entry.any():
                    # 应用斯涅尔定律计算折射方向
                    # n1 * sin_i = n2 * sin_t
                    # 计算sin^2(t) = (n1/n2)^2 * (1 - cos^2(i))
                    ratio = n1 / n2
                    k = 1.0 - ratio * ratio * (1.0 - cos_i * cos_i)
                    
                    # 处理全反射情况（当k < 0时）
                    k = torch.clamp(k, min=0.0)
                    
                    # 计算折射方向: ratio * incident + (ratio * cos_i - sqrt(k)) * normal
                    refracted_dirs = ratio * incident_dirs + (ratio * cos_i - torch.sqrt(k)) * normals
                    
                    # 确保折射方向是单位向量
                    refracted_dirs = F.normalize(refracted_dirs, p=2, dim=-1)
                    
                    # 添加一个小的偏移量，避免数值精度问题导致折射光线起点正好位于表面
                    # 这里的偏移方向是向物体内部（沿着-normal方向）
                    epsilon = 1e-4  # 小偏移量
                    refracted_origins = valid_points - epsilon * normals
                    
                    # 更新有效的射线原点和方向
                    new_rays_o_flat[i:end_idx][intersection_mask] = refracted_origins
                    new_rays_d_flat[i:end_idx][intersection_mask] = refracted_dirs
                    
                    # 更新有效交点掩码
                    valid_intersection_mask[i:end_idx][intersection_mask] = True
            
            # 清空CUDA缓存
            torch.cuda.empty_cache()
        
        # 第一阶段处理结束
        logger.info(f"第一次折射处理完成，找到 {valid_intersection_mask.sum().item()}/{valid_intersection_mask.numel()} 个有效折射点")
        
        # 如果没有有效的折射点，就不执行第二次折射计算
        if valid_intersection_mask.sum().item() == 0:
            logger.warning("没有找到有效的第一次折射点，跳过第二次折射计算")
            # 使用原始射线
            new_rays_o = rays_o_flat.reshape(batch_size, H, W, 3)
            new_rays_d = rays_d_flat.reshape(batch_size, H, W, 3)
        else:
            # 为第二次折射准备数据
            # 创建存储第二次折射后射线的数组
            final_rays_o_flat = new_rays_o_flat.clone()
            final_rays_d_flat = new_rays_d_flat.clone()
            
            # 记录第二次折射的有效点
            second_valid_mask = torch.zeros_like(valid_intersection_mask)
            
            # 只对第一次折射有效的点执行第二次折射查找
            valid_indices = torch.where(valid_intersection_mask.squeeze(-1))[0]
            
            # 按批次处理有效的折射射线
            for i in tqdm(range(0, len(valid_indices), iter_size), desc="Ray Refracting (Second)", unit="batch"):
                end_idx = min(i + iter_size, len(valid_indices))
                batch_indices = valid_indices[i:end_idx]
                
                # 获取第一次折射后的射线
                refracted_o_batch = new_rays_o_flat[batch_indices]
                refracted_d_batch = new_rays_d_flat[batch_indices]
                
                # 获取近远平面
                near, far = self.dataset.near_far_from_sphere(refracted_o_batch, refracted_d_batch)
                
                # 在射线上均匀采样256个点（比第一次更多，以确保捕获到出射点）
                n_samples = 256
                z_vals = torch.linspace(0.0, 1.0, n_samples, device=refracted_o_batch.device)
                z_vals = near + (far - near) * z_vals[None, :]  # [batch_size, n_samples]
                
                # 计算采样点的3D坐标
                pts = refracted_o_batch.unsqueeze(1) + z_vals.unsqueeze(-1) * refracted_d_batch.unsqueeze(1)
                pts = pts.reshape(-1, 3)  # [batch_size * n_samples, 3]
                
                # 评估SDF值
                with torch.no_grad():
                    sdf_values = renderer.sdf_network.sdf(pts)
                sdf_values = sdf_values.reshape(-1, n_samples)  # [batch_size, n_samples]
                
                # 计算相邻采样点之间的SDF符号变化
                sdf_sign_change = (sdf_values[:, 1:] * sdf_values[:, :-1]) < 0  # [batch_size, n_samples-1]
                
                # 找到SDF值从负到正变化的点（从物体内部射出）
                # 首先我们知道第一个点在物体内部（SDF应为负）
                # 寻找第一个SDF值变为正的位置
                exit_mask = sdf_sign_change & (sdf_values[:, :-1] < 0) & (sdf_values[:, 1:] > 0)
                
                # 对于每条有效射线，查找第一个出射点
                first_exit_indices = torch.zeros(exit_mask.shape[0], dtype=torch.long, device=exit_mask.device)
                for ray_idx in range(exit_mask.shape[0]):
                    exit_points = torch.where(exit_mask[ray_idx])[0]
                    if len(exit_points) > 0:
                        first_exit_indices[ray_idx] = exit_points[0]
                    else:
                        # 如果没有找到出射点，设置为最后一个有效索引
                        first_exit_indices[ray_idx] = n_samples - 2
                
                # 找到哪些射线实际上有出射点
                has_exit = torch.any(exit_mask, dim=1)
                
                # 如果有任何射线找到了出射点
                if has_exit.any():
                    # 对有出射点的射线，计算精确的出射交点
                    ray_indices = torch.where(has_exit)[0]
                    
                    # 获取出射点前后的SDF值
                    sdf_before = sdf_values[ray_indices, first_exit_indices[ray_indices]]
                    sdf_after = sdf_values[ray_indices, first_exit_indices[ray_indices] + 1]
                    
                    # 获取出射点前后的z值
                    z_before = z_vals[ray_indices, first_exit_indices[ray_indices]]
                    z_after = z_vals[ray_indices, first_exit_indices[ray_indices] + 1]
                    
                    # 线性插值计算精确的出射点z值
                    exit_z = z_before - sdf_before * (z_after - z_before) / (sdf_after - sdf_before)
                    
                    # 计算出射点位置
                    exit_points = refracted_o_batch[ray_indices] + exit_z.unsqueeze(-1) * refracted_d_batch[ray_indices]
                    
                    # 计算出射点处的法向量
                    exit_pts = exit_points.clone()
                    exit_pts.requires_grad_(True)
                    
                    # 计算出射点处的梯度
                    with torch.enable_grad():
                        sdf_at_exit = renderer.sdf_network.sdf(exit_pts)
                        exit_gradients = torch.autograd.grad(
                            sdf_at_exit,
                            exit_pts,
                            grad_outputs=torch.ones_like(sdf_at_exit),
                            create_graph=False,
                            retain_graph=False
                        )[0]
                    
                    # 归一化法向量（注意：SDF梯度指向物体外部）
                    exit_normals = F.normalize(exit_gradients, p=2, dim=-1)
                    
                    # 获取入射方向（第一次折射后的方向）
                    incident_dirs = refracted_d_batch[ray_indices]
                    
                    # 计算入射角的余弦值（因为现在是从内部射向外部，所以法线与入射方向夹角应为锐角）
                    cos_i = torch.sum(incident_dirs * exit_normals, dim=-1, keepdim=True)
                    
                    # 处理从内部射向外部的情况（cos_i > 0）
                    valid_exit = cos_i > 0
                    
                    if valid_exit.any():
                        # 应用斯涅尔定律计算第二次折射方向（从玻璃到空气）
                        # 注意折射率比值反转：n2/n1
                        ratio = n2 / n1
                        k = 1.0 - ratio * ratio * (1.0 - cos_i * cos_i)
                        
                        # 处理全反射情况
                        total_internal_reflection = k < 0
                        k = torch.clamp(k, min=0.0)
                        
                        # 计算折射方向
                        exit_refracted_dirs = ratio * incident_dirs - (ratio * cos_i - torch.sqrt(k)) * exit_normals
                        
                        # 对于全反射的情况，计算反射方向
                        exit_reflected_dirs = incident_dirs - 2.0 * cos_i * exit_normals
                        
                        # 合并折射和反射方向
                        if total_internal_reflection.any():
                            exit_dirs = torch.where(
                                total_internal_reflection,
                                exit_reflected_dirs,
                                exit_refracted_dirs
                            )
                        else:
                            exit_dirs = exit_refracted_dirs
                        
                        # 确保方向是单位向量
                        exit_dirs = F.normalize(exit_dirs, p=2, dim=-1)
                        
                        # 添加一个小的偏移量，避免数值精度问题
                        epsilon = 1e-4
                        exit_origins = exit_points + epsilon * exit_normals
                        
                        # 将出射点和方向存储到原始数组中对应位置
                        valid_exit_indices = batch_indices[ray_indices[valid_exit.squeeze(-1)]]
                        final_rays_o_flat[valid_exit_indices] = exit_origins[valid_exit.squeeze(-1)]
                        final_rays_d_flat[valid_exit_indices] = exit_dirs[valid_exit.squeeze(-1)]
                        
                        # 更新第二次折射的有效掩码
                        second_valid_mask[valid_exit_indices] = True
                
                # 清空CUDA缓存
                torch.cuda.empty_cache()
            
            # 将rays_o_flat放到没有缩放的世界坐标系中
            final_rays_o_flat = final_rays_o_flat * scale_mat[0, 0] + scale_mat[:3, 3]
            
            # 重塑回原始形状
            new_rays_o = final_rays_o_flat.reshape(batch_size, H, W, 3)
            new_rays_d = final_rays_d_flat.reshape(batch_size, H, W, 3)
            
            # 输出找到的折射点数量
            num_first_refracted = valid_intersection_mask.sum().item()
            num_second_refracted = second_valid_mask.sum().item()
            total_rays = valid_intersection_mask.numel()
            
            logger.info(f"第一次折射: {num_first_refracted}/{total_rays} ({100.0*num_first_refracted/total_rays:.2f}%) 个有效点")
            logger.info(f"第二次折射: {num_second_refracted}/{total_rays} ({100.0*num_second_refracted/total_rays:.2f}%) 个有效点")
        
        # 关键修改：确保返回的射线是不带梯度的新副本
        with torch.no_grad():
            # 创建完全分离的拷贝，不带任何梯度历史
            detached_rays_o = new_rays_o.detach().clone()
            detached_rays_d = new_rays_d.detach().clone()
            
            # 更新gpu_batch中的射线为无梯度版本
            gpu_batch.rays_ori = detached_rays_o
            gpu_batch.rays_dir = detached_rays_d
        
        # 清理不再需要的变量
        if 'rays_o_flat' in locals(): del rays_o_flat
        if 'rays_d_flat' in locals(): del rays_d_flat
        if 'new_rays_o_flat' in locals(): del new_rays_o_flat
        if 'new_rays_d_flat' in locals(): del new_rays_d_flat
        if 'valid_intersection_mask' in locals(): del valid_intersection_mask
        if 'new_rays_o' in locals(): del new_rays_o
        if 'new_rays_d' in locals(): del new_rays_d
        if 'final_rays_o_flat' in locals(): del final_rays_o_flat
        if 'final_rays_d_flat' in locals(): del final_rays_d_flat
        if 'second_valid_mask' in locals(): del second_valid_mask
        
        # 强制进行垃圾回收和清理CUDA缓存
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        return gpu_batch

    @torch.no_grad()  # 使用装饰器确保整个render_all都在no_grad下执行，除了显式调用test_render_from_neto的部分
    def render_all(self):
        """Render all the images in the test dataset and log the metrics."""
        
        # 判断模型使用的是哪种渲染方法
        render_method = self.model.conf.render.method
        is_3dgrt = render_method == "3dgrt"
        is_3dgut = render_method == "3dgut"

        # Criterions that we log during training
        criterions = {"psnr": PeakSignalNoiseRatio(data_range=1).to("cuda")}

        if self.compute_extra_metrics:
            criterions |= {
                "ssim": StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda"),
                "lpips": LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda"),
            }

        output_path_renders = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "renders")
        os.makedirs(output_path_renders, exist_ok=True)

        if self.save_gt:
            output_path_gt = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "gt")
            os.makedirs(output_path_gt, exist_ok=True)

        psnr = []
        ssim = []
        lpips = []
        inference_time = []
        test_images = []

        best_psnr = -1.0
        worst_psnr = 2**16 * 1.0

        best_psnr_img = None
        best_psnr_img_gt = None

        worst_psnr_img = None
        worst_psnr_img_gt = None

        logger.start_progress(task_name="Rendering", total_steps=len(self.dataloader), color="orange1")

        for iteration, batch in enumerate(self.dataloader):

            # Get the GPU-cached batch
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)
            
            # 针对不同渲染器可能需要不同的预处理
            if is_3dgrt:
                gpu_batch = self.rays_to_world(gpu_batch)
                
                # 暂时退出no_grad上下文，仅用于处理test_render_from_neto
                # 这里不使用with torch.enable_grad()因为@torch.no_grad()装饰器的优先级更高
                # 而是重新获取一个临时的上下文管理器实例
                _no_grad_context = torch.no_grad()
                _no_grad_context.__exit__(None, None, None)
                try:
                    # 执行需要梯度的操作
                    gpu_batch = self.test_render_from_neto(gpu_batch)
                finally:
                    # 恢复no_grad上下文
                    _no_grad_context.__enter__()
            
            elif is_3dgut:
                # 3dgut需要确保intrinsics已正确设置
                if gpu_batch.intrinsics is None and gpu_batch.intrinsics_OpenCVPinholeCameraModelParameters is None and gpu_batch.intrinsics_OpenCVFisheyeCameraModelParameters is None:
                    logger.warning(f"Frame {iteration}: 3dgut需要相机内参，但未找到")
            
            
            
            
            
            # print("gpu_batch 内容:")
            # for attr_name in dir(gpu_batch):
            #     # 跳过私有属性和方法
            #     if attr_name.startswith('__') or callable(getattr(gpu_batch, attr_name)):
            #         continue
            #     attr_value = getattr(gpu_batch, attr_name)
            #     if isinstance(attr_value, torch.Tensor):
            #         print(f"  {attr_name} (Tensor): shape={attr_value.shape}, dtype={attr_value.dtype}")
            #     elif attr_value is not None:
            #         print(f"  {attr_name}: {attr_value}")

            # Compute the outputs of a single batch
            outputs = self.model(gpu_batch)
            
            # for key, value in outputs.items():
            #     if isinstance(value, torch.Tensor):
            #         print(f"outputs {key}.shape: {value.shape}")
            #     else:
            #         print(f"outputs {key}: {value}")
                    
            # gpu_batch ____:
            # T_to_world (Tensor): shape=torch.Size([1, 3, 4]), dtype=torch.float32
            # intrinsics: [1111.1110311937682, 1111.1110311937682, 400.0, 400.0]
            # rays_dir (Tensor): shape=torch.Size([1, 800, 800, 3]), dtype=torch.float32
            # rays_ori (Tensor): shape=torch.Size([1, 800, 800, 3]), dtype=torch.float32
            # rgb_gt (Tensor): shape=torch.Size([1, 800, 800, 3]), dtype=torch.float32
            # outputs pred_rgb.shape: torch.Size([1, 800, 800, 3])
            # outputs pred_opacity.shape: torch.Size([1, 800, 800, 1])
            # outputs pred_dist.shape: torch.Size([1, 800, 800, 1])
            # outputs pred_normals.shape: torch.Size([1, 800, 800, 3])
            # outputs hits_count.shape: torch.Size([1, 800, 800, 1])
            # outputs frame_time_ms: 8.245247840881348
            # [INFO] Frame 41, PSNR: 38.63913345336914

            pred_rgb_full = outputs["pred_rgb"]
            rgb_gt_full = gpu_batch.rgb_gt

            # The values are already alpha composited with the background
            torchvision.utils.save_image(
                pred_rgb_full.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_renders, "{0:05d}".format(iteration) + ".png"),
            )
            pred_img_to_write = pred_rgb_full[-1].clip(0, 1.0)
            gt_img_to_write = rgb_gt_full[-1].clip(0, 1.0)

            if self.writer is not None:
                test_images.append(pred_img_to_write)

            if self.save_gt:
                torchvision.utils.save_image(
                    rgb_gt_full.squeeze(0).permute(2, 0, 1),
                    os.path.join(output_path_gt, "{0:05d}".format(iteration) + ".png"),
                )

            # Compute the loss
            psnr_single_img = criterions["psnr"](outputs["pred_rgb"], gpu_batch.rgb_gt).item()
            psnr.append(psnr_single_img)  # evaluation on valid rays only
            logger.info(f"Frame {iteration}, PSNR: {psnr[-1]}")

            if psnr_single_img > best_psnr:
                best_psnr = psnr_single_img
                best_psnr_img = pred_img_to_write
                best_psnr_img_gt = gt_img_to_write

            if psnr_single_img < worst_psnr:
                worst_psnr = psnr_single_img
                worst_psnr_img = pred_img_to_write
                worst_psnr_img_gt = gt_img_to_write

            # evaluate on full image
            ssim.append(
                criterions["ssim"](
                    pred_rgb_full.permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )
            lpips.append(
                criterions["lpips"](
                    pred_rgb_full.clip(0, 1).permute(0, 3, 1, 2),
                    rgb_gt_full.permute(0, 3, 1, 2),
                ).item()
            )

            # Record the time
            inference_time.append(outputs["frame_time_ms"])

            logger.log_progress(task_name="Rendering", advance=1, iteration=f"{str(iteration)}", psnr=psnr[-1])

        logger.end_progress(task_name="Rendering")

        mean_psnr = np.mean(psnr)
        mean_ssim = np.mean(ssim)
        mean_lpips = np.mean(lpips)
        std_psnr = np.std(psnr)
        mean_inference_time = np.mean(inference_time)

        table = dict(
            mean_psnr=mean_psnr,
            mean_ssim=mean_ssim,
            mean_lpips=mean_lpips,
            std_psnr=std_psnr,
        )
        table["mean_inference_time"] = f"{'{:.2f}'.format(mean_inference_time)}" + " ms/frame"
        logger.log_table(f"⭐ Test Metrics - Step {self.global_step}", record=table)

        if self.writer is not None:
            self.writer.add_scalar("psnr/test", mean_psnr, self.global_step)
            self.writer.add_scalar("ssim/test", mean_ssim, self.global_step)
            self.writer.add_scalar("lpips/test", mean_lpips, self.global_step)
            self.writer.add_scalar("time/inference/test", mean_inference_time, self.global_step)

            if len(test_images) > 0:
                self.writer.add_images(
                    "image/pred/test",
                    torch.stack(test_images),
                    self.global_step,
                    dataformats="NHWC",
                )

            if best_psnr_img is not None:
                self.writer.add_images(
                    "image/best_psnr/test",
                    torch.stack([best_psnr_img, best_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

            if worst_psnr_img is not None:
                self.writer.add_images(
                    "image/worst_psnr/test",
                    torch.stack([worst_psnr_img, worst_psnr_img_gt]),
                    self.global_step,
                    dataformats="NHWC",
                )

        return mean_psnr, std_psnr, mean_inference_time


    def init_from_neus(self):
        device = torch.device('cuda')
        # Configuration
        conf_path = "/workspace/sdf/NeTO/Use3DGRUT/confs/silhouette.conf"
        case = "eiko_ball_masked"
        f = open(conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        conf = ConfigFactory.parse_string(conf_text)
        conf['dataset.data_dir'] = conf['dataset.data_dir'].replace('CASE_NAME', case)
        base_exp_dir = conf['general.base_exp_dir']
        
        sdf_network = SDFNetwork(**conf['model.sdf_network']).to(device)
        deviation_network = SingleVarianceNetwork(**conf['model.variance_network']).to(device)
        
        checkpoint_dir = os.path.join(base_exp_dir, 'checkpoints')
        model_list_raw = os.listdir(checkpoint_dir)
        model_list = []
        for model_name in model_list_raw:
            if model_name[-3:] == 'pth':
                model_list.append(model_name)
        model_list.sort()
        latest_model_name = model_list[-1]
        checkpoint = torch.load(os.path.join(checkpoint_dir, latest_model_name), map_location=device)
        sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        renderer = NeuSRenderer(sdf_network,deviation_network,**conf['model.neus_renderer'])
        
        return renderer

    def render_all_neto(self):        
        """Render all the images in the test dataset and log the metrics."""        
        # 判断模型使用的是哪种渲染方法
        render_method = self.model.conf.render.method
        is_3dgrt = render_method == "3dgrt"
        is_3dgut = render_method == "3dgut"
        output_path_renders = os.path.join(self.out_dir, f"ours_{int(self.global_step)}", "renders")
        os.makedirs(output_path_renders, exist_ok=True)
        logger.start_progress(task_name="Rendering", total_steps=len(self.dataloader), color="orange1")
        
        # 创建NeTO/NeuS Renderer实例
        renderer = self.init_from_neus()
        
        for iteration, batch in enumerate(self.dataloader):
            # Get the GPU-cached batch
            gpu_batch = self.dataset.get_gpu_batch_with_intrinsics(batch)            
            # 针对不同渲染器可能需要不同的预处理
            if is_3dgrt:
                gpu_batch = self.rays_to_world(gpu_batch)
                # gpu_batch = self.test_render_from_neto(gpu_batch, renderer)
                gpu_batch = self.test_render_refractive_from_neto(gpu_batch, renderer)
            elif is_3dgut:
                # 3dgut需要确保intrinsics已正确设置
                if gpu_batch.intrinsics is None and gpu_batch.intrinsics_OpenCVPinholeCameraModelParameters is None and gpu_batch.intrinsics_OpenCVFisheyeCameraModelParameters is None:
                    logger.warning(f"Frame {iteration}: 3dgut需要相机内参，但未找到")
            # print("gpu_batch 内容:")
            # for attr_name in dir(gpu_batch):
            #     # 跳过私有属性和方法
            #     if attr_name.startswith('__') or callable(getattr(gpu_batch, attr_name)):
            #         continue
            #     attr_value = getattr(gpu_batch, attr_name)
            #     if isinstance(attr_value, torch.Tensor):
            #         print(f"  {attr_name} (Tensor): shape={attr_value.shape}, dtype={attr_value.dtype}")
            #     elif attr_value is not None:
            #         print(f"  {attr_name}: {attr_value}")

            # Compute the outputs of a single batch
            outputs = self.model(gpu_batch)
            
            # for key, value in outputs.items():
            #     if isinstance(value, torch.Tensor):
            #         print(f"outputs {key}.shape: {value.shape}")
            #     else:
            #         print(f"outputs {key}: {value}")
                    
            # gpu_batch ____:
            # T_to_world (Tensor): shape=torch.Size([1, 3, 4]), dtype=torch.float32
            # intrinsics: [1111.1110311937682, 1111.1110311937682, 400.0, 400.0]
            # rays_dir (Tensor): shape=torch.Size([1, 800, 800, 3]), dtype=torch.float32
            # rays_ori (Tensor): shape=torch.Size([1, 800, 800, 3]), dtype=torch.float32
            # rgb_gt (Tensor): shape=torch.Size([1, 800, 800, 3]), dtype=torch.float32
            # outputs pred_rgb.shape: torch.Size([1, 800, 800, 3])
            # outputs pred_opacity.shape: torch.Size([1, 800, 800, 1])
            # outputs pred_dist.shape: torch.Size([1, 800, 800, 1])
            # outputs pred_normals.shape: torch.Size([1, 800, 800, 3])
            # outputs hits_count.shape: torch.Size([1, 800, 800, 1])
            # outputs frame_time_ms: 8.245247840881348
            # [INFO] Frame 41, PSNR: 38.63913345336914

            rgb_gt = gpu_batch.rgb_gt.squeeze(0)
            rgb_predict = outputs["pred_rgb"].squeeze(0)
            
            concated = torch.cat([rgb_gt, rgb_predict], dim=1)  # 水平拼接两张图片
            # 保存拼接后的图片
            torchvision.utils.save_image(
                concated.squeeze(0).permute(2, 0, 1),
                os.path.join(output_path_renders, "{0:05d}_concat".format(iteration) + ".png"),
            )