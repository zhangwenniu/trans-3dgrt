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
import cv2
import logging
import time
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 默认日志级别为INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler('training.log')  # 同时保存到文件
    ]
)
logger = logging.getLogger('threedgrut')

# 从threedgrut导入
import threedgrut.datasets as datasets
from threedgrut.datasets import NeRFDataset, ColmapDataset, ScannetppDataset
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.logger import logger as threedgrut_logger  # 保留原有logger以避免冲突
from threedgrut.utils.misc import create_summary_writer
from threedgrut.datasets.utils import MultiEpochsDataLoader
from threedgrut.trace_refract_as_tnsr import RefractTracer
from threedgrut.trace_refract_inplace import RefractTracerInplace

# NeUS相关导入
import sys
sys.path.append(os.path.abspath("./sdf/NeTO/Use3DGRUT"))
import torch.nn.functional as F
from models_silhouette.fields import SDFNetwork, SingleVarianceNetwork
from models_silhouette.renderer import NeuSRenderer
from pyhocon import ConfigFactory

# 使用新的推荐API设置默认张量类型和设备
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class Trainer:
    def __init__(
        self, model, conf, global_step, out_dir
    ) -> None:
        """
        训练器类的初始化方法，简化版本
        
        参数:
        - model: 要训练的模型实例
        - conf: 配置信息
        - global_step: 全局训练步数
        - out_dir: 输出目录，用于保存结果
        """
        # 保存基本参数
        self.model = model
        self.out_dir = out_dir 
        self.conf = conf
        self.global_step = global_step
        
        # 设置背景颜色
        if conf.model.background.color == "black":
            self.bg_color = torch.zeros((3,), dtype=torch.float32, device="cuda")
        elif conf.model.background.color == "white":
            self.bg_color = torch.ones((3,), dtype=torch.float32, device="cuda")
        else:
            assert False, f"{conf.model.background.color} is not a supported background color."
        
        # 初始化训练相关变量
        self.iter_step = 0
        self.n_epochs = 1000  # 默认训练轮数，可以从配置中读取
        
        self.init_dataloaders(self.conf)
        self.init_from_neus()

    def init_dataloaders(self, conf):
        train_dataset, val_dataset = datasets.make(name=conf.dataset.type, config=conf, ray_jitter=None)
        
        # 禁用pin_memory，因为数据集可能已经返回CUDA张量
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            num_workers=0,  # 设置为0以避免CUDA在多进程中初始化的问题
            batch_size=1,
            shuffle=True,
            pin_memory=False,  # 禁用pin_memory，因为数据可能已经在CUDA上
            persistent_workers=False,
            generator=torch.Generator(device='cuda')
        )
        
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            num_workers=0,  # 设置为0以避免CUDA在多进程中初始化的问题
            batch_size=1,
            shuffle=False,
            pin_memory=False,  # 禁用pin_memory，因为数据可能已经在CUDA上
            persistent_workers=False,
            generator=torch.Generator(device='cuda')
        )
        
        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataset = val_dataset
        self.val_dataloader = val_dataloader

    @classmethod
    def from_checkpoint(cls, checkpoint_path, out_dir):
        """
        从检查点加载模型和配置，简化版本
        
        参数:
        - checkpoint_path: 检查点文件的路径
        - out_dir: 输出目录
        
        返回:
        - 配置好的训练器实例
        """
        # 加载检查点文件
        logger.info(f"从检查点加载Gaussian Splatting模型: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        global_step = checkpoint["global_step"]

        # 从检查点获取配置信息
        conf = checkpoint["config"]
        
        # 初始化模型
        model = MixtureOfGaussians(conf)
        model.init_from_checkpoint(checkpoint)
        model.build_acc()
        logger.info(f"Mixture Of Gaussians模型加载完成，全局步数: {global_step}")

        # 返回训练器实例
        return cls(
            model=model,
            conf=conf,
            global_step=global_step,
            out_dir=out_dir,
        )
        
    
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
        self.base_exp_dir = conf['general.base_exp_dir']
        
        self.sdf_network = SDFNetwork(**conf['model.sdf_network']).to(device)
        self.deviation_network = SingleVarianceNetwork(**conf['model.variance_network']).to(device)
        
        checkpoint_dir = os.path.join(self.base_exp_dir, 'checkpoints')
        model_list_raw = os.listdir(checkpoint_dir)
        model_list = []
        for model_name in model_list_raw:
            if model_name[-3:] == 'pth':
                model_list.append(model_name)
        model_list.sort()
        # latest_model_name = model_list[-1]
        latest_model_name = "ckpt_300000.pth"
        
        checkpoint = torch.load(os.path.join(checkpoint_dir, latest_model_name), map_location=device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.iter_step = checkpoint['iter_step']
        self.renderer = NeuSRenderer(self.sdf_network, self.deviation_network, **conf['model.neus_renderer'])
        
        # 创建优化器
        params_to_train = list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        
        self.optimizer = torch.optim.Adam(params_to_train, lr=0.0001)
        return self.renderer
    
    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'optimizerNoColor': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.out_dir, 'checkpoints'), exist_ok=True)
        save_path = os.path.join(self.out_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step))
        torch.save(checkpoint, save_path)
        logger.info(f"保存NeuS检查点: {self.iter_step}")
        logger.info(f"NeuS检查点保存路径: {save_path}")


    def run_render(self):
        for idx in range(len(list(self.val_dataloader))):
            self.save_visualization(30000, batch_idx=idx, resolution_level=2)
            
        
        import time
        for iter_idx, batch in enumerate(self.train_dataloader):
            start_time = time.time()
            
            # 步骤1: 获取GPU批次数据
            t1_start = time.time()
            gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)
            t1_end = time.time()
            
            # 步骤2: 转换到世界坐标系
            t2_start = time.time()
            gpu_batch = self.rays_to_world(gpu_batch)
            t2_end = time.time()
            
            # 步骤3: 应用折射光线追踪
            t3_start = time.time()
            gpu_batch = self.test_render_refractive_from_tracer(gpu_batch, self.renderer)                        
            t3_end = time.time()
            
            # 步骤4: 模型渲染
            t4_start = time.time()
            outputs = self.model(gpu_batch, train=False, frame_id=self.iter_step)
            t4_end = time.time()

            rgb_gt = gpu_batch.rgb_gt
            pred_rgb = outputs['pred_rgb']
            
            # 步骤5: 图像处理和保存
            t5_start = time.time()
            # 将预测图像和真实图像拼接在一起
            H, W = rgb_gt.shape[1:3]
            combined_img = torch.cat([pred_rgb.reshape(H, W, 3), rgb_gt.reshape(H, W, 3)], dim=1)
            
            # 转换为numpy数组并调整为0-255范围
            combined_img_np = (combined_img.detach().cpu().numpy() * 255).astype(np.uint8)
            
            # 创建保存目录
            vis_dir = os.path.join(self.out_dir, 'visualization')
            os.makedirs(vis_dir, exist_ok=True)
            
            # 保存拼接图像
            img_path = os.path.join(vis_dir, f'render_comparison_{self.iter_step:06d}_{iter_idx}.png')
            cv2.imwrite(img_path, combined_img_np[..., ::-1])  # BGR转RGB
            t5_end = time.time()
            
            # 计算各步骤耗时
            total_time = time.time() - start_time
            t1_time = t1_end - t1_start
            t2_time = t2_end - t2_start
            t3_time = t3_end - t3_start
            t4_time = t4_end - t4_start
            t5_time = t5_end - t5_start
            
            # 输出耗时信息
            logger.info(f"已保存渲染对比图: {img_path}")
            logger.info(f"渲染耗时统计 (总计: {total_time:.4f}秒):")
            logger.info(f"  - 获取GPU批次: {t1_time:.4f}秒 ({t1_time/total_time*100:.1f}%)")
            logger.info(f"  - 坐标系转换: {t2_time:.4f}秒 ({t2_time/total_time*100:.1f}%)")
            logger.info(f"  - 折射光线追踪: {t3_time:.4f}秒 ({t3_time/total_time*100:.1f}%)")
            logger.info(f"  - 模型渲染: {t4_time:.4f}秒 ({t4_time/total_time*100:.1f}%)")
            logger.info(f"  - 图像处理与保存: {t5_time:.4f}秒 ({t5_time/total_time*100:.1f}%)")

    def train_silhouette(self, gpu_batch):
        rays_o = gpu_batch.rays_ori
        rays_d = gpu_batch.rays_dir
        rgb_gt = gpu_batch.rgb_gt
        mask = gpu_batch.mask
        
        batch_size, H, W, _ = rays_o.shape
        
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        rgb_gt_flat = rgb_gt.reshape(-1, 3)
        mask_flat = mask.reshape(-1, 1)
        
        scale_mat = self.train_dataset.scale_mat
        rays_o_flat = (rays_o_flat - scale_mat[:3, 3]) / scale_mat[0, 0]
        
        sample_size = 512
        perm = torch.randperm(len(rays_o_flat), device=rays_o_flat.device)
        selected_indices = perm[:sample_size]
        
        selected_rays_o = rays_o_flat[selected_indices]
        selected_rays_d = rays_d_flat[selected_indices]
        selected_mask = mask_flat[selected_indices]
        
        near, far = self.near_far_from_sphere(selected_rays_o, selected_rays_d)
        render_out = self.renderer.render(selected_rays_o, selected_rays_d, near, far)
        
        weights_sum = render_out['weights_sum']
        
        # 计算silhouette loss        
        silhouette_loss = torch.nn.functional.binary_cross_entropy(weights_sum.clip(1e-3, 1-1e-3).squeeze(), (selected_mask > 0.5).float().squeeze()) / rays_o.shape[0]
        
        # 计算eikonal loss
        eikonal_loss = render_out['eikonal_loss']
        
        # 计算法向平滑度损失
        with torch.enable_grad():
            # 从renderer获取表面点（使用sdf零级等值面的点）
            surface_points = selected_rays_o + weights_sum.reshape(-1, 1) * selected_rays_d
            mask_surface = (selected_mask > 0.5).squeeze(-1)
            if torch.sum(mask_surface) > 0:
                surface_points = surface_points[mask_surface]
                # 在表面点附近生成随机扰动点
                surface_points_neig = surface_points + (torch.rand_like(surface_points) - 0.5) * 0.01
                pp = torch.cat([surface_points, surface_points_neig], dim=0)
                surface_grad = self.renderer.sdf_network.gradient(pp)
                surface_points_normal = torch.nn.functional.normalize(surface_grad, p=2, dim=-1)
                
                N = surface_points_normal.shape[0] // 2
                diff_norm = torch.norm(surface_points_normal[:N] - surface_points_normal[N:], dim=-1)
                normal_smoothness_loss = torch.mean(diff_norm)  # 可以通过配置调整权重
            else:
                normal_smoothness_loss = torch.tensor(0.0, device=rays_o.device)
        
        loss = silhouette_loss * 0.1 + eikonal_loss * 0.1 + normal_smoothness_loss * 0.005
        
        return loss
        
    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = torch.clamp(mid - 1.0, min=0.0)
        far = mid + 1.0
        # shape: (-1, 1)
        return near, far
        

    def generate_train_batch_as_tnsr(self, gpu_batch, renderer):
        """使用RayTracing类实现光线折射渲染，并进行随机采样与法线平滑度计算。
        
        参数:
            gpu_batch: 包含射线信息的批次数据
            renderer: 渲染器对象，包含SDF网络
            
        返回:
            gpu_batch: 更新后的批次数据，包含折射后的射线和采样掩码
            normal_smoothness_loss: 法线平滑度损失
        """
        # 获取射线原点和方向
        rays_o = gpu_batch.rays_ori
        rays_d = gpu_batch.rays_dir
        rgb_gt = gpu_batch.rgb_gt
        mask = gpu_batch.mask

        # 获取批次大小和图像尺寸
        batch_size, H, W, _ = rays_o.shape
        
        # 设置折射率
        n1 = 1.0003  # 空气的折射率
        n2 = 1.51  # 玻璃的折射率
        
        # 创建RefractTracer实例
        ray_tracer = RefractTracer(ior_air=n1, ior_object=n2, n_samples=64)
        
        # 扁平化射线以便处理
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        rgb_gt_flat = rgb_gt.reshape(-1, 3)
        mask_flat = mask.reshape(-1, 1)
        
        # 确保射线方向是单位向量
        rays_d_flat = F.normalize(rays_d_flat, p=2, dim=-1)
        
        # 引入坐标系变换，缩放到单位球空间里
        scale_mat = self.train_dataset.scale_mat
        rays_o_flat = (rays_o_flat - scale_mat[:3, 3]) / scale_mat[0, 0]
        
        logger.debug(f"origin after scale: {rays_o_flat[0]}")
        logger.debug(f"origin norm to [0, 0, 0]: {torch.norm(rays_o_flat[0])}")
        
        # 只处理mask区域内的射线
        mask_indices = torch.where(mask_flat.squeeze(-1))[0]
        
        # 随机采样像素进行折射计算
        sample_size = min(512, mask_indices.shape[0])
        if mask_indices.shape[0] > sample_size:
            # 随机选择sample_size个像素
            perm = torch.randperm(mask_indices.shape[0], device=mask_indices.device)
            selected_indices = perm[:sample_size]
            mask_indices = mask_indices[selected_indices]
        
        logger.debug(f"随机采样 {mask_indices.shape[0]} 个像素进行折射计算")
        
        # 提取被选中的射线
        masked_rays_o = rays_o_flat[mask_indices]
        masked_rays_d = rays_d_flat[mask_indices]
        masked_rgb_gt = rgb_gt_flat[mask_indices]

        # 调用RefractTracer的forward函数，获取返回的字典
        ret_dict = ray_tracer(masked_rays_o, masked_rays_d, renderer.sdf_network)
        
        # 从字典中获取需要的值
        new_rays_o = ret_dict["rays_o"]
        new_rays_d = ret_dict["rays_d"]
        selected_indicies_final = ret_dict["selected_indicies_final"]
        rays_o_reflect = ret_dict["rays_o_reflect"]
        rays_d_reflect = ret_dict["rays_d_reflect"]
        attenuate_1 = ret_dict["attenuate_1"]
        attenuate_2 = ret_dict["attenuate_2"]
        
        # 获取第一次和第二次表面点
        first_surface_points = ret_dict["first_surface_points"]
        second_surface_points = ret_dict["second_surface_points"]
        
        # 创建追踪掩码，标识哪些射线成功进行了两次折射
        batch_tracing_mask = torch.zeros(masked_rays_o.shape[0], dtype=torch.bool, device=masked_rays_o.device)
        reflect_mask = torch.zeros_like(batch_tracing_mask)
        
        if len(selected_indicies_final) > 0:
            batch_tracing_mask[selected_indicies_final] = True
            reflect_mask[selected_indicies_final] = True
        
        # --- 计算法线平滑度损失 ---
        normal_smoothness_loss = torch.tensor(0.0, device=masked_rays_o.device)
        
        # 收集所有表面点
        surface_points_list = []

        # 添加第一次相交的表面点（如果有）
        if len(first_surface_points) > 0:
            surface_points_list.append(first_surface_points)

        # 添加第二次相交的表面点（如果有）
        if len(second_surface_points) > 0:
            surface_points_list.append(second_surface_points)

        # 只要找到了任意表面点就计算法线平滑度
        if len(surface_points_list) > 0:
            with torch.enable_grad():
                # 合并所有表面点
                surface_points = torch.cat(surface_points_list, dim=0)
                
                # # 在表面点附近生成随机扰动点
                # surface_points_neig = surface_points + (torch.rand_like(surface_points) - 0.5) * 0.01
                
                # # 将所有点合并为一个批次
                # pp = torch.cat([surface_points, surface_points_neig], dim=0)
                
                # # 计算梯度（法线方向）
                # surface_grad = renderer.sdf_network.gradient(pp)
                # surface_points_normal = torch.nn.functional.normalize(surface_grad, p=2, dim=-1)
                
                # # 计算法线平滑度损失
                # N = surface_points_normal.shape[0] // 2
                # diff_norm = torch.norm(surface_points_normal[:N] - surface_points_normal[N:], dim=-1)
                # normal_smoothness_loss = torch.mean(diff_norm)
                
                normal_smoothness_loss = self.compute_smoothness_loss(surface_points, renderer)
                
                # 记录日志
                logger.debug(f"法线平滑度损失: {normal_smoothness_loss.item():.6f}, 表面点数量: {surface_points.shape[0]}")

        # 设置valid_mask为成功进行两次折射的射线
        if batch_tracing_mask.sum() > 0:
            final_rays_o = new_rays_o[batch_tracing_mask]
            final_rays_d = new_rays_d[batch_tracing_mask]
            final_rgb_gt = masked_rgb_gt[batch_tracing_mask]
        
            # 将处理后的射线变换回原始坐标系
            final_rays_o = final_rays_o * scale_mat[0, 0] + scale_mat[:3, 3]
            
            final_rays_o = final_rays_o.view(1, 1, -1, 3)
            final_rays_d = final_rays_d.view(1, 1, -1, 3)
            final_rgb_gt = final_rgb_gt.view(1, 1, -1, 3)
        else:
            return None, torch.tensor(0.0, device=masked_rays_o.device)
        
        # 输出统计信息
        num_traced = batch_tracing_mask.sum().item()
        num_reflected = reflect_mask.sum().item()
        total_rays = mask_indices.shape[0]
        
        logger.debug(f"折射处理完成: {num_traced}/{total_rays} ({100.0*num_traced/total_rays:.2f}%) 射线成功折射")
        logger.debug(f"反射处理完成: {num_reflected}/{total_rays} ({100.0*num_reflected/total_rays:.2f}%) 射线发生反射")
        
        # 更新gpu_batch，保留梯度流
        gpu_batch.rays_ori = final_rays_o
        gpu_batch.rays_dir = final_rays_d
        gpu_batch.rgb_gt = final_rgb_gt
        
        return gpu_batch, normal_smoothness_loss
    
    def run_training(self):
        # 计算总迭代次数用于进度显示
        total_iterations = self.n_epochs * len(self.train_dataloader)
        
        # 创建整体训练进度条
        overall_progress = tqdm(total=total_iterations, desc="整体训练进度", unit="iter")
        overall_progress.update(self.iter_step)  # 更新已经完成的步数
        val_batch_idx = 0
        for epoch_idx in range(self.n_epochs):
            # 为每个epoch创建一个进度条
            epoch_progress = tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx+1}/{self.n_epochs}", leave=False)
            
            # 记录这个epoch的平均损失
            epoch_losses = []
            epoch_start_time = time.time()
            
            for batch in epoch_progress:
                # 步骤1: 获取GPU批次数据
                gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)
                # 步骤2: 转换到世界坐标系
                gpu_batch = self.rays_to_world(gpu_batch)
                
                loss = torch.tensor(0.0, device='cuda')
                
                # 计算silhouette loss以及eikonal_loss
                silhouette_losses = self.train_silhouette(gpu_batch)
                loss += silhouette_losses
                
                # 应用折射光线追踪 - 现在返回gpu_batch和法线平滑度损失
                gpu_batch, normal_smoothness_loss = self.generate_train_batch_as_tnsr(gpu_batch, self.renderer)
                
                # 检查是否有符合条件的射线（经过两次折射）
                if gpu_batch is not None:
                    # 步骤4: 模型渲染 - 只执行一次
                    outputs = self.model(gpu_batch, train=True, frame_id=self.iter_step)
                    pred_rgb_full = outputs["pred_rgb"]
                    rgb_gt_full = gpu_batch.rgb_gt
                    
                    l1_loss = torch.abs(pred_rgb_full - rgb_gt_full).mean()
                    
                    # 添加Eikonal损失和法线平滑度损失
                    loss = silhouette_losses + l1_loss + normal_smoothness_loss * 0.005  # 调整权重系数
                else:
                    # 如果当前批次没有符合条件的射线，记录并跳过
                    epoch_progress.set_postfix({'status': 'skipped - no valid rays'})
                    logger.warning(f"Epoch {epoch_idx+1}/{self.n_epochs}, Iter {self.iter_step}: No valid rays found, skipping")

                # 更新进度条信息
                epoch_progress.set_postfix({
                    'loss': f"{loss.item():.4f}"
                })
                epoch_losses.append(loss.item())
                
                # 记录训练信息
                if self.iter_step % 100 == 0:
                    logger.info(
                        f"Epoch {epoch_idx+1}/{self.n_epochs}, Iter {self.iter_step}, "
                        f"Loss: {loss.item():.4f}"
                    )      


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.iter_step += 1
                overall_progress.update(1)  # 更新整体进度条
                
                # 定期保存检查点
                if self.iter_step % 1000 == 0:
                    self.save_checkpoint()
                
                # 定期保存可视化结果 
                if self.iter_step % 100 == 0:
                    resolution_level = 2
                    self.save_visualization(self.iter_step, val_batch_idx, resolution_level)
                    val_batch_idx = (val_batch_idx + 1 ) % len(self.val_dataloader)

            # Epoch结束，计算平均损失并输出
            if epoch_losses:
                avg_loss = sum(epoch_losses) / len(epoch_losses)
                epoch_time = time.time() - epoch_start_time
                logger.info(f"Epoch {epoch_idx+1}/{self.n_epochs} 完成，平均损失: {avg_loss:.4f}，用时: {epoch_time:.2f}秒")
        
        # 训练结束，关闭进度条
        overall_progress.close()
        logger.info(f"训练完成！总迭代次数: {self.iter_step}")
    
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

    @staticmethod
    def set_log_level(level):
        """
        设置日志级别
        
        参数:
        - level: 日志级别，可以是字符串('INFO', 'DEBUG', 'WARNING', 'ERROR', 'CRITICAL')
                 或者logging模块中的级别常量
        """
        if isinstance(level, str):
            level = level.upper()
            level_dict = {
                'DEBUG': logging.DEBUG,
                'INFO': logging.INFO,
                'WARNING': logging.WARNING,
                'ERROR': logging.ERROR,
                'CRITICAL': logging.CRITICAL
            }
            if level in level_dict:
                level = level_dict[level]
            else:
                logger.warning(f"未知的日志级别: {level}，使用INFO级别")
                level = logging.INFO
                
        logger.setLevel(level)
        # 同时更新处理器的级别
        for handler in logger.handlers:
            handler.setLevel(level)
            
        logger.info(f"日志级别设置为: {logging.getLevelName(level)}")
        
    def test_render_refractive_from_tracer(self, gpu_batch, renderer, resolution_level=1):
        """使用RayTracing类实现光线折射渲染。
        
        参数:
            gpu_batch: 包含射线信息的批次数据
            renderer: 渲染器对象，包含SDF网络
            
        返回:
            gpu_batch: 更新后的批次数据，包含折射后的射线
        """
        # 获取射线原点和方向
        rays_o = gpu_batch.rays_ori
        rays_d = gpu_batch.rays_dir
        rgb_gt = gpu_batch.rgb_gt
        mask = gpu_batch.mask
        
        # 可选的下采样因子
        downsample_factor = resolution_level
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
        
        # 创建RefractTracer实例
        ray_tracer = RefractTracerInplace(ior_air=n1, ior_object=n2, n_samples=128)
        
        # 扁平化射线以便处理
        rays_o_flat = rays_o.reshape(-1, 3)
        rays_d_flat = rays_d.reshape(-1, 3)
        mask_flat = mask.reshape(-1, 1)
        
        # 确保射线方向是单位向量
        rays_d_flat = F.normalize(rays_d_flat, p=2, dim=-1)
        
        # 引入坐标系变换，缩放到单位球空间里
        scale_mat = self.train_dataset.scale_mat
        rays_o_flat = (rays_o_flat - scale_mat[:3, 3]) / scale_mat[0, 0]
        
        # 只处理mask区域内的射线
        mask_indices = torch.where(mask_flat.squeeze(-1))[0]
        logger.debug(f"只处理mask区域中的 {mask_indices.shape[0]}/{rays_o_flat.shape[0]} ({100.0*mask_indices.shape[0]/rays_o_flat.shape[0]:.2f}%) 射线")
        
        # 创建包含所有射线的副本
        processed_rays_o = rays_o_flat.clone()
        processed_rays_d = rays_d_flat.clone()
        
        # 提取mask区域的射线
        masked_rays_o = rays_o_flat[mask_indices]
        masked_rays_d = rays_d_flat[mask_indices]
        
        # 处理mask区域的射线
        from tqdm import tqdm
        
        # 每次处理的批次大小
        iter_size = 512
        tracing_mask_flat = torch.zeros(rays_o_flat.shape[0], dtype=torch.bool, device=rays_o_flat.device)
        reflect_mask_flat = torch.zeros(rays_o_flat.shape[0], dtype=torch.bool, device=rays_o_flat.device)
        reflect_dir_flat = rays_d_flat.clone()
        reflect_rate_flat = torch.zeros(rays_o_flat.shape[0], 1, device=rays_o_flat.device)
        refract_rate_flat = torch.zeros(rays_o_flat.shape[0], 1, device=rays_o_flat.device)
        
        # 按批次处理mask区域的射线
        for i in tqdm(range(0, mask_indices.shape[0], iter_size), desc="Ray Tracing with Refraction", unit="batch"):
            end_idx = min(i + iter_size, mask_indices.shape[0])
            batch_indices = mask_indices[i:end_idx]
            
            # 获取当前批次的射线
            batch_rays_o = rays_o_flat[batch_indices]
            batch_rays_d = rays_d_flat[batch_indices]
            
            # 应用ray_tracing_with_refraction函数
            new_rays_o, new_rays_d, reflect_dir, reflect_mask, reflect_rate, refract_rate, batch_tracing_mask = ray_tracer.ray_tracing_with_refraction(
                batch_rays_o, batch_rays_d, renderer.sdf_network
            )
            
            # 更新处理后的射线
            processed_rays_o[batch_indices] = new_rays_o
            processed_rays_d[batch_indices] = new_rays_d
            
            # 更新追踪掩码
            tracing_mask_flat[batch_indices] = batch_tracing_mask
            reflect_mask_flat[batch_indices] = reflect_mask
            
            # 更新反射方向和系数
            reflect_dir_flat[batch_indices] = reflect_dir
            reflect_rate_flat[batch_indices] = reflect_rate
            refract_rate_flat[batch_indices] = refract_rate
            
            # 清空CUDA缓存
            torch.cuda.empty_cache()
        
        # 将处理后的射线变换回原始坐标系
        processed_rays_o = processed_rays_o * scale_mat[0, 0] + scale_mat[:3, 3]
        
        # 重塑为原始形状
        processed_rays_o = processed_rays_o.reshape(batch_size, H, W, 3)
        processed_rays_d = processed_rays_d.reshape(batch_size, H, W, 3)
        
        # 输出统计信息
        num_traced = tracing_mask_flat.sum().item()
        num_reflected = reflect_mask_flat.sum().item()
        total_rays = mask_indices.shape[0]
        
        logger.debug(f"折射处理完成: {num_traced}/{total_rays} ({100.0*num_traced/total_rays:.2f}%) 射线成功折射")
        logger.debug(f"反射处理完成: {num_reflected}/{total_rays} ({100.0*num_reflected/total_rays:.2f}%) 射线发生反射")
        
        # 确保返回的射线是不带梯度的
        with torch.no_grad():
            # 创建完全分离的拷贝
            detached_rays_o = processed_rays_o.detach().clone()
            detached_rays_d = processed_rays_d.detach().clone()
            
            # 更新gpu_batch
            gpu_batch.rays_ori = detached_rays_o
            gpu_batch.rays_dir = detached_rays_d
        
        # 清理不再需要的变量
        if 'rays_o_flat' in locals(): del rays_o_flat
        if 'rays_d_flat' in locals(): del rays_d_flat
        if 'processed_rays_o' in locals(): del processed_rays_o
        if 'processed_rays_d' in locals(): del processed_rays_d
        if 'tracing_mask_flat' in locals(): del tracing_mask_flat
        if 'reflect_mask_flat' in locals(): del reflect_mask_flat
        if 'reflect_dir_flat' in locals(): del reflect_dir_flat
        if 'reflect_rate_flat' in locals(): del reflect_rate_flat
        if 'refract_rate_flat' in locals(): del refract_rate_flat
        
        # 强制垃圾回收和清理CUDA缓存
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        return gpu_batch
   

    def save_visualization(self, iteration, batch_idx=-1, resolution_level=1):
        """
        从验证集获取数据并保存渲染结果，使用matplotlib进行可视化
        
        参数:
        - iteration: 当前迭代次数，用于命名文件
        - batch_idx: 批次索引，为-1时随机选择
        - resolution_level: 图像分辨率级别，1表示原始分辨率，>1表示降采样
        """
        import numpy as np
        import os
        
        # 确保输出目录存在
        vis_dir = os.path.join(self.out_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 根据迭代次数确定性地选择不同批次
        val_batches = list(self.val_dataloader)
        if batch_idx < 0:
            # 随机选择一个批次
            batch_idx = np.random.randint(0, len(val_batches))
        batch = val_batches[batch_idx]
        
        # 从验证集获取数据
        try: 
            # 步骤1: 获取GPU批次数据
            gpu_batch = self.val_dataset.get_gpu_batch_with_intrinsics(batch)
            
            # 步骤2: 转换到世界坐标系
            gpu_batch = self.rays_to_world(gpu_batch)
            
            # 步骤3: 应用折射光线追踪 - 使用test_render版本
            gpu_batch = self.test_render_refractive_from_tracer(gpu_batch, self.renderer, resolution_level)
            
            # 步骤4: 模型渲染
            outputs = self.model(gpu_batch, train=False, frame_id=iteration)
            
            # 获取预测图像和ground truth
            pred_rgb = outputs["pred_rgb"].detach().cpu()  # [B, H, W, 3]
            gt_rgb = gpu_batch.rgb_gt.detach().cpu()       # [B, H, W, 3]
            
            # 获取掩码并确保形状正确
            mask = gpu_batch.mask.detach().cpu() > 0.5  # [B, H, W, 1]
            
            # 只处理批次中的第一个图像
            pred_img = pred_rgb[0].numpy()    # [H, W, 3]
            gt_img = gt_rgb[0].numpy()        # [H, W, 3]
            current_mask = mask[0]            # [H, W, 1]
            
            # 确保掩码形状与图像兼容
            mask_bool = current_mask.squeeze(-1).numpy()  # 从[H, W, 1]转换为[H, W]
            
            # 计算三种PSNR
            # 1. 有效区域的PSNR (mask区域)
            if torch.sum(current_mask) > 0:
                # 使用布尔索引来选择掩码区域的像素
                pred_masked = pred_rgb[0][current_mask.squeeze(-1)]  # 选择掩码为True的像素
                gt_masked = gt_rgb[0][current_mask.squeeze(-1)]      # 同样的像素位置
                
                mask_mse = torch.mean(torch.square(pred_masked - gt_masked))
                mask_psnr = -10 * torch.log10(mask_mse + 1e-8)
                mask_psnr_value = mask_psnr.item()
            else:
                mask_psnr_value = 0.0
            
            # 2. 非掩码区域的PSNR
            non_mask = ~current_mask.squeeze(-1)
            if torch.sum(non_mask) > 0:
                pred_non_masked = pred_rgb[0][non_mask]  # 选择掩码为False的像素
                gt_non_masked = gt_rgb[0][non_mask]      # 同样的像素位置
                
                non_mask_mse = torch.mean(torch.square(pred_non_masked - gt_non_masked))
                non_mask_psnr = -10 * torch.log10(non_mask_mse + 1e-8)
                non_mask_psnr_value = non_mask_psnr.item()
            else:
                non_mask_psnr_value = 0.0
            
            # 3. 全图的PSNR
            full_mse = torch.mean(torch.square(pred_rgb[0] - gt_rgb[0]))
            full_psnr = -10 * torch.log10(full_mse + 1e-8)
            full_psnr_value = full_psnr.item()
            
            # 创建matplotlib图像
            plt.figure(figsize=(12, 6))
            gs = GridSpec(2, 2, height_ratios=[9, 1])
            
            # 预测图像
            ax1 = plt.subplot(gs[0, 0])
            ax1.imshow(pred_img)
            ax1.set_title('Predict Image', fontsize=14)
            ax1.axis('off')
            
            # Ground Truth图像
            ax2 = plt.subplot(gs[0, 1])
            ax2.imshow(gt_img)
            ax2.set_title('Ground Truth', fontsize=14)
            ax2.axis('off')
            
            # PSNR信息展示区域
            ax3 = plt.subplot(gs[1, :])
            ax3.axis('off')
            ax3.text(0.5, 0.5, f'Mask PSNR: {mask_psnr_value:.2f}dB | Non-mask PSNR: {non_mask_psnr_value:.2f}dB | Full PSNR: {full_psnr_value:.2f}dB',
                    ha='center', va='center', fontsize=12, fontweight='bold')
            
            # 调整布局
            plt.tight_layout()
            
            # 保存图像
            img_path = os.path.join(vis_dir, f'iter_{iteration:06d}_{batch_idx:02d}.png')
            plt.savefig(img_path, dpi=150)
            plt.close()
            
            # 步骤1: 获取GPU批次数据
            gpu_batch = self.val_dataset.get_gpu_batch_with_intrinsics(batch)
            
            # 步骤2: 转换到世界坐标系
            gpu_batch = self.rays_to_world(gpu_batch)
            
            gpu_batch.rays_ori = gpu_batch.rays_ori[:, ::4, ::4, :]
            gpu_batch.rays_dir = gpu_batch.rays_dir[:, ::4, ::4, :]
            gpu_batch.mask = gpu_batch.mask[:, ::4, ::4, :]
            
            # 通过self.renderer渲染weights_sum以及gradients
            batch_size, H, W, _ = gpu_batch.rays_ori.shape

            # 初始化结果数组
            rendered_mask = np.zeros((H, W), dtype=np.float32)
            rendered_normals = np.zeros((H, W, 3), dtype=np.float32)

            # 设置批处理大小，避免显存不足
            render_batch_size = 1024

            # 扁平化射线数据
            rays_o_flat = gpu_batch.rays_ori.reshape(-1, 3)
            rays_d_flat = gpu_batch.rays_dir.reshape(-1, 3)
            
            scale_mat = self.train_dataset.scale_mat
            rays_o_flat = (rays_o_flat - scale_mat[:3, 3] ) / scale_mat[0, 0]

            # 计算迭代次数
            num_pixels = H * W
            num_batches = (num_pixels + render_batch_size - 1) // render_batch_size

            # 分批渲染
            logger.info(f"开始分批渲染，总共 {num_batches} 批次")
            for i in tqdm(range(num_batches), desc="渲染进度"):
                # 计算当前批次的起始和结束索引
                start_idx = i * render_batch_size
                end_idx = min((i + 1) * render_batch_size, num_pixels)
                
                # 获取当前批次的射线
                batch_rays_o = rays_o_flat[start_idx:end_idx]
                batch_rays_d = rays_d_flat[start_idx:end_idx]
                
                # 计算near和far
                near, far = self.near_far_from_sphere(batch_rays_o, batch_rays_d)
                
                # 使用renderer渲染
                render_results = self.renderer.render(
                    batch_rays_o, 
                    batch_rays_d, 
                    near, 
                    far, 
                    cos_anneal_ratio=0.0
                )
                
                # 获取weights_sum和gradients
                weights_sum = render_results['weights_sum'].detach().cpu().numpy()
                weights = render_results['weights'].detach()
                gradients = render_results['gradients'].detach()
                
                # 根据weights和gradients计算法线
                # 正确处理维度关系
                curr_batch_size = weights.shape[0]  # 当前批次的实际大小
                n_samples = weights.shape[1]  # 样本数量
                
                # 确保weights的维度与gradients匹配
                weights_expanded = weights[:, :, None]  # [curr_batch_size, n_samples, 1]
                
                # 计算加权梯度
                weighted_gradients = weights_expanded * gradients  # [curr_batch_size, n_samples, 3]
                aggregated_gradients = torch.sum(weighted_gradients, dim=1)  # [curr_batch_size, 3]
                
                normals = torch.clamp((aggregated_gradients + 1.0) * 0.5, 0.0, 1.0)
                normals = normals.cpu().numpy()
                
                # 将当前批次结果存入结果数组
                pixel_indices = np.arange(start_idx, end_idx)
                row_indices = pixel_indices // W
                col_indices = pixel_indices % W
                
                # 确保权重在合理范围内
                weights_sum_clipped = np.clip(weights_sum.squeeze(), 0.0, 1.0)
                rendered_mask[row_indices, col_indices] = weights_sum_clipped
                rendered_normals[row_indices, col_indices] = normals
                
                # 清理GPU内存
                torch.cuda.empty_cache()

            # 确保掩码值在合理范围内
            rendered_mask = np.clip(rendered_mask, 0.0, 1.0)

            # 创建三通道掩码图像（灰度图）
            rendered_mask_rgb = np.stack([rendered_mask] * 3, axis=-1)

            # 获取ground truth掩码
            gt_mask = gpu_batch.mask[0].squeeze(-1).detach().cpu().numpy().astype(np.float32)
            gt_mask_rgb = np.stack([gt_mask] * 3, axis=-1)

            # 在创建拼接图像前再次确保所有图像数据在有效范围内
            rendered_mask = np.clip(rendered_mask, 0.0, 1.0)
            rendered_normals = np.clip(rendered_normals, 0.0, 1.0)
            gt_mask_rgb = np.clip(gt_mask_rgb, 0.0, 1.0)

            # 创建拼接图像
            plt.figure(figsize=(18, 6))

            # 第一列：法线图
            plt.subplot(1, 3, 1)
            plt.imshow(rendered_normals)
            plt.title('Surface Normals', fontsize=14)
            plt.axis('off')

            # 第二列：渲染的掩码
            plt.subplot(1, 3, 2)
            plt.imshow(rendered_mask_rgb)
            plt.title('Predicted Mask', fontsize=14)
            plt.axis('off')

            # 第三列：ground truth掩码
            plt.subplot(1, 3, 3)
            plt.imshow(gt_mask_rgb)
            plt.title('Ground Truth Mask', fontsize=14)
            plt.axis('off')

            # 调整布局
            plt.tight_layout()

            # 保存拼接图像
            geo_vis_path = os.path.join(vis_dir, f'geo_vis_{iteration:06d}_{batch_idx:02d}.png')
            plt.savefig(geo_vis_path, dpi=150)
            plt.close()

            logger.info(f"保存几何可视化结果到 {geo_vis_path}")
            
        except Exception as e:
            logger.error(f"保存可视化结果时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    def compute_smoothness_loss(self, surface_points, renderer, n_samples=16):
        """计算平滑度损失
        
        Args:
            surface_points: 表面点 [N, 3]
            renderer: 包含SDF网络的渲染器
            n_samples: 每个表面点周围采样的点数
        """
        with torch.enable_grad():
            # 计算表面点的法线（梯度）
            surface_normals = renderer.sdf_network.gradient(surface_points)
            surface_normals = F.normalize(surface_normals, p=2, dim=-1)
            
            # 在表面点周围随机采样点
            # 使用正态分布在法线方向和切向方向采样
            batch_size = surface_points.shape[0]
            
            # 创建局部坐标系（使用法线作为一个轴）
            # 找到与法线正交的两个向量作为切向基
            random_vec = torch.randn_like(surface_normals)
            tangent = F.normalize(torch.cross(surface_normals, random_vec), dim=-1)
            bitangent = torch.cross(surface_normals, tangent)
            
            # 在局部空间采样点
            # 使用较小的标准差在法线方向采样，较大的标准差在切向采样
            sigma_normal = 0.01  # 法线方向的标准差
            sigma_tangent = 0.05  # 切向的标准差
            
            local_samples = []
            for i in range(n_samples):
                # 生成局部空间的随机偏移
                normal_offset = torch.randn(batch_size, 1, device=surface_points.device) * sigma_normal
                tangent_offset = torch.randn(batch_size, 1, device=surface_points.device) * sigma_tangent
                bitangent_offset = torch.randn(batch_size, 1, device=surface_points.device) * sigma_tangent
                
                # 将偏移转换到世界空间
                offset = (surface_normals * normal_offset + 
                         tangent * tangent_offset + 
                         bitangent * bitangent_offset)
                
                # 生成采样点
                sample_points = surface_points + offset
                local_samples.append(sample_points)
            
            # 将所有采样点堆叠在一起
            local_samples = torch.stack(local_samples, dim=1)  # [N, n_samples, 3]
            local_samples = local_samples.reshape(-1, 3)  # [N*n_samples, 3]
            
            # 计算所有点的SDF值和梯度
            sdf_values = renderer.sdf_network.sdf(local_samples)
            sample_normals = renderer.sdf_network.gradient(local_samples)
            sample_normals = F.normalize(sample_normals, p=2, dim=-1)
            
            # 重塑回批次形式
            sdf_values = sdf_values.reshape(batch_size, n_samples)
            sample_normals = sample_normals.reshape(batch_size, n_samples, 3)
            local_samples = local_samples.reshape(batch_size, n_samples, 3)
            
            # 计算密度平滑损失 Ld
            # 计算每个采样点到表面的归一化向量
            directions = local_samples - surface_points.unsqueeze(1)  # [N, n_samples, 3]
            distances = torch.norm(directions, dim=-1, keepdim=True)  # [N, n_samples, 1]
            directions = directions / (distances + 1e-8)
            
            # 计算与法线的点积
            normal_alignment = torch.abs(torch.sum(
                directions * surface_normals.unsqueeze(1), dim=-1))  # [N, n_samples]
            
            # 使用SDF值作为密度的代理
            density_weights = torch.exp(-torch.abs(sdf_values))
            Ld = torch.mean(density_weights * normal_alignment)
            
            # 计算法线平滑损失 Ln
            # 计算采样点法线与表面法线的角度
            normal_dot = torch.sum(
                sample_normals * surface_normals.unsqueeze(1), dim=-1)  # [N, n_samples]
            normal_angle = (1 - normal_dot) / 2
            
            # 同样使用密度权重
            Ln = torch.mean(density_weights * normal_angle)
            
            # 总损失
            lambda_d = 0.1  # 密度平滑权重
            lambda_n = 0.1  # 法线平滑权重
            smoothness_loss = lambda_d * Ld + lambda_n * Ln
            
            return smoothness_loss