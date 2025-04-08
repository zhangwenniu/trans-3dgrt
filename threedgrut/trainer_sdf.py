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
        self, conf
    ) -> None:
        """
        训练器类的初始化方法，简化版本
        
        参数:
        - conf: 配置信息
        - out_dir: 输出目录，用于保存结果
        """
        # 保存基本参数
        self.conf = conf
        self.out_dir = os.path.join(conf.out_dir, conf.experiment_name) 
        
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
        
    
    def init_from_neus(self):
        device = torch.device('cuda')
        # Configuration
        conf_path = "/workspace/sdf/NeTO/Use3DGRUT/confs/silhouette.conf"

        f = open(conf_path)
        conf_text = f.read()
        f.close()

        conf = ConfigFactory.parse_string(conf_text)

        self.sdf_network = SDFNetwork(**conf['model.sdf_network']).to(device)
        self.deviation_network = SingleVarianceNetwork(**conf['model.variance_network']).to(device)
        
        self.iter_step = 0
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
                    resolution_level = 4
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
        
    def save_visualization(self, iteration, batch_idx=-1, resolution_level=4):
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
            
            gpu_batch.rays_ori = gpu_batch.rays_ori[:, ::resolution_level, ::resolution_level, :]
            gpu_batch.rays_dir = gpu_batch.rays_dir[:, ::resolution_level, ::resolution_level, :]
            gpu_batch.mask = gpu_batch.mask[:, ::resolution_level, ::resolution_level, :]
            
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




