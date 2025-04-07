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
        # params_to_train += list(self.deviation_network.parameters())
        
        self.optimizer = torch.optim.Adam(params_to_train, lr=0.01)
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
    

    def run_render(self):
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
            # gpu_batch = self.test_render_refractive_from_neto(gpu_batch, self.renderer)
            # gpu_batch = self.test_render_refractive_from_tracer(gpu_batch, self.renderer)
            gpu_batch = self.generate_refractive_rays_from_sphere_parameter(gpu_batch)
            
            
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


    def generate_train_batch_from_tracer(self, gpu_batch, renderer):
        """使用RayTracing类实现光线折射渲染，并进行随机采样。
        
        参数:
            gpu_batch: 包含射线信息的批次数据
            renderer: 渲染器对象，包含SDF网络
            
        返回:
            gpu_batch: 更新后的批次数据，包含折射后的射线和采样掩码
        """
        # 获取射线原点和方向
        rays_o = gpu_batch.rays_ori
        rays_d = gpu_batch.rays_dir
        rgb_gt = gpu_batch.rgb_gt
        mask = gpu_batch.mask

        # 下采样因子
        downsample_factor = 4  # 增加下采样因子减少计算量
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
        ray_tracer = RefractTracer(ior_air=n1, ior_object=n2, n_samples=128, train=True)
        
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
        
        # 随机采样512个像素进行折射计算
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
        
        # 应用ray_tracing_with_refraction函数，并保留梯度链
        new_rays_o, new_rays_d, reflect_dir, reflect_mask, reflect_rate, refract_rate, batch_tracing_mask = ray_tracer.ray_tracing_with_refraction(
            masked_rays_o, masked_rays_d, renderer.sdf_network
        )
        
        # 创建valid_mask，初始全为False
        valid_mask_flat = torch.zeros(rays_o_flat.shape[0], dtype=torch.bool, device=rays_o_flat.device)
        
        # 获取成功折射的射线索引
        update_indices = mask_indices[batch_tracing_mask]
        
        # 设置valid_mask为成功进行两次折射的射线
        if update_indices.shape[0] > 0:
            valid_mask_flat = torch.scatter(valid_mask_flat, 0, update_indices, True)
        
        # 构建最终处理后的射线（包含未处理的和已处理的）
        # 创建最终张量，而不是修改原始张量
        final_rays_o = rays_o_flat
        final_rays_d = rays_d_flat
        
        # 只更新成功折射的光线（使用scatter避免原地修改）
        if update_indices.shape[0] > 0:
            final_rays_o = torch.scatter(final_rays_o, 0, 
                        update_indices.unsqueeze(1).repeat(1, 3), 
                        new_rays_o[batch_tracing_mask])
            
            final_rays_d = torch.scatter(final_rays_d, 0,
                        update_indices.unsqueeze(1).repeat(1, 3),
                        new_rays_d[batch_tracing_mask])
        
        # 将处理后的射线变换回原始坐标系
        final_rays_o = final_rays_o * scale_mat[0, 0] + scale_mat[:3, 3]
        
        # 重塑为原始形状
        final_rays_o = final_rays_o.view(batch_size, H, W, 3)
        final_rays_d = final_rays_d.view(batch_size, H, W, 3)
        valid_mask = valid_mask_flat.view(batch_size, H, W)
        
        # 输出统计信息
        num_traced = batch_tracing_mask.sum().item()
        num_reflected = reflect_mask.sum().item()
        total_rays = mask_indices.shape[0]
        
        logger.debug(f"折射处理完成: {num_traced}/{total_rays} ({100.0*num_traced/total_rays:.2f}%) 射线成功折射")
        logger.debug(f"反射处理完成: {num_reflected}/{total_rays} ({100.0*num_reflected/total_rays:.2f}%) 射线发生反射")
        
        # 更新gpu_batch，保留梯度流
        gpu_batch.rays_ori = final_rays_o
        gpu_batch.rays_dir = final_rays_d
        gpu_batch.valid_mask = valid_mask
        
        return gpu_batch
    
    def generate_train_batch_by_renderer_with_check(self, gpu_batch, renderer):
        """使用RayTracing类实现光线折射渲染，并进行随机采样。
        
        参数:
            gpu_batch: 包含射线信息的批次数据
            renderer: 渲染器对象，包含SDF网络
            
        返回:
            gpu_batch: 更新后的批次数据，包含折射后的射线和采样掩码
        """
        try:
            # 获取射线原点和方向
            rays_o = gpu_batch.rays_ori
            rays_d = gpu_batch.rays_dir
            rgb_gt = gpu_batch.rgb_gt
            mask = gpu_batch.mask

            # 获取批次大小和图像尺寸
            batch_size, H, W, _ = rays_o.shape
            
            # 检查掩码和射线形状是否匹配
            if mask.shape[:3] != rays_o.shape[:3]:
                logger.error(f"掩码形状不匹配: mask={mask.shape}, rays_o={rays_o.shape}")
                return None
            
            # 设置折射率
            n1 = 1.0003  # 空气的折射率
            n2 = 1.51  # 玻璃的折射率
            
            # 创建RefractTracer实例
            ray_tracer = RefractTracer(ior_air=n1, ior_object=n2, n_samples=128, train=True)
            
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
            
            # 检查掩码中是否存在有效值
            if torch.sum(mask_flat) == 0:
                logger.warning("没有有效的掩码区域，跳过当前批次")
                return None
                
            # 只处理mask区域内的射线
            mask_indices = torch.where(mask_flat.squeeze(-1))[0]
            
            # 安全检查：确保mask_indices不为空
            if mask_indices.shape[0] == 0:
                logger.warning("mask_indices为空，跳过当前批次")
                return None
                
            # 安全检查：确保索引在有效范围内
            max_index = mask_indices.max().item()
            if max_index >= rays_o_flat.shape[0]:
                logger.error(f"索引越界: max_index={max_index}, rays_o_flat.shape[0]={rays_o_flat.shape[0]}")
                return None
            
            # 随机采样512个像素进行折射计算
            sample_size = min(512, mask_indices.shape[0])
            if mask_indices.shape[0] > sample_size:
                # 随机选择sample_size个像素
                perm = torch.randperm(mask_indices.shape[0], device=mask_indices.device)
                selected_indices = perm[:sample_size]
                mask_indices = mask_indices[selected_indices]
            
            logger.debug(f"随机采样 {mask_indices.shape[0]} 个像素进行折射计算")
            
            # 安全检查：记录形状信息
            logger.debug(f"rays_o_flat.shape={rays_o_flat.shape}, mask_indices.shape={mask_indices.shape}")
            
            # 提取被选中的射线
            try:
                masked_rays_o = rays_o_flat[mask_indices]
                masked_rays_d = rays_d_flat[mask_indices]
                masked_rgb_gt = rgb_gt_flat[mask_indices]
            except IndexError as e:
                logger.error(f"射线索引错误: {e}")
                logger.error(f"mask_indices范围: min={mask_indices.min().item()}, max={mask_indices.max().item()}")
                logger.error(f"rays_o_flat.shape={rays_o_flat.shape}")
                return None
                
            # 应用ray_tracing_with_refraction函数，并保留梯度链
            try:
                new_rays_o, new_rays_d, reflect_dir, reflect_mask, reflect_rate, refract_rate, batch_tracing_mask = ray_tracer.ray_tracing_with_refraction(
                    masked_rays_o, masked_rays_d, renderer.sdf_network
                )
            except Exception as e:
                logger.error(f"折射追踪错误: {e}")
                return None

            # 安全检查：确保batch_tracing_mask和mask_indices形状匹配
            if batch_tracing_mask.shape[0] != mask_indices.shape[0]:
                logger.error(f"掩码形状不匹配: batch_tracing_mask={batch_tracing_mask.shape}, mask_indices={mask_indices.shape}")
                return None
                
            # 安全检查：检查batch_tracing_mask是否有任何为True的元素
            if not torch.any(batch_tracing_mask):
                logger.warning("没有成功折射的射线，跳过当前批次")
                return None

            # 获取成功折射的射线索引
            update_indices = mask_indices[batch_tracing_mask]
            
            # 安全检查：确保update_indices不为空
            if update_indices.shape[0] == 0:
                logger.warning("没有有效的update_indices，跳过当前批次")
                return None
                
            # 安全检查：确保索引不超出范围
            if update_indices.max().item() >= masked_rays_o.shape[0]:
                logger.error(f"update_indices越界: max={update_indices.max().item()}, masked_rays_o.shape[0]={masked_rays_o.shape[0]}")
                return None
                
            # 安全检查：确保new_rays_o等与batch_tracing_mask兼容
            if new_rays_o.shape[0] < batch_tracing_mask.sum().item():
                logger.error(f"数据形状不匹配: new_rays_o.shape={new_rays_o.shape}, batch_tracing_mask.sum()={batch_tracing_mask.sum().item()}")
                return None
            
            # 设置valid_mask为成功进行两次折射的射线
            if update_indices.shape[0] > 0:
                try:
                    # 安全索引：确保是有效范围
                    batch_indices = batch_tracing_mask.nonzero().squeeze(-1)
                    
                    final_rays_o = new_rays_o[batch_indices]
                    final_rays_d = new_rays_d[batch_indices]
                    final_rgb_gt = masked_rgb_gt[update_indices]
                
                    # 将处理后的射线变换回原始坐标系
                    final_rays_o = final_rays_o * scale_mat[0, 0] + scale_mat[:3, 3]
                    
                    # 安全检查：记录改变形状前的维度
                    logger.debug(f"改变形状前: final_rays_o.shape={final_rays_o.shape}")
                    
                    final_rays_o = final_rays_o.view(1, 1, -1, 3)
                    final_rays_d = final_rays_d.view(1, 1, -1, 3)
                    final_rgb_gt = final_rgb_gt.view(1, 1, -1, 3)
                    
                    # 安全检查：确保形状匹配
                    if final_rays_o.shape != final_rays_d.shape or final_rays_o.shape != final_rgb_gt.shape:
                        logger.error(f"形状不匹配: final_rays_o={final_rays_o.shape}, final_rays_d={final_rays_d.shape}, final_rgb_gt={final_rgb_gt.shape}")
                        return None
                except Exception as e:
                    logger.error(f"处理成功折射射线时发生错误: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    return None
            else:
                logger.warning("没有成功折射的射线，跳过当前批次")
                return None
                    
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
            
            return gpu_batch
            
        except Exception as e:
            logger.error(f"generate_train_batch_by_renderer发生未捕获异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def generate_train_batch_by_renderer(self, gpu_batch, renderer):
        """使用RayTracing类实现光线折射渲染，并进行随机采样。
        
        参数:
            gpu_batch: 包含射线信息的批次数据
            renderer: 渲染器对象，包含SDF网络
            
        返回:
            gpu_batch: 更新后的批次数据，包含折射后的射线和采样掩码
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
        ray_tracer = RefractTracer(ior_air=n1, ior_object=n2, n_samples=128, train=True)
        
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
        
        # 只处理mask区域内的射线
        mask_indices = torch.where(mask_flat.squeeze(-1))[0]
        
        # 随机采样512个像素进行折射计算
        sample_size = min(16, mask_indices.shape[0])
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
        
        with torch.no_grad():
            sdfs = renderer.sdf_network.sdf(masked_rays_o)
            logger.info(f"sdfs: {sdfs}")
        
        gradients = renderer.sdf_network.gradient(masked_rays_o).detach()
        gradients_norm = torch.norm(gradients, dim=-1)
        logger.info(f"gradients_norm: {gradients_norm}")

        
        
        # 应用ray_tracing_with_refraction函数，并保留梯度链
        new_rays_o, new_rays_d, reflect_dir, reflect_mask, reflect_rate, refract_rate, batch_tracing_mask, eikonal_loss = ray_tracer.ray_tracing_with_refraction(
            masked_rays_o, masked_rays_d, renderer.sdf_network
        )

 
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
            return None
            
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
        
        
        return gpu_batch

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
            surface_points = selected_rays_o
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
                normal_smoothness_loss = torch.mean(diff_norm) * 0.1  # 可以通过配置调整权重
            else:
                normal_smoothness_loss = torch.tensor(0.0, device=rays_o.device)
        
        # 添加强先验约束，确保球体附近的SDF值接近真实值，且梯度模长为1
        ball_center_position = [-0.02497456, 0.05139336, 0.04607311]
        ball_center_position = torch.tensor(ball_center_position, dtype=torch.float32, device=rays_o.device)
        radius = 0.15571212930017542
        radius = torch.tensor(radius, dtype=torch.float32, device=rays_o.device)

        # 在球半径2倍的空间内随机采样512个点
        num_samples = 512
        # 随机生成点的方法：在立方体中均匀采样然后保留在球形区域内的点
        sample_range = 2.0 * radius
        random_points = (torch.rand(num_samples, 3, device=rays_o.device) * 2 - 1) * sample_range
        random_points += ball_center_position  # 以球心为中心

        # 计算每个点的真实SDF值：到球心的距离减去半径
        points_to_center = random_points - ball_center_position
        distances = torch.norm(points_to_center, dim=1)
        true_sdf_values = distances - radius  # 正值表示点在球外，负值表示点在球内

        # 使用需要梯度的计算
        with torch.enable_grad():
            # 让随机点需要梯度，以便计算梯度模长
            random_points.requires_grad_(True)
            
            # 计算模型预测的SDF值
            pred_sdf_values = self.renderer.sdf_network.sdf(random_points)
            
            # 计算SDF值的梯度
            grad_outputs = torch.ones_like(pred_sdf_values)
            gradients = torch.autograd.grad(
                outputs=pred_sdf_values,
                inputs=random_points,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]
            
            # 计算梯度模长
            gradients_norm = torch.norm(gradients, dim=-1)
            
            # 计算预测值与真实值之间的损失
            sdf_loss = torch.mean((pred_sdf_values - true_sdf_values) ** 2)
            
            # 梯度模长约束：要求模长为1
            gradient_loss = torch.mean((gradients_norm - 1.0) ** 2)
            
            # 组合损失
            hyper_knowledge_loss = sdf_loss * 10.0 + gradient_loss * 1.0  # 可以调整权重

        # 记录统计信息
        mean_sdf_error = torch.mean(torch.abs(pred_sdf_values.detach() - true_sdf_values)).item()
        mean_gradient_error = torch.mean(torch.abs(gradients_norm.detach() - 1.0)).item()

        logger.debug(f"球体周围空间的平均SDF误差: {mean_sdf_error:.6f}")
        logger.debug(f"球体周围空间的平均梯度模长误差: {mean_gradient_error:.6f}")
        
        # 更新总损失
        loss = hyper_knowledge_loss
        
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
        """使用RayTracing类实现光线折射渲染，并进行随机采样。
        
        参数:
            gpu_batch: 包含射线信息的批次数据
            renderer: 渲染器对象，包含SDF网络
            
        返回:
            gpu_batch: 更新后的批次数据，包含折射后的射线和采样掩码
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
        ray_tracer = RefractTracer(ior_air=n1, ior_object=n2, n_samples=128, train=True)
        
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
        
        # 打印ret_dict中的所有值
        logger.info("RefractTracer返回字典内容:")
        for key, value in ret_dict.items():
            if isinstance(value, torch.Tensor):
                if value.numel() > 0:  # 检查张量是否为空
                    logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    logger.info(f"  {key}: 空张量")
            elif isinstance(value, list):
                if len(value) > 0:  # 检查列表是否为空
                    logger.info(f"  {key}: 列表长度={len(value)}")
                else:
                    logger.info(f"  {key}: 空列表")
            else:
                logger.info(f"  {key}: {value}")
        
        # 从字典中获取需要的值
        new_rays_o = ret_dict["rays_o"]
        new_rays_d = ret_dict["rays_d"]
        selected_indicies_final = ret_dict["selected_indicies_final"]
        rays_o_reflect = ret_dict["rays_o_reflect"]
        rays_d_reflect = ret_dict["rays_d_reflect"]
        attenuate_1 = ret_dict["attenuate_1"]
        attenuate_2 = ret_dict["attenuate_2"]
        
        # 创建追踪掩码，标识哪些射线成功进行了两次折射
        batch_tracing_mask = torch.zeros(masked_rays_o.shape[0], dtype=torch.bool, device=masked_rays_o.device)
        reflect_mask = torch.zeros_like(batch_tracing_mask)
        
        if len(selected_indicies_final) > 0:
            batch_tracing_mask[selected_indicies_final] = True
            reflect_mask[selected_indicies_final] = True
 
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
            return None
            
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
        
        return gpu_batch
    
    def generate_refractive_rays_from_sphere_parameter(self, gpu_batch):
        """
        基于已知的球体参数（中心位置和半径）生成折射光线。
        这个函数模拟光线穿过透明玻璃球时的折射效果。
        
        参数:
            gpu_batch: 包含射线信息的批次数据
            
        返回:
            gpu_batch: 更新后的批次数据，包含折射后的射线
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
        n2 = 1.51    # 玻璃的折射率
        
        # 球体参数
        ball_center_position = torch.tensor([-0.02497456, 0.05139336, 0.04607311], 
                                            dtype=torch.float32, device=rays_o.device)
        radius = torch.tensor(0.15571212930017542, 
                             dtype=torch.float32, device=rays_o.device)
        scale_mat = self.train_dataset.scale_mat
        ball_center_position += (scale_mat[:3, 3]).reshape(-1)
        ball_center_position *= (scale_mat[0, 0])
        
        radius *= scale_mat[0, 0]

        ball_center_position = torch.tensor([0, 0, -1], 
                                            dtype=torch.float32, device=rays_o.device)
        radius = torch.tensor(1.5, 
                             dtype=torch.float32, device=rays_o.device)
        
        # 扁平化射线以便处理
        rays_o_flat = rays_o.reshape(-1, 3)  # [batch_size*H*W, 3]
        rays_d_flat = rays_d.reshape(-1, 3)  # [batch_size*H*W, 3]
        rgb_gt_flat = rgb_gt.reshape(-1, 3)  # [batch_size*H*W, 3]
        
        # 确保射线方向是单位向量
        rays_d_flat = F.normalize(rays_d_flat, p=2, dim=-1)
        
        # 初始化结果数组
        hit_mask = torch.zeros(rays_o_flat.shape[0], dtype=torch.bool, device=rays_o.device)
        new_rays_o = rays_o_flat.clone()
        new_rays_d = rays_d_flat.clone()
        
        # 计算射线与球体的交点
        # 射线方程: p(t) = o + t*d
        # 球体方程: ||p - c||^2 = r^2
        # 展开: ||o + t*d - c||^2 = r^2
        oc = rays_o_flat - ball_center_position  # 射线原点到球心的向量
        
        # 二次方程系数
        a = torch.sum(rays_d_flat * rays_d_flat, dim=1)  # 方向向量是单位向量，所以a=1
        b = 2.0 * torch.sum(oc * rays_d_flat, dim=1)
        c = torch.sum(oc * oc, dim=1) - radius * radius
        
        # 计算判别式
        discriminant = b * b - 4 * a * c
        
        # 射线与球相交的情况
        valid_mask = discriminant > 0
        
        if torch.sum(valid_mask) > 0:
            # 只处理相交的射线
            valid_a = a[valid_mask]
            valid_b = b[valid_mask]
            valid_disc = discriminant[valid_mask]
            
            # 计算交点参数t（选择较小的t值，即第一个交点）
            sqrt_disc = torch.sqrt(valid_disc)
            t1 = (-valid_b - sqrt_disc) / (2.0 * valid_a)
            t2 = (-valid_b + sqrt_disc) / (2.0 * valid_a)
            
            # 确保t1是入射点，t2是出射点
            valid_t1 = t1 > 0  # 只考虑射线前方的交点
            
            # 找到所有有效的交点
            hit_indices = torch.where(valid_mask)[0][valid_t1]
            hit_mask[hit_indices] = True
            
            if torch.sum(hit_mask) > 0:
                # 计算入射点
                valid_t1_values = t1[valid_t1]
                valid_t2_values = t2[valid_t1]
                
                hit_rays_o = rays_o_flat[hit_indices]
                hit_rays_d = rays_d_flat[hit_indices]
                
                # 计算入射点和出射点
                entry_points = hit_rays_o + valid_t1_values.unsqueeze(1) * hit_rays_d
                exit_points = hit_rays_o + valid_t2_values.unsqueeze(1) * hit_rays_d
                
                # 计算入射点法线（从球心指向表面的单位向量）
                entry_normals = F.normalize(entry_points - ball_center_position, p=2, dim=1)
                exit_normals = F.normalize(exit_points - ball_center_position, p=2, dim=1)
                # 出射点法线指向球体外部，需要取负值
                exit_normals = -exit_normals
                
                # 第一次折射（空气->玻璃）
                cos_theta1 = torch.sum(hit_rays_d * (-entry_normals), dim=1, keepdim=True)
                
                # 计算入射光线在表面的投影
                perpendicular1 = hit_rays_d + entry_normals * cos_theta1
                
                # 应用斯涅尔定律
                sin_theta1 = torch.norm(perpendicular1, dim=1, keepdim=True)
                sin_theta2 = sin_theta1 * (n1 / n2)
                
                # 检查全反射情况
                total_internal_reflection1 = sin_theta2 > 1.0
                
                # 如果没有全反射，计算折射方向
                cos_theta2 = torch.sqrt(1 - sin_theta2 * sin_theta2)
                refracted_d1 = (perpendicular1 * (n1 / n2)) - (entry_normals * cos_theta2)
                
                # 归一化折射方向
                refracted_d1 = F.normalize(refracted_d1, p=2, dim=1)
                
                # 第二次折射（玻璃->空气）
                cos_theta3 = torch.sum(refracted_d1 * (-exit_normals), dim=1, keepdim=True)
                
                # 计算第二次入射光线在表面的投影
                perpendicular2 = refracted_d1 + exit_normals * cos_theta3
                
                # 应用斯涅尔定律
                sin_theta3 = torch.norm(perpendicular2, dim=1, keepdim=True)
                sin_theta4 = sin_theta3 * (n2 / n1)
                
                # 检查全反射情况
                total_internal_reflection2 = sin_theta4 > 1.0
                
                # 如果没有全反射，计算折射方向
                cos_theta4 = torch.sqrt(1 - sin_theta4 * sin_theta4)
                refracted_d2 = (perpendicular2 * (n2 / n1)) - (exit_normals * cos_theta4)
                
                # 归一化第二次折射方向
                refracted_d2 = F.normalize(refracted_d2, p=2, dim=1)
                
                # 只更新没有发生全反射的射线
                valid_refraction = ~(total_internal_reflection1 | total_internal_reflection2)
                valid_indices = hit_indices[valid_refraction.squeeze(-1)]
                
                if len(valid_indices) > 0:
                    # 更新射线起点和方向
                    new_rays_o[valid_indices] = exit_points[valid_refraction.squeeze(-1)]
                    new_rays_d[valid_indices] = refracted_d2[valid_refraction.squeeze(-1)]
        
        # 重塑为原始形状
        new_rays_o = new_rays_o.reshape(batch_size, H, W, 3)
        new_rays_d = new_rays_d.reshape(batch_size, H, W, 3)
        
        # 输出统计信息
        num_hit = hit_mask.sum().item()
        total_rays = rays_o_flat.shape[0]
        
        logger.debug(f"射线与球体相交: {num_hit}/{total_rays} ({100.0*num_hit/total_rays:.2f}%) 射线穿过球体")
        
        # 更新gpu_batch
        gpu_batch.rays_ori = new_rays_o
        gpu_batch.rays_dir = new_rays_d
        
        return gpu_batch

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
                
                # 计算silhouette loss以及eikonal_loss
                silhouette_losses = self.train_silhouette(gpu_batch)
                
                # 步骤3: 应用折射光线追踪 - 现在返回的是更新后的gpu_batch
                # gpu_batch = self.generate_train_batch_as_tnsr(gpu_batch, self.renderer)
                
                loss = silhouette_losses
                
                # 检查是否有符合条件的射线（经过两次折射）
                if gpu_batch is not None:
                    # # 步骤4: 模型渲染 - 只执行一次
                    # outputs = self.model(gpu_batch, train=True, frame_id=self.iter_step)
                    # pred_rgb_full = outputs["pred_rgb"]
                    # rgb_gt_full = gpu_batch.rgb_gt
                    
                    # l1_loss = torch.abs(pred_rgb_full - rgb_gt_full).mean()
                    
                    # # 添加Eikonal损失
                    # loss = loss + l1_loss
                    pass
                else:
                    # 如果当前批次没有符合条件的射线，记录并跳过
                    epoch_progress.set_postfix({'status': 'skipped - no valid rays'})
                    logger.info(f"Epoch {epoch_idx+1}/{self.n_epochs}, Iter {self.iter_step}: No valid rays found, skipping")

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
            
            # 添加可视化掩码的功能
            plt.figure(figsize=(6, 6))
            plt.imshow(mask_bool, cmap='Greens')
            plt.title('Mask Region', fontsize=14)
            plt.axis('off')
            
            # 保存掩码图像
            mask_path = os.path.join(vis_dir, f'mask_{iteration:06d}.png')
            plt.savefig(mask_path, dpi=150)
            plt.close()
            
            logger.info(f"保存可视化结果到 {img_path}")
            logger.info(f"掩码区域PSNR: {mask_psnr_value:.2f}dB, 非掩码区域PSNR: {non_mask_psnr_value:.2f}dB, 全图PSNR: {full_psnr_value:.2f}dB")
            
        except Exception as e:
            logger.error(f"保存可视化结果时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())