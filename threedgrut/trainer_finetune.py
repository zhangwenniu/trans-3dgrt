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
from threedgrut.trace_refract import RefractTracer
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
        self.n_epochs = 100  # 默认训练轮数，可以从配置中读取
        
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
        
        self.optimizer = torch.optim.Adam(params_to_train, lr=0.1)
        return self.renderer
    
    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'optimizerNoColor': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        save_path = os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step))
        torch.save(checkpoint, save_path)
        logger.info(f"保存NeuS检查点: {self.iter_step}")
        logger.info(f"NeuS检查点保存路径: {save_path}")

    def test_render_refractive_from_tracer(self, gpu_batch, renderer):
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
        downsample_factor = 2
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
        
            # # =============================
            # inter_points = final_rays_o[valid_mask_flat]
            # gradients = self.sdf_network.gradient(inter_points)
            # gradient_error = (torch.linalg.norm(gradients.reshape(-1, 3), ord=2,
            #                                     dim=-1) - 1.0) ** 2
            # gradient_error = gradient_error.sum()
            # gpu_batch.gradient_error = gradient_error
            # logger.info(f"interpoints sdf: {self.sdf_network.sdf(inter_points).mean()}")
            # logger.info(f"gradient_error = {gradient_error}")
            # # =============================

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
        # 应用ray_tracing_with_refraction函数，并保留梯度链
        new_rays_o, new_rays_d, reflect_dir, reflect_mask, reflect_rate, refract_rate, batch_tracing_mask = ray_tracer.ray_tracing_with_refraction(
            masked_rays_o, masked_rays_d, renderer.sdf_network
        )

        # 获取成功折射的射线索引
        update_indices = mask_indices[batch_tracing_mask]
        
        # 设置valid_mask为成功进行两次折射的射线
        if update_indices.shape[0] > 0:
            final_rays_o = new_rays_o[update_indices]
            final_rays_d = new_rays_d[update_indices]
            final_rgb_gt = masked_rgb_gt[update_indices]
        
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




    def run_training(self):
        DEBUG = False
        if DEBUG:
            for batch in self.train_dataloader:
                logger.info("============测试损失：简单求和开始================ #")
                gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)
                logger.info(f"1. gpu_batch.rays_ori.requires_grad: {gpu_batch.rays_ori.requires_grad}")
                logger.info(f"1. gpu_batch.rays_dir.requires_grad: {gpu_batch.rays_dir.requires_grad}")
                # gpu_batch.rays_ori.requires_grad = True
                # gpu_batch.rays_dir.requires_grad = True
                
                gpu_batch = self.rays_to_world(gpu_batch)

                logger.info(f"2. gpu_batch.rays_ori.requires_grad: {gpu_batch.rays_ori.requires_grad}")
                logger.info(f"2. gpu_batch.rays_dir.requires_grad: {gpu_batch.rays_dir.requires_grad}")

                gpu_batch.rays_ori.requires_grad = True
                gpu_batch.rays_dir.requires_grad = True
                
                logger.info(f"3. gpu_batch.rays_ori.requires_grad: {gpu_batch.rays_ori.requires_grad}")
                logger.info(f"3. gpu_batch.rays_dir.requires_grad: {gpu_batch.rays_dir.requires_grad}")
                                                
                gpu_batch = self.generate_train_batch_from_tracer(gpu_batch, self.renderer)
                
                
                logger.info(f"4. gpu_batch.rays_ori.requires_grad: {gpu_batch.rays_ori.requires_grad}")
                logger.info(f"4. gpu_batch.rays_dir.requires_grad: {gpu_batch.rays_dir.requires_grad}")
                
                outputs = self.model(gpu_batch, train=True, frame_id=self.iter_step)
                pred_rgb_full = outputs["pred_rgb"]
                test_loss = pred_rgb_full.sum()
                test_loss.backward()
                logger.info(f"5. test_loss: {test_loss}")
                logger.info(f"5. pred_rgb_full.grad: {pred_rgb_full.grad}")
                logger.info(f"5. self.model.positions.grad: {self.model.positions.grad}")
                logger.info(f"5. gpu_batch.rays_ori.grad: {gpu_batch.rays_ori.grad}")
                logger.info(f"5. gpu_batch.rays_dir.grad: {gpu_batch.rays_dir.grad}")
                logger.info("============测试损失：简单求和结束================ #")
            
            return
        
        
        
        
        
        from tqdm import tqdm
        import time
        
        # 计算总迭代次数用于进度显示
        total_iterations = self.n_epochs * len(self.train_dataloader)
        
        # 创建整体训练进度条
        overall_progress = tqdm(total=total_iterations, desc="整体训练进度", unit="iter")
        overall_progress.update(self.iter_step)  # 更新已经完成的步数
        
        for epoch_idx in range(self.n_epochs):
            # 为每个epoch创建一个进度条
            epoch_progress = tqdm(self.train_dataloader, desc=f"Epoch {epoch_idx+1}/{self.n_epochs}", leave=False)
            
            # 记录这个epoch的平均损失
            epoch_losses = []
            epoch_start_time = time.time()
            
            for batch in epoch_progress:
                # 定期保存可视化结果 
                if self.iter_step % 1000 == 0:
                    resolution_level = 2
                    self.save_visualization(self.iter_step, resolution_level)
                # 步骤1: 获取GPU批次数据
                gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)
                # # 步骤2: 转换到世界坐标系
                # gpu_batch = self.rays_to_world(gpu_batch)
                # gpu_batch.rays_ori.requires_grad = True
                # gpu_batch.rays_dir.requires_grad = True
                # # 步骤3: 应用折射光线追踪
                # gpu_batch = self.generate_train_batch_by_renderer(gpu_batch, self.renderer)
                # 检查是否有符合条件的射线（经过两次折射）
                if gpu_batch is not None:
                    logger.info(f"5. gpu_batch.rays_ori.requires_grad = {gpu_batch.rays_ori.requires_grad}")
                    logger.info(f"5. gpu_batch.rays_dir.requires_grad = {gpu_batch.rays_dir.requires_grad}")
                    # 步骤4: 模型渲染 - 只执行一次
                    outputs = self.model(gpu_batch, train=True, frame_id=self.iter_step)
                    pred_rgb_full = outputs["pred_rgb"]
                    rgb_gt_full = gpu_batch.rgb_gt
                    
                    l1_loss = torch.abs(pred_rgb_full - rgb_gt_full).mean()
                    loss = l1_loss * 0.001
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    # 调试测试 - 验证计算图连接
                    logger.info(f"pred_rgb_full需要梯度: {pred_rgb_full.requires_grad}")
                    logger.info(f"pred_rgb_full是叶节点: {pred_rgb_full.is_leaf}")
                    logger.info(f"计算图检查: {pred_rgb_full._grad_fn}")
                    logger.info(f"loss: {loss}")
                    logger.info(f"loss.grad: {loss.grad}")
                    
                    for name, param in self.sdf_network.named_parameters():
                        if param.grad is not None:
                            logger.info(f"{name} 的梯度已回传 (梯度范数: {param.grad.norm().item():.6f})")
                        else:
                            logger.info(f"{name} 的梯度未回传！")
                        

                    # 更新进度条信息
                    epoch_progress.set_postfix({
                        'loss': f"{loss.item():.4f}", 
                        # 'valid_rays': torch.sum(valid_mask).item()
                    })
                    epoch_losses.append(loss.item())
                    
                    # 记录训练信息
                    if self.iter_step % 100 == 0:
                        logger.info(
                            f"Epoch {epoch_idx+1}/{self.n_epochs}, Iter {self.iter_step}, "
                            f"Loss: {loss.item():.4f},"
                            # f"Valid rays: {torch.sum(valid_mask).item()}"
                        )      
                else:
                    # 如果当前批次没有符合条件的射线，记录并跳过
                    epoch_progress.set_postfix({'status': 'skipped - no valid rays'})
                    logger.info(f"Epoch {epoch_idx+1}/{self.n_epochs}, Iter {self.iter_step}: No valid rays found, skipping")
                    continue
                
                self.iter_step += 1
                overall_progress.update(1)  # 更新整体进度条
                
                # 定期保存检查点
                if self.iter_step % 1000 == 0:
                    self.save_checkpoint()
            
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

    def save_visualization(self, iteration, resolution_level=1):
        """
        从验证集获取数据并保存渲染结果，类似run_render的流程
        使用test_render_refractive_from_tracer进行全分辨率渲染
        
        参数:
        - iteration: 当前迭代次数，用于命名文件
        - resolution_level: 图像分辨率级别，1表示原始分辨率，>1表示降采样
        """
        import numpy as np
        import os
        import cv2
        
        # 确保输出目录存在
        vis_dir = os.path.join(self.out_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        # 从验证集获取数据
        try:
            # 尝试从验证集获取一个batch
            if hasattr(self, 'val_dataloader') and self.val_dataloader is not None:
                batch = next(iter(self.val_dataloader))
            else:
                # 如果没有验证集，则使用训练集
                batch = next(iter(self.train_dataloader))
            
            # 步骤1: 获取GPU批次数据
            gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)
            
            # 步骤2: 转换到世界坐标系
            gpu_batch = self.rays_to_world(gpu_batch)
            
            # 步骤3: 应用折射光线追踪 - 使用test_render版本
            gpu_batch = self.test_render_refractive_from_tracer(gpu_batch, self.renderer)
            
            # 步骤4: 模型渲染
            outputs = self.model(gpu_batch, train=False, frame_id=iteration)
            
            # 获取预测图像和ground truth
            pred_rgb = outputs["pred_rgb"].detach().cpu()  # [B, H, W, 3]
            gt_rgb = gpu_batch.rgb_gt.detach().cpu()       # [B, H, W, 3]
            
            # 获取掩码并确保形状正确
            mask = gpu_batch.mask.detach().cpu() > 0.5  # [B, H, W, 1]
            
            # 如果需要降采样
            if resolution_level > 1:
                B, H, W, C = pred_rgb.shape
                new_H, new_W = H // resolution_level, W // resolution_level
                
                # 降采样预测和ground truth图像
                pred_rgb_resized = torch.nn.functional.interpolate(
                    pred_rgb.permute(0, 3, 1, 2), 
                    size=(new_H, new_W), 
                    mode='bilinear'
                ).permute(0, 2, 3, 1)
                
                gt_rgb_resized = torch.nn.functional.interpolate(
                    gt_rgb.permute(0, 3, 1, 2), 
                    size=(new_H, new_W), 
                    mode='bilinear'
                ).permute(0, 2, 3, 1)
                
                mask_resized = torch.nn.functional.interpolate(
                    mask.float().permute(0, 3, 1, 2), 
                    size=(new_H, new_W), 
                    mode='nearest'
                ).permute(0, 2, 3, 1).bool()
            else:
                pred_rgb_resized = pred_rgb
                gt_rgb_resized = gt_rgb
                mask_resized = mask
            
            # 只处理批次中的第一个图像
            pred_img = pred_rgb_resized[0]    # [H, W, 3]
            gt_img = gt_rgb_resized[0]        # [H, W, 3]
            current_mask = mask_resized[0]    # [H, W, 1]
            
            # 将预测图像和真实图像拼接在一起
            H, W = pred_img.shape[:2]
            combined_img = torch.cat([pred_img, gt_img], dim=1)  # [H, 2*W, 3]
            
            # 计算两种PSNR
            # 1. 有效区域的PSNR (mask区域)
            # 确保掩码形状与图像兼容
            mask_bool = current_mask.squeeze(-1)  # 从[H, W, 1]转换为[H, W]
            
            if torch.sum(mask_bool) > 0:
                # 使用布尔索引来选择掩码区域的像素
                pred_masked = pred_img[mask_bool]  # 选择掩码为True的像素
                gt_masked = gt_img[mask_bool]      # 同样的像素位置
                
                mask_mse = torch.mean(torch.square(pred_masked - gt_masked))
                mask_psnr = -10 * torch.log10(mask_mse + 1e-8)
                mask_psnr_value = mask_psnr.item()
            else:
                mask_psnr_value = 0.0
            
            # 2. 全图的PSNR
            full_mse = torch.mean(torch.square(pred_img - gt_img))
            full_psnr = -10 * torch.log10(full_mse + 1e-8)
            full_psnr_value = full_psnr.item()
            
            # 转换为numpy数组并调整为0-255范围
            combined_img_np = (combined_img.numpy() * 255).astype(np.uint8)
            
            # 在图像上添加PSNR信息
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(combined_img_np, f'Mask PSNR: {mask_psnr_value:.2f}dB', 
                        (10, 30), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_img_np, f'Full PSNR: {full_psnr_value:.2f}dB', 
                        (10, 60), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            
            # 在图像上添加标题
            cv2.putText(combined_img_np, 'Prediction', 
                        (W//4, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(combined_img_np, 'Ground Truth', 
                        (W + W//4, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            # 保存拼接图像
            img_path = os.path.join(vis_dir, f'iter_{iteration:06d}.png')
            cv2.imwrite(img_path, combined_img_np[..., ::-1])  # BGR转RGB
            
            # 添加可视化掩码的功能
            # 创建一个掩码可视化图像
            mask_vis = np.zeros((H, W, 3), dtype=np.uint8)
            mask_vis[mask_bool.numpy()] = [0, 255, 0]  # 将掩码区域标为绿色
            
            # 保存掩码图像
            mask_path = os.path.join(vis_dir, f'mask_{iteration:06d}.png')
            cv2.imwrite(mask_path, mask_vis)
            
            logger.info(f"保存可视化结果到 {img_path}")
            logger.info(f"Mask PSNR: {mask_psnr_value:.2f}dB, Full PSNR: {full_psnr_value:.2f}dB")
            
        except Exception as e:
            logger.error(f"保存可视化结果时出错: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

