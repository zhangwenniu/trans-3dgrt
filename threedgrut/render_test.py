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

        dataloader = torch.utils.data.DataLoader(dataset, num_workers=8, batch_size=1, shuffle=False, collate_fn=None)
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

    @torch.no_grad()
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
                
                
                # ----------------- 添加镜面球体反射实验 -----------------
                # 球体参数：中心(0,0,0)，半径2
                sphere_center = torch.zeros(1, 1, 3, device=rays_ori_world_flat.device)
                sphere_radius = 1
                
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
