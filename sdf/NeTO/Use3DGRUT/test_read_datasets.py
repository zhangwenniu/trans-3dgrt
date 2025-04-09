#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from addict import Dict

from datasets import ColmapDataset
from datasets.utils import MultiEpochsDataLoader


def main():
    # 数据集路径，请替换为您自己的Colmap数据集路径
    dataset_path = "data/eiko_ball_masked/"
    
    # 创建Colmap数据集
    train_dataset = ColmapDataset(
        dataset_path,
        split="train",          # 使用训练集分割
        downsample_factor=1.0,  # 可以设置为>1的值来下采样图像
        ray_jitter=None         # 训练时可以添加光线抖动，这里不使用
    )
    
    # 创建验证集
    val_dataset = ColmapDataset(
        dataset_path,
        split="val",
        downsample_factor=1.0
    )
    
    # 打印数据集信息
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # 获取场景信息
    scene_extent = train_dataset.get_scene_extent()
    scene_bbox = train_dataset.get_scene_bbox()
    
    print(f"Scene extent: {scene_extent}")
    print(f"Scene bbox: {scene_bbox}")
    
    # 创建数据加载器
    train_dataloader = MultiEpochsDataLoader(
        train_dataset,
        num_workers=4,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
    )
    
    # 检查两种方法是否给出相同的结果
    are_equal = np.allclose(train_dataset.camera_centers, train_dataset.poses[:, :3, 3])
    print(f"camera_centers and poses[:, :3, 3] are equal: {are_equal}")
    
    # 检查near_far_from_sphere的结果
    near, far = train_dataset.near_far_from_sphere()
    print(f"near: {near}")
    print(f"far: {far}")
    
    # 加载一个批次
    for batch_idx, batch in enumerate(train_dataloader):
        print(f"Read batch {batch_idx + 1}")
        
        # 将批次数据移动到GPU并获取带内参的批次
        gpu_batch = train_dataset.get_gpu_batch_with_intrinsics(batch)
        
        # 提取关键数据
        rays_ori = gpu_batch.rays_ori      # 光线原点，形状为 [B, H, W, 3]
        rays_dir = gpu_batch.rays_dir      # 光线方向，形状为 [B, H, W, 3]
        rgb_gt = gpu_batch.rgb_gt         # 目标RGB值，形状为 [B, H, W, 3]
        mask = gpu_batch.mask            # 掩码，形状为 [B, H, W, 1]
        T_to_world = gpu_batch.T_to_world  # 世界坐标变换矩阵，形状为 [B, 4, 4]
        
        # 打印数据形状
        print(f"rays_ori.shape: {rays_ori.shape}")
        print(f"rays_dir.shape: {rays_dir.shape}")
        print(f"rgb_gt.shape: {rgb_gt.shape}")
        if mask is not None:
            print(f"mask.shape: {mask.shape}")
        print(f"T_to_world.shape: {T_to_world.shape}")
        
        # 显示第一张图像
        img = rgb_gt[0].cpu().numpy()  # 将张量转换为numpy数组
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        plt.title("Sample Image")
        plt.savefig("sample_image.png")
        print(f"Saved sample_image.png")
        
        # 如果有掩码，显示掩码
        if mask is not None:
            mask_img = mask[0].cpu().numpy()
            plt.figure(figsize=(10, 10))
            plt.imshow(mask_img.squeeze(), cmap='gray')
            plt.title("Sample Mask")
            plt.savefig("sample_mask.png")
            print(f"Saved sample_mask.png")
            
            masked_image = ((img + mask_img) / 2.0)
            plt.figure(figsize=(10, 10))
            plt.imshow(masked_image)
            plt.title("Masked Image")
            plt.savefig("masked_image.png")
            print(f"Saved masked_image.png")
        
        # 检查相机内参
        # ColmapDataset 中的内参是作为字典或特定属性存储的
        for key in vars(gpu_batch):
            if key.startswith('intrinsics_'):
                print(f"Found camera intrinsics: {key}")
                camera_params = getattr(gpu_batch, key)
                if isinstance(camera_params, dict):
                    print(f"Camera parameters: {camera_params}")
        
        # 只处理第一个批次
        break


if __name__ == "__main__":
    main()
