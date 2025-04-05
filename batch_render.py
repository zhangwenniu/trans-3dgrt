#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import numpy as np
import torch
from pathlib import Path
from headless_render import HeadlessRenderer, nullcontext

def create_orbit_camera_poses(
    num_views: int, 
    radius: float, 
    height: float, 
    device: str = "cuda",
    center: list = [0.0, 0.0, 0.0],
    tilt: float = 0.0
) -> list:
    """
    创建环绕场景的相机姿态
    
    Args:
        num_views: 视角数量
        radius: 相机到中心点的距离
        height: 相机垂直高度
        device: 计算设备
        center: 中心点坐标
        tilt: 相机倾斜角度（弧度）
        
    Returns:
        相机姿态列表
    """
    camera_poses = []
    center_tensor = torch.tensor(center, device=device)
    
    for i in range(num_views):
        # 计算相机位置（绕z轴旋转）
        angle = i * 2 * np.pi / num_views
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        
        # 应用倾斜角度
        xz_dist = np.sqrt(x**2 + z**2)
        xz_angle = np.arctan2(z, x)
        xz_angle += tilt
        x = xz_dist * np.cos(xz_angle)
        z = xz_dist * np.sin(xz_angle)
        
        # 创建从当前位置看向原点的相机姿态
        camera_pos = torch.tensor([x, y, z], device=device)
        look_at = center_tensor
        up = torch.tensor([0.0, 1.0, 0.0], device=device)
        
        # 计算相机坐标系
        forward = torch.nn.functional.normalize(look_at - camera_pos, dim=0)
        right = torch.nn.functional.normalize(torch.cross(up, forward), dim=0)
        up = torch.cross(forward, right)
        
        # 构建相机姿态矩阵
        camera_pose = torch.stack([right, up, forward, camera_pos], dim=1)
        camera_poses.append(camera_pose)
    
    return camera_poses

def create_path_camera_poses(
    keyframes: list, 
    num_frames: int, 
    device: str = "cuda",
    smooth: bool = True
) -> list:
    """
    从关键帧创建相机路径
    
    Args:
        keyframes: 关键帧相机姿态列表
        num_frames: 总帧数
        device: 计算设备
        smooth: 是否平滑插值
        
    Returns:
        相机姿态列表
    """
    camera_poses = []
    
    # 确保有至少两个关键帧
    if len(keyframes) < 2:
        raise ValueError("至少需要两个关键帧来创建路径")
    
    # 计算每段的帧数
    segment_frames = num_frames // (len(keyframes) - 1)
    extra_frames = num_frames % (len(keyframes) - 1)
    
    # 生成相机路径
    for i in range(len(keyframes) - 1):
        start_pose = keyframes[i]
        end_pose = keyframes[i + 1]
        
        # 这段的帧数
        frames_in_segment = segment_frames + (1 if i < extra_frames else 0)
        
        # 为这段生成相机姿态
        for j in range(frames_in_segment):
            # 线性插值
            t = j / frames_in_segment
            
            # 平滑插值（使用平方正弦函数）
            if smooth:
                t = np.sin(t * np.pi / 2) ** 2
            
            # 分解相机姿态
            start_right = start_pose[:, 0]
            start_up = start_pose[:, 1]
            start_forward = start_pose[:, 2]
            start_pos = start_pose[:, 3]
            
            end_right = end_pose[:, 0]
            end_up = end_pose[:, 1]
            end_forward = end_pose[:, 2]
            end_pos = end_pose[:, 3]
            
            # 插值位置
            pos = start_pos * (1 - t) + end_pos * t
            
            # 插值方向（确保归一化）
            right = torch.nn.functional.normalize(start_right * (1 - t) + end_right * t, dim=0)
            forward = torch.nn.functional.normalize(start_forward * (1 - t) + end_forward * t, dim=0)
            # 重新计算up以确保正交性
            up = torch.cross(forward, right)
            up = torch.nn.functional.normalize(up, dim=0)
            # 确保right垂直于up和forward
            right = torch.cross(up, forward)
            
            # 构建相机姿态
            camera_pose = torch.stack([right, up, forward, pos], dim=1)
            camera_poses.append(camera_pose)
    
    return camera_poses

def main():
    parser = argparse.ArgumentParser(description='3D高斯批量渲染工具')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--output_dir', type=str, default='renders', help='输出目录')
    parser.add_argument('--mode', type=str, default='orbit', choices=['orbit', 'path'], help='相机模式: orbit(环绕), path(路径)')
    parser.add_argument('--height', type=int, default=720, help='输出图像高度')
    parser.add_argument('--width', type=int, default=1280, help='输出图像宽度')
    parser.add_argument('--fov', type=float, default=45.0, help='视场角（度）')
    parser.add_argument('--num_views', type=int, default=36, help='视角数量')
    parser.add_argument('--radius', type=float, default=3.0, help='相机到原点的距离')
    parser.add_argument('--camera_height', type=float, default=0.5, help='相机高度')
    parser.add_argument('--center_x', type=float, default=0.0, help='场景中心X坐标')
    parser.add_argument('--center_y', type=float, default=0.0, help='场景中心Y坐标')
    parser.add_argument('--center_z', type=float, default=0.0, help='场景中心Z坐标')
    parser.add_argument('--tilt', type=float, default=0.0, help='相机倾斜角度（度）')
    parser.add_argument('--step', type=int, default=1, help='渲染间隔（每n帧渲染一次）')
    
    args = parser.parse_args()
    
    # 转换倾斜角度为弧度
    tilt_rad = np.radians(args.tilt)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # 初始化渲染器
        renderer = HeadlessRenderer(args.config, args.checkpoint)
        
        # 创建相机姿态列表
        if args.mode == 'orbit':
            camera_poses = create_orbit_camera_poses(
                num_views=args.num_views,
                radius=args.radius,
                height=args.camera_height,
                device=renderer.device,
                center=[args.center_x, args.center_y, args.center_z],
                tilt=tilt_rad
            )
        elif args.mode == 'path':
            # 这里示范如何创建一个简单的预定义路径
            # 实际使用时，用户可以根据需要修改此处的关键帧定义
            keyframes = [
                # 关键帧1：前视图
                torch.tensor([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.5],
                    [0.0, 0.0, 1.0, -3.0]
                ], device=renderer.device),
                
                # 关键帧2：右侧视图
                torch.tensor([
                    [0.0, 0.0, 1.0, 3.0],
                    [0.0, 1.0, 0.0, 0.5],
                    [-1.0, 0.0, 0.0, 0.0]
                ], device=renderer.device),
                
                # 关键帧3：后视图
                torch.tensor([
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.5],
                    [0.0, 0.0, -1.0, 3.0]
                ], device=renderer.device),
                
                # 关键帧4：左侧视图
                torch.tensor([
                    [0.0, 0.0, -1.0, -3.0],
                    [0.0, 1.0, 0.0, 0.5],
                    [1.0, 0.0, 0.0, 0.0]
                ], device=renderer.device),
                
                # 回到关键帧1
                torch.tensor([
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.5],
                    [0.0, 0.0, 1.0, -3.0]
                ], device=renderer.device),
            ]
            
            camera_poses = create_path_camera_poses(
                keyframes=keyframes,
                num_frames=args.num_views,
                device=renderer.device,
                smooth=True
            )
        
        # 应用渲染间隔
        if args.step > 1:
            camera_poses = camera_poses[::args.step]
            
        # 渲染所有视角
        total_views = len(camera_poses)
        for i, camera_pose in enumerate(camera_poses):
            print(f"正在渲染视角 {i+1}/{total_views} [{i*100/total_views:.1f}%]")
            
            try:
                # 渲染图像
                image = renderer.render_image(
                    height=args.height,
                    width=args.width,
                    fov=args.fov,
                    camera_pose=camera_pose
                )
                
                # 保存图像
                output_path = os.path.join(args.output_dir, f"view_{i:04d}.png")
                renderer.save_image(image, output_path)
            except Exception as e:
                print(f"渲染视角 {i+1} 失败: {e}")
                continue
        
        print(f"批量渲染完成. 已生成 {total_views} 个图像到 {args.output_dir} 目录")
        
        # 生成视频创建命令提示
        print("\n如果您想将图像合成为视频，可以使用以下FFmpeg命令:")
        print(f"ffmpeg -framerate 30 -i {args.output_dir}/view_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 output.mp4")
    
    except Exception as e:
        print(f"批量渲染过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 