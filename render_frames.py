#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os
import argparse
import numpy as np
import torch
import cv2
from tqdm import tqdm
from playground.playground import Playground

def setup_trajectory(playground, trajectory_file=None, num_frames=60, orbit=True, 
                     camera_path=None, output_dir='output_frames'):
    """设置渲染轨迹"""
    
    if trajectory_file:
        # 从文件加载轨迹
        trajectory = np.load(trajectory_file)
        eyes = trajectory['eyes']
        targets = trajectory['targets']
        ups = trajectory['ups']
        
    elif camera_path:
        # 从预定义路径加载
        try:
            path_data = np.load(camera_path)
            eyes = path_data['eyes']
            targets = path_data['targets']
            ups = path_data['ups']
        except:
            print(f"无法加载摄像机路径: {camera_path}，使用默认轨迹")
            return setup_trajectory(playground, num_frames=num_frames, orbit=True, output_dir=output_dir)
    
    elif orbit:
        # 创建绕场景中心的轨迹
        # 获取初始视图
        view_params = playground.initial_view_params
        initial_eye = view_params.get_position()
        initial_target = np.array([0.0, 0.0, 0.0])  # 假设中心是场景原点
        initial_up = np.array([0.0, 1.0, 0.0])
        
        # 计算环绕轨迹
        radius = np.linalg.norm(initial_eye - initial_target)
        eyes = []
        targets = []
        ups = []
        
        for i in range(num_frames):
            angle = 2 * np.pi * i / num_frames
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            eye = np.array([x, initial_eye[1], z]) + initial_target
            eyes.append(eye)
            targets.append(initial_target)
            ups.append(initial_up)
            
        eyes = np.stack(eyes)
        targets = np.stack(targets)
        ups = np.stack(ups)
    
    else:
        # 默认使用初始视图的单个帧
        view_params = playground.initial_view_params
        eye = view_params.get_position()
        target = view_params.get_target()
        up = view_params.get_up()
        
        eyes = np.expand_dims(eye, axis=0)
        targets = np.expand_dims(target, axis=0)
        ups = np.expand_dims(up, axis=0)
    
    # 准备轨迹格式
    trajectory = []
    for eye, target, up in zip(eyes, targets, ups):
        trajectory.append((eye, target, up))
    
    # 设置playground轨迹参数
    playground.trajectory = trajectory
    playground.continuous_trajectory = False
    playground.frames_between_cameras = 1
    playground.trajectory_output_path = os.path.join(output_dir, 'trajectory.mp4')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    return eyes, targets, ups, output_dir

def render_frames(gs_object, output_dir='output_frames', width=1920, height=1080, 
                  trajectory_file=None, camera_path=None, num_frames=60, 
                  mesh_assets_folder=None, default_gs_config='apps/colmap_3dgrt.yaml',
                  quality='high', use_dof=False, focus_z=1.0, spp=64,
                  orbit=True):
    """渲染一系列图像帧"""
    
    # 创建Playground实例
    if mesh_assets_folder is None:
        mesh_assets_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'playground', 'assets')
    
    # 创建playground实例但不显示UI
    playground = Playground(
        gs_object,
        mesh_assets_folder,
        default_gs_config,
        buffer_mode="device2device",
        suppress_ui=True  # 抑制UI初始化
    )
    
    # 设置渲染参数
    playground.window_w = width
    playground.window_h = height
    
    # 设置渲染质量
    if quality == 'high':
        playground.use_spp = True
        playground.spp.spp = spp
        playground.use_optix_denoiser = True
    elif quality == 'medium':
        playground.use_spp = True
        playground.spp.spp = 16
        playground.use_optix_denoiser = True
    else:  # 'fast'
        playground.use_spp = False
        playground.use_optix_denoiser = False
    
    # 设置景深
    playground.use_depth_of_field = use_dof
    if use_dof:
        playground.depth_of_field.focus_z = focus_z
        playground.depth_of_field.aperture_size = 0.1
        playground.depth_of_field.spp = 16
    
    # 设置轨迹
    eyes, targets, ups, output_dir = setup_trajectory(
        playground, 
        trajectory_file, 
        num_frames,
        orbit,
        camera_path,
        output_dir
    )
    
    # 渲染每一帧并保存
    os.makedirs(output_dir, exist_ok=True)
    
    for frame_idx, (eye, target, up) in enumerate(tqdm(zip(eyes, targets, ups), total=len(eyes))):
        # 设置摄像机位置
        playground.set_camera(eye, target, up)
        
        # 渲染图像
        rgb, _ = playground.render_from_camera(eye, target, up, playground.window_w, playground.window_h)
        
        # 继续渲染直到所有渐进效果完成
        while playground.has_progressive_effects_to_render():
            rgb, _ = playground.render_from_camera(eye, target, up, playground.window_w, playground.window_h)
        
        # 处理图像数据并保存
        data = rgb[0].clip(0, 1).detach().cpu().numpy()
        data = (data * 255).astype(np.uint8)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        
        # 保存图像
        output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(output_path, data)
        
        print(f"已保存图像: {output_path}")
    
    print(f"渲染完成。所有图像已保存至: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="3DGRUT Playground非交互式渲染工具")
    
    # 必需参数
    parser.add_argument('--gs_object', type=str, required=True,
                       help="预训练的3DGRT检查点路径，.pt/.ingp/.ply文件")
    
    # 输出配置
    parser.add_argument('--output_dir', type=str, default='output_frames',
                       help="保存渲染图像的目录")
    parser.add_argument('--width', type=int, default=1920,
                       help="输出图像宽度")
    parser.add_argument('--height', type=int, default=1080,
                       help="输出图像高度")
    
    # 轨迹配置
    parser.add_argument('--trajectory_file', type=str, default=None,
                       help="包含eyes, targets, ups的NPZ轨迹文件")
    parser.add_argument('--camera_path', type=str, default=None,
                       help="预定义摄像机路径文件")
    parser.add_argument('--num_frames', type=int, default=60,
                       help="要渲染的帧数(仅在生成轨迹时使用)")
    parser.add_argument('--orbit', action='store_true',
                       help="生成环绕场景中心的轨迹")
    
    # 渲染配置
    parser.add_argument('--mesh_assets', type=str, default=None,
                       help="包含网格资产的文件夹路径(.obj或.glb格式)")
    parser.add_argument('--default_gs_config', type=str, default='apps/colmap_3dgrt.yaml',
                       help="用于.ingp, .ply文件或非3DGRT训练的.pt文件的默认配置名称")
    parser.add_argument('--quality', type=str, choices=['fast', 'medium', 'high'], default='high',
                       help="渲染质量预设")
    parser.add_argument('--use_dof', action='store_true',
                       help="启用景深效果")
    parser.add_argument('--focus_z', type=float, default=1.0,
                       help="景深焦点距离")
    parser.add_argument('--spp', type=int, default=64,
                       help="每像素采样数(仅在高质量模式下使用)")
    
    args = parser.parse_args()
    
    render_frames(
        gs_object=args.gs_object,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        trajectory_file=args.trajectory_file,
        camera_path=args.camera_path,
        num_frames=args.num_frames,
        mesh_assets_folder=args.mesh_assets,
        default_gs_config=args.default_gs_config,
        quality=args.quality,
        use_dof=args.use_dof,
        focus_z=args.focus_z,
        spp=args.spp,
        orbit=args.orbit
    )

if __name__ == "__main__":
    main() 