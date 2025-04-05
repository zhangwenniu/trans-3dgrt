# render_with_glass_sphere.py
import os
import cv2
import torch
import numpy as np
import polyscope as ps
from playground.playground import Playground, OptixPrimitiveTypes, ObjectTransform

def render_scene_with_glass_sphere(
    gs_object_path, 
    output_image_path, 
    sphere_position=[0, 0, 0], 
    sphere_scale=1.0, 
    refractive_index=1.33,
    camera_position=None,
    camera_target=None,
    render_width=1920,
    render_height=1080
):
    """
    渲染带有透明玻璃球的场景并将结果保存为图像
    
    参数:
    - gs_object_path: 预训练模型路径 (.pt, .ingp, .ply)
    - output_image_path: 输出图像保存路径
    - sphere_position: 玻璃球位置 [x, y, z]
    - sphere_scale: 玻璃球缩放比例
    - refractive_index: 折射率（默认为水的折射率1.33）
    - camera_position: 相机位置 [x, y, z]
    - camera_target: 相机目标点 [x, y, z]
    - render_width: 渲染宽度
    - render_height: 渲染高度
    """
    # 初始化polyscope但不显示窗口
    ps.init()
    ps.set_program_name("Headless Renderer")
    ps.set_print_enabled(False)
    ps.set_ground_plane_mode("none")
    ps.set_window_size(render_width, render_height)
    ps.set_automatically_compute_scene_extents(False)
    
    # 初始化Playground
    playground = Playground(
        gs_object=gs_object_path,
        mesh_assets_folder=os.path.join(os.path.dirname(__file__), 'playground', 'assets'),
        default_config='apps/colmap_3dgrt.yaml',
        buffer_mode="device2device"
    )
    
    # 配置渲染参数
    playground.window_w = render_width
    playground.window_h = render_height
    playground.use_optix_denoiser = True
    playground.gamma_correction = 1.0
    playground.max_pbr_bounces = 15
    
    # 修改球体位置和属性
    device = playground.scene_mog.device
    for name, obj in playground.primitives.objects.items():
        if "Sphere" in name:
            # 设置位置
            transform = obj.transform
            transform.reset()
            transform.translate(torch.tensor(sphere_position, device=device))
            transform.scale(sphere_scale)
            
            # 设置折射率
            obj.refractive_index = refractive_index
            obj.refractive_index_tensor[:] = refractive_index
            
            # 更新BVH
            playground.primitives.recompute_stacked_buffers()
            playground.primitives.rebuild_bvh_if_needed(force=True, rebuild=True)
            playground.is_force_canvas_dirty = True
            break
    
    # 设置相机位置
    if camera_position is not None and camera_target is not None:
        camera_up = [0, 1, 0]  # 默认向上方向
        ps.look_at_dir(camera_position, camera_target, camera_up)
    
    # 强制重建BVH
    playground.rebuild_bvh(playground.scene_mog)
    
    # 渲染图像
    with torch.no_grad():
        rgb, _ = playground.render_from_current_ps_view(window_w=render_width, window_h=render_height)
        
        # 确保所有渐进式效果都已渲染完成
        while playground.has_progressive_effects_to_render():
            rgb, _ = playground.render_from_current_ps_view(window_w=render_width, window_h=render_height)
        
        # 保存图像
        data = rgb[0].clip(0, 1).detach().cpu().numpy()
        data = (data * 255).astype(np.uint8)
        data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.imwrite(output_image_path, data)
    
    print(f"图像已保存到 {output_image_path}")
    
    # 清理资源
    ps.shutdown()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="渲染带有透明玻璃球的场景并保存为图像")
    parser.add_argument("--gs_object", required=True, type=str, help="预训练模型路径")
    parser.add_argument("--output", required=True, type=str, help="输出图像保存路径")
    parser.add_argument("--position", nargs=3, type=float, default=[0, 0, 0], help="玻璃球位置 [x y z]")
    parser.add_argument("--scale", type=float, default=1.0, help="玻璃球大小缩放")
    parser.add_argument("--refractive_index", type=float, default=1.33, help="折射率（1.0-2.0）")
    parser.add_argument("--camera_position", nargs=3, type=float, help="相机位置 [x y z]")
    parser.add_argument("--camera_target", nargs=3, type=float, help="相机目标位置 [x y z]")
    parser.add_argument("--width", type=int, default=1920, help="渲染宽度")
    parser.add_argument("--height", type=int, default=1080, help="渲染高度")
    
    args = parser.parse_args()
    
    # 如果未指定相机参数，将使用默认相机视角
    render_scene_with_glass_sphere(
        gs_object_path=args.gs_object,
        output_image_path=args.output,
        sphere_position=args.position,
        sphere_scale=args.scale,
        refractive_index=args.refractive_index,
        camera_position=args.camera_position,
        camera_target=args.camera_target,
        render_width=args.width,
        render_height=args.height
    )