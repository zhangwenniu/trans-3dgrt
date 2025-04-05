import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from gaussians_loader import GaussiansModelLoader
from neus_loader import NeuSModelLoader


CHECKPOINT_PATH = "../runs/eiko_ball_masked_expanded_3dgrt/eiko_ball_masked_expanded-2703_050608/ckpt_last.pt"

def test_load_gaussians():
    """测试加载3D高斯模型"""
    try:
        # 这里需要根据实际情况替换checkpoint路径
        checkpoint_path = CHECKPOINT_PATH
        
        print("加载3D高斯模型...")
        gaussians_loader = GaussiansModelLoader(checkpoint_path=checkpoint_path)
        
        print("获取高斯模型...")
        model = gaussians_loader.get_model()
        
        if model is not None:
            print("高斯模型加载成功")
            
            # 获取模型中的高斯数量
            if hasattr(model, 'num_gaussians'):
                print(f"高斯数量: {model.num_gaussians}")
            elif hasattr(model, 'means'):
                print(f"高斯数量: {len(model.means)}")
            
        return True
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
        return False

def test_gaussians_rendering():
    """测试3D高斯模型的渲染功能"""
    try:
        # 这里需要根据实际情况替换checkpoint路径
        checkpoint_path = CHECKPOINT_PATH
        
        print("加载3D高斯模型...")
        gaussians_loader = GaussiansModelLoader(checkpoint_path=checkpoint_path)
        
        # 创建测试光线
        n_rays = 5
        print(f"创建{n_rays}条测试光线...")
        
        # 创建包含必要属性的批处理对象
        class BatchObject:
            pass
        
        # 初始化批处理对象
        batch = BatchObject()
        
        # 设置相机外参矩阵：从相机到世界坐标系的变换
        # 假设相机在z轴负方向上，看向原点
        batch.T_to_world = torch.eye(4, dtype=torch.float32, device='cuda')
        batch.T_to_world[2, 3] = -5.0  # 相机位置在z=-5
        batch.T_to_world = batch.T_to_world[:3, :4].unsqueeze(0)  # 形状为[1, 3, 4]
        
        # 设置相机内参
        batch.intrinsics = [1111.11, 1111.11, 400.0, 400.0]
        
        # 创建图像分辨率
        H, W = 32, 32  # 使用小分辨率进行测试
        
        # 创建从不同位置向场景发射的光线
        # 注意：这些光线是在世界坐标系下
        batch.rays_ori = torch.zeros((1, H, W, 3), device='cuda')
        batch.rays_dir = torch.zeros((1, H, W, 3), device='cuda')
        
        # 生成图像平面上的网格
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device='cuda'),
            torch.linspace(-1, 1, W, device='cuda'),
            indexing='ij'
        )
        
        # 计算每个像素的光线方向（相机坐标系）
        fx, fy, cx, cy = batch.intrinsics
        camera_dirs = torch.stack([
            (x - cx) / fx,
            -(y - cy) / fy,  # 注意：y方向需要取反
            -torch.ones_like(x)  # z方向指向相机前方
        ], dim=-1)
        
        # 归一化方向
        camera_dirs = F.normalize(camera_dirs, p=2, dim=-1)
        
        # 将相机坐标系的光线方向转换到世界坐标系
        # 只需要应用旋转部分，不需要平移
        rotation = batch.T_to_world[0, :, :3]  # 形状为[3, 3]
        
        # 对每个像素进行转换
        for h in range(H):
            for w in range(W):
                cam_dir = camera_dirs[h, w]
                # 应用旋转转换到世界坐标系
                world_dir = torch.matmul(rotation, cam_dir)
                # 归一化
                world_dir = F.normalize(world_dir, p=2, dim=0)
                # 存储结果
                batch.rays_dir[0, h, w] = world_dir
        
        # 设置光线原点（所有像素共用同一个原点，即相机中心）
        camera_pos = batch.T_to_world[0, :, 3]  # 相机在世界坐标系中的位置
        batch.rays_ori[0, :, :] = camera_pos.view(1, 1, 3).expand(H, W, 3)
        
        # 添加rgb_gt，虽然只是用于测试
        batch.rgb_gt = torch.zeros((1, H, W, 3), device='cuda')
        
        print("渲染高斯场景...")
        outputs = gaussians_loader.render(batch)
        
        if 'pred_rgb' in outputs:
            print("渲染成功")
            print(f"渲染RGB形状: {outputs['pred_rgb'].shape}")
            print(f"渲染RGB值 (前5个像素):\n{outputs['pred_rgb'][0, :5, 0]}")
            
            if 'pred_dist' in outputs:
                print(f"渲染深度形状: {outputs['pred_dist'].shape}")
                print(f"渲染深度值 (前5个像素):\n{outputs['pred_dist'][0, :5, 0]}")
            
            if 'pred_normals' in outputs:
                print(f"渲染法线形状: {outputs['pred_normals'].shape}")
                print(f"渲染法线值 (前5个像素):\n{outputs['pred_normals'][0, :5, 0]}")
                
        return True
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
        return False

def test_coordinate_transform():
    """测试坐标系转换"""
    try:
        # 加载NeuS模型以获取坐标转换函数
        print("加载NeuS模型...")
        neus_loader = NeuSModelLoader()
        
        # 设置坐标变换矩阵
        # 假设NeuS模型的局部坐标是在半径为1的单位球内
        # 而世界坐标系要大得多
        scale = 10.0  # 假设世界坐标比NeuS局部坐标大10倍
        translation = torch.tensor([0.0, 0.0, 0.0], device='cuda')  # 假设两个坐标系的原点是重合的
        
        # 创建4x4变换矩阵
        transform_mat = torch.eye(4, dtype=torch.float32, device='cuda')
        transform_mat[0, 0] = transform_mat[1, 1] = transform_mat[2, 2] = scale
        transform_mat[:3, 3] = translation
        
        # 设置NeuS的坐标变换矩阵
        neus_loader.set_scale_mat(transform_mat)
        print(f"设置坐标变换矩阵:\n{transform_mat}")
        
        # 创建测试点
        world_points = torch.tensor([
            [5.0, 0.0, 0.0],  # 世界坐标中在x=5的位置
            [0.0, 5.0, 0.0],  # 世界坐标中在y=5的位置
            [0.0, 0.0, 5.0],  # 世界坐标中在z=5的位置
            [0.0, 0.0, 0.0],  # 世界坐标中的原点
        ], device='cuda')
        
        print("世界坐标点:")
        print(world_points)
        
        # 将世界坐标转换为NeuS局部坐标
        neus_points = neus_loader.to_neus_from_world(world_points)
        print("转换为NeuS局部坐标:")
        print(neus_points)
        
        # 将局部坐标转换回世界坐标
        recovered_points = neus_loader.to_world_from_neus(neus_points)
        print("转换回世界坐标:")
        print(recovered_points)
        
        # 验证转换的准确性
        error = torch.abs(world_points - recovered_points).max()
        print(f"转换误差: {error}")
        assert error < 1e-5, "坐标转换误差过大"
        
        # 测试折射光线的转换
        # 1. 假设我们有NeuS局部坐标系下的折射光线
        neus_ray_origins = torch.tensor([
            [0.5, 0.0, 0.0],  # 局部坐标中在x=0.5的位置
            [0.0, 0.5, 0.0],  # 局部坐标中在y=0.5的位置
            [0.0, 0.0, 0.5],  # 局部坐标中在z=0.5的位置
        ], device='cuda')
        
        neus_ray_directions = torch.tensor([
            [1.0, 0.0, 0.0],  # 沿x轴正方向
            [0.0, 1.0, 0.0],  # 沿y轴正方向
            [0.0, 0.0, 1.0],  # 沿z轴正方向
        ], device='cuda')
        neus_ray_directions = F.normalize(neus_ray_directions, p=2, dim=1)
        
        print("NeuS局部坐标下的光线起点:")
        print(neus_ray_origins)
        print("NeuS局部坐标下的光线方向:")
        print(neus_ray_directions)
        
        # 2. 将它们转换到世界坐标系
        world_ray_origins = neus_loader.to_world_from_neus(neus_ray_origins)
        # 方向向量需要特殊处理：只需要应用旋转，不需要平移
        world_ray_directions = neus_loader.to_world_direction_from_neus(neus_ray_directions)
        
        print("转换到世界坐标下的光线起点:")
        print(world_ray_origins)
        print("转换到世界坐标下的光线方向:")
        print(world_ray_directions)
        
        # 3. 将世界坐标系的光线转换回局部坐标系
        recovered_neus_origins = neus_loader.to_neus_from_world(world_ray_origins)
        recovered_neus_directions = neus_loader.to_neus_direction_from_world(world_ray_directions)
        
        print("转换回NeuS局部坐标的光线起点:")
        print(recovered_neus_origins)
        print("转换回NeuS局部坐标的光线方向:")
        print(recovered_neus_directions)
        
        # 验证转换的准确性
        origin_error = torch.abs(neus_ray_origins - recovered_neus_origins).max()
        direction_error = torch.abs(neus_ray_directions - recovered_neus_directions).max()
        
        print(f"光线起点转换误差: {origin_error}")
        print(f"光线方向转换误差: {direction_error}")
        
        assert origin_error < 1e-5, "光线起点坐标转换误差过大"
        assert direction_error < 1e-5, "光线方向坐标转换误差过大"
        
        return True
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
        return False

def test_refraction_with_coordinate_transform():
    """测试带坐标转换的折射"""
    try:
        # 加载NeuS模型
        print("加载NeuS模型...")
        neus_loader = NeuSModelLoader()
        
        # 设置坐标变换矩阵，使NeuS模型与世界坐标系对齐
        scale = 10.0
        transform_mat = torch.eye(4, dtype=torch.float32, device='cuda')
        transform_mat[0, 0] = transform_mat[1, 1] = transform_mat[2, 2] = scale
        neus_loader.set_scale_mat(transform_mat)
        
        # 创建世界坐标系下的测试光线
        n_rays = 5
        print(f"创建{n_rays}条世界坐标系下的测试光线...")
        
        # 世界坐标下的光线起点（距离较远）
        world_origins = torch.zeros((n_rays, 3), device='cuda')
        world_origins[:, 0] = torch.linspace(-3.0, 3.0, n_rays)  # x轴分布
        world_origins[:, 2] = -15.0  # 从z轴负方向射入
        
        # 世界坐标下的光线方向（指向原点）
        world_directions = torch.zeros((n_rays, 3), device='cuda')
        for i in range(n_rays):
            direction = -world_origins[i]  # 指向原点
            world_directions[i] = F.normalize(direction, p=2, dim=0)
        
        # 1. 将世界坐标系下的光线转换到NeuS局部坐标系
        print("将世界坐标系光线转换到NeuS局部坐标系...")
        neus_origins = neus_loader.to_neus_from_world(world_origins)
        neus_directions = neus_loader.to_neus_direction_from_world(world_directions)
        
        print("世界坐标系光线起点:")
        print(world_origins)
        print("转换到NeuS局部坐标系:")
        print(neus_origins)
        
        print("世界坐标系光线方向:")
        print(world_directions)
        print("转换到NeuS局部坐标系:")
        print(neus_directions)
        
        # 2. 在NeuS局部坐标系下计算折射
        print("在NeuS局部坐标系下计算折射...")
        # 计算近远平面
        near, far = neus_loader.near_far_from_sphere(neus_origins, neus_directions)
        print(f"近平面: {near}")
        print(f"远平面: {far}")
        
        # 计算折射
        refracted_neus_origins, refracted_neus_directions, valid_mask = neus_loader.compute_refractive_rays(
            neus_origins, neus_directions
        )
        
        print(f"有效折射数量: {valid_mask.sum().item()}/{n_rays}")
        if valid_mask.sum() > 0:
            print("NeuS坐标系下折射光线起点:")
            print(refracted_neus_origins[valid_mask])
            print("NeuS坐标系下折射光线方向:")
            print(refracted_neus_directions[valid_mask])
        
        # 3. 将折射后的光线从NeuS局部坐标系转换回世界坐标系
        print("将折射光线转换回世界坐标系...")
        refracted_world_origins = neus_loader.to_world_from_neus(refracted_neus_origins)
        refracted_world_directions = neus_loader.to_world_direction_from_neus(refracted_neus_directions)
        
        if valid_mask.sum() > 0:
            print("世界坐标系下折射光线起点:")
            print(refracted_world_origins[valid_mask])
            print("世界坐标系下折射光线方向:")
            print(refracted_world_directions[valid_mask])
        
        # 4. 验证：再转换回局部坐标系，应该与原始折射光线一致
        print("验证坐标转换的一致性...")
        recovered_neus_origins = neus_loader.to_neus_from_world(refracted_world_origins)
        recovered_neus_directions = neus_loader.to_neus_direction_from_world(refracted_world_directions)
        
        if valid_mask.sum() > 0:
            origin_error = torch.abs(refracted_neus_origins[valid_mask] - recovered_neus_origins[valid_mask]).max()
            direction_error = torch.abs(refracted_neus_directions[valid_mask] - recovered_neus_directions[valid_mask]).max()
            
            print(f"折射光线起点转换误差: {origin_error}")
            print(f"折射光线方向转换误差: {direction_error}")
            
            assert origin_error < 1e-5, "折射光线起点坐标转换误差过大"
            assert direction_error < 1e-5, "折射光线方向坐标转换误差过大"
        
        return True
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 注意：需要更新正确的checkpoint路径才能运行这些测试
    print("=== 测试加载3D高斯模型 ===")
    test_load_gaussians()
    
    print("\n=== 测试高斯渲染 ===")
    test_gaussians_rendering()
    
    print("\n=== 测试坐标转换 ===")
    test_coordinate_transform()
    
    print("\n=== 测试带坐标转换的折射 ===")
    test_refraction_with_coordinate_transform() 