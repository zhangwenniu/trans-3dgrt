import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from neus_loader import NeuSModelLoader
import torch.nn.functional as F

def test_sdf_values():
    """测试SDF值计算"""
    # 加载NeuS模型
    neus_loader = NeuSModelLoader()
    
    # 创建测试点
    x = torch.linspace(-1, 1, 100, device='cuda')
    y = torch.zeros_like(x)
    z = torch.zeros_like(x)
    points = torch.stack([x, y, z], dim=1)
    
    # 计算SDF值
    sdf_values = neus_loader.get_sdf_value(points)
    
    # 打印SDF值
    print(f"SDF值范围: [{sdf_values.min().item()}, {sdf_values.max().item()}]")
    
    # 绘制SDF值
    plt.figure(figsize=(10, 5))
    plt.plot(x.cpu().numpy(), sdf_values.cpu().numpy(), label='SDF Values')
    plt.axhline(y=0, color='r', linestyle='-', label='Zero Line')
    plt.title('SDF Values along X-axis')
    plt.xlabel('X')
    plt.ylabel('SDF')
    plt.grid(True)
    plt.legend()
    
    # 保存图像
    os.makedirs("./temp", exist_ok=True)
    plt.savefig("./temp/sdf_values.png")
    print("SDF值可视化已保存到 ./temp/sdf_values.png")

def test_intersection():
    """测试光线与SDF表面的交点计算"""
    try:
        # 加载NeuS模型
        print("加载NeuS模型...")
        neus_loader = NeuSModelLoader()
        
        # 创建更简单的测试光线（从更远的距离发射，确保与物体相交）
        n_rays = 5
        print(f"创建{n_rays}条测试光线...")
        
        # 创建从不同位置向原点方向发射的光线
        origins = torch.zeros((n_rays, 3), device='cuda')
        origins[:, 0] = torch.linspace(-0.5, 0.5, n_rays)
        origins[:, 2] = -2.0  # 从z轴负方向射入
        
        # 所有光线都指向原点
        directions = torch.zeros((n_rays, 3), device='cuda')
        for i in range(n_rays):
            # 计算从origins[i]指向原点(0,0,0)的方向
            direction = -origins[i]  # 原点减去起点
            directions[i] = F.normalize(direction, p=2, dim=0)  # 归一化
        
        print("计算交点...")
        # 计算交点
        intersection_points, intersection_normals, valid_mask = neus_loader.compute_intersection(origins, directions)
        
        # 打印结果
        print(f"有效交点数量: {valid_mask.sum().item()}/{n_rays}")
        print(f"有效交点掩码: {valid_mask}")
        
        if valid_mask.sum() > 0:
            print("\n交点位置:")
            print(intersection_points[valid_mask])
            print("\n法向量:")
            print(intersection_normals[valid_mask])
            
            # 计算交点到原点的距离，验证交点确实在SDF表面附近
            distances = torch.norm(intersection_points[valid_mask], dim=1)
            print("\n交点到原点的距离:")
            print(distances)
            
            # 验证法向量是否为单位向量
            normal_lengths = torch.norm(intersection_normals[valid_mask], dim=1)
            print("\n法向量长度 (应接近1):")
            print(normal_lengths)
            
        return True
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
        return False

def test_refraction():
    """测试折射计算"""
    try:
        # 加载NeuS模型
        print("加载NeuS模型...")
        neus_loader = NeuSModelLoader()
        
        # 为了测试方便，明确设置几何对象的边界球半径
        neus_loader.object_bounding_sphere = 0.7
        
        # 创建更简单的测试光线（从更远的距离发射，确保与物体相交）
        n_rays = 5
        print(f"创建{n_rays}条测试光线...")
        
        # 创建从不同位置向原点方向发射的光线
        origins = torch.zeros((n_rays, 3), device='cuda')
        origins[:, 0] = torch.linspace(-0.3, 0.3, n_rays)  # 小范围，增加相交概率
        origins[:, 2] = -1.5  # 从z轴负方向射入
        
        # 所有光线都指向原点
        directions = torch.zeros((n_rays, 3), device='cuda')
        for i in range(n_rays):
            # 计算从origins[i]指向原点(0,0,0)的方向
            direction = -origins[i]  # 原点减去起点
            directions[i] = F.normalize(direction, p=2, dim=0)  # 归一化
        
        # 测试近远平面计算
        print("计算近远平面...")
        near, far = neus_loader.near_far_from_sphere(origins, directions)
        if near.max() > 0:
            print(f"至少有一条光线与包围球相交，近平面: {near}")
        else:
            print("警告: 没有光线与包围球相交")
            
        print("计算折射...")
        # 计算折射
        refracted_origins, refracted_directions, valid_mask = neus_loader.compute_refractive_rays(origins, directions)
        
        # 打印结果
        print(f"有效折射数量: {valid_mask.sum().item()}/{n_rays}")
        print(f"有效折射掩码: {valid_mask}")
        
        if valid_mask.sum() > 0:
            print("\n折射光线起点:")
            print(refracted_origins[valid_mask])
            print("\n折射光线方向:")
            print(refracted_directions[valid_mask])
            
            # 计算折射方向的长度，验证它们是单位向量
            direction_lengths = torch.norm(refracted_directions[valid_mask], dim=1)
            print("\n折射方向长度 (应接近1):")
            print(direction_lengths)
            
        return True
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
        return False

def test_full_refraction():
    """测试完整折射路径计算"""
    try:
        # 加载NeuS模型
        print("加载NeuS模型...")
        neus_loader = NeuSModelLoader()
        
        # 为了测试方便，明确设置几何对象的边界球半径
        neus_loader.object_bounding_sphere = 0.7
        
        # 创建更简单的测试光线（从更远的距离发射，确保与物体相交）
        n_rays = 5
        print(f"创建{n_rays}条测试光线...")
        
        # 创建从不同位置向原点方向发射的光线
        origins = torch.zeros((n_rays, 3), device='cuda')
        origins[:, 0] = torch.linspace(-0.3, 0.3, n_rays)  # 小范围，增加相交概率
        origins[:, 2] = -1.5  # 从z轴负方向射入
        
        # 所有光线都指向原点
        directions = torch.zeros((n_rays, 3), device='cuda')
        for i in range(n_rays):
            # 计算从origins[i]指向原点(0,0,0)的方向
            direction = -origins[i]  # 原点减去起点
            directions[i] = F.normalize(direction, p=2, dim=0)  # 归一化
        
        # 测试近远平面计算
        print("计算近远平面...")
        near, far = neus_loader.near_far_from_sphere(origins, directions)
        if near.max() > 0:
            print(f"至少有一条光线与包围球相交，近平面: {near}")
        else:
            print("警告: 没有光线与包围球相交")
        
        print("计算完整折射路径...")
        # 计算完整折射路径
        exit_origins, exit_directions, valid_mask = neus_loader.compute_full_refractive_path(origins, directions)
        
        # 打印结果
        print(f"有效完整折射数量: {valid_mask.sum().item()}/{n_rays}")
        print(f"有效折射掩码: {valid_mask}")
        
        if valid_mask.sum() > 0:
            print("\n出射光线起点:")
            print(exit_origins[valid_mask])
            print("\n出射光线方向:")
            print(exit_directions[valid_mask])
            
            # 计算出射方向的长度，验证它们是单位向量
            direction_lengths = torch.norm(exit_directions[valid_mask], dim=1)
            print("\n出射方向长度 (应接近1):")
            print(direction_lengths)
            
        return True
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
        return False

def setup_neus_with_scale():
    """设置带有坐标缩放的NeuS模型"""
    # 加载NeuS模型
    print("加载NeuS模型...")
    neus_loader = NeuSModelLoader()
    
    # 创建自定义的缩放矩阵
    scale = 0.5  # 将模型缩小一半
    translation = torch.tensor([0.2, 0.1, -0.1], device='cuda')  # 添加一些平移
    
    # 创建4x4缩放矩阵
    scale_mat = torch.eye(4, dtype=torch.float32, device='cuda')
    scale_mat[0, 0] = scale_mat[1, 1] = scale_mat[2, 2] = scale
    scale_mat[:3, 3] = translation
    
    # 设置缩放矩阵
    neus_loader.set_scale_mat(scale_mat)
    print(f"已设置缩放矩阵:\n{scale_mat}")
    
    return neus_loader

def test_coordinate_transforms():
    """测试坐标转换功能"""
    try:
        # 设置带有缩放的NeuS模型
        neus_loader = setup_neus_with_scale()
        
        # 创建测试点
        points = torch.tensor([
            [1.0, 0.0, 0.0],  # 单位x轴
            [0.0, 1.0, 0.0],  # 单位y轴
            [0.0, 0.0, 1.0],  # 单位z轴
            [0.0, 0.0, 0.0],  # 原点
        ], device='cuda')
        
        # 将点从世界坐标转换到局部坐标
        neus_points = neus_loader.to_neus_from_world(points)
        print("世界坐标点:")
        print(points)
        print("转换到NeuS局部坐标:")
        print(neus_points)
        
        # 将点从局部坐标转换回世界坐标
        world_points = neus_loader.to_world_from_neus(neus_points)
        print("转换回世界坐标:")
        print(world_points)
        
        # 计算误差
        error = torch.abs(points - world_points).max()
        print(f"转换误差: {error}")
        assert error < 1e-5, "坐标转换误差过大"
        
        return True
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
        return False

def test_scale_intersection():
    """测试带有缩放的交点计算"""
    try:
        # 设置带有缩放的NeuS模型
        neus_loader = setup_neus_with_scale()
        
        # 创建测试光线
        n_rays = 5
        print(f"创建{n_rays}条测试光线...")
        
        # 创建从不同位置向原点方向发射的光线
        origins = torch.zeros((n_rays, 3), device='cuda')
        origins[:, 0] = torch.linspace(-0.5, 0.5, n_rays)
        origins[:, 2] = -2.0  # 从z轴负方向射入
        
        # 所有光线都指向模型坐标中的原点(考虑平移)
        directions = torch.zeros((n_rays, 3), device='cuda')
        model_origin = neus_loader.scale_mat[:3, 3]  # 模型中心在世界坐标中的位置
        
        for i in range(n_rays):
            # 计算从origins[i]指向模型中心的方向
            direction = model_origin - origins[i]
            directions[i] = F.normalize(direction, p=2, dim=0)  # 归一化
        
        # 测试近远平面计算
        print("计算近远平面...")
        near, far = neus_loader.near_far_from_sphere(origins, directions)
        print(f"近平面值: {near}")
        print(f"远平面值: {far}")
        
        print("计算交点...")
        # 计算交点，显式传入near和far
        intersection_points, intersection_normals, valid_mask = neus_loader.compute_intersection(
            origins, directions, near, far
        )
        
        # 打印结果
        print(f"有效交点数量: {valid_mask.sum().item()}/{n_rays}")
        print(f"有效交点掩码: {valid_mask}")
        
        if valid_mask.sum() > 0:
            print("\n交点位置:")
            print(intersection_points[valid_mask])
            print("\n法向量:")
            print(intersection_normals[valid_mask])
            
            # 计算交点到模型中心的距离
            distances = torch.norm(intersection_points[valid_mask] - model_origin, dim=1)
            print("\n交点到模型中心的距离:")
            print(distances)
            
        return True
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 测试SDF值计算 ===")
    test_sdf_values()
    
    print("\n=== 测试坐标转换 ===")
    test_coordinate_transforms()
    
    print("\n=== 测试光线交点计算 ===")
    test_intersection()
    
    print("\n=== 测试带缩放的交点计算 ===")
    test_scale_intersection()
    
    print("\n=== 测试折射计算 ===")
    test_refraction()
    
    print("\n=== 测试完整折射路径计算 ===")
    test_full_refraction() 