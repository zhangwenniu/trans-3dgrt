import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh  # 添加trimesh导入
import logging
import json
from PIL import Image
from gaussians_loader import GaussiansModelLoader
from neus_loader import NeuSModelLoader

# 设置日志
logger = logging.getLogger(__name__)

class RefractiveRenderer(nn.Module):
    """
    折射渲染器：结合NeuS模型的折射和3DGS背景渲染
    """
    def __init__(
        self,
        neus_model_path,
        gaussians_model_path,
        scene_path=None,  # 添加场景路径参数
        n1=1.0003,  # 空气的折射率
        n2=1.5,     # 玻璃/透明物体的折射率
        device=None,
        auto_scale=True  # 是否自动计算缩放矩阵
    ):
        """
        初始化折射渲染器

        Args:
            neus_model_path: NeuS模型的路径
            gaussians_model_path: 3D高斯模型的路径
            scene_path: 场景数据路径，用于加载稀疏点云
            n1: 外部介质的折射率（通常是空气）
            n2: 透明物体的折射率
            device: 运行设备
            auto_scale: 是否自动计算NeuS模型与世界坐标系的变换关系
        """
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载NeuS模型
        print(f"加载NeuS模型：{neus_model_path}")
        self.neus_loader = NeuSModelLoader(checkpoint_path=neus_model_path)
        
        # 加载3D高斯模型
        print(f"加载3D高斯模型：{gaussians_model_path}")
        self.gaussians_loader = GaussiansModelLoader(checkpoint_path=gaussians_model_path)
        
        # 设置折射率
        self.n1 = n1
        self.n2 = n2
        
        # 场景路径
        self.scene_path = scene_path
        
        # 自动计算坐标变换矩阵
        if auto_scale and scene_path is not None:
            self.compute_and_set_scale_matrix_from_ply()
        
        print("折射渲染器初始化完成")
    
    def compute_and_set_scale_matrix_from_ply(self):
        """
        从sparse_points_interest.ply文件计算并设置坐标变换矩阵
        """
        try:
            # 构建ply文件路径
            ply_path = os.path.join(self.scene_path, 'sparse_points_interest.ply')
            
            if not os.path.exists(ply_path):
                logger.warning(f"兴趣区域PLY文件未找到: {ply_path}")
                print(f"警告: 未找到PLY文件 {ply_path}，无法自动计算坐标变换")
                return
            
            print(f"从文件加载稀疏点云: {ply_path}")
            # 加载点云
            pcd = trimesh.load(ply_path)
            vertices = torch.from_numpy(pcd.vertices).float().to(self.device)
            
            # 计算边界框和中心
            bbox_max = torch.max(vertices, dim=0).values
            bbox_min = torch.min(vertices, dim=0).values
            center = (bbox_max + bbox_min) * 0.5
            
            # 计算包围球半径
            radius = torch.norm(vertices - center.unsqueeze(0), p=2, dim=-1).max()
            
            print(f"点云中心: {center.cpu().numpy()}")
            print(f"点云边界半径: {radius.item()}")
            
            # 创建变换矩阵 (世界坐标系 -> NeuS局部坐标系)
            # NeuS模型假设对象位于半径为1的单位球内
            # 因此我们需要缩放和平移场景点云
            scale_factor = 0.9 / radius.item()  # 略小于1以确保所有点都在单位球内
            
            # 创建变换矩阵
            scale_mat = torch.eye(4, dtype=torch.float32, device=self.device)
            scale_mat[0, 0] = scale_mat[1, 1] = scale_mat[2, 2] = scale_factor
            scale_mat[:3, 3] = -center * scale_factor  # 平移到原点
            
            # 求逆得到 NeuS局部坐标系 -> 世界坐标系 的变换
            world_to_neus_mat = scale_mat
            neus_to_world_mat = torch.inverse(world_to_neus_mat)
            
            # 设置变换矩阵
            self.set_scale_mat(neus_to_world_mat)
            print(f"自动计算的坐标变换矩阵 (NeuS -> 世界):\n{neus_to_world_mat}")
            
            return neus_to_world_mat
            
        except Exception as e:
            logger.error(f"计算坐标变换矩阵时出错: {e}")
            print(f"错误: 计算坐标变换矩阵失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def set_scale_mat(self, scale_mat):
        """
        设置NeuS模型与世界坐标系之间的变换矩阵

        Args:
            scale_mat: 4x4变换矩阵
        """
        self.neus_loader.set_scale_mat(scale_mat)
        print(f"设置坐标变换矩阵:\n{scale_mat}")
    
    def render(self, batch, render_refractive=True):
        """
        执行渲染 (优化版本，使用批量折射计算)

        Args:
            batch: 包含必要信息的批处理对象
            render_refractive: 是否渲染折射效果，如果为False则只渲染正常的3D高斯
            
        Returns:
            outputs: 渲染结果
        """
        # 首先，使用3D高斯模型渲染原始场景（无折射）
        normal_outputs = self.gaussians_loader.render(batch)
        
        if not render_refractive:
            return normal_outputs
        
        # 获取相机光线
        camera_rays_ori = batch.rays_ori  # [B, H, W, 3]
        camera_rays_dir = batch.rays_dir  # [B, H, W, 3]
        
        # 获取批处理大小和图像尺寸
        B, H, W, _ = camera_rays_ori.shape
        
        # 为每个相机光线创建结果存储张量
        # 最终颜色
        final_colors = normal_outputs['pred_rgb'].clone()  # 初始化为普通渲染结果
        # 是否经过折射（对于判断哪些像素展示折射结果很有用）
        is_refracted = torch.zeros((B, H, W, 1), dtype=torch.bool, device=self.device)
        
        # 处理每个批次
        for b in range(B):
            # 提取当前批次的光线并展平
            rays_ori = camera_rays_ori[b].reshape(-1, 3)  # [H*W, 3]
            rays_dir = camera_rays_dir[b].reshape(-1, 3)  # [H*W, 3]
            
            # 将世界坐标系的光线转换到NeuS局部坐标系
            neus_rays_ori = self.neus_loader.to_neus_from_world(rays_ori)
            neus_rays_dir = self.neus_loader.to_neus_direction_from_world(rays_dir)
            
            # 内存优化：将大分辨率图像的光线分成更小的批次处理
            pixels_total = H * W
            batch_size = 10000  # 每次处理的光线数量，调整以适应GPU内存
            
            # 创建存储最终结果的张量
            temp_exit_world_origins = torch.zeros_like(rays_ori)
            temp_exit_world_directions = torch.zeros_like(rays_dir)
            temp_valid_refraction = torch.zeros(pixels_total, dtype=torch.bool, device=self.device)
            
            # 分批处理所有光线
            for i in range(0, pixels_total, batch_size):
                end_i = min(i + batch_size, pixels_total)
                current_batch_size = end_i - i
                
                # 使用批量折射计算获取完整折射路径（使用优化后的批量版本）
                # 注意：传递max_batch_size参数以进一步优化内存使用
                exit_neus_origins_batch, exit_neus_directions_batch, valid_refraction_batch = self.neus_loader.compute_batch_full_refractive_path(
                    neus_rays_ori[i:end_i], neus_rays_dir[i:end_i], 
                    self.n1, self.n2,
                    max_batch_size=5000  # 更小的内部批次大小
                )
                
                # 将折射出射光线转换回世界坐标系
                world_origins_batch = self.neus_loader.to_world_from_neus(exit_neus_origins_batch)
                world_directions_batch = self.neus_loader.to_world_direction_from_neus(exit_neus_directions_batch)
                
                # 存储批次结果
                temp_exit_world_origins[i:end_i] = world_origins_batch
                temp_exit_world_directions[i:end_i] = world_directions_batch
                temp_valid_refraction[i:end_i] = valid_refraction_batch
                
                # 手动释放内存
                torch.cuda.empty_cache()
            
            # 记录哪些光线经过折射
            valid_mask_reshaped = temp_valid_refraction.reshape(H, W)
            is_refracted[b, :, :, 0] = valid_mask_reshaped
            
            if temp_valid_refraction.sum() > 0:
                # 创建折射批次对象，仅包含有效折射光线
                refracted_batch = self._create_efficient_refracted_batch(
                    batch, b, temp_exit_world_origins, temp_exit_world_directions, temp_valid_refraction, H, W
                )
                
                # 使用折射光线执行3D高斯渲染
                refracted_outputs = self.gaussians_loader.render(refracted_batch)
                
                # 获取折射渲染结果并应用到最终颜色
                refracted_rgb = refracted_outputs['pred_rgb'][0]  # [H, W, 3]
                
                # 只更新有折射的像素
                final_colors[b, valid_mask_reshaped] = refracted_rgb[valid_mask_reshaped]
        
        # 构建输出字典
        outputs = {
            'pred_rgb': final_colors,
            'is_refracted': is_refracted
        }
        
        # 复制来自normal_outputs的其他输出
        for key in normal_outputs:
            if key != 'pred_rgb' and key not in outputs:
                outputs[key] = normal_outputs[key]
        
        return outputs
    
    def _create_efficient_refracted_batch(self, original_batch, batch_idx, refracted_origins, refracted_directions, valid_mask, H, W):
        """
        创建用于折射渲染的批处理对象（优化版本，只包含有效折射光线）

        Args:
            original_batch: 原始批处理对象
            batch_idx: 当前处理的批次索引
            refracted_origins: 折射光线起点 [H*W, 3]
            refracted_directions: 折射光线方向 [H*W, 3]
            valid_mask: 有效折射掩码 [H*W]
            H, W: 图像高度和宽度
            
        Returns:
            refracted_batch: 包含折射光线的批处理对象
        """
        # 创建新的批处理对象
        class BatchObject:
            pass
        
        refracted_batch = BatchObject()
        
        # 复制原始批处理的属性
        refracted_batch.T_to_world = original_batch.T_to_world[batch_idx:batch_idx+1].clone()
        refracted_batch.intrinsics = original_batch.intrinsics.copy() if hasattr(original_batch, 'intrinsics') else None
        
        # 创建折射光线数组
        reshaped_origins = refracted_origins.reshape(H, W, 3)
        reshaped_directions = refracted_directions.reshape(H, W, 3)
        
        # 设置光线
        refracted_batch.rays_ori = reshaped_origins.unsqueeze(0)  # [1, H, W, 3]
        refracted_batch.rays_dir = reshaped_directions.unsqueeze(0)  # [1, H, W, 3]
        
        # 添加其他必要属性
        refracted_batch.rgb_gt = torch.zeros((1, H, W, 3), device=self.device)
        
        return refracted_batch

    def load_camera_data(self, view_index=0):
        """
        从场景数据加载相机位姿数据
        
        Args:
            view_index: 要加载的视图索引
            
        Returns:
            camera_data: 相机数据字典，包含内参、外参等信息
        """
        if not self.scene_path:
            raise ValueError("未设置场景路径，无法加载相机数据")
        
        # 加载相机位姿数据
        try:
            # 方法1: 尝试加载COLMAP格式的数据
            # 尝试读取COLMAP的相机参数
            try:
                # 首先尝试读取二进制格式
                sparse_dir = os.path.join(self.scene_path, "sparse/0")
                cameras_extrinsic_file = os.path.join(sparse_dir, "images.bin")
                cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.bin")
                if os.path.exists(cameras_extrinsic_file) and os.path.exists(cameras_intrinsic_file):
                    # 导入COLMAP读取函数
                    from threedgrut.datasets.utils import (
                        read_colmap_extrinsics_binary,
                        read_colmap_intrinsics_binary,
                        qvec_to_so3
                    )
                    print(f"从COLMAP二进制文件加载相机参数: {sparse_dir}")
                    cam_extrinsics = read_colmap_extrinsics_binary(cameras_extrinsic_file)
                    cam_intrinsics = read_colmap_intrinsics_binary(cameras_intrinsic_file)
                else:
                    # 尝试读取文本格式
                    cameras_extrinsic_file = os.path.join(sparse_dir, "images.txt")
                    cameras_intrinsic_file = os.path.join(sparse_dir, "cameras.txt")
                    if os.path.exists(cameras_extrinsic_file) and os.path.exists(cameras_intrinsic_file):
                        from threedgrut.datasets.utils import (
                            read_colmap_extrinsics_text,
                            read_colmap_intrinsics_text,
                            qvec_to_so3
                        )
                        print(f"从COLMAP文本文件加载相机参数: {sparse_dir}")
                        cam_extrinsics = read_colmap_extrinsics_text(cameras_extrinsic_file)
                        cam_intrinsics = read_colmap_intrinsics_text(cameras_intrinsic_file)
                    else:
                        raise FileNotFoundError(f"未找到COLMAP相机参数文件: {sparse_dir}")
                
                # 检查视图索引是否有效
                if view_index >= len(cam_extrinsics):
                    raise ValueError(f"视图索引{view_index}超出范围，总共{len(cam_extrinsics)}个视图")
                
                # 获取当前视图的外参和内参
                extr = cam_extrinsics[view_index]
                intr = cam_intrinsics[extr.camera_id - 1]
                
                # 构建相机数据
                # 从四元数和平移向量构建相机姿态
                R = qvec_to_so3(extr.qvec)
                T = np.array(extr.tvec)
                W2C = np.zeros((4, 4), dtype=np.float32)
                W2C[:3, 3] = T
                W2C[:3, :3] = R
                W2C[3, 3] = 1.0
                C2W = np.linalg.inv(W2C)  # 从世界到相机的变换
                
                # 转换为PyTorch张量
                C2W_tensor = torch.tensor(C2W, dtype=torch.float32, device=self.device)
                
                # 获取图像分辨率
                # 获取图像目录
                imgs_dir = os.path.join(self.scene_path, "images")
                if not os.path.exists(imgs_dir):
                    # 尝试其他可能的目录名
                    for dir_name in ["images_1", "images_2", "images_4", "images_8"]:
                        imgs_dir = os.path.join(self.scene_path, dir_name)
                        if os.path.exists(imgs_dir):
                            break
                
                # 获取当前视图的图像文件名
                img_filename = os.path.basename(extr.name)
                img_path = os.path.join(imgs_dir, img_filename)
                
                # 获取图像尺寸
                if os.path.exists(img_path):
                    from PIL import Image
                    img = Image.open(img_path)
                    W, H = img.size
                else:
                    # 如果找不到图像，使用内参中的尺寸
                    H, W = intr.height, intr.width
                
                # 处理内参
                if intr.model == "SIMPLE_PINHOLE":
                    fx = fy = intr.params[0]
                    cx = cy = 0
                    if len(intr.params) >= 3:
                        cx, cy = intr.params[1:3]
                    else:
                        cx, cy = W/2, H/2
                elif intr.model == "PINHOLE":
                    fx, fy = intr.params[0:2]
                    cx, cy = intr.params[2:4] if len(intr.params) >= 4 else (W/2, H/2)
                else:
                    # 对于其他相机模型，简化处理
                    fx = fy = intr.params[0]
                    cx, cy = W/2, H/2
                    print(f"警告：不支持的相机模型 {intr.model}，使用简化内参")
                
                # 创建相机数据
                camera_data = {
                    'intrinsics': [float(fx), float(fy), float(cx), float(cy)],
                    'T_to_world': C2W_tensor[:3, :4],  # 只需要前3行4列
                    'W': W,
                    'H': H,
                    'file_path': img_path
                }
                
                return camera_data
                
            except (ImportError, FileNotFoundError) as e:
                print(f"无法加载COLMAP格式数据: {e}")
                pass
            
            # 方法2: 尝试加载transforms.json (NeRF格式)
            transforms_file = os.path.join(self.scene_path, "transforms.json")
            if os.path.exists(transforms_file):
                with open(transforms_file, 'r') as f:
                    transforms_data = json.load(f)
                
                if view_index >= len(transforms_data['frames']):
                    raise ValueError(f"视图索引{view_index}超出范围，总共{len(transforms_data['frames'])}个视图")
                
                # 提取相机数据
                frame_data = transforms_data['frames'][view_index]
                camera_data = {
                    'file_path': frame_data['file_path'],
                    'transform_matrix': torch.tensor(frame_data['transform_matrix'], 
                                                   dtype=torch.float32, 
                                                   device=self.device)
                }
                
                # 提取相机内参
                if 'camera_angle_x' in transforms_data:
                    # 针对NERF格式的数据
                    W = H = transforms_data.get('w', transforms_data.get('width', 800))
                    camera_angle_x = transforms_data['camera_angle_x']
                    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
                    camera_data['intrinsics'] = [float(focal), float(focal), W/2.0, H/2.0]
                    camera_data['W'] = W
                    camera_data['H'] = H
                
                # 从transform_matrix中提取T_to_world
                camera_data['T_to_world'] = camera_data['transform_matrix'][:3, :4]
                
                return camera_data
            
            # 方法3: 尝试加载相机参数文件 (NeuS格式)
            cameras_file = os.path.join(self.scene_path, "cameras.npz")
            if os.path.exists(cameras_file):
                camera_dict = np.load(cameras_file)
                world_mats_np = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(len(camera_dict.files) // 4)]
                scale_mats_np = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(len(camera_dict.files) // 4)]
                
                if view_index >= len(world_mats_np):
                    raise ValueError(f"视图索引{view_index}超出范围，总共{len(world_mats_np)}个视图")
                
                # 转换为PyTorch张量
                world_mat = torch.from_numpy(world_mats_np[view_index]).to(self.device)
                scale_mat = torch.from_numpy(scale_mats_np[view_index]).to(self.device)
                
                # 相机内参在world_mat的左上角3x3
                intrinsic = world_mat[:3, :3]
                # 相机外参 = scale_mat * world_mat
                pose = torch.matmul(scale_mat, world_mat)
                pose = pose[:3, :4]  # 取前3行4列
                
                # 提取相机参数
                fx, fy = intrinsic[0, 0], intrinsic[1, 1]
                cx, cy = intrinsic[0, 2], intrinsic[1, 2]
                
                # 设置图像尺寸（如果文件中没有，使用默认值）
                H = W = camera_dict.get('height', camera_dict.get('H', 800))
                
                camera_data = {
                    'intrinsics': [float(fx), float(fy), float(cx), float(cy)],
                    'T_to_world': pose,
                    'W': W,
                    'H': H
                }
                
                return camera_data
            
            # 如果上述都不存在，尝试其他格式...
            raise FileNotFoundError("在场景路径中未找到支持的相机参数文件")
            
        except Exception as e:
            print(f"加载相机数据时出错: {e}")
            raise
    
    def render_view(self, view_index=0, H=None, W=None, render_refractive=True, output_dir="render_output"):
        """
        渲染特定视角的图像
        
        Args:
            view_index: 要渲染的视图索引
            H: 图像高度，如果为None则使用相机数据中的高度
            W: 图像宽度，如果为None则使用相机数据中的宽度
            render_refractive: 是否渲染折射效果
            output_dir: 输出目录
            
        Returns:
            outputs: 渲染结果
        """
        # 加载相机数据
        print(f"加载视图 {view_index} 的相机数据...")
        camera_data = self.load_camera_data(view_index)
        
        # 使用相机数据中的分辨率（除非明确指定）
        H = H or camera_data.get('H', 512)
        W = W or camera_data.get('W', 512)
        
        # 创建批处理对象
        class BatchObject:
            pass
        
        batch = BatchObject()
        
        # 设置相机外参
        batch.T_to_world = camera_data['T_to_world'].unsqueeze(0)  # 形状为[1, 3, 4]
        
        # 设置相机内参
        batch.intrinsics = camera_data['intrinsics']
        
        print(f"相机内参: {batch.intrinsics}")
        print(f"相机外参:\n{batch.T_to_world[0]}")
        print(f"渲染分辨率: {W}x{H}")
        
        # 创建光线
        batch.rays_ori = torch.zeros((1, H, W, 3), device=self.device)
        batch.rays_dir = torch.zeros((1, H, W, 3), device=self.device)
        
        # 生成图像平面上的网格
        y, x = torch.meshgrid(
            torch.linspace(0, H-1, H, device=self.device),
            torch.linspace(0, W-1, W, device=self.device),
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
        
        # 添加rgb_gt（虽然只是用于测试）
        batch.rgb_gt = torch.zeros((1, H, W, 3), device=self.device)
        
        # 执行渲染
        print("执行渲染...")
        outputs = self.render(batch, render_refractive=render_refractive)
        
        # 打印结果
        print(f"渲染RGB形状: {outputs['pred_rgb'].shape}")
        if render_refractive:
            print(f"经过折射的像素数量: {outputs['is_refracted'].sum().item()}")
            print(f"折射像素比例: {outputs['is_refracted'].sum().item() / (H*W):.2%}")
        
        # 保存渲染结果
        os.makedirs(output_dir, exist_ok=True)
        
        # 将张量转换为PIL图像并保存
        def save_tensor_as_image(tensor, filename):
            # 确保值在[0,1]范围内
            tensor = torch.clamp(tensor, 0, 1)
            # 从CUDA转移到CPU，从FloatTensor转换为uint8
            img_np = (tensor[0].detach().cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(img_np)
            img.save(os.path.join(output_dir, filename))
            print(f"图像已保存: {os.path.join(output_dir, filename)}")
        
        # 保存渲染结果
        suffix = "_refractive" if render_refractive else "_normal"
        save_tensor_as_image(outputs['pred_rgb'], f"view_{view_index}{suffix}.png")
        
        # 如果是折射渲染，保存折射掩码
        if render_refractive and 'is_refracted' in outputs:
            refraction_mask = outputs['is_refracted'].float()
            try:
                save_tensor_as_image(refraction_mask, f"view_{view_index}_mask.png")
            except Exception as e:
                print(f"保存折射掩码时出错: {e}")
                # 可能需要特殊处理这个掩码数据
                mask_np = refraction_mask[0].detach().cpu().numpy()
                mask_img = Image.fromarray((mask_np * 255).astype(np.uint8))
                mask_img.save(os.path.join(output_dir, f"view_{view_index}_mask.png"))
                print(f"掩码图像已保存: {os.path.join(output_dir, f'view_{view_index}_mask.png')}")
        
        return outputs

def test_batch_refractive_rendering():
    """测试优化后的批量折射渲染"""
    # 这里需要根据实际情况替换模型路径和场景路径
    neus_model_path = "/workspace/sdf/NeTO/Use3DGRUT/exp/eiko_ball_masked/silhouette/checkpoints/ckpt_300000.pth"
    gaussians_model_path = "/workspace/runs/eiko_ball_masked_expanded_3dgrt/eiko_ball_masked_expanded-2703_050608/ckpt_last.pt"
    scene_path = "/workspace/sdf/NeTO/Use3DGRUT/data/eiko_ball_masked"  # 场景路径
    
    import time
    
    try:
        # 设置低分辨率以减少内存使用
        H, W = 400, 400  # 降低分辨率以防止OOM
        view_index = 0  # 选择第一个视角
        
        # 设置CUDA内存分配配置
        try:
            torch.cuda.set_per_process_memory_fraction(0.8)  # 限制使用80%的GPU内存
        except:
            print("无法设置CUDA内存限制")
        
        # 初始化折射渲染器
        print("初始化折射渲染器...")
        renderer = RefractiveRenderer(
            neus_model_path=neus_model_path,
            gaussians_model_path=gaussians_model_path,
            scene_path=scene_path,
            auto_scale=True
        )
        
        # 创建输出目录
        output_dir = "batch_refraction_test/render_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载相机数据
        print(f"加载视图 {view_index} 的相机数据...")
        camera_data = renderer.load_camera_data(view_index)
        
        # 创建批处理对象
        class BatchObject:
            pass
        
        batch = BatchObject()
        
        # 设置相机外参
        batch.T_to_world = camera_data['T_to_world'].unsqueeze(0)  # 形状为[1, 3, 4]
        
        # 设置相机内参
        batch.intrinsics = camera_data['intrinsics']
        
        # 创建光线
        batch.rays_ori = torch.zeros((1, H, W, 3), device=renderer.device)
        batch.rays_dir = torch.zeros((1, H, W, 3), device=renderer.device)
        
        # 生成图像平面上的网格
        y, x = torch.meshgrid(
            torch.linspace(0, H-1, H, device=renderer.device),
            torch.linspace(0, W-1, W, device=renderer.device),
            indexing='ij'
        )
        
        # 计算每个像素的光线方向（相机坐标系）
        fx, fy, cx, cy = batch.intrinsics
        
        # 调整内参以适应新的分辨率
        scale_x = H / camera_data.get('H', H)
        scale_y = W / camera_data.get('W', W)
        fx *= scale_x
        fy *= scale_y
        cx *= scale_x
        cy *= scale_y
        
        camera_dirs = torch.stack([
            (x - cx) / fx,
            -(y - cy) / fy,  # 注意：y方向需要取反
            -torch.ones_like(x)  # z方向指向相机前方
        ], dim=-1)
        
        # 归一化方向
        camera_dirs = F.normalize(camera_dirs, p=2, dim=-1)
        
        # 将相机坐标系的光线方向转换到世界坐标系
        rotation = batch.T_to_world[0, :, :3]  # 形状为[3, 3]
        
        # 批量转换所有光线方向
        camera_dirs_flat = camera_dirs.reshape(-1, 3)  # [H*W, 3]
        world_dirs_flat = torch.matmul(camera_dirs_flat, rotation.T)  # [H*W, 3]
        world_dirs_flat = F.normalize(world_dirs_flat, p=2, dim=-1)
        world_dirs = world_dirs_flat.reshape(H, W, 3)
        batch.rays_dir[0] = world_dirs
        
        # 设置光线原点（所有像素共用同一个原点，即相机中心）
        camera_pos = batch.T_to_world[0, :, 3]  # 相机在世界坐标系中的位置
        batch.rays_ori[0, :, :] = camera_pos.view(1, 1, 3).expand(H, W, 3)
        
        # 添加rgb_gt（虽然只是用于测试）
        batch.rgb_gt = torch.zeros((1, H, W, 3), device=renderer.device)
        
        # 使用普通渲染作为基准
        print("执行普通渲染（不带折射）...")
        torch.cuda.synchronize()
        start_time = time.time()
        normal_outputs = renderer.render(batch, render_refractive=False)
        torch.cuda.synchronize()
        normal_render_time = time.time() - start_time
        print(f"普通渲染时间: {normal_render_time:.4f} 秒")
        
        # 保存普通渲染结果
        save_tensor_as_image(normal_outputs['pred_rgb'], f"normal_render_{H}x{W}.png", output_dir)
        
        # 清理缓存释放内存
        torch.cuda.empty_cache()
        
        # 执行优化后的批量折射渲染
        print("执行优化后的批量折射渲染...")
        torch.cuda.synchronize()
        start_time = time.time()
        batch_outputs = renderer.render(batch, render_refractive=True)
        torch.cuda.synchronize()
        batch_render_time = time.time() - start_time
        print(f"批量折射渲染时间: {batch_render_time:.4f} 秒")
        
        # 打印渲染统计信息
        refracted_pixels = batch_outputs['is_refracted'].sum().item()
        total_pixels = H * W
        print(f"渲染分辨率: {W}x{H} ({total_pixels} 像素)")
        print(f"折射像素数量: {refracted_pixels} ({refracted_pixels/total_pixels:.2%})")
        
        # 保存批量折射渲染结果
        save_tensor_as_image(batch_outputs['pred_rgb'], f"batch_refractive_render_{H}x{W}.png", output_dir)
        
        # 保存折射掩码
        refraction_mask = batch_outputs['is_refracted'].float()
        save_tensor_as_image(refraction_mask, f"refraction_mask_{H}x{W}.png", output_dir)
        
        # 创建对比图像
        diff = torch.abs(batch_outputs['pred_rgb'] - normal_outputs['pred_rgb'])
        save_tensor_as_image(diff, f"render_diff_{H}x{W}.png", output_dir)
        
        # 显示性能提升
        speedup = max(1.0, normal_render_time / batch_render_time)
        print(f"加速比: {speedup:.2f}x")
        
        return True
    
    except Exception as e:
        import traceback
        print(f"错误: {e}")
        traceback.print_exc()
        return False

def save_tensor_as_image(tensor, filename, output_dir="render_output"):
    """保存张量为图像文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    if tensor.shape[-1] == 1:  # 灰度图（例如掩码）
        # 确保值在[0,1]范围内
        tensor = torch.clamp(tensor, 0, 1)
        # 扩展到3通道
        tensor = tensor.repeat(1, 1, 1, 3)
    
    # 确保值在[0,1]范围内
    tensor = torch.clamp(tensor, 0, 1)
    # 从CUDA转移到CPU，从FloatTensor转换为uint8
    img_np = (tensor[0].detach().cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    img.save(os.path.join(output_dir, filename))
    print(f"图像已保存: {os.path.join(output_dir, filename)}")

if __name__ == "__main__":
    # test_refractive_rendering()
    test_batch_refractive_rendering() 