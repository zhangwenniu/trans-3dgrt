from . import ColmapDataset
from . import NeRFDatasetWithMask
import torch
import numpy as np
import cv2 as cv
from .utils import MultiEpochsDataLoader

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs     


class ColmapDatasetAdapter:
    """
    适配器类，将threedgrut中的ColmapDataset转换为与NeTO兼容的接口
    """
    def __init__(self, conf):
        """
        使用NeTO的配置格式初始化
        
        Args:
            conf: NeTO的配置对象
        """
        self.conf = conf
        self.data_dir = conf.get_string('data_dir')
        self.device = torch.device('cuda')
        
        # 从NeTO配置中提取ColmapDataset需要的参数
        downsample_factor = conf.get_float('downsample_factor', default=1.0)
        ray_jitter = None  # 可以从配置中提取
        
        # 初始化ColmapDataset
        self.colmap_dataset = ColmapDataset(
            path=self.data_dir,
            device=self.device,
            split="train",
            downsample_factor=downsample_factor,
            ray_jitter=ray_jitter
        )
        
        self.n_images = len(self.colmap_dataset)
        
        # 创建数据加载器 - 这是修改的核心部分
        self.dataloader = MultiEpochsDataLoader(
            self.colmap_dataset,
            num_workers=0,  # 将num_workers设置为0以禁用多进程
            batch_size=1,
            shuffle=False,  # 不打乱数据，保持索引一致性
            pin_memory=False,  # 禁用pin_memory，因为数据集返回的已经是CUDA张量
            persistent_workers=False  # num_workers为0时不需要persistent_workers
        )
        
        # 预加载所有GPU批次，避免重复计算
        self.gpu_batches = []
        self._preload_gpu_batches()
        
        
        # 获取场景信息
        self.object_bbox_min = torch.tensor([-1.01, -1.01, -1.01], dtype=torch.float32)
        self.object_bbox_max = torch.tensor([1.01, 1.01, 1.01], dtype=torch.float32)
        
        # 从ColmapDataset提取并存储图像和掩码尺寸
        self.H = self.colmap_dataset.image_h
        self.W = self.colmap_dataset.image_w
        
        # 提取并存储相机参数
        # 使用与原始NeTO相似的属性命名
        self.focal = None  # 将在_extract_camera_params中设置
        self.intrinsics_all = []
        self.pose_all = []
        self._extract_camera_params()
        
        # scale_mat是转换用于NeTO的缩放矩阵，4x4矩阵，[0, 0] = [1, 1] = [2, 2] = length_scale, [3, 3] = 1.0
        # scale_mat[:3, 3] = 场景中心在原世界坐标系下的坐标
        self.scale_mat = self.colmap_dataset.scale_mat
        
        # 设置masks和light_masks从ColmapDataset获取
        self.masks = self._get_masks()
        self.light_masks = self._get_light_masks()

    def _preload_gpu_batches(self):
        """预加载所有GPU批次"""
        self.gpu_batches = []
        # 遍历数据加载器，获取所有批次
        for idx, batch in enumerate(self.dataloader):
            # 将批次数据移动到GPU并获取含内参的批次
            gpu_batch = self.colmap_dataset.get_gpu_batch_with_intrinsics(batch)
            self.gpu_batches.append(gpu_batch)
        
    def _extract_camera_params(self):
        """从预加载的GPU批次中提取相机参数"""
        for idx, gpu_batch in enumerate(self.gpu_batches):
            # 提取相机内参
            camera_name = None
            camera_params = None
            for key in vars(gpu_batch):
                if key.startswith('intrinsics_'):
                    camera_name = key
                    camera_params = getattr(gpu_batch, key)
                    break
            
            if camera_params:
                # 构建4x4内参矩阵
                intrinsic = torch.eye(4, device=self.device)
                intrinsic[0, 0] = torch.tensor(camera_params['focal_length'][0], device=self.device)
                intrinsic[1, 1] = torch.tensor(camera_params['focal_length'][1], device=self.device)
                intrinsic[0, 2] = torch.tensor(camera_params['principal_point'][0], device=self.device)
                intrinsic[1, 2] = torch.tensor(camera_params['principal_point'][1], device=self.device)
                self.intrinsics_all.append(intrinsic)
                
                # 保存第一个相机的焦距作为统一焦距
                if idx == 0:
                    self.focal = torch.tensor(camera_params['focal_length'][0], device=self.device)
            
            # 提取相机外参
            pose = gpu_batch.T_to_world
            self.pose_all.append(pose)
        
        # 将列表转为张量
        if self.intrinsics_all:
            self.intrinsics_all = torch.stack(self.intrinsics_all)
            self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        if self.pose_all:
            self.pose_all = torch.stack(self.pose_all)


    def _get_masks(self):
        """从预加载的GPU批次中提取掩码"""
        masks = []
        for gpu_batch in self.gpu_batches:
            mask = gpu_batch.mask
            masks.append(mask)
        return torch.stack(masks) if masks else None

    def _get_light_masks(self):
        """提取或生成光照掩码"""
        # 如果ColmapDataset没有光照掩码，我们可以使用普通掩码
        return self._get_masks()

    def gen_rays_at(self, idx, resolution_level=1):
        """
        生成指定图像索引的所有光线
        
        Args:
            idx: 图像索引
            resolution_level: 分辨率级别，1为全分辨率
            
        Returns:
            rays_o, rays_d, ray_points: 光线起点、方向和点
        """
        # 获取预加载的批次
        gpu_batch = self.gpu_batches[idx]
        
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
        
        # 将射线转为缩放后的世界坐标系，因为我们假设感兴趣的场景在单位球内
        # rays_dir_world_flat 不变，由于场景缩放只涉及平移和缩放，不涉及旋转，所以rays_dir_world_flat 不变
        # rays_ori_world_flat 需要先平移，再缩放
        rays_ori_world_flat = (rays_ori_world_flat - self.scale_mat[:3, 3]) / self.scale_mat[0, 0]
        
        # 重塑回原始形状
        rays_ori_world = rays_ori_world_flat.reshape(batch_size, H, W, 3)  # [batch_size, H, W, 3]
        rays_dir_world = rays_dir_world_flat.reshape(batch_size, H, W, 3)  # [batch_size, H, W, 3]

        rays_ori_world = rays_ori_world[0]
        rays_dir_world = rays_dir_world[0]
        # 如果需要降低分辨率
        if resolution_level > 1:
            rays_ori_world = rays_ori_world[::resolution_level, ::resolution_level, :]
            rays_dir_world = rays_dir_world[::resolution_level, ::resolution_level, :]
        
        # 原始NeTO可能还返回ray_points
        # 这里我们使用零张量，或者可以实现更合适的计算
        ray_points = torch.zeros_like(rays_ori_world)
        
        return rays_ori_world, rays_dir_world, ray_points

    def gen_random_rays_at(self, idx, batch_size):
        """
        在指定图像索引生成随机光线
        
        Args:
            idx: 图像索引
            batch_size: 光线数量
            
        Returns:
            光线批次, pixel_x, pixel_y
        """
        # 获取预加载的批次
        gpu_batch = self.gpu_batches[idx]
        
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
        
        # 将射线转为缩放后的世界坐标系，因为我们假设感兴趣的场景在单位球内
        # rays_dir_world_flat 不变，由于场景缩放只涉及平移和缩放，不涉及旋转，所以rays_dir_world_flat 不变
        # rays_ori_world_flat 需要先平移，再缩放
        rays_ori_world_flat = (rays_ori_world_flat - self.scale_mat[:3, 3]) / self.scale_mat[0, 0]
        
        
        # 重塑回原始形状
        rays_ori_world = rays_ori_world_flat.reshape(batch_size, H, W, 3)  # [batch_size, H, W, 3]
        rays_dir_world = rays_dir_world_flat.reshape(batch_size, H, W, 3)  # [batch_size, H, W, 3]
        
        
        
        # 获取图像尺寸
        H, W = gpu_batch.rays_ori.shape[1:3]

        # 随机选择像素
        pixels_x = torch.randint(low=0, high=W, size=[batch_size], device=self.device)
        pixels_y = torch.randint(low=0, high=H, size=[batch_size], device=self.device)
        
        
        # 提取选定位置的光线
        rays_o = rays_ori_world[0, pixels_y, pixels_x, :]  # [batch_size, 3]
        rays_d = rays_dir_world[0, pixels_y, pixels_x, :]  # [batch_size, 3]
        
        
        # 提取掩码
        if gpu_batch.mask is not None:
            mask = gpu_batch.mask[0, pixels_y, pixels_x]  # [batch_size, 1]
        else:
            mask = torch.ones((batch_size, 1), device=self.device)
        
        # 生成有效掩码和光线点(根据原始NeTO需求)
        valid_mask = torch.ones_like(mask)
        ray_point = torch.zeros_like(rays_o)  # 可根据需要修改
        
        # 组合为NeTO期望的格式
        result = torch.cat([rays_o, rays_d, ray_point, mask, valid_mask], dim=-1)
        
        return result, pixels_x.cpu(), pixels_y.cpu()

    def gen_ray_masks_near(self, idx, batch_size):
        """
        生成靠近掩码边界的随机光线
        
        Args:
            idx: 图像索引
            batch_size: 光线数量
            
        Returns:
            光线批次, pixel_x, pixel_y
        """
        # 获取预加载的批次
        gpu_batch = self.gpu_batches[idx]
        mask = gpu_batch.mask[0]  # [H, W, 1]
        
        # 计算掩码边界
        mask_np = mask.cpu().numpy()
        y_indices, x_indices = np.where(mask_np[:, :, 0] > 0.5)
        if len(y_indices) == 0:  # 如果掩码为空
            return self.gen_random_rays_at(idx, batch_size)
            
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        # 添加边界
        margin = 150  # 与原始NeTO一致
        y_min = max(y_min - margin, 0)
        y_max = min(y_max + margin, self.H - 1)
        x_min = max(x_min - margin, 0)
        x_max = min(x_max + margin, self.W - 1)
        
        # 在边界区域随机采样
        pixels_y = torch.randint(low=y_min, high=y_max + 1, size=[batch_size], device=self.device)
        pixels_x = torch.randint(low=x_min, high=x_max + 1, size=[batch_size], device=self.device)
        
        # 提取光线和掩码
        rays_o = gpu_batch.rays_ori[0, pixels_y, pixels_x]  # [batch_size, 3]
        rays_d = gpu_batch.rays_dir[0, pixels_y, pixels_x]  # [batch_size, 3]
        mask = gpu_batch.mask[0, pixels_y, pixels_x]  # [batch_size, 1]
        
        # 生成有效掩码和光线点
        valid_mask = torch.ones_like(mask)
        ray_point = torch.zeros_like(rays_o)  # 根据需要可修改
        
        # 组合结果
        result = torch.cat([rays_o, rays_d, ray_point, mask, valid_mask], dim=-1)
        
        return result, pixels_x.cpu(), pixels_y.cpu()

    def near_far_from_sphere(self, rays_o, rays_d):
        """
        计算从球体到的近远平面距离
        
        Args:
            rays_o: 光线起点 [batch_size, 3]
            rays_d: 光线方向 [batch_size, 3]
            
        Returns:
            near, far: 近平面和远平面距离
        """
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far
    
    def mask_at(self, idx, resolution_level=1):
        """
        获取指定图像索引的掩码
        
        Args:
            idx: 图像索引
            resolution_level: 分辨率级别
            
        Returns:
            掩码图像
        """
        # 获取预加载的批次
        gpu_batch = self.gpu_batches[idx]
        mask = gpu_batch.mask[0].cpu().numpy()  # [H, W, 1]
        
        # 如果需要降低分辨率
        if resolution_level > 1:
            mask = mask[::resolution_level, ::resolution_level]
        return mask

class NeRFWithMaskDatasetAdapter:
    """
    适配器类，将threedgrut中的NeRFDatasetWithMask转换为与NeTO兼容的接口
    """
    def __init__(self, conf):
        """
        使用NeTO的配置格式初始化
        
        Args:
            conf: NeTO的配置对象
        """
        self.conf = conf
        self.data_dir = conf.get_string('data_dir')
        self.device = torch.device('cuda')
        
        # 从NeTO配置中提取NeRFDatasetWithMask需要的参数
        bg_color = conf.get_string('background_color', default='black')
        ray_jitter = None  # 可以从配置中提取
        
        # 初始化NeRFDatasetWithMask
        self.nerf_dataset = NeRFDatasetWithMask(
            path=self.data_dir,
            device=self.device,
            split="train",
            return_masks=True,
            ray_jitter=ray_jitter,
            bg_color=bg_color
        )
        
        self.n_images = len(self.nerf_dataset)
        
        # 创建数据加载器
        self.dataloader = MultiEpochsDataLoader(
            self.nerf_dataset,
            num_workers=0,  # 将num_workers设置为0以禁用多进程
            batch_size=1,
            shuffle=False,  # 不打乱数据，保持索引一致性
            pin_memory=False,  # 禁用pin_memory，因为数据集返回的已经是CUDA张量
            persistent_workers=False  # num_workers为0时不需要persistent_workers
        )
        
        # 预加载所有GPU批次，避免重复计算
        self.gpu_batches = []
        self._preload_gpu_batches()
        
        # 获取场景信息
        scene_bbox = self.nerf_dataset.get_scene_bbox()
        scene_center = self.nerf_dataset.get_center()
        length_scale = self.nerf_dataset.get_length_scale()
        
        # 创建缩放矩阵
        self.scale_mat = torch.eye(4, dtype=torch.float32, device=self.device)
        self.scale_mat[0, 0] = self.scale_mat[1, 1] = self.scale_mat[2, 2] = length_scale
        self.scale_mat[:3, 3] = scene_center
        
        # 设置碰撞边界框（略大于场景边界框）
        padding = 0.01  # 添加一点点边距
        self.object_bbox_min = scene_bbox[0] - padding
        self.object_bbox_max = scene_bbox[1] + padding
        
        # 从NeRFDatasetWithMask提取并存储图像和掩码尺寸
        self.H = self.nerf_dataset.image_h
        self.W = self.nerf_dataset.image_w
        
        # 提取并存储相机参数
        # 使用与原始NeTO相似的属性命名
        self.focal = None  # 将在_extract_camera_params中设置
        self.intrinsics_all = []
        self.pose_all = []
        self._extract_camera_params()
        
        # 设置masks和light_masks
        self.masks = self._get_masks()
        self.light_masks = self._get_light_masks()

    def _preload_gpu_batches(self):
        """预加载所有GPU批次"""
        self.gpu_batches = []
        # 遍历数据加载器，获取所有批次
        for idx, batch in enumerate(self.dataloader):
            # 将批次数据移动到GPU并获取含内参的批次
            gpu_batch = self.nerf_dataset.get_gpu_batch_with_intrinsics(batch)
            self.gpu_batches.append(gpu_batch)
        
    def _extract_camera_params(self):
        """从预加载的GPU批次中提取相机参数"""
        for idx, gpu_batch in enumerate(self.gpu_batches):
            # NeRF数据集的内参存储在intrinsics字段中，它是一个包含[fx, fy, cx, cy]的列表
            if hasattr(gpu_batch, 'intrinsics'):
                fx, fy, cx, cy = gpu_batch.intrinsics
                
                # 构建4x4内参矩阵
                intrinsic = torch.eye(4, device=self.device)
                intrinsic[0, 0] = torch.tensor(fx, device=self.device)
                intrinsic[1, 1] = torch.tensor(fy, device=self.device)
                intrinsic[0, 2] = torch.tensor(cx, device=self.device)
                intrinsic[1, 2] = torch.tensor(cy, device=self.device)
                self.intrinsics_all.append(intrinsic)
                
                # 保存第一个相机的焦距作为统一焦距
                if idx == 0:
                    self.focal = torch.tensor(fx, device=self.device)
            
            # 提取相机外参
            pose = gpu_batch.T_to_world
            self.pose_all.append(pose)
        
        # 将列表转为张量
        if self.intrinsics_all:
            self.intrinsics_all = torch.stack(self.intrinsics_all)
            self.intrinsics_all_inv = torch.inverse(self.intrinsics_all)
        if self.pose_all:
            self.pose_all = torch.stack(self.pose_all)
    
    def _get_masks(self):
        """从预加载的GPU批次中提取掩码"""
        masks = []
        for gpu_batch in self.gpu_batches:
            # NeRFDatasetWithMask中掩码字段为'mask'
            if hasattr(gpu_batch, 'mask'):
                masks.append(gpu_batch.mask)
        
        return torch.stack(masks) if masks else None

    def _get_light_masks(self):
        """提取或生成光照掩码，使用普通掩码"""
        return self._get_masks()

    def gen_rays_at(self, idx, resolution_level=1):
        """
        生成指定图像索引的所有光线
        
        Args:
            idx: 图像索引
            resolution_level: 分辨率级别，1为全分辨率
            
        Returns:
            rays_o, rays_d, ray_points: 光线起点、方向和点
        """
        # 获取预加载的批次
        gpu_batch = self.gpu_batches[idx]
        
        # 获取参数形状
        batch_size, H, W, _ = gpu_batch.rays_ori.shape
        
        # 提取T_to_world矩阵
        T = gpu_batch.T_to_world  # [3, 4]
        if T.dim() == 2:
            T = T.unsqueeze(0)  # [1, 3, 4]
        
        # 处理射线原点(rays_ori)转换
        rays_ori_flat = gpu_batch.rays_ori.reshape(batch_size, H*W, 3)  # [batch_size, H*W, 3]
        
        # 应用旋转和平移
        rays_ori_rotated = torch.bmm(rays_ori_flat, T[:, :, :3].transpose(1, 2))  # [batch_size, H*W, 3]
        translation = T[:, :, 3].unsqueeze(1).expand(-1, H*W, -1)  # [batch_size, H*W, 3]
        rays_ori_world_flat = rays_ori_rotated + translation  # [batch_size, H*W, 3]
        
        # 处理射线方向(rays_dir)转换
        rays_dir_flat = gpu_batch.rays_dir.reshape(batch_size, H*W, 3)  # [batch_size, H*W, 3]
        rays_dir_world_flat = torch.bmm(rays_dir_flat, T[:, :, :3].transpose(1, 2))  # [batch_size, H*W, 3]
        
        # 确保射线方向归一化
        rays_dir_world_flat = torch.nn.functional.normalize(rays_dir_world_flat, dim=2)
        
        # 将射线转为缩放后的世界坐标系（单位球内）
        rays_ori_world_flat = (rays_ori_world_flat - self.scale_mat[:3, 3]) / self.scale_mat[0, 0]
        
        # 重塑回原始形状
        rays_ori_world = rays_ori_world_flat.reshape(batch_size, H, W, 3)  # [batch_size, H, W, 3]
        rays_dir_world = rays_dir_world_flat.reshape(batch_size, H, W, 3)  # [batch_size, H, W, 3]

        rays_ori_world = rays_ori_world[0]  # [H, W, 3]
        rays_dir_world = rays_dir_world[0]  # [H, W, 3]
        
        # 如果需要降低分辨率
        if resolution_level > 1:
            rays_ori_world = rays_ori_world[::resolution_level, ::resolution_level, :]
            rays_dir_world = rays_dir_world[::resolution_level, ::resolution_level, :]
        
        # 原始NeTO也需要ray_points，这里使用零张量
        ray_points = torch.zeros_like(rays_ori_world)
        
        return rays_ori_world, rays_dir_world, ray_points

    def gen_random_rays_at(self, idx, batch_size):
        """
        在指定图像索引生成随机光线
        
        Args:
            idx: 图像索引
            batch_size: 光线数量
            
        Returns:
            光线批次, pixel_x, pixel_y
        """
        # 获取预加载的批次
        gpu_batch = self.gpu_batches[idx]
        
        # 获取图像尺寸
        H, W = self.H, self.W
        
        # 随机选择像素
        pixels_x = torch.randint(low=0, high=W, size=[batch_size], device=self.device)
        pixels_y = torch.randint(low=0, high=H, size=[batch_size], device=self.device)
        
        # 获取世界坐标系下的光线
        rays_o_world, rays_d_world, _ = self.gen_rays_at(idx)  # [H, W, 3]
        
        # 提取选定位置的光线
        rays_o = rays_o_world[pixels_y, pixels_x, :]  # [batch_size, 3]
        rays_d = rays_d_world[pixels_y, pixels_x, :]  # [batch_size, 3]
        
        # 提取掩码
        if hasattr(gpu_batch, 'mask'):
            mask = gpu_batch.mask[0, pixels_y, pixels_x, 0]  # [batch_size]
            mask = mask.unsqueeze(-1)  # [batch_size, 1]
        else:
            mask = torch.ones((batch_size, 1), device=self.device)
        
        # 生成有效掩码和光线点
        valid_mask = torch.ones_like(mask)
        ray_point = torch.zeros_like(rays_o)  # 占位符
        
        # 组合为NeTO期望的格式
        result = torch.cat([rays_o, rays_d, ray_point, mask, valid_mask], dim=-1)
        
        return result, pixels_x.cpu(), pixels_y.cpu()

    def gen_ray_masks_near(self, idx, batch_size):
        """
        生成靠近掩码边界的随机光线
        
        Args:
            idx: 图像索引
            batch_size: 光线数量
            
        Returns:
            光线批次, pixel_x, pixel_y
        """
        # 获取预加载的批次
        gpu_batch = self.gpu_batches[idx]
        
        # 检查是否有掩码
        if not hasattr(gpu_batch, 'mask'):
            return self.gen_random_rays_at(idx, batch_size)
        
        mask = gpu_batch.mask[0]  # [H, W, 1]
        
        # 将mask转移到CPU计算边界
        mask_np = mask.cpu().numpy()
        mask_2d = mask_np[:, :, 0]  # [H, W]
        
        # 计算掩码边界
        y_indices, x_indices = np.where(mask_2d > 0.5)
        if len(y_indices) == 0:  # 如果掩码为空
            return self.gen_random_rays_at(idx, batch_size)
            
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        # 添加边界
        margin = 50  # 边界宽度
        y_min = max(y_min - margin, 0)
        y_max = min(y_max + margin, self.H - 1)
        x_min = max(x_min - margin, 0)
        x_max = min(x_max + margin, self.W - 1)
        
        # 在边界区域随机采样
        pixels_y = torch.randint(low=y_min, high=y_max + 1, size=[batch_size], device=self.device)
        pixels_x = torch.randint(low=x_min, high=x_max + 1, size=[batch_size], device=self.device)
        
        # 获取世界坐标系下的光线
        rays_o_world, rays_d_world, _ = self.gen_rays_at(idx)  # [H, W, 3]
        
        # 提取选定位置的光线
        rays_o = rays_o_world[pixels_y, pixels_x, :]  # [batch_size, 3]
        rays_d = rays_d_world[pixels_y, pixels_x, :]  # [batch_size, 3]
        
        # 提取掩码值
        mask = gpu_batch.mask[0, pixels_y, pixels_x, 0]  # [batch_size]
        mask = mask.unsqueeze(-1)  # [batch_size, 1]
        
        # 生成有效掩码和光线点
        valid_mask = torch.ones_like(mask)
        ray_point = torch.zeros_like(rays_o)  # 占位符
        
        # 组合结果
        result = torch.cat([rays_o, rays_d, ray_point, mask, valid_mask], dim=-1)
        
        return result, pixels_x.cpu(), pixels_y.cpu()

    def near_far_from_sphere(self, rays_o, rays_d):
        """
        计算从球体到的近远平面距离
        
        Args:
            rays_o: 光线起点 [batch_size, 3]
            rays_d: 光线方向 [batch_size, 3]
            
        Returns:
            near, far: 近平面和远平面距离
        """
        # 假设场景被归一化到一个单位球体中
        a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        return near, far
    
    def mask_at(self, idx, resolution_level=1):
        """
        获取指定图像索引的掩码
        
        Args:
            idx: 图像索引
            resolution_level: 分辨率级别
            
        Returns:
            掩码图像
        """
        # 获取预加载的批次
        gpu_batch = self.gpu_batches[idx]
        
        # 检查是否有掩码
        if not hasattr(gpu_batch, 'mask'):
            return np.ones((self.H, self.W, 1))
        
        mask = gpu_batch.mask[0].cpu().numpy()  # [H, W, 1]
        
        # 如果需要降低分辨率
        if resolution_level > 1:
            mask = mask[::resolution_level, ::resolution_level]
            
        return mask


