from . import ColmapDataset
from . import NeRFDatasetWithMask
import torch
import numpy as np
import cv2 as cv
from .utils import MultiEpochsDataLoader
import os
import glob

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
        
        # scale_mat是转换用于NeTO的缩放矩阵，4x4矩阵，[0, 0] = [1, 1] = [2, 2] = length_scale, [3, 3] = 1.0
        # scale_mat[:3, 3] = 场景中心在原世界坐标系下的坐标
        self.scale_mat = self.colmap_dataset.scale_mat

    def _preload_gpu_batches(self):
        """预加载所有GPU批次"""
        self.gpu_batches = []
        # 遍历数据加载器，获取所有批次
        for idx, batch in enumerate(self.dataloader):
            # 将批次数据移动到GPU并获取含内参的批次
            gpu_batch = self.colmap_dataset.get_gpu_batch_with_intrinsics(batch)
            self.gpu_batches.append(gpu_batch)

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
        mask = gpu_batch.mask[0]
        rgb_gt = gpu_batch.rgb_gt[0]        
        if resolution_level > 1:
            rays_ori_world = rays_ori_world[::resolution_level, ::resolution_level, :]
            rays_dir_world = rays_dir_world[::resolution_level, ::resolution_level, :]
            mask = mask[::resolution_level, ::resolution_level, :]
            rgb_gt = rgb_gt[::resolution_level, ::resolution_level, :]
            
            if rgb_gt.max() > 1.5:
                rgb_gt = rgb_gt / 255.0
            if mask.max() > 1.5:
                mask = mask > 0.5
        
        # 原始NeTO可能还返回ray_points
        # 这里我们使用零张量，或者可以实现更合适的计算

        return rays_ori_world, rays_dir_world, mask, rgb_gt

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
        
        mask = gpu_batch.mask[0, pixels_y, pixels_x]  # [batch_size, 1]
        rgb_gt = gpu_batch.rgb_gt[0, pixels_y, pixels_x, :]  # [batch_size, 3]
        
        return rays_o, rays_d, mask, rgb_gt

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
        split = conf.get_string('split', default='train')
        return_masks = conf.get_bool('return_masks', default=True)
        downsample_factor = conf.get_float('downsample_factor', default=1.0)
        ray_jitter = None  # 可以从配置中提取
        bg_color = conf.get_string('bg_color', default=None)
        
        # 初始化NeRFDatasetWithMask
        self.nerf_dataset = NeRFDatasetWithMask(
            path=self.data_dir,
            device=self.device,
            split=split,
            return_masks=return_masks,
            ray_jitter=ray_jitter,
            bg_color=bg_color
        )
        
        self.n_images = len(self.nerf_dataset)
        
        # 创建数据加载器 - 与ColmapDatasetAdapter相同的配置
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
        self.object_bbox_min = torch.tensor([-1.01, -1.01, -1.01], dtype=torch.float32)
        self.object_bbox_max = torch.tensor([1.01, 1.01, 1.01], dtype=torch.float32)
        
        # 从NeRFDatasetWithMask提取并存储图像和掩码尺寸
        self.H = self.nerf_dataset.image_h
        self.W = self.nerf_dataset.image_w
        
        # scale_mat是转换用于NeTO的缩放矩阵
        scene_center = self.nerf_dataset.get_center()
        length_scale = self.nerf_dataset.get_length_scale()
        
        self.scale_mat = torch.eye(4, dtype=torch.float32, device=self.device)
        self.scale_mat[0, 0] = self.scale_mat[1, 1] = self.scale_mat[2, 2] = length_scale
        self.scale_mat[:3, 3] = scene_center

    def _preload_gpu_batches(self):
        """预加载所有GPU批次"""
        self.gpu_batches = []
        # 遍历数据加载器，获取所有批次
        for idx, batch in enumerate(self.dataloader):
            # 将批次数据移动到GPU并获取含内参的批次
            gpu_batch = self.nerf_dataset.get_gpu_batch_with_intrinsics(batch)
            self.gpu_batches.append(gpu_batch)

    def gen_rays_at(self, idx, resolution_level=1):
        """
        生成指定图像索引的所有光线
        
        Args:
            idx: 图像索引
            resolution_level: 分辨率级别，1为全分辨率
            
        Returns:
            rays_o, rays_d, mask, rgb_gt: 光线起点、方向、掩码和真实RGB值
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
        mask = gpu_batch.mask[0] if hasattr(gpu_batch, 'mask') else torch.ones((H, W, 1), device=self.device)
        rgb_gt = gpu_batch.rgb_gt[0]
        
        if resolution_level > 1:
            rays_ori_world = rays_ori_world[::resolution_level, ::resolution_level, :]
            rays_dir_world = rays_dir_world[::resolution_level, ::resolution_level, :]
            mask = mask[::resolution_level, ::resolution_level, :]
            rgb_gt = rgb_gt[::resolution_level, ::resolution_level, :]
            
            if rgb_gt.max() > 1.5:
                rgb_gt = rgb_gt / 255.0
            if mask.max() > 1.5:
                mask = mask > 0.5
        
        return rays_ori_world, rays_dir_world, mask, rgb_gt

    def gen_random_rays_at(self, idx, batch_size):
        """
        在指定图像索引生成随机光线
        
        Args:
            idx: 图像索引
            batch_size: 光线数量
            
        Returns:
            rays_o, rays_d, mask, rgb_gt: 光线起点、方向、掩码和真实RGB值
        """
        # 获取预加载的批次
        gpu_batch = self.gpu_batches[idx]
        
        # 获取参数形状
        orig_batch_size, H, W, _ = gpu_batch.rays_ori.shape
        
        # 提取T_to_world矩阵
        T = gpu_batch.T_to_world[:, :3, :]  # [batch_size, 3, 4]
        
        # 处理射线原点(rays_ori)转换
        # 首先重塑rays_ori以便于矩阵乘法
        rays_ori_flat = gpu_batch.rays_ori.reshape(orig_batch_size, H*W, 3)  # [batch_size, H*W, 3]
        
        # 应用旋转(前3x3部分)和平移(最后一列)
        # 对于位置向量，我们需要完整的变换(旋转+平移)
        rays_ori_rotated = torch.bmm(rays_ori_flat, T[:, :, :3].transpose(1, 2))  # [batch_size, H*W, 3]
        
        # 添加平移部分
        translation = T[:, :, 3].unsqueeze(1).expand(-1, H*W, -1)  # [batch_size, H*W, 3]
        rays_ori_world_flat = rays_ori_rotated + translation  # [batch_size, H*W, 3]
        
        # 处理射线方向(rays_dir)转换
        # 首先重塑rays_dir以便于矩阵乘法
        rays_dir_flat = gpu_batch.rays_dir.reshape(orig_batch_size, H*W, 3)  # [batch_size, H*W, 3]
        
        # 对于方向向量，我们只需要应用旋转部分(前3x3部分)，不需要平移
        rays_dir_world_flat = torch.bmm(rays_dir_flat, T[:, :, :3].transpose(1, 2))  # [batch_size, H*W, 3]
        
        # 确保射线方向归一化
        rays_dir_world_flat = torch.nn.functional.normalize(rays_dir_world_flat, dim=2)
        
        # 将射线转为缩放后的世界坐标系，因为我们假设感兴趣的场景在单位球内
        # rays_dir_world_flat 不变，由于场景缩放只涉及平移和缩放，不涉及旋转，所以rays_dir_world_flat 不变
        # rays_ori_world_flat 需要先平移，再缩放
        rays_ori_world_flat = (rays_ori_world_flat - self.scale_mat[:3, 3]) / self.scale_mat[0, 0]
        
        # 重塑回原始形状
        rays_ori_world = rays_ori_world_flat.reshape(orig_batch_size, H, W, 3)  # [batch_size, H, W, 3]
        rays_dir_world = rays_dir_world_flat.reshape(orig_batch_size, H, W, 3)  # [batch_size, H, W, 3]
        
        # 获取图像尺寸
        H, W = gpu_batch.rays_ori.shape[1:3]

        # 随机选择像素
        pixels_x = torch.randint(low=0, high=W, size=[batch_size], device=self.device)
        pixels_y = torch.randint(low=0, high=H, size=[batch_size], device=self.device)
        
        # 提取选定位置的光线
        rays_o = rays_ori_world[0, pixels_y, pixels_x, :]  # [batch_size, 3]
        rays_d = rays_dir_world[0, pixels_y, pixels_x, :]  # [batch_size, 3]
        
        # 提取掩码和RGB真值
        if hasattr(gpu_batch, 'mask'):
            mask = gpu_batch.mask[0, pixels_y, pixels_x]  # [batch_size, 1]
        else:
            mask = torch.ones((batch_size, 1), device=self.device)
            
        rgb_gt = gpu_batch.rgb_gt[0, pixels_y, pixels_x, :]  # [batch_size, 3]
        
        return rays_o, rays_d, mask, rgb_gt

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
        
        if hasattr(gpu_batch, 'mask'):
            mask = gpu_batch.mask[0].cpu().numpy()  # [H, W, 1]
        else:
            # 如果没有掩码，返回全1掩码
            H, W, _ = gpu_batch.rgb_gt.shape[1:]
            mask = np.ones((H, W, 1), dtype=np.float32)
        
        # 如果需要降低分辨率
        if resolution_level > 1:
            mask = mask[::resolution_level, ::resolution_level]
        
        return mask 