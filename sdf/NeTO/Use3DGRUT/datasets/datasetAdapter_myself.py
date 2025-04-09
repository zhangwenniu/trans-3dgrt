from . import ColmapDataset
import torch
import numpy as np
import cv2 as cv
from .utils import MultiEpochsDataLoader

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

class DatasetAdapter:
    def __init__(self, conf):
        super(DatasetAdapter, self).__init__()
        print('Load data: Begin')
        
        self.device = torch.device('cuda')
        self.conf = conf
        self.data_dir = conf.get_string('data_dir')
        
        # 数据集路径，请替换为您自己的Colmap数据集路径
        dataset_path = self.data_dir
        
        # 创建Colmap数据集
        self.train_dataset = ColmapDataset(
            dataset_path,
            split="train",          # 使用训练集分割
            downsample_factor=1.0,  # 可以设置为>1的值来下采样图像
            ray_jitter=None         # 训练时可以添加光线抖动，这里不使用
        )
        
        self.n_images = len(self.train_dataset)
        self.H = self.train_dataset.image_h
        self.W = self.train_dataset.image_w
        
        # # 创建验证集
        # self.val_dataset = ColmapDataset(
        #     dataset_path,
        #     split="val",
        #     downsample_factor=1.0
        # )
        
        # 打印数据集信息
        print(f"Train dataset size: {len(self.train_dataset)}")
        # print(f"Validation dataset size: {len(self.val_dataset)}")
        
        
        # 创建数据加载器
        train_dataloader = MultiEpochsDataLoader(
            self.train_dataset,
            num_workers=0,
            batch_size=1,
            shuffle=False,
            pin_memory=False,
            persistent_workers=False,
        )
        
        self.cached_train_dataloader = list(train_dataloader)
        
        # self.dataloader = MultiEpochsDataLoader(
        #     self.colmap_dataset,
        #     num_workers=0,  # 将num_workers设置为0以禁用多进程
        #     batch_size=1,
        #     shuffle=False,  # 不打乱数据，保持索引一致性
        #     pin_memory=False,  # 禁用pin_memory，因为数据集返回的已经是CUDA张量
        #     persistent_workers=False  # num_workers为0时不需要persistent_workers
        # )
        
        # 获取场景信息
        object_bbox = self.train_dataset.get_scene_bbox()
        self.object_bbox_min = object_bbox[0].cpu().numpy()
        self.object_bbox_max = object_bbox[1].cpu().numpy()
        
        self.scale_mat = self.train_dataset.scale_mat
        print("self.object_bbox_min: ", self.object_bbox_min)
        print("self.object_bbox_max: ", self.object_bbox_max)
        print("self.scale_mat: ", self.scale_mat)

    
    def gen_random_rays_at(self, img_idx, batch_size):
        # 获取数据批次
        batch = self.cached_train_dataloader[img_idx]
        gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)
        
        # 提取关键数据
        rays_ori = gpu_batch.rays_ori[0]      # 光线原点，形状为 [B, H, W, 3]
        rays_dir = gpu_batch.rays_dir[0]      # 光线方向，形状为 [B, H, W, 3]
        rgb_gt = gpu_batch.rgb_gt[0]         # 目标RGB值，形状为 [B, H, W, 3]
        mask = gpu_batch.mask[0]            # 掩码，形状为 [B, H, W, 1]
        # T_to_world用于将相机坐标系转换为世界坐标系
        T_to_world = gpu_batch.T_to_world[0]  # 形状为 [4, 4]
        
        # 计算或加载scale_mat，用于归一化世界坐标
        scale_mat = self.scale_mat  # 形状为 [4, 4]
        
        # 随机选择像素点
        pixels_x = torch.randint(low=0, high=self.W, size=[batch_size])
        pixels_y = torch.randint(low=0, high=self.H, size=[batch_size])
        
        # 采样光线原点、方向和目标值
        rays_o_cam = rays_ori[(pixels_y, pixels_x)]  # 相机坐标系下的光线原点
        rays_d_cam = rays_dir[(pixels_y, pixels_x)]  # 相机坐标系下的光线方向
        mask_samples = mask[(pixels_y, pixels_x)]
        rgb_gt_samples = rgb_gt[(pixels_y, pixels_x)]
        
        # 步骤1: 相机坐标系 → 原始世界坐标系
        # 提取T_to_world的旋转和平移部分
        rotation = T_to_world[:3, :3]
        translation = T_to_world[:3, 3]
        
        # 应用旋转和平移到光线原点
        rays_o_world = torch.matmul(rays_o_cam, rotation.T) + translation
        
        # 只应用旋转到光线方向
        rays_d_world = torch.matmul(rays_d_cam, rotation.T)
        rays_d_world = torch.nn.functional.normalize(rays_d_world, dim=1)
        
        # # 步骤2: 原始世界坐标系 → 归一化世界坐标系
        # # 提取scale_mat的旋转、缩放和平移
        # scale_rotation = scale_mat[:3, :3]  # 实际上是缩放和旋转的组合
        # scale_translation = scale_mat[:3, 3]
        
        # # 应用逆变换 (从原始世界坐标系到归一化世界坐标系)
        # # 公式: p_normalized = scale_rotation_inv @ (p_world - scale_translation)
        # scale_rotation_inv = torch.inverse(scale_rotation)
        
        # # 应用到光线原点
        # rays_o_normalized = torch.matmul(rays_o_world - scale_translation, scale_rotation_inv.T)
        
        # # 应用到光线方向 (仅旋转部分，不含平移)
        # rays_d_normalized = torch.matmul(rays_d_world, scale_rotation_inv.T)
        # rays_d_normalized = torch.nn.functional.normalize(rays_d_normalized, dim=1)
        
        
        
        # 返回归一化空间中的光线数据
        return torch.cat([rays_o_world, rays_d_world, mask_samples, rgb_gt_samples], dim=-1)
        
    def near_far_from_sphere(self, rays_o, rays_d):
        # 解释：由于已经将场景归一化到单位球内，所以near和far的计算可以采用NeuS的计算方式
        # a = torch.sum(rays_d ** 2, dim=-1, keepdim=True)
        # b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        # mid = 0.5 * (-b) / a
        # near = mid - 1.0
        # far = mid + 1.0
        # return near, far
        batch_size = rays_o.shape[0]

        near_far_by_camera_position = self.train_dataset.near_far_from_sphere()

        near = near_far_by_camera_position[0] * torch.ones_like(rays_o[..., :1])
        far = near_far_by_camera_position[1] * torch.ones_like(rays_o[..., :1])
        return near, far
    
    def mask_at(self, idx, resolution_level=1):
        batch = self.cached_train_dataloader[idx]
        gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)
        
        # 提取关键数据
        mask = gpu_batch.mask[0]         # 掩码，形状为 [B, H, W, 1]
        if resolution_level > 1:
            mask = mask[::resolution_level, ::resolution_level]
        return mask

    def image_at(self, idx, resolution_level=1):
        batch = self.cached_train_dataloader[idx]
        gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)
        
        # 提取关键数据
        rgb_gt = gpu_batch.rgb_gt[0]         # 目标RGB值，形状为 [B, H, W, 3]
        if resolution_level > 1:
            rgb_gt = rgb_gt[::resolution_level, ::resolution_level]
        return rgb_gt
    
    def gen_rays_at(self, img_idx, resolution_level=1):
        """
        Generate rays at world space from one camera.
        """
        
        # 获取数据批次
        batch = self.cached_train_dataloader[img_idx]
        gpu_batch = self.train_dataset.get_gpu_batch_with_intrinsics(batch)
        
        # 提取关键数据
        rays_ori = gpu_batch.rays_ori[0]      # 光线原点，形状为 [B, H, W, 3]
        rays_dir = gpu_batch.rays_dir[0]      # 光线方向，形状为 [B, H, W, 3]
        rgb_gt = gpu_batch.rgb_gt[0]         # 目标RGB值，形状为 [B, H, W, 3]
        mask = gpu_batch.mask[0]            # 掩码，形状为 [B, H, W, 1]
        
        # T_to_world用于将相机坐标系转换为世界坐标系
        T_to_world = gpu_batch.T_to_world[0]  # 形状为 [4, 4]
        
        # 计算或加载scale_mat，用于归一化世界坐标
        scale_mat = self.scale_mat  # 形状为 [4, 4]
        
        # 基于resolution_level，生成像素点
        l = resolution_level
        tx = torch.linspace(0, self.W - 1, self.W // l)  # 保持为float
        ty = torch.linspace(0, self.H - 1, self.H // l)  # 保持为float
        pixels_x, pixels_y = torch.meshgrid(tx, ty)
        
        # 只在用作索引时转换为long类型
        pixels_x_idx = pixels_x.long()
        pixels_y_idx = pixels_y.long()
        
        # 采样光线原点、方向和目标值
        rays_o_cam = rays_ori[pixels_y_idx, pixels_x_idx, :]  # 相机坐标系下的光线原点
        rays_d_cam = rays_dir[pixels_y_idx, pixels_x_idx, :]  # 相机坐标系下的光线方向
        mask_samples = mask[pixels_y_idx, pixels_x_idx, :]
        rgb_gt_samples = rgb_gt[pixels_y_idx, pixels_x_idx, :]
        
        # 步骤1: 相机坐标系 → 原始世界坐标系
        # 提取T_to_world的旋转和平移部分
        rotation = T_to_world[:3, :3]
        translation = T_to_world[:3, 3]
        
        # 应用旋转和平移到光线原点
        rays_o_world = torch.matmul(rays_o_cam, rotation.T) + translation
        
        # 只应用旋转到光线方向
        rays_d_world = torch.matmul(rays_d_cam, rotation.T)
        rays_d_world = torch.nn.functional.normalize(rays_d_world, dim=1)
        
        # # 步骤2: 原始世界坐标系 → 归一化世界坐标系
        # # 提取scale_mat的旋转、缩放和平移
        # scale_rotation = scale_mat[:3, :3]  # 实际上是缩放和旋转的组合
        # scale_translation = scale_mat[:3, 3]
        
        # # 应用逆变换 (从原始世界坐标系到归一化世界坐标系)
        # # 公式: p_normalized = scale_rotation_inv @ (p_world - scale_translation)
        # scale_rotation_inv = torch.inverse(scale_rotation)
        
        # # 应用到光线原点
        # rays_o_normalized = torch.matmul(rays_o_world - scale_translation, scale_rotation_inv.T)
        
        # # 应用到光线方向 (仅旋转部分，不含平移)
        # rays_d_normalized = torch.matmul(rays_d_world, scale_rotation_inv.T)
        # rays_d_normalized = torch.nn.functional.normalize(rays_d_normalized, dim=1)
        
        # 返回归一化空间中的光线数据
        H, W = len(ty), len(tx)
        
        rays_o = rays_o_world.reshape(W, H, 3).permute(1, 0, 2)  # 调整为正确的图像布局
        rays_d = rays_d_world.reshape(W, H, 3).permute(1, 0, 2)
        mask = mask_samples.reshape(W, H, 1).permute(1, 0, 2)
        rgb_gt = rgb_gt_samples.reshape(W, H, 3).permute(1, 0, 2)
        
        return rays_o, rays_d, mask, rgb_gt


        