# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import torch
from torch.utils.data import Dataset

import cv2
import imageio
from PIL import Image
import numpy as np

from einops import rearrange
from kornia import create_meshgrid

from threedgrut.utils.logger import logger

from .protocols import Batch, BoundedMultiViewDataset, DatasetVisualization
from .utils import create_camera_visualization, get_center_and_diag


class NeRFDatasetWithMask(Dataset, BoundedMultiViewDataset, DatasetVisualization):
    def __init__(self, path, device="cuda", split="train", return_masks=True, ray_jitter=None, bg_color=None):
        self.root_dir = path
        self.device = device
        self.split = split
        self.return_masks = return_masks  # 默认返回mask
        self.ray_jitter = ray_jitter
        self.bg_color = bg_color

        self.read_intrinsics()
        self.read_meta(split)
        self.center, self.length_scale, self.scene_bbox = self.compute_spatial_extents()

        # GPU-cached camera rays
        directions = NeRFDatasetWithMask.__get_ray_directions(
            self.image_h,
            self.image_w,
            torch.tensor(self.K, device=self.device),
            device=self.device,
            ray_jitter=self.ray_jitter,
        )
        self.rays_o_cam = torch.zeros((1, self.image_h, self.image_w, 3), dtype=torch.float32, device=self.device)
        self.rays_d_cam = directions.reshape((1, self.image_h, self.image_w, 3)).contiguous()

        assert self.colors.dtype == np.uint8, "RGB image must be of type uint8"
        if self.return_masks and hasattr(self, 'masks'):
            assert self.masks.dtype == np.uint8, "Mask image must be of type uint8"
            
        self.scale_mat = self.compute_scale_mat()
        
    @torch.no_grad()
    def compute_scale_mat(self):
        """
        return: 
            4x4 缩放矩阵, [0, 0] = [1, 1] = [2, 2] = length_scale, [3, 3] = 1.0
            [:3, 3] = 场景中心在原世界坐标系下的坐标
        """
        # 如果需要将场景放在一个单位球内，也就是NeuS的假设
        scale_mat = torch.diag(torch.tensor([self.length_scale, self.length_scale, self.length_scale, 1.0], dtype=torch.float32))
        scale_mat[:3, 3] = self.center.float()  # 确保是float32
        return scale_mat

    def read_intrinsics(self):
        transform_file = os.path.join(self.root_dir, self.split, "transforms.json")
        with open(transform_file, "r") as f:
            meta = json.load(f)

        # !! Assumptions !!
        # 1. All images have the same intrinsics
        # 2. Principal point is at canvas center
        # 3. Camera has no distortion params
        first_frame_path = meta["frames"][0]["file_path"]
        img_path = os.path.join(self.root_dir, self.split, first_frame_path)

        # Check if the image path has an extension
        if os.path.exists(img_path):
            self.suffix = ""
        elif os.path.exists(img_path + ".png"):
            self.suffix = ".png"
        elif os.path.exists(img_path + ".jpg"):
            self.suffix = ".jpg"
        else:
            raise FileNotFoundError(f"Image path {img_path} does not exist.")

        frame = Image.open(img_path + self.suffix)

        w = frame.width
        h = frame.height
        self.img_wh = (w, h)

        fx = fy = 0.5 * w / np.tan(0.5 * meta["camera_angle_x"])

        self.K = np.float32([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]])
        self.intrinsics = [fx, fy, w / 2, h / 2]

    def read_meta(self, split):
        self.colors = []
        self.masks = []
        self.poses = []

        transform_file = os.path.join(self.root_dir, split, "transforms.json")
        with open(transform_file, "r") as f:
            frames = json.load(f)["frames"]

        cam_centers = []
        for frame in logger.track(frames, description=f"Load Dataset ({split})", color="salmon1"):
            c2w = np.array(frame["transform_matrix"])[:3, :4]
            c2w[:, 1:3] *= -1  # [right up back] to [right down front]
            cam_centers.append(c2w[:3, 3])
            self.poses.append(c2w)

            # 读取RGB图像
            img_path = os.path.join(self.root_dir, split, f"{frame['file_path']}") + self.suffix
            img = NeRFDatasetWithMask.__read_image(img_path, self.img_wh, return_alpha=False, bg_color=self.bg_color)
            self.colors.append(img)
            
            # 读取mask图像
            if self.return_masks and 'silhouette_path' in frame:
                mask_path = os.path.join(self.root_dir, split, f"{frame['silhouette_path']}") + self.suffix
                if os.path.exists(mask_path):
                    mask = NeRFDatasetWithMask.__read_mask(mask_path, self.img_wh)
                    self.masks.append(mask)
                else:
                    logger.warning(f"Mask file not found: {mask_path}")
                    # 如果找不到mask文件，创建一个全黑的mask
                    h, w = self.img_wh[1], self.img_wh[0]
                    white_mask = np.zeros((h*w, 1), dtype=np.uint8)
                    self.masks.append(white_mask)

        self.camera_centers = np.array(cam_centers)

        # https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/__init__.py#L69
        _, diagonal = get_center_and_diag(self.camera_centers)
        self.cameras_extent = diagonal * 1.1

        if len(self.colors) > 0:
            self.colors = np.stack(self.colors)  # (N_images, H, W, 3)

        if len(self.masks) > 0 and self.return_masks:
            self.masks = np.stack(self.masks)  # (N_images, H, W, 1)

        self.poses = np.array(self.poses).astype(np.float32)  # (N_images, 3, 4)

    @torch.no_grad()
    def compute_spatial_extents(self):
        camera_origins = torch.FloatTensor(self.poses[:, :, 3])
        center = camera_origins.mean(dim=0)
        dists = torch.linalg.norm(camera_origins - center[None, :], dim=-1)
        mean_dist = torch.mean(dists)  # mean distance between of cameras from center
        bbox_min = torch.min(camera_origins, dim=0).values
        bbox_max = torch.max(camera_origins, dim=0).values
        return center, mean_dist, (bbox_min, bbox_max)

    def get_length_scale(self):
        return self.length_scale

    def get_center(self):
        return self.center

    def get_scene_bbox(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.scene_bbox

    def get_scene_extent(self):
        return self.cameras_extent

    def get_observer_points(self):
        return self.camera_centers

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx) -> dict:
        out_shape = (1, self.image_h, self.image_w, 3)
        result = {
            "data": torch.tensor(self.colors[idx]).reshape(out_shape),
            "pose": torch.tensor(self.poses[idx]).unsqueeze(0),
        }
        
        # 添加mask数据
        if self.return_masks and hasattr(self, 'masks'):
            mask_shape = (1, self.image_h, self.image_w, 1)
            result["mask"] = torch.tensor(self.masks[idx]).reshape(mask_shape)
        
        return result

    def get_gpu_batch_with_intrinsics(self, batch):
        """
        将数据批次移至GPU设备并添加相机内参信息，为神经渲染准备完整的输入数据。
        
        Args:
            batch (dict): 输入的批次数据，包含以下键值对：
                - "data" (torch.Tensor): RGB图像数据，形状为(1, H, W, 3)，数据范围[0, 255]，类型为uint8
                - "pose" (torch.Tensor): 相机外参矩阵，形状为(1, 3, 4)，表示相机到世界坐标系的变换
                - "mask" (torch.Tensor, 可选): 物体掩码数据，形状为(1, H, W, 1)，数据范围[0, 255]，
                                             类型为uint8，值为255表示前景，0表示背景
            
        Returns:
            Batch: 包含以下属性的数据对象：
                - rgb_gt (torch.Tensor): GPU上的RGB图像数据，形状为(1, H, W, 3)，
                                        数据范围[0, 1]，类型为float32
                - rays_ori (torch.Tensor): 光线原点，形状为(1, H, W, 3)，表示相机坐标系中的零向量
                - rays_dir (torch.Tensor): 光线方向，形状为(1, H, W, 3)，是归一化的方向向量（单位长度）
                - T_to_world (torch.Tensor): 相机到世界坐标系的变换矩阵，形状为(3, 4)
                - intrinsics (list): 相机内参列表[fx, fy, cx, cy]，其中：
                    - fx, fy: 焦距（像素单位）
                    - cx, cy: 主点坐标（像素单位）
                - mask (torch.Tensor, 可选): GPU上的掩码数据，形状为(1, H, W, 1)，
                                           数据范围[0, 1]，类型为float32
        
        注意:
            - 所有输入张量会被移至指定的GPU设备
            - RGB图像和掩码数据会被归一化到[0, 1]范围
            - 如果self.return_masks为False或batch中不包含"mask"键，则返回的Batch对象不会包含mask属性
        """

        data = batch["data"][0].to(self.device, non_blocking=True) / 255.0
        pose = batch["pose"][0].to(self.device, non_blocking=True)
        assert data.dtype == torch.float32
        assert pose.dtype == torch.float32

        sample = {
            "rgb_gt": data,
            "rays_ori": self.rays_o_cam,
            "rays_dir": self.rays_d_cam,
            "T_to_world": pose,
            "intrinsics": self.intrinsics,
        }
        
        # 添加mask数据
        if self.return_masks and "mask" in batch:
            mask = batch["mask"][0].to(self.device, non_blocking=True) / 255.0
            assert mask.dtype == torch.float32
            sample["mask"] = mask
        
        return Batch(**sample)

    @property
    def image_h(self):
        return self.img_wh[1]

    @property
    def image_w(self):
        return self.img_wh[0]

    def create_dataset_camera_visualization(self):
        # just one global intrinsic mat for now
        intrinsics = self.K

        cam_list = []
        for i_cam, pose in enumerate(self.poses):
            trans_mat = np.eye(4)
            trans_mat[:3, :4] = pose
            trans_mat_world_to_camera = np.linalg.inv(trans_mat)

            # these cameras follow the opposite convention from polyscope
            camera_convention_rot = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            trans_mat_world_to_camera = camera_convention_rot @ trans_mat_world_to_camera

            w = self.image_w
            h = self.image_h

            f_w = intrinsics[0, 0]
            f_h = intrinsics[1, 1]

            fov_w = 2.0 * np.arctan(0.5 * w / f_w)
            fov_h = 2.0 * np.arctan(0.5 * h / f_h)

            rgb = self.colors[i_cam].reshape(h, w, 3) / np.float32(255.0)
            assert rgb.dtype == np.float32, "RGB image must be of type float32, but got {}".format(rgb.dtype)

            cam_data = {
                "ext_mat": trans_mat_world_to_camera,
                "w": w,
                "h": h,
                "fov_w": fov_w,
                "fov_h": fov_h,
                "rgb_img": rgb,
                "split": self.split,
            }
            
            # 添加mask可视化
            if self.return_masks and hasattr(self, 'masks'):
                mask = self.masks[i_cam].reshape(h, w, 1) / np.float32(255.0)
                assert mask.dtype == np.float32
                cam_data["mask_img"] = mask
                
            cam_list.append(cam_data)

        create_camera_visualization(cam_list)

    @staticmethod
    @torch.cuda.amp.autocast(dtype=torch.float32)
    def __get_ray_directions(H, W, K, device="cpu", ray_jitter=None, return_uv=False, flatten=True):
        """
        Get ray directions for all pixels in camera coordinate [right down front].
        Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
                ray-tracing-generating-camera-rays/standard-coordinate-systems

        Inputs:
            H, W: image height and width
            K: (3, 3) camera intrinsics
            ray_jitter: Optional RayJitter component, for whether the ray passes randomly inside the pixel
            return_uv: whether to return uv image coordinates

        Outputs: (shape depends on @flatten)
            directions: (H, W, 3) or (H*W, 3), the direction of the rays in camera coordinate
            uv: (H, W, 2) or (H*W, 2) image coordinates
        """
        grid = create_meshgrid(H, W, False, device=device)[0]  # (H, W, 2)
        u, v = grid.unbind(-1)

        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        if ray_jitter is None:  # pass by the center
            directions = torch.stack([(u - cx + 0.5) / fx, (v - cy + 0.5) / fy, torch.ones_like(u)], -1)
        else:
            jitter = ray_jitter(u.shape)
            directions = torch.stack(
                [((u + jitter[:, :, 0]) - cx) / fx, ((v + jitter[:, :, 1]) - cy) / fy, torch.ones_like(u)], -1
            )
        if flatten:
            directions = directions.reshape(-1, 3)
            grid = grid.reshape(-1, 2)

        if return_uv:
            return directions, grid

        return torch.nn.functional.normalize(directions, dim=-1)

    @staticmethod
    @torch.cuda.amp.autocast(dtype=torch.float32)
    def __get_rays(directions, c2w):
        """
        Get ray origin and directions in world coordinate for all pixels in one image.
        Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
                ray-tracing-generating-camera-rays/standard-coordinate-systems

        Inputs:
            directions: (N, 3) ray directions in camera coordinate
            c2w: (3, 4) or (N, 3, 4) transformation matrix from camera coordinate to world coordinate

        Outputs:
            rays_o: (N, 3), the origin of the rays in world coordinate
            rays_d: (N, 3), the direction of the rays in world coordinate
        """
        if c2w.ndim == 2:
            # Rotate ray directions from camera coordinate to the world coordinate
            rays_d = directions @ c2w[:, :3].T
        else:
            rays_d = rearrange(directions, "n c -> n 1 c") @ rearrange(c2w[..., :3], "n a b -> n b a")
            rays_d = rearrange(rays_d, "n 1 c -> n c")
        # The origin of all rays is the camera origin in world coordinate
        rays_o = c2w[..., 3].expand_as(rays_d)

        return rays_o, rays_d

    @staticmethod
    def __read_image(img_path, img_wh, return_alpha=False, bg_color=None):
        img = imageio.imread(img_path).astype(np.float32) / 255.0
        # img[..., :3] = srgb_to_linear(img[..., :3])

        # Below assume image is float32
        if img.shape[2] == 4:  # blend A to RGB
            if return_alpha:
                alpha = img[:, :, -1]
            if bg_color is None:
                img = img[..., :3]
            elif bg_color == "black":
                img = img[..., :3] * img[..., -1:]
            elif bg_color == "white":
                img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
            else:
                assert False, f"{bg_color} is not a supported background color."

        img = cv2.resize(img, img_wh)
        img = rearrange(img, "h w c -> (h w) c")

        # Convert to uint8 again
        img = (img * 255.0).astype(np.uint8)
        assert img.dtype == np.uint8, "Image must be uint8"

        if return_alpha:
            alpha = cv2.resize(alpha, img_wh)
            alpha = rearrange(alpha, "h w -> (h w)")
            return img, alpha
        else:
            return img
            
    @staticmethod
    def __read_mask(mask_path, img_wh):
        """
        读取并处理mask图像
        
        参数:
            mask_path: mask图像路径
            img_wh: 目标图像宽高
            
        返回:
            处理后的mask图像，形状为(H*W, 1)，类型为uint8
        """
        mask = imageio.imread(mask_path).astype(np.float32) / 255.0
        
        # 处理不同类型的mask图像
        if len(mask.shape) == 3:
            if mask.shape[2] == 4:  # RGBA格式，依然对RGB通道取平均
                mask = np.mean(mask, axis=2)
            elif mask.shape[2] == 3:  # RGB格式，灰度处理
                mask = np.mean(mask, axis=2)
            # 其他情况保持不变（可能是单通道重复三次）
            else:
                mask = mask[:, :, 0]  # 取第一个通道
        # 如果已经是单通道的，不需要额外处理
        
        # 调整大小
        mask = cv2.resize(mask, img_wh)
        
        # 重排形状
        mask = mask.reshape(-1, 1)
        
        # 二值化处理（可选）
        # mask = (mask > 0.5).astype(np.float32)
        
        # 转回uint8
        mask = (mask * 255.0).astype(np.uint8)
        
        return mask
