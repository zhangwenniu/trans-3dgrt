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

import os
import copy

import numpy as np
from PIL import Image
import trimesh

import torch
from torch.utils.data import Dataset

from threedgrut.utils.logger import logger

from .protocols import Batch, BoundedMultiViewDataset, DatasetVisualization
from .utils import (
    create_camera_visualization,
    get_center_and_diag,
    pinhole_camera_rays,
    compute_max_radius,
    qvec_to_so3,
    read_colmap_extrinsics_binary,
    read_colmap_extrinsics_text,
    read_colmap_intrinsics_binary,
    read_colmap_intrinsics_text,
)
from .camera_models import (
    ShutterType,
    OpenCVPinholeCameraModelParameters,
    OpenCVFisheyeCameraModelParameters,
    image_points_to_camera_rays,
    pixels_to_image_points,
)


class ColmapDataset(Dataset, BoundedMultiViewDataset, DatasetVisualization):
    def __init__(self, path, device="cuda", split="train", downsample_factor=1, ray_jitter=None):
        self.path = path
        self.device = device
        self.split = split
        self.downsample_factor = downsample_factor
        self.ray_jitter = ray_jitter
        
        # 添加mask文件夹路径
        self.mask_folder = "mask"  # 可以考虑将其作为参数传入
        self.mask_expanded_folder = "mask_expanded"

        # GPU cache of processed camera intrinsics
        self.intrinsics = {}

        # Get the scene data
        self.load_intrinsics_and_extrinsics()
        self.get_scene_info()
        self.load_camera_data()

        llff_test_split = 8
        indices = np.arange(self.n_frames)
        if split == "train":
            indices = np.mod(indices, llff_test_split) != 0
        else:
            indices = np.mod(indices, llff_test_split) == 0
        self.poses = self.poses[indices].astype(np.float32)  # poses are numpy arrays
        self.image_data = self.image_data[indices]  # image_data are UINT8 umpy arrays
        assert self.image_data.dtype == np.uint8, "Image data must be of type uint8"
        
        self.mask_data = self.mask_data[indices]
        assert self.mask_data.dtype == np.uint8, "Mask data must be of type uint8"
        
        self.mask_expanded_data = self.mask_expanded_data[indices]
        assert self.mask_expanded_data.dtype == np.uint8, "Mask expanded data must be of type uint8"

        self.camera_centers = self.camera_centers[indices]
        self.center, self.length_scale, self.scene_bbox = self.compute_spatial_extents()

        # Update the number of frames to only include the samples from the split
        self.n_frames = self.poses.shape[0]
        
        self.scale_mat = self.compute_scale_mat()
        

    def load_intrinsics_and_extrinsics(self):
        try:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.bin")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.bin")
            self.cam_extrinsics = read_colmap_extrinsics_binary(cameras_extrinsic_file)
            self.cam_intrinsics = read_colmap_intrinsics_binary(cameras_intrinsic_file)
        except:
            cameras_extrinsic_file = os.path.join(self.path, "sparse/0", "images.txt")
            cameras_intrinsic_file = os.path.join(self.path, "sparse/0", "cameras.txt")
            self.cam_extrinsics = read_colmap_extrinsics_text(cameras_extrinsic_file)
            self.cam_intrinsics = read_colmap_intrinsics_text(cameras_intrinsic_file)

    def get_images_folder(self):
        downsample_suffix = "" if self.downsample_factor == 1 else f"_{self.downsample_factor}"
        return f"images{downsample_suffix}"

    def get_scene_info(self):
        self.image_h = 0
        self.image_w = 0
        self.n_frames = len(self.cam_extrinsics)
        image_path = os.path.join(self.path, self.get_images_folder(), os.path.basename(self.cam_extrinsics[0].name))
        image = np.asarray(Image.open(image_path))
        self.image_h = image.shape[0]
        self.image_w = image.shape[1]
        self.scaling_factor = int(
            round(self.cam_intrinsics[self.cam_extrinsics[0].camera_id - 1].height / self.image_h)
        )

    def load_camera_data(self):
        """Load the camera data and generate rays for each camera."""

        # Generate UV coordinates
        u = np.tile(np.arange(self.image_w), self.image_h)
        v = np.arange(self.image_h).repeat(self.image_w)
        out_shape = (1, self.image_h, self.image_w, 3)

        def create_pinhole_camera(focalx, focaly):
            params = OpenCVPinholeCameraModelParameters(
                resolution=np.array([self.image_w, self.image_h], dtype=np.int64),
                shutter_type=ShutterType.GLOBAL,
                principal_point=np.array([self.image_w, self.image_h], dtype=np.float32) / 2,
                focal_length=np.array([focalx, focaly], dtype=np.float32),
                radial_coeffs=np.zeros((6,), dtype=np.float32),
                tangential_coeffs=np.zeros((2,), dtype=np.float32),
                thin_prism_coeffs=np.zeros((4,), dtype=np.float32),
            )
            rays_o_cam, rays_d_cam = pinhole_camera_rays(
                u, v, focalx, focaly, self.image_w, self.image_h, self.ray_jitter
            )
            return (
                params.to_dict(),
                torch.tensor(rays_o_cam, dtype=torch.float32, device=self.device).reshape(out_shape),
                torch.tensor(rays_d_cam, dtype=torch.float32, device=self.device).reshape(out_shape),
                type(params).__name__,
            )

        def create_fisheye_camera(params):
            resolution = np.array([self.image_w, self.image_h]).astype(np.int64)
            principal_point = params[2:4].astype(np.float32)
            focal_length = params[0:2].astype(np.float32)
            radial_coeffs = params[4:].astype(np.float32)
            # Estimate max angle for fisheye
            max_radius_pixels = compute_max_radius(resolution.astype(np.float64), principal_point)
            fov_angle_x = 2.0 * max_radius_pixels / focal_length[0]
            fov_angle_y = 2.0 * max_radius_pixels / focal_length[1]
            max_angle = np.max([fov_angle_x, fov_angle_y]) / 2.0

            params = OpenCVFisheyeCameraModelParameters(
                principal_point=principal_point,
                focal_length=focal_length,
                radial_coeffs=radial_coeffs,
                resolution=resolution,
                # NOTE: fixed max angle might not apply to all datasets / better estimate from available data
                max_angle=max_angle,
                shutter_type=ShutterType.GLOBAL,
            )
            pixel_coords = torch.tensor(np.stack([u, v], axis=1), dtype=torch.int32, device=self.device)
            image_points = pixels_to_image_points(pixel_coords)
            rays_d_cam = image_points_to_camera_rays(params, image_points, device=self.device)
            rays_o_cam = torch.zeros_like(rays_d_cam)
            return (
                params.to_dict(),
                torch.tensor(rays_o_cam, dtype=torch.float32, device=self.device).reshape(out_shape),
                torch.tensor(rays_d_cam, dtype=torch.float32, device=self.device).reshape(out_shape),
                type(params).__name__,
            )

        for intr in self.cam_intrinsics:
            height = intr.height
            width = intr.width
            assert abs(height / self.scaling_factor - self.image_h) <= 1
            assert abs(width / self.scaling_factor - self.image_w) <= 1

            if intr.model == "SIMPLE_PINHOLE":
                focal_length = intr.params[0] / self.scaling_factor
                self.intrinsics[intr.id] = create_pinhole_camera(focal_length, focal_length)

            elif intr.model == "PINHOLE":
                focal_length_x = intr.params[0] / self.scaling_factor
                focal_length_y = intr.params[1] / self.scaling_factor
                self.intrinsics[intr.id] = create_pinhole_camera(focal_length_x, focal_length_y)

            elif intr.model == "OPENCV_FISHEYE":
                params = copy.deepcopy(intr.params)
                params[:4] = params[:4] / self.scaling_factor
                self.intrinsics[intr.id] = create_fisheye_camera(params)

            else:
                assert (
                    False
                ), f"Colmap camera model '{intr.model}' not handled: Only undistorted datasets (PINHOLE, SIMPLE_PINHOLE or OPENCV_FISHEYE cameras) supported!"

        self.poses = []
        self.image_data = []
        self.mask_data = []  # 添加mask数据列表
        self.mask_expanded_data = [] # 添加mask_expanded数据列表

        cam_centers = []
        for extr in logger.track(self.cam_extrinsics, description=f"Load Dataset ({self.split})", color="salmon1"):
            R = qvec_to_so3(extr.qvec)
            T = np.array(extr.tvec)
            W2C = np.zeros((4, 4), dtype=np.float32)
            W2C[:3, 3] = T
            W2C[:3, :3] = R
            W2C[3, 3] = 1.0
            C2W = np.linalg.inv(W2C)
            self.poses.append(C2W)
            cam_centers.append(C2W[:3, 3])
            
            # 加载图像数据
            image_basename = os.path.basename(extr.name)
            image_path = os.path.join(self.path, self.get_images_folder(), image_basename)
            image_data = np.asarray(Image.open(image_path))
            assert image_data.dtype == np.uint8, "Image data must be of type uint8"
            
            # 如果图像是单通道，转为三通道
            if len(image_data.squeeze().shape) == 2:
                image_data = np.repeat(image_data[:, :, None], 3, axis=-1)
            
            self.image_data.append(image_data)
            
            # 加载mask数据
            mask_path = os.path.join(self.path, self.mask_folder, image_basename)
            if os.path.exists(mask_path):
                mask_data = np.asarray(Image.open(mask_path))
                # 如果mask是彩色图像，将其转换为灰度图
                if len(mask_data.shape) == 3 and mask_data.shape[2] == 3:
                    mask_data = np.mean(mask_data, axis=2).astype(np.uint8)
                # 确保mask是二值图像（0或255）
                mask_data = (mask_data > 127).astype(np.uint8) * 255
            else:
                # 如果没有找到对应的mask文件，创建全1的mask
                logger.warning(f"Mask file not found: {mask_path}, using default mask")
                mask_data = np.zeros((self.image_h, self.image_w), dtype=np.uint8)
            
            # 加载mask_expanded数据
            mask_expanded_path = os.path.join(self.path, self.mask_expanded_folder, image_basename)
            if os.path.exists(mask_path):
                mask_expanded_data = np.asarray(Image.open(mask_expanded_path))
                # 如果mask是彩色图像，将其转换为灰度图
                if len(mask_expanded_data.shape) == 3 and mask_expanded_data.shape[2] == 3:
                    mask_expanded_data = np.mean(mask_expanded_data, axis=2).astype(np.uint8)
                # 确保mask是二值图像（0或255）
                mask_expanded_data = (mask_expanded_data > 127).astype(np.uint8) * 255
            else:
                # 如果没有找到对应的mask文件，创建全1的mask
                logger.warning(f"Mask_expanded file not found: {mask_expanded_path}, using default mask")
                mask_expanded_data = np.zeros((self.image_h, self.image_w), dtype=np.uint8)
                
            
            self.mask_data.append(mask_data)
            self.mask_expanded_data.append(mask_expanded_data)

        self.camera_centers = np.array(cam_centers)
        _, diagonal = get_center_and_diag(self.camera_centers)
        self.cameras_extent = diagonal * 1.1

        self.poses = np.stack(self.poses)
        self.image_data = np.stack(self.image_data)
        self.mask_data = np.stack(self.mask_data)
        self.mask_expanded_data = np.stack(self.mask_expanded_data)
        
    
    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        # shape: (-1, 1)
        return near, far

    @torch.no_grad()
    def compute_spatial_extents(self, spatial_compute_type='interest_region'):
        def from_pose():
            camera_origins = torch.FloatTensor(self.poses[:, :3, 3])
            center = camera_origins.mean(dim=0)
            dists = torch.linalg.norm(camera_origins - center[None, :], dim=-1)
            mean_dist = torch.mean(dists)  # mean distance between of cameras from center
            bbox_min = torch.min(camera_origins, dim=0).values
            bbox_max = torch.max(camera_origins, dim=0).values
            return center, mean_dist, (bbox_min, bbox_max)

        def from_interest_region():
            ply_path = os.path.join(self.path, 'sparse_points_interest.ply')
            if not os.path.exists(ply_path):
                logger.warning(f"Interest region PLY file not found: {ply_path}")
                raise FileNotFoundError(f"Interest region PLY file not found: {ply_path}")
            pcd = trimesh.load(ply_path)
            vertices = torch.from_numpy(pcd.vertices)  # 首先将vertices转换为torch张量
            bbox_max = torch.max(vertices, dim=0).values
            bbox_min = torch.min(vertices, dim=0).values
            center = (bbox_max + bbox_min) * 0.5
            radius = torch.norm(vertices - center.unsqueeze(0), p=2, dim=-1).max()
            return center, radius, (bbox_min, bbox_max)
        
        if spatial_compute_type == 'pose':
            try:
                return from_pose()
            except Exception as e:
                logger.warning(f"Error computing spatial extents from poses: {e}")
                return from_interest_region()
        elif spatial_compute_type == 'interest_region':
            return from_interest_region()
        else:
            raise ValueError(f"Invalid spatial compute type: {spatial_compute_type}")
    
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

    def get_intrinsics_idx(self, extr_idx: int):
        return self.cam_extrinsics[extr_idx].camera_id

    def __len__(self) -> int:
        return self.n_frames

    def __getitem__(self, idx) -> dict:
        out_shape = (1, self.image_h, self.image_w, 3)
        mask_shape = (1, self.image_h, self.image_w, 1)
        mask_expanded_shape = (1, self.image_h, self.image_w, 1)
        
        return {
            "data": torch.tensor(self.image_data[idx], dtype=torch.float32).reshape(out_shape),
            "mask": torch.tensor(self.mask_data[idx][..., None], dtype=torch.float32).reshape(mask_shape),
            "mask_expanded": torch.tensor(self.mask_expanded_data[idx][..., None], dtype=torch.float32).reshape(mask_shape),
            "pose": torch.tensor(self.poses[idx], dtype=torch.float32).unsqueeze(0),
            "intr": self.get_intrinsics_idx(idx),
        }

    def get_gpu_batch_with_intrinsics(self, batch):
        """Add the intrinsics to the batch and move data to GPU."""

        data = batch["data"][0].to(self.device, non_blocking=True) / 255.0
        mask = batch["mask"][0].to(self.device, non_blocking=True) / 255.0  # 转换为0-1范围
        mask_expanded = batch["mask_expanded"][0].to(self.device, non_blocking=True) / 255.0
        pose = batch["pose"][0].to(self.device, non_blocking=True)
        intr = batch["intr"][0].item()
        assert data.dtype == torch.float32
        assert pose.dtype == torch.float32

        camera_params_dict, rays_ori, rays_dir, camera_name = self.intrinsics[intr]

        sample = {
            "rgb_gt": data,
            "mask": mask,  # 添加mask到sample中
            "mask_expanded": mask_expanded,
            "rays_ori": rays_ori,
            "rays_dir": rays_dir,
            "T_to_world": pose,
            f"intrinsics_{camera_name}": camera_params_dict,
        }
        return Batch(**sample)

    def create_dataset_camera_visualization(self):
        """Create a visualization of the dataset cameras."""

        cam_list = []  # just one global intrinsic mat for now

        for i_cam, pose in enumerate(self.poses):
            trans_mat = pose
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

            intr, _, _, _ = self.intrinsics[self.get_intrinsics_idx(i_cam)]

            f_w = intr["focal_length"][0]
            f_h = intr["focal_length"][1]

            fov_w = 2.0 * np.arctan(0.5 * w / f_w)
            fov_h = 2.0 * np.arctan(0.5 * h / f_h)

            rgb = self.image_data[i_cam].reshape(h, w, 3) / np.float32(255.0)
            assert rgb.dtype == np.float32, "RGB image must be of type float32, but got {}".format(rgb.dtype)

            cam_list.append(
                {
                    "ext_mat": trans_mat_world_to_camera,
                    "w": w,
                    "h": h,
                    "fov_w": fov_w,
                    "fov_h": fov_h,
                    "rgb_img": rgb,
                    "split": self.split,
                }
            )

        create_camera_visualization(cam_list)
