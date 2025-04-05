import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import SDFNetwork, SingleVarianceNetwork
from models.renderer import NeuSRenderer


class Runner:
    def __init__(self, conf_path, mode='train', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        self.case = case
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        
        
        dataset_start_loading_time = time.time()
        self.dataset = Dataset(self.conf['dataset'])
        dataset_end_loading_time = time.time()
        print(f"Dataset loading time: {dataset_end_loading_time - dataset_start_loading_time} seconds")
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        params_to_train = []
        self.trans_shape_sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        params_to_train += list(self.trans_shape_sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.trans_shape_renderer = NeuSRenderer(self.trans_shape_sdf_network,
                                     self.deviation_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
            model_list = []
            for model_name in model_list_raw:
                if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                    model_list.append(model_name)
            model_list.sort()
            latest_model_name = model_list[-1]

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()
            
    def calculate_loss_from_rendering(self, rays_o, rays_d, mask):
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
        render_out = self.trans_shape_renderer.render(rays_o, rays_d, near, far)
        
        # weights.shape: [Batch_size, 1]
        # gradients.shape: [Batch_size, sample_num, 3]
        weights_sum = render_out['weights_sum']
        gradients = render_out['gradients']
        
        # 计算silhouette loss        
        silhouette_loss = torch.nn.functional.binary_cross_entropy(weights_sum.clip(1e-3, 1-1e-3), (mask > 0.5).float()) / rays_o.shape[0]
        
        # 计算颜色损失
        color_loss = torch.nn.functional.l1_loss(weights_sum, (mask > 0.5).float(), reduction='mean')
        
        # 计算eikonal loss
        eikonal_loss = render_out['eikonal_loss']
        
        # 计算总损失
        loss = silhouette_loss + color_loss + eikonal_loss
        
        return loss

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        render_idx = 0
        for iter_i in tqdm(range(res_step)):
            data = self.dataset.gen_random_rays_at(image_perm[self.iter_step % len(image_perm)], self.batch_size)

            rays_o, rays_d, mask = data[:, :3], data[:, 3: 6], data[:, 9: 10]

            loss = self.calculate_loss_from_rendering(rays_o, rays_d, mask)
            

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1
            
            freq = 1000
            self.report_freq = freq
            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss, self.optimizer.param_groups[0]['lr']))

            self.save_freq = freq
            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()
                
            self.render_freq = freq
            if self.iter_step % self.render_freq == 0:
                self.render_silhouette(idx=render_idx, resolution_level=8)
                render_idx = (render_idx + 1) % self.dataset.n_images

            self.val_mesh_freq = freq
            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.iter_step / self.anneal_end])

    def update_learning_rate(self):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.trans_shape_sdf_network.load_state_dict(checkpoint['trans_shape_sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'trans_shape_sdf_network_fine': self.trans_shape_sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))


    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        logging.info("Validate mesh for case {}, iter: {}, resolution: {}, threshold: {}".format(self.case, self.iter_step, resolution, threshold))
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)

        vertices, triangles =\
            self.trans_shape_renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        if world_space:
            vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', 'iter-{:0>8d}-resolution-{}.ply'.format(self.iter_step, resolution)))

        logging.info('End')
        
    def render_silhouette(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}, resolution: {}'.format(self.iter_step, idx, resolution_level))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        
        rays_o, rays_d, _ = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        
        out_weights = []
        # 为tqdm添加更多信息显示
        total_batches = len(rays_o)
        progress_bar = tqdm(
            zip(rays_o, rays_d),
            total=total_batches,
            desc=f"{self.case} 渲染剪影 (相机 {idx})",
            bar_format='{l_bar}{bar:30}{r_bar}',
            unit="批次",
            postfix={"分辨率": resolution_level, "迭代": self.iter_step}
        )
        for rays_o_batch, rays_d_batch in progress_bar:
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            render_out = self.trans_shape_renderer.render(rays_o_batch, rays_d_batch, near, far)
            
            # weights.shape: [Batch_size, 1]
            # gradients.shape: [Batch_size, sample_num, 3]
            weights = render_out['weights_sum']
            out_weights.append(weights.detach().cpu().numpy())
            
        out_weights = np.concatenate(out_weights, axis=0)
        out_weights = out_weights.reshape(H, W)

        silhouette_img = out_weights
        
        silhouette_img = (silhouette_img / (silhouette_img.max() + 1e-3)).clip(0, 1)
        silhouette_img = (silhouette_img * 255).astype(np.uint8)
        silhouette_ground_truth = self.dataset.mask_at(idx, resolution_level=resolution_level)

        # 解决维度不匹配问题的更强解决方案
        print(f"silhouette_img shape: {silhouette_img.shape}")
        print(f"silhouette_ground_truth shape before: {silhouette_ground_truth.shape}")
        
        # 确保 ground truth 是二维的
        if len(silhouette_ground_truth.shape) == 3:
            # 如果是三维的，移除最后一个维度或者取第一个通道
            if silhouette_ground_truth.shape[2] == 1:
                silhouette_ground_truth = silhouette_ground_truth[:, :, 0]
            else:
                # 如果有多个通道，可以取第一个通道或转为灰度
                silhouette_ground_truth = silhouette_ground_truth[:, :, 0]
        
        print(f"silhouette_ground_truth shape after: {silhouette_ground_truth.shape}")
        
        # 确保两个图像尺寸相同
        if silhouette_img.shape != silhouette_ground_truth.shape:
            silhouette_ground_truth = cv.resize(silhouette_ground_truth, 
                                               (silhouette_img.shape[1], silhouette_img.shape[0]),
                                               interpolation=cv.INTER_NEAREST)

        silhouette_path = os.path.join(self.base_exp_dir, 'silhouette')
        to_save_img = np.concatenate([silhouette_img, silhouette_ground_truth], axis=1)
        os.makedirs(silhouette_path, exist_ok=True)
        cv.imwrite(os.path.join(silhouette_path, 'iter-{:0>8d}-idx-{}-resolution-{}.png'.format(self.iter_step, idx, resolution_level)), to_save_img)
        return 
    
    def render_normal(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}, resolution: {}'.format(self.iter_step, idx, resolution_level))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        
        rays_o, rays_d, _ = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size)
        
        out_weights = []
        out_normals = []
        # 为tqdm添加更多信息显示
        total_batches = len(rays_o)
        progress_bar = tqdm(
            zip(rays_o, rays_d),
            total=total_batches,
            desc=f"{self.case} 渲染法向量 (相机 {idx})",
            bar_format='{l_bar}{bar:30}{r_bar}',
            unit="批次",
            postfix={"分辨率": resolution_level, "迭代": self.iter_step}
        )
        for rays_o_batch, rays_d_batch in progress_bar:
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)
            render_out = self.trans_shape_renderer.render(rays_o_batch, rays_d_batch, near, far)
            
            # weights.shape: [Batch_size, sample_num]
            # weights_sum.shape: [Batch_size, 1]
            # gradients.shape: [Batch_size, sample_num, 3]
            weights = render_out['weights']
            
            gradients = render_out['gradients']
            weights_sum = render_out['weights_sum']
            normals = (gradients * weights.unsqueeze(-1)).sum(dim=1, keepdim=False)
            normals = normals / (normals.norm(dim=-1, keepdim=True) + 1e-3)

            out_weights.append(weights_sum.detach().cpu().numpy())
            out_normals.append(normals.detach().cpu().numpy())

        out_weights = np.concatenate(out_weights, axis=0)
        out_normals = np.concatenate(out_normals, axis=0)

        out_weights = out_weights.reshape(H, W)
        out_normals = out_normals.reshape(H, W, 3)

        silhouette_img = out_weights        
        silhouette_img = (silhouette_img / (silhouette_img.max() + 1e-3)).clip(0, 1)
        silhouette_img = (silhouette_img * 255).astype(np.uint8)
        
        normals_img = out_normals.clip(0, 1)
        normals_img = (normals_img * 255).astype(np.uint8)

        silhouette_ground_truth = self.dataset.mask_at(idx, resolution_level=resolution_level)

        # 解决维度不匹配问题的更强解决方案
        print(f"silhouette_img shape: {silhouette_img.shape}")
        print(f"silhouette_ground_truth shape before: {silhouette_ground_truth.shape}")
        
        # 确保 ground truth 是二维的
        if len(silhouette_ground_truth.shape) == 3:
            # 如果是三维的，移除最后一个维度或者取第一个通道
            if silhouette_ground_truth.shape[2] == 1:
                silhouette_ground_truth = silhouette_ground_truth[:, :, 0]
            else:
                # 如果有多个通道，可以取第一个通道或转为灰度
                silhouette_ground_truth = silhouette_ground_truth[:, :, 0]
        
        print(f"silhouette_ground_truth shape after: {silhouette_ground_truth.shape}")
        
        # 确保两个图像尺寸相同
        if silhouette_img.shape != silhouette_ground_truth.shape:
            silhouette_ground_truth = cv.resize(silhouette_ground_truth, 
                                               (silhouette_img.shape[1], silhouette_img.shape[0]),
                                               interpolation=cv.INTER_NEAREST)

        normals_path = os.path.join(self.base_exp_dir, 'normals')
        to_save_img = np.concatenate([normals_img, silhouette_img.repeat(3, axis=1), silhouette_ground_truth.repeat(3, axis=1)], axis=0)
        os.makedirs(normals_path, exist_ok=True)
        cv.imwrite(os.path.join(normals_path, 'iter-{:0>8d}-idx-{}-resolution-{}.png'.format(self.iter_step, idx, resolution_level)), to_save_img)
        return 


if __name__ == '__main__':
    print('Hello Wenniu')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=False, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == 'render_silhouette':
        for idx in range(runner.dataset.n_images):
            runner.render_silhouette(idx=idx, resolution_level=8)
    elif args.mode == 'render_normals':
        for idx in range(runner.dataset.n_images):
            runner.render_normal(idx=idx, resolution_level=8)