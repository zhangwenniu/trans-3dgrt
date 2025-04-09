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
# from datasets.datasetAdapter import DatasetAdapter as Dataset
# from datasets.datasetAdapter_neus import ColmapDatasetAdapter as Dataset
from datasets.datasetAdapter_neus import NeRFWithMaskDatasetAdapter as Dataset
from models_silhouette.fields import SDFNetwork, SingleVarianceNetwork
from models_silhouette.renderer import NeuSRenderer
from pathlib import Path


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
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.init_epoch = self.conf.get_int('train.init_epoch')
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')

        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')
        self.uncertain_map = self.conf.get_bool('train.uncertain_map')

        self.views = self.conf.get_float('train.views', default=72)

        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.extIOR =  self.conf.get_float('train.extIOR', default=1.0003)
        self.intIOR = self.conf.get_float('train.intIOR', default=1.4723)
        self.decay_rate  = self.conf.get_float('train.decay_rate', default=0.1)
        self.n_samples = self.conf.get_int('model.neus_renderer.n_samples', default=64)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.refract_weight = self.conf.get_float('train.refract_weight')
        self.mask_l1_weight = self.conf.get_float('train.mask_l1_weight')
        self.mask_binary_weight = self.conf.get_float('train.mask_binary_weight')
        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.uncertain_masks = []
        self.writer = None

        # Networks
        params_to_train = []
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.sdf_network,
                                     self.deviation_network,
                                     **self.conf['model.neus_renderer'])

        # Load checkpoint
        latest_model_name = None
        if is_continue:
            checkpoint_dir = Path(self.base_exp_dir) / 'checkpoints'
            if checkpoint_dir.exists():
                model_list_raw = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
                model_list = []
                for model_name in model_list_raw:
                    if model_name[-3:] == 'pth' and int(model_name[5:-4]) <= self.end_iter:
                        model_list.append(model_name)
                model_list.sort()
                latest_model_name = model_list[-1]
            else:
                print('Checkpoints Folder "{}" does not exists. Train from scratch. '.format(checkpoint_dir))

        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()
            
            
    def calculate_loss_from_rendering(self, rays_o, rays_d, mask, rgb_gt):
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
        render_out = self.renderer.render(rays_o, rays_d, near, far)
        
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
            indx = image_perm[self.iter_step % len(image_perm)]
            
            rays_o, rays_d, mask, rgb_gt = self.dataset.gen_random_rays_at(indx, self.batch_size)
            loss = self.calculate_loss_from_rendering(rays_o, rays_d, mask, rgb_gt)
            
            self.iter_step += 1
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            
            if self.iter_step % self.report_freq == 0:
                print(self.base_exp_dir)
                print('iter:{:8>d} loss = {} lr={}'.format(self.iter_step, loss,
                                                           self.optimizer.param_groups[0]['lr']))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()
                
            if self.iter_step < 50000:
                render_freq = 1000
                self.val_mesh_freq = 2000
            else:
                render_freq = 5000
                self.val_mesh_freq = 5000
                
            if self.iter_step % render_freq == 0:
                self.render_silhouette(render_idx, 8)
                render_idx = (render_idx + 1) % self.dataset.n_images

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh(world_space=True, resolution=64, threshold=args.mcube_threshold)

            self.update_learning_rate()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

            # 每隔一定迭代次数清理显存
            if self.iter_step % 1000 == 0:
                torch.cuda.empty_cache()

    def get_image_perm(self):
        views = self.dataset.n_images
        return torch.randperm(views)

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

        # for g in self.optimizer.param_groups:
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
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.iter_step = checkpoint['iter_step']
        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'optimizerNoColor': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_mesh(self, world_space=False, resolution=256, threshold=0.0):
        bound_min = self.dataset.object_bbox_min
        bound_max = self.dataset.object_bbox_max

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        
        if world_space:
            scale_mat = self.dataset.scale_mat
            if isinstance(scale_mat, torch.Tensor):
                scale_mat = scale_mat.cpu().numpy()
            vertices = vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]

        mesh = trimesh.Trimesh(vertices, triangles)
        save_path = os.path.join(self.base_exp_dir, 'meshes', '{}_resolution_{}_{:0>8d}.ply'.format('world_space' if world_space else 'object_space', resolution, self.iter_step))
        print(save_path)
        mesh.export(save_path)

        logging.info('End')

    def render_silhouette(self, idx=-1, resolution_level=8):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}, resolution: {}'.format(self.iter_step, idx, resolution_level))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        
        rays_o, rays_d, mask, rgb_gt = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        print("rays_o.shape: ", rays_o.shape)
        print("rays_d.shape: ", rays_d.shape)
        
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
            render_out = self.renderer.render(rays_o_batch, rays_d_batch, near, far)
            
            # weights.shape: [Batch_size, 1]
            # gradients.shape: [Batch_size, sample_num, 3]
            weights = render_out['weights_sum'].detach().cpu().numpy()
            out_weights.append(weights)
        
        out_weights = np.concatenate(out_weights, axis=0)
        out_weights = out_weights.reshape(H, W)

        silhouette_img = out_weights
        
        silhouette_img = (silhouette_img).clip(0, 1)
        silhouette_img = (silhouette_img * 255).astype(np.uint8)
        silhouette_img = silhouette_img[:, :, np.newaxis]
        silhouette_img = np.repeat(silhouette_img, 3, axis=2)
        
        mask_img = mask
        print("mask_img.shape: ", mask_img.shape)
        if isinstance(mask_img, torch.Tensor):
            mask_img = mask_img.detach().cpu().numpy()
        if mask_img.max() < 1.1:
            mask_img = (mask_img * 255).astype(np.uint8)
        if len(mask_img.shape) == 2:
            mask_img = mask_img[:, :, np.newaxis]
        mask_img = np.repeat(mask_img, 3, axis=2)
        
        rgb_img = rgb_gt.detach().cpu().numpy()
        if rgb_img.max() < 1.1:
            rgb_img = (rgb_img * 255).astype(np.uint8)
        rgb_img = rgb_img[:, :, [2, 1, 0]]
        
        silhouette_path = os.path.join(self.base_exp_dir, 'silhouette')
        to_save_img = np.concatenate([silhouette_img, mask_img, rgb_img], axis=1)
        os.makedirs(silhouette_path, exist_ok=True)
        cv.imwrite(os.path.join(silhouette_path, 'iter-{:0>8d}-idx-{}-resolution-{}.png'.format(self.iter_step, idx, resolution_level)), to_save_img)
        return 

    def validate_loader(self):
        save_folder = os.path.join(self.base_exp_dir, 'validate_loader')
        os.makedirs(save_folder, exist_ok=True)
        for i in tqdm(range(self.dataset.n_images), desc="validating loader"):
            mask = self.dataset.mask_at(i, resolution_level=8)
            mask = mask.detach().cpu().numpy()
            if mask.max() < 1.1:
                mask = (mask * 255).astype(np.uint8)
            cv.imwrite(os.path.join(save_folder, 'mask, iter-{:0>8d}-idx-{}-resolution-{}.png'.format(self.iter_step, i, 8)), mask)
            image = self.dataset.image_at(i, resolution_level=8)
            image = image.detach().cpu().numpy()
            if image.max() < 1.1:
                image = (image * 255).astype(np.uint8)
            # 图片格式是RGB，cv存储的时候需要转换为BGR
            image = image[:, :, [2, 1, 0]]
            cv.imwrite(os.path.join(save_folder, 'image, iter-{:0>8d}-idx-{}-resolution-{}.png'.format(self.iter_step, i, 8)), image)
    
    def validate_loader_by_rays_at(self):
        save_folder = os.path.join(self.base_exp_dir, 'validate_loader_by_rays_at')
        os.makedirs(save_folder, exist_ok=True)
        for i in tqdm(range(self.dataset.n_images), desc="validating loader by rays_at"):
            rays_o, rays_d, mask, rgb_gt = self.dataset.gen_rays_at(i, resolution_level=8)
            mask = mask.detach().cpu().numpy()
            if mask.max() < 1.1:
                mask = (mask * 255).astype(np.uint8)
            mask = np.repeat(mask, 3, axis=2)
            cv.imwrite(os.path.join(save_folder, 'mask, iter-{:0>8d}-idx-{}-resolution-{}.png'.format(self.iter_step, i, 8)), mask)
            image = rgb_gt.detach().cpu().numpy()
            if image.max() < 1.1:
                image = (image * 255).astype(np.uint8)
            image = image[:, :, [2, 1, 0]]
            cv.imwrite(os.path.join(save_folder, 'image, iter-{:0>8d}-idx-{}-resolution-{}.png'.format(self.iter_step, i, 8)), image)


if __name__ == '__main__':
    print('Hello Zhangwenniu')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='pig')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=True, resolution=512, threshold=args.mcube_threshold)
    elif args.mode == 'validate_loader':
        runner.validate_loader()
    elif args.mode == 'validate_loader_by_rays_at':
        runner.validate_loader_by_rays_at()
