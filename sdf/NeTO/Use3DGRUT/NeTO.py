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
from datasets.datasetAdapter import ColmapDatasetAdapter as Dataset
from models.fields import  SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer
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
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        params_to_train += list(self.nerf_outside.parameters())
        params_to_train += list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                     self.sdf_network,
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

    def initial_model(self, data):
        rays_o, rays_d, ray_point, mask, valid_mask = data[:, :3], data[:, 3: 6], \
                                                      data[:, 6: 9], data[:, 9: 10], data[:, 10:11][..., 0]
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
        background_rgb = None
        if self.use_white_bkgd:
            background_rgb = torch.ones([1, 3])
            
        mask = (mask > 0.5).float()

        mask_sum = mask.sum() + 1e-5
        
        render_out = self.renderer.render(rays_o, rays_d, near, far,
                                          background_rgb=background_rgb,
                                          cos_anneal_ratio=self.get_cos_anneal_ratio())
        gradient_error = render_out['gradient_error']
        weight_max = render_out['weight_max']
        weight_sum = render_out['weight_sum']
        s_val = render_out['s_val']
        cdf_fine = render_out['cdf_fine']
        
        mask_l1_loss = torch.abs(weight_sum - mask).mean()
        eikonal_loss = gradient_error
        mask_binary_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
        loss = eikonal_loss * self.igr_weight + mask_l1_loss * self.mask_l1_weight + mask_binary_loss * self.mask_binary_weight  
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.iter_step += 1

        self.writer.add_scalar('Loss/loss', loss, self.iter_step)
        self.writer.add_scalar('Loss/mask_l1_loss', mask_l1_loss, self.iter_step)
        self.writer.add_scalar('Loss/mask_binary_loss', mask_binary_loss, self.iter_step)
        self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
        self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
        self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
        self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
        return loss


    def detail_recontruction(self, data):
        rays_o, rays_d, ray_point, mask, valid_mask = data[:, :3], data[:, 3: 6], \
                                                      data[:, 6: 9], data[:, 9: 10], data[:, 10:11][..., 0]
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
        background_rgb = None
        valid_mask = valid_mask.bool()
        if self.use_white_bkgd:
            background_rgb = torch.ones([1, 3])

        if self.mask_weight > 0.0:
            mask = (mask > 0.5).float()
        else:
            mask = torch.ones_like(mask)
        mask_sum = mask.sum() + 1e-5
        render_out = self.renderer.render(rays_o, rays_d, near, far,
                                          background_rgb=background_rgb,
                                          cos_anneal_ratio=self.get_cos_anneal_ratio())
        gradient_error = render_out['gradient_error']
        weight_max = render_out['weight_max']
        weight_sum = render_out['weight_sum']
        s_val = render_out['s_val']
        cdf_fine = render_out['cdf_fine']
        normal_1 = render_out['gradients']
        inter_point = render_out['inter_point']

        l_t1, attenuate1, totalReflectMask1 = self.refraction(rays_d, normal_1,
                                                              eta1=1.0003, eta2=1.4723)
        rays_o = inter_point + l_t1 * 2
        rays_d = -l_t1
        near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)
        render_out = self.renderer.render(rays_o, rays_d, near, far,
                                          background_rgb=background_rgb,
                                          cos_anneal_ratio=self.get_cos_anneal_ratio()
                                          )
        normal_2 = render_out['gradients']
        inter_point2 = render_out['inter_point']
        weight_sum2 = render_out['weight_sum']
        render_out_dir2, attenuate2, totalReflectMask2 = self.refraction(-rays_d, -normal_2,
                                                                         eta1=1.4723, eta2=1.0003)

        check_sign, check_sdf = self.check_sdf_val(inter_point, inter_point2)
        check_sdf = (check_sdf > 1e-3).sum(1)
        occlusion_mask = check_sdf == 0



        valid_mask = valid_mask & (~totalReflectMask1[:, 0]) & (~totalReflectMask2[:, 0]) & (occlusion_mask)

        target = ray_point - inter_point2.detach()
        target = target / target.norm(dim=1, keepdim=True)
        diff = (render_out_dir2 - target)
        ray_loss = (diff[valid_mask]).pow(2).sum()

        eikonal_loss = gradient_error
        mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
        loss = mask_loss * self.mask_weight +\
               eikonal_loss * self.igr_weight + \
               ray_loss * self.refract_weight
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.iter_step += 1

        self.writer.add_scalar('Loss/loss', loss, self.iter_step)
        self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
        self.writer.add_scalar('Loss/ray_loss', ray_loss, self.iter_step)
        self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
        self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
        self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
        self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)

        return loss

    def check_sdf_val(self, intersection1, interseciont2):
        with torch.no_grad():

            first_second = (interseciont2 - intersection1)

            rays_d = first_second / torch.linalg.norm(first_second, ord=2, dim=-1, keepdim=True)

            rays_o = intersection1
            z_vals = torch.linspace(0.0, 1.0, self.renderer.n_samples)

            check_z_vals = torch.linalg.norm(interseciont2-intersection1, ord=2, dim=-1, keepdim=True) * z_vals[None, :]

            pts = rays_o[:, None, :] + rays_d[:, None, :] * check_z_vals[..., :, None]
            check_sdf = self.sdf_network.sdf(pts.reshape(-1, 3)).reshape(-1, self.renderer.n_samples)

            for i in range(self.renderer.up_sample_steps//2):
                new_check_z_vals = self.renderer.up_sample_occulsion(rays_o,
                                                            rays_d,
                                                            check_z_vals,
                                                            check_sdf,
                                                            self.renderer.n_importance // (self.renderer.up_sample_steps//2),
                                                            64 * 2 ** i)
                check_z_vals, check_sdf = self.renderer.cat_z_vals(rays_o,
                                                          rays_d,
                                                          check_z_vals,
                                                          new_check_z_vals,
                                                          check_sdf,
                                                          last=False)

            occlusion_sign = check_sdf.sign().detach()
        return occlusion_sign, check_sdf


    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()
        
        render_idx = 0
        
        for iter_i in tqdm(range(res_step)):
            indx = image_perm[self.iter_step % len(image_perm)]
            if self.iter_step > self.init_epoch:
                data, _, _ = self.dataset.gen_ray_masks_near(indx, self.batch_size)
                loss = self.detail_recontruction(data)

            else:
                data, _, _ = self.dataset.gen_random_rays_at(indx, self.batch_size)
                loss = self.initial_model(data)

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

    def refraction(self, l, normal, eta1, eta2):
        # l n x 3 x imHeight x imWidth
        # normal n x 3 x imHeight x imWidth
        # eta1 float
        # eta2 float
        cos_theta = torch.sum(l * (-normal), dim=1).unsqueeze(1)  # [10, 1, 192, 256] 鏍囬噺
        i_p = l + normal * cos_theta
        t_p = eta1 / eta2 * i_p

        t_p_norm = torch.sum(t_p * t_p, dim=1)
        totalReflectMask = (t_p_norm.detach() > 0.999999).unsqueeze(1)

        t_i = torch.sqrt(1 - torch.clamp(t_p_norm, 0, 0.999999)).unsqueeze(1).expand_as(normal) * (-normal)
        t = t_i + t_p
        t = t / torch.sqrt(torch.clamp(torch.sum(t * t, dim=1), min=1e-10)).unsqueeze(1)

        cos_theta_t = torch.sum(t * (-normal), dim=1).unsqueeze(1)

        e_i = (cos_theta_t * eta2 - cos_theta * eta1) / \
              torch.clamp(cos_theta_t * eta2 + cos_theta * eta1, min=1e-10)
        e_p = (cos_theta_t * eta1 - cos_theta * eta2) / \
              torch.clamp(cos_theta_t * eta1 + cos_theta * eta2, min=1e-10)

        attenuate = torch.clamp(0.5 * (e_i * e_i + e_p * e_p), 0, 1).detach()

        return t, attenuate, totalReflectMask

    def reflection(self, l, normal):
        # l n x 3 x imHeight x imWidth
        # normal n x 3 x imHeight x imWidth
        # eta1 float
        # eta2 float
        cos_theta = torch.sum(l * (-normal), dim=1).unsqueeze(1)
        r_p = l + normal * cos_theta
        r_p_norm = torch.clamp(torch.sum(r_p * r_p, dim=1), 0, 0.999999)
        r_i = torch.sqrt(1 - r_p_norm).unsqueeze(1).expand_as(normal) * normal
        r = r_p + r_i
        r = r / torch.sqrt(torch.clamp(torch.sum(r * r, dim=1), min=1e-10).unsqueeze(1))

        return r

    def get_image_perm(self):
        # 确保self.views是整数
        views = int(self.views)
        
        if views == self.dataset.n_images:
            return torch.randperm(self.dataset.n_images)
        elif self.dataset.n_images % views == 0:
            return torch.linspace(0, self.dataset.n_images-1, self.dataset.n_images)[::
                   int(self.dataset.n_images//views)].int()
        elif views == 20:
            # 如果特定的20视图不适用于当前的数据集大小，返回回退方案
            if self.dataset.n_images < 69:  # 原始代码中最大索引是68
                # 如果视图数量不足，返回均匀间隔的视图
                return torch.linspace(0, self.dataset.n_images-1, 20).int()
            else:
                return torch.tensor([0, 4, 8, 12, 16, 18, 20, 24, 28, 32, 36, 40, 44, 48, 52, 54, 56, 60, 64, 68])
        else:
            # 默认情况：在所有可用图像中随机选择固定数量的视图
            # 如果views大于可用图像，则使用所有图像
            if views > self.dataset.n_images:
                return torch.arange(0, self.dataset.n_images)
            else:
                # 均匀选择视图
                return torch.linspace(0, self.dataset.n_images-1, views).int()

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
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.iter_step = checkpoint['iter_step']
        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'nerf': self.nerf_outside.state_dict(),
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
        
        rays_o, rays_d, ray_points = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
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
            weights = render_out['weight_sum'].detach().cpu().numpy()
            out_weights.append(weights)
        
        out_weights = np.concatenate(out_weights, axis=0)
        out_weights = out_weights.reshape(H, W)

        silhouette_img = out_weights
        
        silhouette_img = (silhouette_img).clip(0, 1)
        silhouette_img = (silhouette_img * 255).astype(np.uint8)
        silhouette_img = silhouette_img[:, :, np.newaxis]
        
        mask_img = self.dataset.mask_at(idx, resolution_level=resolution_level)
        print("mask_img.shape: ", mask_img.shape)
        if isinstance(mask_img, torch.Tensor):
            mask_img = mask_img.detach().cpu().numpy()
        if mask_img.max() < 1.1:
            mask_img = (mask_img * 255).astype(np.uint8)
        if len(mask_img.shape) == 2:
            mask_img = mask_img[:, :, np.newaxis]
        
        silhouette_path = os.path.join(self.base_exp_dir, 'silhouette')
        to_save_img = np.concatenate([silhouette_img, mask_img], axis=1)
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
            cv.imwrite(os.path.join(save_folder, 'mask, iter-{:0>8d}-idx-{}-resolution-{}.png'.format(self.iter_step, i, 8)), mask)
            image = rgb_gt.detach().cpu().numpy()
            image = (image * 255).astype(np.uint8)
            cv.imwrite(os.path.join(save_folder, 'image, iter-{:0>8d}-idx-{}-resolution-{}.png'.format(self.iter_step, i, 8)), image)

if __name__ == '__main__':
    print('Hello ZC')

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
