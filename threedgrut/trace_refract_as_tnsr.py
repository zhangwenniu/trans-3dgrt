import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import logging

logger = logging.getLogger(__name__)



class RefractTracer(nn.Module):
    def __init__(
            self, ior_air, ior_object,  
            n_samples=64, n_upsample=64, upsample=True,
        ):
        super(RefractTracer, self).__init__()
        self.ior_air = ior_air
        self.ior_object = ior_object
        self.n_samples = n_samples
        self.n_upsample = n_upsample
        self.upsample = upsample
        self.cal_grad = GradFunction.apply
        
    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = torch.clamp(mid - 1.0, min=0.0)
        far = mid + 1.0
        # shape: (-1, 1)
        return near, far
    
    def forward(self, rays_o, rays_d, sdf_network):
        rays_o, rays_d = copy.deepcopy(rays_o), copy.deepcopy(rays_d)
        rays_o_reflect, rays_d_reflect = copy.deepcopy(rays_o), copy.deepcopy(rays_d)
        
        valid_mask, intersection_depth, normals = self.intersection_with_sdf(rays_o, rays_d, sdf_network, enter_object=True)
        
        valid_ray_count = valid_mask.sum().item()
        logger.debug(f"First intersection found {valid_ray_count}/{rays_o.shape[0]} valid rays ({100.0*valid_ray_count/rays_o.shape[0]:.2f}%)")
        
        ret_dict = {
            "rays_o": rays_o,
            "rays_d": rays_d,
            "selected_indicies_final": [],
            "rays_o_reflect": rays_o_reflect,
            "rays_d_reflect": rays_d_reflect,
            "attenuate_1": [],
            "attenuate_2": [],
            "first_surface_points": [],
            "second_surface_points": []
        }
        
        if valid_mask.any():
            normals = normals.to(rays_o.device)
            normals = normals[valid_mask]
            
            valid_mask = valid_mask.to(rays_o.device)
            hit_rays_o = copy.deepcopy(rays_o[valid_mask])
            hit_rays_d = copy.deepcopy(rays_d[valid_mask])
            
            init_depth = intersection_depth[valid_mask].to(rays_o.device)
            logger.debug(f"First intersection depths: min={init_depth.min().item():.4f}, max={init_depth.max().item():.4f}, mean={init_depth.mean().item():.4f}")
            
            in_dir = hit_rays_d
            first_surface_points = hit_rays_o + init_depth.unsqueeze(1) * hit_rays_d
            
            # gradient calulating # 
            logger.debug("Computing gradient at first surface points")
            s1 = sdf_network.sdf(first_surface_points).squeeze()
            logger.debug(f"SDF at first surface: min={s1.min().item():.6f}, max={s1.max().item():.6f}, mean={s1.mean().item():.6f}")
            
            inputs = [s1, first_surface_points, in_dir, normals, init_depth]
            # 打印inputs中各元素的形状
            logger.debug(f"Inputs shapes:")
            logger.debug(f"  s1: {s1.shape}")
            logger.debug(f"  first_surface_points: {first_surface_points.shape}")
            logger.debug(f"  in_dir: {in_dir.shape}")
            logger.debug(f"  normals: {normals.shape}")
            logger.debug(f"  init_depth: {init_depth.shape}")
            init_depth = self.cal_grad(*inputs)
            logger.debug(f"Refined depths: min={init_depth.min().item():.4f}, max={init_depth.max().item():.4f}, mean={init_depth.mean().item():.4f}")
            
            first_surface_points = hit_rays_o + init_depth.unsqueeze(1) * hit_rays_d
            # gradient calulating # 
            
            # first refraction #
            logger.debug("Computing first refraction")
            refraction_direction_1, attenuate_1, totalReflectMask_1 = self.refraction(in_dir, normals, self.ior_air, self.ior_object)
            
            total_reflect_count = totalReflectMask_1.sum().item()
            logger.debug(f"Total internal reflection at first surface: {total_reflect_count}/{valid_ray_count} rays ({100.0*total_reflect_count/valid_ray_count:.2f}%)")
            logger.debug(f"First attenuation factors: min={attenuate_1.min().item():.4f}, max={attenuate_1.max().item():.4f}, mean={attenuate_1.mean().item():.4f}")
            
            # first reflection #
            logger.debug("Computing first reflection")
            reflection_direction_1 = self.reflection(in_dir, normals)
            
            hit_rays_o = first_surface_points.detach()
            hit_rays_d = refraction_direction_1.detach()
              
            
            # second intersection # 
            logger.debug("Computing second intersection")
            valid_mask_2, intersection_depth_2, normals_2 = self.intersection_with_sdf(hit_rays_o, hit_rays_d, sdf_network, enter_object=False)
            
            valid_ray_count_2 = valid_mask_2.sum().item()
            logger.debug(f"Second intersection found {valid_ray_count_2}/{valid_ray_count} valid rays ({100.0*valid_ray_count_2/valid_ray_count:.2f}%)")
            
            if valid_mask_2.any():
                normals_2 = normals_2.to(rays_o.device)
                normals_2 = normals_2[valid_mask_2]
                
                valid_mask_2 = valid_mask_2.to(rays_o.device)
                
                init_depth_2 = intersection_depth_2[valid_mask_2].to(rays_o.device)
                logger.debug(f"Second intersection depths: min={init_depth_2.min().item():.4f}, max={init_depth_2.max().item():.4f}, mean={init_depth_2.mean().item():.4f}")
                
                second_surface_points = hit_rays_o[valid_mask_2] + init_depth_2.unsqueeze(1) * hit_rays_d[valid_mask_2]
                in_dir_2 = hit_rays_d[valid_mask_2]
                
                selected_indicies_1 = torch.where(valid_mask)[0]
                selected_indicies_final = selected_indicies_1[valid_mask_2]
                logger.debug(f"Final successful ray count: {len(selected_indicies_final)}/{rays_o.shape[0]} ({100.0*len(selected_indicies_final)/rays_o.shape[0]:.2f}%)")
                
                # gradient calculating # 
                logger.debug("Computing gradient at second surface points")
                s2 = sdf_network.sdf(second_surface_points).squeeze()
                logger.debug(f"SDF at second surface: min={s2.min().item():.6f}, max={s2.max().item():.6f}, mean={s2.mean().item():.6f}")
                
                inputs_2 = [s2, second_surface_points, in_dir_2, normals_2, init_depth_2]
                logger.debug(f"Inputs2 shapes:")
                logger.debug(f"  s2: {s2.shape}")
                logger.debug(f"  second_surface_points: {second_surface_points.shape}")
                logger.debug(f"  in_dir_2: {in_dir_2.shape}")
                logger.debug(f"  normals_2: {normals_2.shape}")
                logger.debug(f"  init_depth_2: {init_depth_2.shape}")
                init_depth_2 = self.cal_grad(*inputs_2)
                logger.debug(f"Refined second depths: min={init_depth_2.min().item():.4f}, max={init_depth_2.max().item():.4f}, mean={init_depth_2.mean().item():.4f}")
                
                second_surface_points = hit_rays_o[valid_mask_2] + init_depth_2.unsqueeze(1) * hit_rays_d[valid_mask_2]
                # gradient calculating # 
                
                # second refraction # 
                logger.debug("Computing second refraction")
                refraction_direction_2, attenuate_2, totalReflectMask_2 = self.refraction(in_dir_2, -normals_2, self.ior_object, self.ior_air)
                
                total_reflect_count_2 = totalReflectMask_2.sum().item()
                logger.debug(f"Total internal reflection at second surface: {total_reflect_count_2}/{valid_ray_count_2} rays ({100.0*total_reflect_count_2/valid_ray_count_2:.2f}%)")
                logger.debug(f"Second attenuation factors: min={attenuate_2.min().item():.4f}, max={attenuate_2.max().item():.4f}, mean={attenuate_2.mean().item():.4f}")
                
                rays_o[selected_indicies_final] = second_surface_points
                rays_d[selected_indicies_final] = refraction_direction_2
                
                rays_o_reflect[selected_indicies_final] = first_surface_points[valid_mask_2].detach()
                rays_d_reflect[selected_indicies_final] = reflection_direction_1[valid_mask_2]
                
                logger.debug("Ray tracing completed successfully")
                
                ret_dict = {
                    "rays_o": rays_o,
                    "rays_d": rays_d,
                    "selected_indicies_final": selected_indicies_final,
                    "rays_o_reflect": rays_o_reflect,
                    "rays_d_reflect": rays_d_reflect,
                    "attenuate_1": attenuate_1[valid_mask_2],
                    "attenuate_2": attenuate_2,
                    "first_surface_points": first_surface_points.detach(),
                    "second_surface_points": second_surface_points.detach()
                }
                
                return ret_dict
            else:
                logger.warning("No valid second intersections found, ray tracing incomplete")
        else:
            logger.warning("No valid first intersections found, ray tracing incomplete")
            
        return ret_dict
                
                
                
            
        
    def intersection_with_sdf(self, rays_o, rays_d, sdf_network, enter_object):
        with torch.no_grad():
            batch_size = rays_o.shape[0]

            # 传递进来的是已经筛选过的射线
            near, far = self.near_far_from_sphere(rays_o, rays_d) # shape: (batch_size, 1)

            # 在每条射线上均匀采样self.n_samples个点
            z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=rays_o.device)
            z_vals = near + (far - near) * z_vals[None, :]  # [batch_size, n_samples]
            
            # 计算采样点的3D坐标
            pts = rays_o.unsqueeze(1) + z_vals.unsqueeze(-1) * rays_d.unsqueeze(1)
            pts = pts.reshape(-1, 3)  # [batch_size * n_samples, 3]
            
            sdf_values = sdf_network.sdf(pts)
            sdf_values = sdf_values.reshape(batch_size, self.n_samples)  # [batch_size, n_samples]
            
            # 计算相邻采样点之间的SDF差值
            sdf_sign_change = (sdf_values[:, 1:] * sdf_values[:, :-1]) <= 0  # [batch_size, n_samples-1]
            sign_change_count = sdf_sign_change.sum(dim=1)
            logger.debug(f"Sign changes per ray: min={sign_change_count.min().item()}, max={sign_change_count.max().item()}, avg={sign_change_count.float().mean().item():.2f}")
            
            # 找到第一个符号变化的点
            first_sign_change = torch.argmax(sdf_sign_change.float(), dim=1)  # [batch_size]
            
            # 使用线性插值计算精确的交点位置
            batch_indices_local = torch.arange(sdf_values.shape[0], device=sdf_values.device)
            sdf_before = sdf_values[batch_indices_local, first_sign_change]
            sdf_after = sdf_values[batch_indices_local, first_sign_change + 1]
            z_before = z_vals[batch_indices_local, first_sign_change]
            z_after = z_vals[batch_indices_local, first_sign_change + 1]
            
            logger.debug(f"Initial approximation: SDF before={sdf_before.mean().item():.4f}, SDF after={sdf_after.mean().item():.4f}")
            
            if self.upsample:
                logger.debug(f"Performing upsampling with {self.n_upsample} additional samples")
                # 在z_before和z_after之间均匀采样n_upsample个点
                z_fine = torch.linspace(0.0, 1.0, self.n_upsample, device=rays_o.device)
                z_fine = z_before.unsqueeze(-1) + (z_after - z_before).unsqueeze(-1) * z_fine[None, :]  # [batch_size, n_upsample]
                
                # 计算精细采样点的3D坐标
                pts_fine = rays_o.unsqueeze(1) + z_fine.unsqueeze(-1) * rays_d.unsqueeze(1)
                pts_fine = pts_fine.reshape(-1, 3)  # [batch_size * n_upsample, 3]
                
                # 评估精细采样点的SDF值
                sdf_values_fine = sdf_network.sdf(pts_fine)
                sdf_values_fine = sdf_values_fine.reshape(batch_size, self.n_upsample)  # [batch_size, n_upsample]
                
                # 计算相邻采样点之间的SDF差值
                sdf_sign_change_fine = (sdf_values_fine[:, 1:] * sdf_values_fine[:, :-1]) <= 0  # [batch_size, n_upsample-1]
                sign_change_count_fine = sdf_sign_change_fine.sum(dim=1)
                logger.debug(f"Fine sign changes per ray: min={sign_change_count_fine.min().item()}, max={sign_change_count_fine.max().item()}, avg={sign_change_count_fine.float().mean().item():.2f}")
                
                # 找到第一个符号变化的点
                first_sign_change_fine = torch.argmax(sdf_sign_change_fine.float(), dim=1)  # [batch_size]
                # 如果没有找到符号发生变化，就找最接近0的范围
                
                # 使用线性插值计算精确的交点位置
                sdf_before = sdf_values_fine[batch_indices_local, first_sign_change_fine]
                sdf_after = sdf_values_fine[batch_indices_local, first_sign_change_fine + 1]
                z_before = z_fine[batch_indices_local, first_sign_change_fine]
                z_after = z_fine[batch_indices_local, first_sign_change_fine + 1]
                
                logger.debug(f"Fine approximation: SDF before={sdf_before.mean().item():.4f}, SDF after={sdf_after.mean().item():.4f}")
            
            # 如果其中有个值的SDF本身已经比较接近0了，就采用这个点。
            # 线性插值计算交点z值
            intersection_z = z_before - sdf_before * (z_after - z_before) / (sdf_after - sdf_before)
            logger.debug(f"Intersection depths: min={intersection_z.min().item():.4f}, max={intersection_z.max().item():.4f}, mean={intersection_z.mean().item():.4f}")
            
            # 计算交点位置
            intersection_points = rays_o + intersection_z.unsqueeze(-1) * rays_d
            
            # 判断是否从空气进入物体
            # 如果sdf_before >= 0（在物体外部）并且 sdf_after <=  0（在物体内部），说明是从空气进入物体
            from_air_to_object = torch.logical_and(sdf_before >= 0, sdf_after <= 0) # [batch_size]
            from_object_to_air = torch.logical_and(sdf_before <= 0, sdf_after >= 0) # [batch_size]
            air_to_object_count = from_air_to_object.sum().item()
            logger.debug(f"Air to object transitions: {air_to_object_count}/{batch_size} ({100.0*air_to_object_count/batch_size:.2f}%)")
            
            # 判断哪些射线有有效交点
            has_intersection = sdf_sign_change.any(dim=1, keepdim=False)
            intersection_count = has_intersection.sum().item()
            logger.debug(f"Rays with intersections: {intersection_count}/{batch_size} ({100.0*intersection_count/batch_size:.2f}%)")
            
            if enter_object:
                # 如果是从空气进入物体，则需要满足有交点且是从空气到物体的方向
                valid_mask = has_intersection & from_air_to_object
            else:
                # 如果是从物体出射到空气，则需要满足有交点且不是从空气到物体的方向
                valid_mask = has_intersection & (~from_air_to_object)
            
            valid_count = valid_mask.sum().item()
            logger.debug(f"Valid intersections: {valid_count}/{batch_size} ({100.0*valid_count/batch_size:.2f}%)")
        
        # 计算交点处的法向量
        with torch.enable_grad():
            intersection_points = intersection_points.clone().detach().requires_grad_(True)
            intersection_points_sdf = sdf_network.sdf(intersection_points).squeeze()

            d_points = torch.ones_like(
                intersection_points_sdf, requires_grad=False, device=intersection_points.device
            )

            intersection_points_sdf.requires_grad_(True)
        
        verts_grad = torch.autograd.grad(
            outputs=intersection_points_sdf,
            inputs=intersection_points,
            grad_outputs=d_points,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        normals = torch.zeros(batch_size, 3).cuda()
        normals[valid_mask] = verts_grad[valid_mask].to(device=normals.device)
        
        # 检查法向量是否有nan或inf
        if torch.isnan(normals).any() or torch.isinf(normals).any():
            logger.warning("Warning: NaN or Inf detected in normals!")
        
        # 计算法向量长度
        normal_lengths = torch.norm(normals[valid_mask], dim=1) if valid_mask.any() else torch.zeros(0)
        
        # 仅当存在有效法向量时才计算统计信息
        if normal_lengths.numel() > 0:
            logger.debug(f"Normal vector lengths: min={normal_lengths.min().item():.4f}, max={normal_lengths.max().item():.4f}, mean={normal_lengths.mean().item():.4f}")
        else:
            logger.debug("No valid normals found, skipping statistics calculation")
        
        # 归一化法向量
        if valid_mask.any():
            normals[valid_mask] = normals[valid_mask] / normals[valid_mask].norm(2, dim=1).unsqueeze(-1)
        normals_ = normals.detach()
        
        # normals_ = normals # 会导致噪声梯度，不要删掉这个注释

        
        # 打印最终法向量统计
        valid_normals = normals_[valid_mask]
        if valid_normals.shape[0] > 0:
            logger.debug(f"Normalized normals: x_mean={valid_normals[:,0].mean().item():.4f}, y_mean={valid_normals[:,1].mean().item():.4f}, z_mean={valid_normals[:,2].mean().item():.4f}")
        else:
            logger.debug("No valid normalized normals found")
        
        return valid_mask, intersection_z, normals_
        
    def refraction(self, incident_dirs, normals, ior_input, ior_output):
        """计算光线折射方向和衰减系数。
        
        使用斯涅尔定律计算折射光线方向，并基于菲涅耳方程计算衰减系数。
        当发生全反射时，会设置相应的掩码标记。
        
        参数:
            incident_dirs (torch.Tensor): 入射光线方向，形状为 [n, 3]，应为单位向量
            normals (torch.Tensor): 表面法线向量，形状为 [n, 3]，应为单位向量
            ior_input (torch.Tensor): 入射介质的折射率，形状为 [n, 1]
            ior_output (torch.Tensor): 出射介质的折射率，形状为 [n, 1]
            
        返回:
            tuple: 包含以下三个元素的元组:
                - refracted_dirs (torch.Tensor): 折射光线方向，形状为 [n, 3]
                - attenuate (torch.Tensor): 衰减系数，形状为 [n, 1]，范围为 [0, 1]
                - totalReflectMask (torch.Tensor): 全反射掩码，形状为 [n, 1]，类型为 bool
        """
        # 计算入射角余弦值
        cos_theta = torch.sum(incident_dirs * (-normals), dim=1, keepdim=True)  # [n, 1]
        
        # 计算入射光线在表面的投影
        i_p = incident_dirs + normals * cos_theta
        
        # 应用斯涅尔定律计算折射光线在表面的投影
        # ior_ratio形状为[n, 1]，i_p形状为[n, 3]，广播机制会自动处理
        ior_ratio = ior_input / ior_output  # [n, 1]
        t_p = i_p * ior_ratio  # [n, 3]

        # 计算投影部分的平方和
        t_p_norm = torch.sum(t_p * t_p, dim=1)
        
        # 检测全反射情况
        totalReflectMask = (t_p_norm.detach() > 0.999999).unsqueeze(1)

        # 计算折射光线的法线分量
        t_i = torch.sqrt(1 - torch.clamp(t_p_norm, 0, 0.999999)).unsqueeze(1) * (-normals)
        
        # 折射光线 = 法线分量 + 表面投影分量
        t = t_i + t_p
        
        # 归一化折射光线方向
        t = t / torch.sqrt(torch.clamp(torch.sum(t * t, dim=1), min=1e-10)).unsqueeze(1)

        # 计算折射角余弦值
        cos_theta_t = torch.sum(t * (-normals), dim=1, keepdim=True)

        # 计算菲涅耳方程的系数
        e_i = (cos_theta_t * ior_output - cos_theta * ior_input) / \
              torch.clamp(cos_theta_t * ior_output + cos_theta * ior_input, min=1e-10)
        e_p = (cos_theta_t * ior_input - cos_theta * ior_output) / \
              torch.clamp(cos_theta_t * ior_input + cos_theta * ior_output, min=1e-10)

        # 计算衰减系数（反射光强度占的比例）
        attenuate = torch.clamp(0.5 * (e_i * e_i + e_p * e_p), 0, 1).detach()

        return t, attenuate, totalReflectMask

    def reflection(self, incident_dirs, normals):
        """计算光线反射方向。
        
        应用反射定律计算入射光线的反射方向。
        
        参数:
            incident_dirs (torch.Tensor): 入射光线方向，形状为 [n, 3]，应为单位向量
            normals (torch.Tensor): 表面法线向量，形状为 [n, 3]，应为单位向量
            
        返回:
            torch.Tensor: 反射光线方向，形状为 [n, 3]，为单位向量
        """
        # incident_dirs [n, 3] - 入射光线方向
        # normals [n, 3] - 法线向量
        
        # 计算入射角余弦值
        cos_theta = torch.sum(incident_dirs * (-normals), dim=1, keepdim=True)
        
        # 计算入射光线在表面的投影
        r_p = incident_dirs + normals * cos_theta
        
        # 计算投影部分的平方和并裁剪
        r_p_norm = torch.clamp(torch.sum(r_p * r_p, dim=1), 0, 0.999999)
        
        # 计算反射光线的法线分量
        r_i = torch.sqrt(1 - r_p_norm).unsqueeze(1) * normals
        
        # 反射光线 = 投影分量 + 法线分量
        r = r_p + r_i
        
        # 归一化反射光线方向
        r = r / torch.sqrt(torch.clamp(torch.sum(r * r, dim=1), min=1e-10)).unsqueeze(1)

        return r


class GradFunction(torch.autograd.Function):
    ''' Sphere Trace Function class.

    It provides the function to march along given rays to detect the surface
    points for the SDF Network.

    The backward pass is implemented using
    the analytic gradient described in the publication CVPR 2020.
    '''

    @staticmethod
    def forward(ctx, *inputs):
        ''' Performs a forward pass of the Depth function.

        s, delta, do, n

        return delta
        Args:
            input (list): input to forward function
        '''
        s, x, d, n, delta = inputs
        # sdf of first intersection, first_intersection, in-direction, normal, delta

        # Save values for backward pass
        ctx.save_for_backward(d, n, delta)

        return delta

    @staticmethod
    def backward(ctx, grad_output):
        """ Performs the backward pass of the Depth function.
        Args:
            ctx (Pytorch Autograd Context): pytorch autograd context
            grad_output (tensor): gradient outputs
        """
        # import pdb
        di, nj, deltai = ctx.saved_tensors
        # pdb.set_trace()
        # sdf of first intersection, first_intersection, in-direction, normal, delta
        return -grad_output / (nj * di).sum(-1), \
               -grad_output.unsqueeze(-1) * nj / (nj * di).sum(-1, keepdim=True), \
               (-grad_output * deltai).unsqueeze(-1) * nj / (nj * di).sum(-1, keepdim=True), \
               None, None