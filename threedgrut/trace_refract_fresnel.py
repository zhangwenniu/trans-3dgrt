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

        
    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = torch.clamp(mid - 1.0, min=0.0)
        far = mid + 1.0
        # shape: (-1, 1)
        return near, far
    
    def forward(self, rays_o, rays_d, sdf_network):
        batch_size = len(rays_o)
        rays_o, rays_d = copy.deepcopy(rays_o), copy.deepcopy(rays_d)
        
        # 初始化返回值
        no_intersection_mask = torch.ones(batch_size, dtype=torch.bool, device=rays_o.device)
        total_reflection_mask = torch.zeros(batch_size, dtype=torch.bool, device=rays_o.device)
        twice_refraction_mask = torch.zeros(batch_size, dtype=torch.bool, device=rays_o.device)
        four_refraction_mask = torch.zeros(batch_size, dtype=torch.bool, device=rays_o.device)
        
        # 初始化返回张量
        reflect_rays_o = torch.zeros_like(rays_o)
        reflect_rays_d = torch.zeros_like(rays_d)
        refract_rays_o = torch.zeros_like(rays_o)
        refract_rays_d = torch.zeros_like(rays_d)
        reflect_rate = torch.zeros(batch_size, 1, device=rays_o.device)
        refract_rate = torch.zeros(batch_size, 1, device=rays_o.device)
        
        # intersection 1
        valid_mask_1, intersection_points_1, normals_1 = self.intersection_with_sdf(rays_o, rays_d, sdf_network, enter_object=True)
        
        if valid_mask_1.any():
            # 更新无交点掩码
            no_intersection_mask[valid_mask_1] = False
            
            # 计算第一次反射
            in_dir_1 = rays_d[valid_mask_1]
            normals_1 = normals_1[valid_mask_1]
            reflection_direction_1 = self.reflection(in_dir_1, normals_1)
            
            # 保存所有有效交点的反射信息
            reflect_rays_o[valid_mask_1] = intersection_points_1[valid_mask_1]
            reflect_rays_d[valid_mask_1] = reflection_direction_1
            
            # 计算第一次折射
            refraction_direction_1, attenuate_1, totalReflectMask_1 = self.refraction(in_dir_1, normals_1, self.ior_air, self.ior_object)
            
            # 更新全反射掩码
            total_reflection_mask[valid_mask_1] = totalReflectMask_1.squeeze()
            reflect_rate[valid_mask_1] = attenuate_1
            
            # 对非全反射的射线继续追踪
            valid_refract_mask = valid_mask_1 & (~total_reflection_mask)
            if valid_refract_mask.any():
                hit_rays_o_1 = intersection_points_1[valid_refract_mask]
                hit_rays_d_1 = refraction_direction_1[~totalReflectMask_1.squeeze()]
                
                # intersection 2
                valid_mask_2, intersection_points_2, normals_2 = self.intersection_with_sdf(
                    hit_rays_o_1, hit_rays_d_1, sdf_network, enter_object=False
                )
                
                if valid_mask_2.any():
                    # 计算第二次折射
                    in_dir_2 = hit_rays_d_1[valid_mask_2]
                    normals_2 = normals_2[valid_mask_2]
                    refraction_direction_2, attenuate_2, totalReflectMask_2 = self.refraction(
                        in_dir_2, -normals_2, self.ior_object, self.ior_air
                    )
                    
                    # 检查是否还有更多交点
                    hit_rays_o_2 = intersection_points_2[valid_mask_2]
                    hit_rays_d_2 = refraction_direction_2
                    more_intersection_mask, _, _ = self.intersection_with_sdf(
                        hit_rays_o_2, hit_rays_d_2, sdf_network, enter_object=True
                    )
                    
                    # 获取最终的有效索引
                    final_valid_indices = torch.where(valid_refract_mask)[0][valid_mask_2][~more_intersection_mask]
                    
                    # 更新两次折射掩码
                    twice_refraction_mask[final_valid_indices] = True
                    
                    # 更新折射信息
                    refract_rays_o[final_valid_indices] = hit_rays_o_2[~more_intersection_mask]
                    refract_rays_d[final_valid_indices] = hit_rays_d_2[~more_intersection_mask]
                    
                    # 计算最终的折射率
                    final_refract_rate = (1 - attenuate_1[valid_mask_2][~more_intersection_mask]) * \
                                       (1 - attenuate_2[~more_intersection_mask])
                    refract_rate[final_valid_indices] = final_refract_rate
        
        return {
            'batch_size': batch_size,
            'no_intersection_mask': no_intersection_mask,
            'total_reflection_mask': total_reflection_mask,
            'twice_refraction_mask': twice_refraction_mask,
            'four_refraction_mask': four_refraction_mask,
            'reflect_rays_o': reflect_rays_o,
            'reflect_rays_d': reflect_rays_d,
            'refract_rays_o': refract_rays_o,
            'refract_rays_d': refract_rays_d,
            'reflect_rate': reflect_rate,
            'refract_rate': refract_rate
        }
                
                
            
        
    def intersection_with_sdf(self, rays_o, rays_d, sdf_network, enter_object):
        # with torch.no_grad():
        with torch.enable_grad():
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
                        
            # 找到第一个符号变化的点
            first_sign_change = torch.argmax(sdf_sign_change.float(), dim=1)  # [batch_size]
            
            # 使用线性插值计算精确的交点位置
            batch_indices_local = torch.arange(sdf_values.shape[0], device=sdf_values.device)
            sdf_before = sdf_values[batch_indices_local, first_sign_change]
            sdf_after = sdf_values[batch_indices_local, first_sign_change + 1]
            z_before = z_vals[batch_indices_local, first_sign_change]
            z_after = z_vals[batch_indices_local, first_sign_change + 1]
            
            if self.upsample:
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
                
                # 找到第一个符号变化的点
                first_sign_change_fine = torch.argmax(sdf_sign_change_fine.float(), dim=1)  # [batch_size]
                # 如果没有找到符号发生变化，就找最接近0的范围
                
                # 使用线性插值计算精确的交点位置
                sdf_before = sdf_values_fine[batch_indices_local, first_sign_change_fine]
                sdf_after = sdf_values_fine[batch_indices_local, first_sign_change_fine + 1]
                z_before = z_fine[batch_indices_local, first_sign_change_fine]
                z_after = z_fine[batch_indices_local, first_sign_change_fine + 1]
                
            # 如果其中有个值的SDF本身已经比较接近0了，就采用这个点。
            # 线性插值计算交点z值
            intersection_z = z_before - sdf_before * (z_after - z_before) / (sdf_after - sdf_before)

            # 计算交点位置
            intersection_points = rays_o + intersection_z.unsqueeze(-1) * rays_d
            
            # 判断是否从空气进入物体
            # 如果sdf_before >= 0（在物体外部）并且 sdf_after <=  0（在物体内部），说明是从空气进入物体
            from_air_to_object = torch.logical_and(sdf_before >= 0, sdf_after <= 0) # [batch_size]
            from_object_to_air = torch.logical_and(sdf_before <= 0, sdf_after >= 0) # [batch_size]
            air_to_object_count = from_air_to_object.sum().item()

            # 判断哪些射线有有效交点
            has_intersection = sdf_sign_change.any(dim=1, keepdim=False)
            intersection_count = has_intersection.sum().item()

            if enter_object:
                # 如果是从空气进入物体，则需要满足有交点且是从空气到物体的方向
                valid_mask = has_intersection & from_air_to_object
            else:
                # 如果是从物体出射到空气，则需要满足有交点且不是从空气到物体的方向
                valid_mask = has_intersection & (~from_air_to_object)
            
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

        # 归一化法向量
        if valid_mask.any():
            normals[valid_mask] = normals[valid_mask] / normals[valid_mask].norm(2, dim=1).unsqueeze(-1)
        # normals_ = normals.detach()
        
        normals_ = normals # 会导致噪声梯度，不过如果基于菲涅尔定律来回传法向量梯度的话，就必须要保留这个梯度
        
        return valid_mask, intersection_points, normals_
        
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
        attenuate = torch.clamp(0.5 * (e_i * e_i + e_p * e_p), 0, 1)

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