import torch
import torch.nn.functional as F

class RefractTracer:
    def __init__(self, ior_air, ior_object, n_samples=64, n_upsample=64, upsample=True, train=True):
        self.ior_air = ior_air
        self.ior_object = ior_object
        self.n_samples = n_samples
        self.n_upsample = n_upsample
        self.upsample = upsample
        self.train = True
        
    def near_far_from_sphere(self, rays_o, rays_d):
        a = torch.sum(rays_d**2, dim=-1, keepdim=True)
        b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
        mid = 0.5 * (-b) / a
        near = mid - 1.0
        far = mid + 1.0
        # shape: (-1, 1)
        return near, far
    
    def ray_tracing_with_refraction(self, rays_o, rays_d, sdf_network):
        """跟踪光线在透明物体中的折射路径。
        
        此函数模拟光线穿过透明物体时的双重折射过程：第一次从空气进入物体，第二次从物体出射回空气。
        
        参数:
            rays_o (torch.Tensor): 光线起点坐标，形状为 [n, 3]
            rays_d (torch.Tensor): 光线方向向量，形状为 [n, 3]，应为单位向量
            sdf_network (torch.nn.Module): 用于计算有符号距离场(SDF)的网络模型
            
        返回:
            tuple: 包含以下七个元素的元组:
                - new_rays_o (torch.Tensor): 更新后的光线起点，对于成功折射的光线，更新为第二次折射后的位置
                - new_rays_d (torch.Tensor): 更新后的光线方向，对于成功折射的光线，更新为第二次折射后的方向
                - reflect_direction (torch.Tensor): 第一次交点处的反射方向，形状为 [n, 3] ，只有reflect_mask为True的射线才有意义
                - reflect_mask (torch.Tensor): 标识哪些光线与物体相交的掩码，形状为 [n]，布尔类型
                - reflect_rate (torch.Tensor): 反射系数，遵循菲涅耳方程，形状为 [n, 1]，只有reflect_mask为True的射线才有意义
                - refract_rate (torch.Tensor): 折射系数，考虑了两次折射的衰减，形状为 [n, 1]，只有tracing_mask为True的射线才有意义
                - tracing_mask (torch.Tensor): 标识哪些光线成功完成两次折射的掩码，形状为 [n]，布尔类型
        """
        # 创建结果张量 - 默认值为原始光线
        new_rays_o = rays_o.clone()
        new_rays_d = rays_d.clone()
        
        # 初始化追踪掩码，标记哪些光线成功完成了双重折射
        tracing_mask = torch.zeros(rays_o.shape[0], device=rays_o.device, dtype=torch.bool)
        # 第一次光线追踪：从空气进入物体
        intersection_points_1, reflect_dir_ret_1, refract_dir_ret_1, attenuate_1, valid_mask_1, sdf_sign_change_1 = self.ray_tracing(rays_o, rays_d, sdf_network)
        
        # 计算反射率（第一次相交表面的反射强度）
        reflect_rate = attenuate_1
        # 计算折射率（考虑两次相交表面的衰减）
        refract_rate = (1 - attenuate_1)
        
        
        if valid_mask_1.any():            
            # 第二次光线追踪：从物体内部射出
            # 仅处理第一次成功折射的光线
            intersection_points_2, reflect_dir_ret_2, refract_dir_ret_2, attenuate_2, valid_mask_2, sdf_sign_change_2 = self.ray_tracing(intersection_points_1[valid_mask_1], refract_dir_ret_1[valid_mask_1], sdf_network)
            
            if valid_mask_2.any():
                # 获取最终有效的索引
                first_valid_indices = torch.where(valid_mask_1)[0]
                final_valid_indices = first_valid_indices[valid_mask_2]
                
                # 更新tracing_mask
                tracing_mask = torch.scatter(tracing_mask, 0, final_valid_indices, True)
                
                # 更新refract_rate
                update_refract_rate = refract_rate.clone()
                update_refract_rate[final_valid_indices] = refract_rate[final_valid_indices] * (1 - attenuate_2[valid_mask_2])
                refract_rate = update_refract_rate
                
                # 使用scatter_更新光线信息（非原地操作）
                new_rays_o = torch.scatter(new_rays_o, 0, 
                           final_valid_indices.unsqueeze(1).repeat(1, 3), 
                           intersection_points_2[valid_mask_2])
                
                new_rays_d = torch.scatter(new_rays_d, 0,
                           final_valid_indices.unsqueeze(1).repeat(1, 3),
                           refract_dir_ret_2[valid_mask_2])

        # 记录第一次反射的方向和掩码
        reflect_direction = reflect_dir_ret_1
        reflect_mask = sdf_sign_change_1        
        return new_rays_o, new_rays_d, reflect_direction, reflect_mask, reflect_rate, refract_rate, tracing_mask
        

    def ray_tracing(self, rays_o, rays_d, sdf_network):
        batch_size = rays_o.shape[0]
        # 传递进来的是已经筛选过的射线
        near, far = self.near_far_from_sphere(rays_o, rays_d) # shape: (batch_size, 1)
        # 在每条射线上均匀采样128个点
        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=rays_o.device)
        z_vals = near + (far - near) * z_vals[None, :]  # [batch_size, n_samples]
        
        # 计算采样点的3D坐标
        pts = rays_o.unsqueeze(1) + z_vals.unsqueeze(-1) * rays_d.unsqueeze(1)
        pts = pts.reshape(-1, 3)  # [batch_size * n_samples, 3]
        
        # 评估SDF值
        if self.train:
            sdf_values = sdf_network.sdf(pts)
        else:
            with torch.no_grad():
                sdf_values = sdf_network.sdf(pts)
        sdf_values = sdf_values.reshape(batch_size, self.n_samples)  # [batch_size, n_samples]
        
        # 计算相邻采样点之间的SDF差值
        sdf_sign_change = (sdf_values[:, 1:] * sdf_values[:, :-1]) < 0  # [batch_size, n_samples-1]
        
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
            if self.train:
                sdf_values_fine = sdf_network.sdf(pts_fine)
            else:
                with torch.no_grad():
                    sdf_values_fine = sdf_network.sdf(pts_fine)
            sdf_values_fine = sdf_values_fine.reshape(batch_size, self.n_upsample)  # [batch_size, n_upsample]
            
            # 计算相邻采样点之间的SDF差值
            sdf_sign_change_fine = (sdf_values_fine[:, 1:] * sdf_values_fine[:, :-1]) < 0  # [batch_size, n_upsample-1]
            
            # 找到第一个符号变化的点
            first_sign_change_fine = torch.argmax(sdf_sign_change_fine.float(), dim=1)  # [batch_size]
            
            # 使用线性插值计算精确的交点位置
            sdf_before = sdf_values_fine[batch_indices_local, first_sign_change_fine]
            sdf_after = sdf_values_fine[batch_indices_local, first_sign_change_fine + 1]
            z_before = z_fine[batch_indices_local, first_sign_change_fine]
            z_after = z_fine[batch_indices_local, first_sign_change_fine + 1]
        
        # 线性插值计算交点z值
        intersection_z = z_before - sdf_before * (z_after - z_before) / (sdf_after - sdf_before)
        
        # print("intersection_z.shape:", intersection_z.shape)
        # print("intersection_z.unsqueeze(-1).shape:", intersection_z.unsqueeze(-1).shape)
        # 计算交点位置
        intersection_points = rays_o + intersection_z.unsqueeze(-1) * rays_d
        # print("intersection_points.shape:", intersection_points.shape)
        
        
        # 判断是否从空气进入物体
        # 如果sdf_before > 0（在物体外部）或 sdf_after < 0（在物体内部），说明是从空气进入物体
        from_air_to_object = torch.logical_or(sdf_before > 0, sdf_after < 0).unsqueeze(-1)  # [batch_size, 1]
        
        # 判断哪些射线有有效交点
        valid_mask = sdf_sign_change.any(dim=1, keepdim=False)
        
        normals = sdf_network.gradient(intersection_points).reshape(-1, 3).detach()
        normals = F.normalize(normals, dim=-1)
        
        # print("from sdf network calculated normals.shape:", normals.shape)
        
        normals = torch.where(from_air_to_object, normals, -normals)
        
        # print("after torch.where, normals.shape:", normals.shape)
        
        ior_input = torch.where(from_air_to_object, self.ior_air, self.ior_object)
        ior_output = torch.where(from_air_to_object, self.ior_object, self.ior_air)
        
        attenuate_ret = torch.zeros((batch_size, 1), device=rays_o.device)
        
        if valid_mask.any():
            # print("rays_d.shape:", rays_d.shape)
            # print("normals.shape:", normals.shape)
            # print("ior_input.shape:", ior_input.shape)
            # print("ior_output.shape:", ior_output.shape)
            
            # print("valid_mask.shape:", valid_mask.shape)
            # print("valid_mask.sum:", valid_mask.sum())
            # print("rays_d[valid_mask].shape:", rays_d[valid_mask].shape)
            # print("normals[valid_mask].shape:", normals[valid_mask].shape)
            # print("ior_input[valid_mask].shape:", ior_input[valid_mask].shape)
            # print("ior_output[valid_mask].shape:", ior_output[valid_mask].shape)
            refracted_dirs, attenuate, totalReflectMask = self.refraction(rays_d[valid_mask], normals[valid_mask], ior_input[valid_mask], ior_output[valid_mask])
            

            attenuate_ret[valid_mask] = attenuate

            # 更新valid_mask，排除全反射的射线
            valid_indices = torch.where(valid_mask)[0]
            
            if totalReflectMask.any():
                # 获取发生全反射的射线在原始数组中的索引
                total_reflect_indices = valid_indices[totalReflectMask.squeeze(-1)]
                valid_mask[total_reflect_indices] = False
                
                refracted_dirs = refracted_dirs[~totalReflectMask.squeeze(-1)]

        reflect_dir_ret = rays_d.clone()
        refract_dir_ret = rays_d.clone()
        if valid_mask.any():
            reflected_dirs = self.reflection(rays_d[valid_mask], normals[valid_mask])
            # 计算反射光线
            reflect_dir_ret[valid_mask] = reflected_dirs
            refract_dir_ret[valid_mask] = refracted_dirs
            
        # attenuate表示反射光强度占的比例
                
        return intersection_points, reflect_dir_ret, refract_dir_ret, attenuate_ret, valid_mask, sdf_sign_change.any(dim=1)
        
        
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