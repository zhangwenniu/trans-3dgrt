import torch
import torch.nn.functional as F

class RefractTracerInplace:
    def __init__(self, ior_air, ior_object, n_samples=64, n_upsample=64, upsample=True):
        self.ior_air = ior_air
        self.ior_object = ior_object
        self.n_samples = n_samples
        self.n_upsample = n_upsample
        self.upsample = upsample
        
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
                - rays_o (torch.Tensor): 更新后的光线起点，对于成功折射的光线，更新为第二次折射后的位置
                - rays_d (torch.Tensor): 更新后的光线方向，对于成功折射的光线，更新为第二次折射后的方向
                - reflect_direction (torch.Tensor): 第一次交点处的反射方向，形状为 [n, 3] ，只有reflect_mask为True的射线才有意义
                - reflect_mask (torch.Tensor): 标识哪些光线与物体相交的掩码，形状为 [n]，布尔类型
                - reflect_rate (torch.Tensor): 反射系数，遵循菲涅耳方程，形状为 [n, 1]，只有reflect_mask为True的射线才有意义
                - refract_rate (torch.Tensor): 折射系数，考虑了两次折射的衰减，形状为 [n, 1]，只有tracing_mask为True的射线才有意义
                - tracing_mask (torch.Tensor): 标识哪些光线成功完成两次折射的掩码，形状为 [n]，布尔类型
        """
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
                
                # 更新tracing_mask和其他相关张量
                tracing_mask[final_valid_indices] = True
                refract_rate[final_valid_indices] *= (1 - attenuate_2[valid_mask_2])
                rays_o[final_valid_indices] = intersection_points_2[valid_mask_2]
                rays_d[final_valid_indices] = refract_dir_ret_2[valid_mask_2]

        # 记录第一次反射的方向和掩码
        reflect_direction = reflect_dir_ret_1
        reflect_mask = sdf_sign_change_1        
        return rays_o, rays_d, reflect_direction, reflect_mask, reflect_rate, refract_rate, tracing_mask
        

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
        
        normals = sdf_network.gradient(intersection_points).reshape(-1, 3)
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
    


"""
# RefractTracer 光线折射与反射追踪类文档

## 概述

RefractTracer 类是一个高精度的光线追踪实现，专门用于模拟透明物体中的光线折射和反射现象。该类结合了物理光学原理和计算机图形学技术，能够准确模拟光线穿过透明物体时的复杂行为，包括折射、反射、全反射以及菲涅耳效应。

## 核心特性

### 1. 物理精确的光学模拟

- **斯涅尔定律实现**：精确计算不同折射率介质间的光线折射方向
- **菲涅耳方程**：基于物理菲涅耳方程计算反射和折射的能量分布
- **全反射检测**：自动识别和处理临界角情况下的全反射现象
- **双重折射路径**：追踪光线从空气→物体→空气的完整折射路径

### 2. 高效采样与精度优化

- **初始均匀采样**：在射线方向上均匀分布采样点以定位表面
- **自适应细分采样**：可选的精细采样机制，对表面附近区域进行二次高密度采样
- **表面精确定位**：通过线性插值技术精确定位SDF零值交点
- **单位球面优化**：使用单位球体边界快速计算射线的近远平面

### 3. 几何表面精确计算

- **SDF梯度法向量**：利用神经隐式SDF的梯度计算精确的表面法向量
- **自动法向量翻转**：根据光线从空气进入物体或从物体射出自动调整法向量方向
- **矢量规范化**：确保所有方向向量（入射、反射、折射、法线）均为单位向量

### 4. 完整的光线状态追踪

- **交点位置记录**：追踪并记录光线与物体表面的精确交点
- **折射/反射方向**：计算每次交互后的折射和反射方向
- **能量系数**：根据菲涅耳公式计算折射和反射的能量分配
- **状态掩码**：使用多种掩码标记不同状态的光线（有效交点、全反射、成功折射等）

## 关键组件

### 1. ray_tracing_with_refraction 方法
核心方法，协调整个双重折射过程，处理光线从空气进入物体再射出的完整路径。

### 2. ray_tracing 方法
基础光线追踪方法，负责寻找光线与SDF表面的交点，并计算交点处的表面信息。

### 3. refraction 方法
实现斯涅尔定律和菲涅耳方程，计算折射方向和能量分配。

### 4. reflection 方法
实现反射定律，计算反射方向。

## 技术特点

1. **梯度兼容性**：支持保留梯度流，便于与可微渲染框架集成
2. **稳健性处理**：包含全面的边界条件处理和数值稳定性保障
3. **批量处理**：支持批量光线的并行计算
4. **内存效率**：仅在必要时使用精细采样，优化内存使用

## 算法细节

### 光线与SDF表面求交

1. 首先在射线上均匀采样点，评估SDF值
2. 检测SDF值符号变化，定位表面交点区间
3. 可选地进行精细二次采样，提高交点精度 
4. 使用线性插值计算精确交点位置

### 双重折射过程

1. 第一次光线追踪：确定光线从空气进入物体的入射点
2. 计算入射点处的表面法线和折射方向
3. 第二次光线追踪：从入射点沿折射方向追踪至物体内部出射点
4. 计算出射点处的表面法线和第二次折射方向
5. 更新光线起点和方向为最终出射点和出射方向

### 菲涅耳效应计算

1. 计算入射角和折射角
2. 应用物理菲涅耳方程计算反射率和透射率
3. 考虑光线极化效应，计算平行和垂直分量
4. 合并计算最终能量分配系数

## 应用场景

- **真实感透明物体渲染**：提供物理准确的透明物体光学效果
- **SDF形状优化**：通过反向传播优化神经隐式SDF表示
- **光学系统模拟**：模拟透镜、棱镜等光学元件的光线路径
- **材质外观建模**：结合其他光线传输算法实现复杂材质的渲染

## 使用示例

RefractTracer类通常与神经SDF网络结合使用：

```python
# 初始化折射追踪器
ray_tracer = RefractTracer(ior_air=1.0003, ior_object=1.51, n_samples=128)

# 应用ray_tracing_with_refraction函数
new_rays_o, new_rays_d, reflect_dir, reflect_mask, reflect_rate, refract_rate, tracing_mask = ray_tracer.ray_tracing_with_refraction(
    rays_o, rays_d, sdf_network
)

# 筛选成功折射的光线进行渲染
valid_rays_o = new_rays_o[tracing_mask]
valid_rays_d = new_rays_d[tracing_mask]
```

这个RefractTracer类代表了物理光学与计算机图形学的融合，为透明物体的高保真渲染提供了坚实的技术基础。


"""