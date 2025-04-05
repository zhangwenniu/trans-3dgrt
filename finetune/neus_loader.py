import os
import torch
import torch.nn.functional as F
from pyhocon import ConfigFactory

# 设置默认张量类型和设备
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')

# 将NeuS模型相关的路径添加到sys.path
import sys
sys.path.append(os.path.abspath("../sdf/NeTO/Use3DGRUT"))

# 导入NeuS相关模型
from models_silhouette.fields import SDFNetwork, SingleVarianceNetwork
from models_silhouette.renderer import NeuSRenderer

class NeuSModelLoader:
    """加载预训练的NeuS模型的工具类"""
    
    def __init__(self, conf_path=None, case_name=None, checkpoint_path=None):
        """
        初始化NeuS模型加载器
        
        Args:
            conf_path: NeuS配置文件路径，如果为None则使用默认路径
            case_name: 数据集名称，用于替换配置文件中的CASE_NAME
            checkpoint_path: 特定checkpoint路径，如果提供则直接加载该checkpoint
        """
        self.device = torch.device('cuda')
        
        # 设置默认配置路径
        if conf_path is None:
            conf_path = "../sdf/NeTO/Use3DGRUT/confs/silhouette.conf"
        
        # 设置默认case name
        if case_name is None:
            case_name = "eiko_ball_masked"
            
        # 加载配置
        self.conf_path = conf_path
        self.case_name = case_name
        self.conf = self._load_config(conf_path, case_name)
        
        # 初始化模型组件
        self.sdf_network = None
        self.deviation_network = None
        self.renderer = None
        
        # 设置坐标转换矩阵（从世界坐标到模型坐标）
        # 默认使用单位变换
        self.scale_mat = torch.eye(4, dtype=torch.float32, device=self.device)
        
        # 设置球体边界（用于计算光线与物体的交点）
        self.object_bounding_sphere = 1.0
        
        # 加载模型
        self._load_model(checkpoint_path)
        
    def _load_config(self, conf_path, case_name):
        """加载并解析配置文件"""
        with open(conf_path, 'r') as f:
            conf_text = f.read()
            
        # 替换配置中的CASE_NAME
        conf_text = conf_text.replace('CASE_NAME', case_name)
        
        # 解析配置
        conf = ConfigFactory.parse_string(conf_text)
        
        # 替换数据目录
        conf['dataset.data_dir'] = conf['dataset.data_dir'].replace('CASE_NAME', case_name)
        
        return conf
    
    def _load_model(self, checkpoint_path=None):
        """加载模型权重"""
        # 创建网络
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        
        # 确定checkpoint路径
        if checkpoint_path is None:
            # 使用配置文件中的路径
            base_exp_dir = self.conf['general.base_exp_dir']
            checkpoint_dir = os.path.join(base_exp_dir, 'checkpoints')
            
            # 获取最新的checkpoint
            model_list = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
            model_list.sort()
            
            if not model_list:
                raise FileNotFoundError(f"在{checkpoint_dir}中未找到有效的checkpoint文件")
                
            latest_model_name = model_list[-1]
            checkpoint_path = os.path.join(checkpoint_dir, latest_model_name)
        
        # 加载checkpoint
        print(f"加载NeuS模型checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # 加载网络权重
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        
        # 创建渲染器
        self.renderer = NeuSRenderer(
            self.sdf_network,
            self.deviation_network,
            **self.conf['model.neus_renderer']
        )
        
        print("NeuS模型加载完成")
        
    def get_sdf_value(self, points):
        """获取指定点的SDF值"""
        with torch.no_grad():
            sdf = self.sdf_network.sdf(points)
        return sdf
    
    def get_sdf_gradients(self, points):
        """
        获取指定点的SDF梯度（法向量）
        
        Args:
            points: 需要计算梯度的点 [N, 3]
            
        Returns:
            normals: 归一化的梯度（法向量）[N, 3]
        """
        points.requires_grad_(True)
        
        with torch.enable_grad():
            sdf = self.sdf_network.sdf(points)
            
            # 计算梯度
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=points,
                grad_outputs=torch.ones_like(sdf),
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]
        
        # 归一化梯度得到法向量
        normals = F.normalize(gradients, p=2, dim=-1)
        
        # 清除梯度和图
        points.detach_()
        
        return normals
    
    def get_batch_sdf_gradients(self, points, chunk_size=10000):
        """
        批量获取指定点的SDF梯度（法向量），使用分块计算避免OOM
        
        Args:
            points: 需要计算梯度的点 [N, 3]
            chunk_size: 每次处理的最大点数，避免OOM
            
        Returns:
            normals: 归一化的梯度（法向量）[N, 3]
        """
        n_points = points.shape[0]
        normals = torch.zeros_like(points)
        
        # 分块处理以避免OOM
        for i in range(0, n_points, chunk_size):
            end_i = min(i + chunk_size, n_points)
            chunk_points = points[i:end_i].clone().detach().requires_grad_(True)
            
            with torch.enable_grad():
                sdf = self.sdf_network.sdf(chunk_points)
                
                # 计算梯度
                gradients = torch.autograd.grad(
                    outputs=sdf,
                    inputs=chunk_points,
                    grad_outputs=torch.ones_like(sdf),
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True
                )[0]
            
            # 归一化梯度得到法向量
            chunk_normals = F.normalize(gradients, p=2, dim=-1)
            normals[i:end_i] = chunk_normals
        
        return normals
    
    def set_scale_mat(self, scale_mat):
        """设置坐标缩放矩阵"""
        self.scale_mat = scale_mat
        
    def to_world_from_neus(self, points):
        """
        将点从NeuS局部坐标系转换到世界坐标系
        
        Args:
            points: NeuS局部坐标系下的点 [N, 3]
            
        Returns:
            世界坐标系下的点 [N, 3]
        """
        if self.scale_mat is None:
            return points
            
        # 扩展为齐次坐标
        n_points = points.shape[0]
        points_homo = torch.cat([points, torch.ones((n_points, 1), device=points.device)], dim=1)  # [N, 4]
        
        # 应用变换
        world_points_homo = torch.matmul(points_homo, self.scale_mat.transpose(0, 1))  # [N, 4]
        
        # 转回非齐次坐标
        world_points = world_points_homo[:, :3]  # [N, 3]
        
        return world_points
    
    def to_neus_from_world(self, points):
        """
        将点从世界坐标系转换到NeuS局部坐标系
        
        Args:
            points: 世界坐标系下的点 [N, 3]
            
        Returns:
            NeuS局部坐标系下的点 [N, 3]
        """
        if self.scale_mat is None:
            return points
            
        # 计算逆变换矩阵
        inv_scale_mat = torch.inverse(self.scale_mat)
        
        # 扩展为齐次坐标
        n_points = points.shape[0]
        points_homo = torch.cat([points, torch.ones((n_points, 1), device=points.device)], dim=1)  # [N, 4]
        
        # 应用逆变换
        local_points_homo = torch.matmul(points_homo, inv_scale_mat.transpose(0, 1))  # [N, 4]
        
        # 转回非齐次坐标
        local_points = local_points_homo[:, :3]  # [N, 3]
        
        return local_points
        
    def to_world_direction_from_neus(self, directions):
        """
        将方向向量从NeuS局部坐标系转换到世界坐标系
        方向向量只需应用旋转变换，不需要平移
        
        Args:
            directions: NeuS局部坐标系下的方向向量 [N, 3]
            
        Returns:
            世界坐标系下的方向向量 [N, 3]
        """
        if self.scale_mat is None:
            return directions
            
        # 提取旋转/缩放部分（3x3矩阵）
        rotation_scale = self.scale_mat[:3, :3]
        
        # 应用变换（不包括平移）
        world_directions = torch.matmul(directions, rotation_scale.transpose(0, 1))
        
        # 重新归一化方向向量
        world_directions = F.normalize(world_directions, p=2, dim=1)
        
        return world_directions
    
    def to_neus_direction_from_world(self, directions):
        """
        将方向向量从世界坐标系转换到NeuS局部坐标系
        方向向量只需应用旋转的逆变换，不需要平移
        
        Args:
            directions: 世界坐标系下的方向向量 [N, 3]
            
        Returns:
            NeuS局部坐标系下的方向向量 [N, 3]
        """
        if self.scale_mat is None:
            return directions
            
        # 提取旋转/缩放部分（3x3矩阵）并计算逆变换
        rotation_scale = self.scale_mat[:3, :3]
        inv_rotation_scale = torch.inverse(rotation_scale)
        
        # 应用逆变换
        local_directions = torch.matmul(directions, inv_rotation_scale.transpose(0, 1))
        
        # 重新归一化方向向量
        local_directions = F.normalize(local_directions, p=2, dim=1)
        
        return local_directions
    
    def near_far_from_sphere(self, rays_o, rays_d):
        """
        计算射线与包围球的交点，获取最近和最远的交点距离
        
        Args:
            rays_o: 光线起点 [N, 3]
            rays_d: 光线方向 [N, 3]
            
        Returns:
            near: 近平面距离 [N]
            far: 远平面距离 [N]
        """
        # 将光线转换到模型坐标系
        rays_o = self.to_neus_from_world(rays_o)
        
        # 注意：光线方向是向量，只需要应用旋转和缩放，不需要平移
        # 由于我们使用的是简单的缩放和平移，只需要除以缩放因子
        rays_d = rays_d / self.scale_mat[0, 0]
        rays_d = F.normalize(rays_d, p=2, dim=-1)
        
        # 计算射线原点到原点的距离
        rays_o_dot_o = torch.sum(rays_o * rays_o, dim=-1)
        # 计算射线原点与射线方向的点积
        rays_o_dot_d = torch.sum(rays_o * rays_d, dim=-1)
        
        # 计算射线与球体交点的二次方程系数
        # a = 1 (因为rays_d是单位向量)
        # b = 2 * rays_o_dot_d
        # c = rays_o_dot_o - r^2
        b = 2.0 * rays_o_dot_d
        c = rays_o_dot_o - self.object_bounding_sphere ** 2
        
        # 计算判别式
        delta = b * b - 4 * c
        
        # 处理无交点的情况
        mask = delta > 0
        
        # 初始化near和far
        near = torch.zeros_like(rays_o_dot_o)
        far = torch.ones_like(rays_o_dot_o) * 2.0  # 默认设置一个远平面
        
        if mask.any():
            # 计算交点参数
            sqrt_delta = torch.sqrt(torch.clamp(delta[mask], min=0.0))
            near[mask] = -b[mask] - sqrt_delta
            far[mask] = -b[mask] + sqrt_delta
            # 将near和far除以2（因为二次方程系数a=1）
            near = near / 2.0
            far = far / 2.0
            
            # 确保near > 0（射线与球体的交点在射线的正方向）
            near = torch.clamp(near, min=0.0)
        
        return near, far
        
    def compute_batch_intersection(self, ray_origins, ray_directions, near=None, far=None, n_samples=128, n_secant_steps=8, max_batch_size=50000):
        """
        计算光线与SDF表面的交点（批量优化版本）- 内存优化版
        
        Args:
            ray_origins: 光线起点 [batch_size, 3]
            ray_directions: 光线方向 [batch_size, 3]
            near: 近平面距离，如果为None则使用包围球计算
            far: 远平面距离，如果为None则使用包围球计算
            n_samples: 采样点数量
            n_secant_steps: 二分法迭代次数
            max_batch_size: 每批处理的最大光线数量
            
        Returns:
            intersection_points: 交点坐标 [batch_size, 3]
            intersection_normals: 交点法向量 [batch_size, 3]
            valid_mask: 有效交点掩码 [batch_size]
        """
        batch_size = ray_origins.shape[0]
        
        # 将光线转换到模型坐标系
        local_ray_origins = self.to_neus_from_world(ray_origins)
        local_ray_directions = self.to_neus_direction_from_world(ray_directions)
        
        # 如果没有提供near和far，使用包围球计算
        if near is None or far is None:
            near, far = self.near_far_from_sphere(local_ray_origins, local_ray_directions)
        
        # 确保near和far是张量
        if not isinstance(near, torch.Tensor):
            near = torch.tensor(near, dtype=torch.float32, device=self.device).expand(batch_size)
        if not isinstance(far, torch.Tensor):
            far = torch.tensor(far, dtype=torch.float32, device=self.device).expand(batch_size)
        
        # 初始化输出张量
        intersection_points = torch.zeros_like(ray_origins)
        intersection_normals = torch.zeros_like(ray_origins)
        valid_intersection = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # 分批处理光线以减少内存使用
        num_batches = (batch_size + max_batch_size - 1) // max_batch_size  # 向上取整
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * max_batch_size
            end_idx = min((batch_idx + 1) * max_batch_size, batch_size)
            
            # 当前批次的光线
            batch_ray_origins = local_ray_origins[start_idx:end_idx]
            batch_ray_directions = local_ray_directions[start_idx:end_idx]
            batch_near = near[start_idx:end_idx]
            batch_far = far[start_idx:end_idx]
            current_batch_size = batch_ray_origins.shape[0]
            
            # 在射线上均匀采样
            t_vals = torch.linspace(0.0, 1.0, n_samples, device=self.device)
            z_vals = batch_near.unsqueeze(-1) + (batch_far - batch_near).unsqueeze(-1) * t_vals  # [curr_batch, n_samples]
            
            # 计算采样点的3D坐标
            pts = batch_ray_origins.unsqueeze(1) + z_vals.unsqueeze(-1) * batch_ray_directions.unsqueeze(1)  # [curr_batch, n_samples, 3]
            pts_flat = pts.reshape(-1, 3)  # [curr_batch * n_samples, 3]
            
            # 批量评估SDF值 - 内存优化：分批处理点云
            sdf_chunk_size = 100000  # 每次处理的点数量
            num_pts = pts_flat.shape[0]
            sdf_values_list = []
            
            for i in range(0, num_pts, sdf_chunk_size):
                end_i = min(i + sdf_chunk_size, num_pts)
                with torch.no_grad():
                    chunk_sdf = self.sdf_network.sdf(pts_flat[i:end_i])
                sdf_values_list.append(chunk_sdf)
                # 主动释放内存
                torch.cuda.empty_cache()
            
            sdf_values = torch.cat(sdf_values_list, dim=0).reshape(current_batch_size, n_samples)
            
            # 计算相邻采样点之间的SDF符号变化
            signs = torch.sign(sdf_values)
            sign_changes = (signs[:, :-1] * signs[:, 1:]) <= 0  # [curr_batch, n_samples-1]
            
            # 对于每条光线，找到第一个符号变化
            # 首先创建一个掩码，标记第一个符号变化的位置
            first_change_idx = torch.zeros(current_batch_size, n_samples-1, dtype=torch.bool, device=self.device)
            
            # 对每条光线找到第一个符号变化
            for i in range(current_batch_size):
                changes = torch.where(sign_changes[i])[0]
                if changes.shape[0] > 0:
                    first_change_idx[i, changes[0]] = True
                    valid_intersection[start_idx + i] = True
            
            # 提取有效的光线索引
            batch_valid_indices = torch.where(valid_intersection[start_idx:end_idx])[0]
            
            if batch_valid_indices.shape[0] > 0:
                # 提取每条有效光线的第一个符号变化的索引
                interval_indices = torch.where(first_change_idx[batch_valid_indices])[1]
                
                # 获取区间边界的t值和sdf值
                t0 = torch.zeros(batch_valid_indices.shape[0], device=self.device)
                t1 = torch.zeros(batch_valid_indices.shape[0], device=self.device)
                sdf0 = torch.zeros(batch_valid_indices.shape[0], device=self.device)
                sdf1 = torch.zeros(batch_valid_indices.shape[0], device=self.device)
                
                for i, (ray_idx, interval_idx) in enumerate(zip(batch_valid_indices, interval_indices)):
                    t0[i] = z_vals[ray_idx, interval_idx]
                    t1[i] = z_vals[ray_idx, interval_idx + 1]
                    sdf0[i] = sdf_values[ray_idx, interval_idx]
                    sdf1[i] = sdf_values[ray_idx, interval_idx + 1]
                
                # 使用线性插值猜测SDF=0的位置
                t_mid = t0 - sdf0 * (t1 - t0) / (sdf1 - sdf0)
                
                # 使用二分法求根
                lower = t0.clone()
                upper = t1.clone()
                
                # 执行二分法迭代
                for _ in range(n_secant_steps):
                    # 计算中点的坐标
                    mid_points = torch.zeros((batch_valid_indices.shape[0], 3), device=self.device)
                    for i, ray_idx in enumerate(batch_valid_indices):
                        mid_points[i] = batch_ray_origins[ray_idx] + t_mid[i] * batch_ray_directions[ray_idx]
                    
                    # 计算中点的SDF值
                    with torch.no_grad():
                        mid_sdf = self.sdf_network.sdf(mid_points)
                    
                    # 根据SDF的符号更新区间边界
                    update_lower = mid_sdf < 0
                    update_upper = ~update_lower
                    
                    # 更新下边界
                    for i in range(update_lower.shape[0]):
                        if update_lower[i]:
                            lower[i] = t_mid[i]
                        else:
                            upper[i] = t_mid[i]
                    
                    # 更新中点
                    t_mid = 0.5 * (lower + upper)
                
                # 最终的交点位置
                intersection_t = t_mid
                local_intersection_points = torch.zeros((batch_valid_indices.shape[0], 3), device=self.device)
                for i, ray_idx in enumerate(batch_valid_indices):
                    local_intersection_points[i] = batch_ray_origins[ray_idx] + intersection_t[i] * batch_ray_directions[ray_idx]
                
                # 计算交点处的法向量 - 使用批量梯度计算，但控制批处理大小
                norm_chunk_size = 10000  # 每次处理的法线计算点数
                num_points = local_intersection_points.shape[0]
                local_normals_list = []
                
                for i in range(0, num_points, norm_chunk_size):
                    end_i = min(i + norm_chunk_size, num_points)
                    chunk_normals = self.get_batch_sdf_gradients(local_intersection_points[i:end_i])
                    local_normals_list.append(chunk_normals)
                    # 主动释放内存
                    torch.cuda.empty_cache()
                
                local_intersection_normals = torch.cat(local_normals_list, dim=0) if local_normals_list else torch.zeros((0, 3), device=self.device)
                
                # 将交点转换回世界坐标系
                world_intersection_points = self.to_world_from_neus(local_intersection_points)
                
                # 将法向量转换到世界坐标系
                world_intersection_normals = self.to_world_direction_from_neus(local_intersection_normals)
                
                # 将局部批次索引转换为全局索引
                global_valid_indices = batch_valid_indices + start_idx
                
                # 更新输出
                intersection_points[global_valid_indices] = world_intersection_points
                intersection_normals[global_valid_indices] = world_intersection_normals
            
            # 主动释放内存
            del sdf_values, pts, pts_flat, z_vals
            torch.cuda.empty_cache()
        
        return intersection_points, intersection_normals, valid_intersection

    def compute_intersection(self, ray_origins, ray_directions, near=None, far=None, n_samples=128):
        """
        计算光线与SDF表面的交点（调用批量版本，保持兼容性）
        
        Args:
            ray_origins: 光线起点 [batch_size, 3]
            ray_directions: 光线方向 [batch_size, 3]
            near: 近平面距离，如果为None则使用包围球计算
            far: 远平面距离，如果为None则使用包围球计算
            n_samples: 采样点数量
            
        Returns:
            intersection_points: 交点坐标 [batch_size, 3]
            intersection_normals: 交点法向量 [batch_size, 3]
            valid_mask: 有效交点掩码 [batch_size]
        """
        # 调用批量版本
        return self.compute_batch_intersection(ray_origins, ray_directions, near, far, n_samples)
    
    def compute_batch_refractive_rays(self, ray_origins, ray_directions, n1=1.0, n2=1.5):
        """
        批量计算折射光线
        
        Args:
            ray_origins: 光线起点 [batch_size, 3]
            ray_directions: 光线方向 [batch_size, 3]
            n1: 介质1的折射率(空气)
            n2: 介质2的折射率(玻璃)
            
        Returns:
            refracted_origins: 折射光线起点 [batch_size, 3]
            refracted_directions: 折射光线方向 [batch_size, 3]
            valid_mask: 有效折射掩码 [batch_size]
        """
        # 计算光线与SDF的第一个交点（从空气进入玻璃）- 使用批量版本
        entry_points, entry_normals, valid_entry = self.compute_batch_intersection(ray_origins, ray_directions)
        
        # 初始化折射光线参数
        batch_size = ray_origins.shape[0]
        refracted_origins = ray_origins.clone()
        refracted_directions = ray_directions.clone()
        valid_refraction = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # 仅处理有效的交点
        if valid_entry.sum() > 0:
            # 获取有效交点的索引
            valid_indices = torch.where(valid_entry)[0]
            
            # 获取有效交点的法向量和入射方向
            valid_normals = entry_normals[valid_indices]
            valid_dirs = ray_directions[valid_indices]
            valid_points = entry_points[valid_indices]
            
            # 计算入射角的余弦值（入射方向与法线的点积的负值）
            # 注意：以下计算假设射线方向和法向量都已经归一化
            cos_i = torch.sum(-valid_dirs * valid_normals, dim=-1, keepdim=True)  # [n_valid, 1]
            
            # 处理射线从外部进入物体的情况（cos_i > 0）
            from_outside = cos_i > 0
            
            if from_outside.any():
                # 获取从外部进入的射线的索引
                outside_indices = valid_indices[from_outside.squeeze(-1)]
                
                # 获取相应的法向量、方向和入射点
                outside_normals = valid_normals[from_outside.squeeze(-1)]  # [n_outside, 3]
                outside_dirs = valid_dirs[from_outside.squeeze(-1)]        # [n_outside, 3]
                outside_points = valid_points[from_outside.squeeze(-1)]    # [n_outside, 3]
                outside_cos_i = cos_i[from_outside]                        # [n_outside, 1]
                
                # 应用斯涅尔定律计算折射方向
                ratio = n1 / n2  # 标量
                
                # 计算 1 - (n1/n2)^2 * (1 - cos^2(i))
                k = 1.0 - ratio * ratio * (1.0 - outside_cos_i * outside_cos_i)  # [n_outside, 1]
                
                # 处理全反射情况
                k = torch.clamp(k, min=0.0)
                
                # 计算折射方向系数
                sqrt_k = torch.sqrt(k)  # [n_outside, 1]
                coef = ratio * outside_cos_i - sqrt_k  # [n_outside, 1]
                
                # 确保系数正确广播
                # 这里重要的是确保coef的形状是[n_outside, 1]，这样它会正确地广播到[n_outside, 3]
                # 当与outside_normals（形状[n_outside, 3]）相乘时
                refracted_dirs = torch.zeros_like(outside_dirs)  # [n_outside, 3]
                
                # 折射方向计算: n1/n2 * incident - (n1/n2 * cos(i) - sqrt(1 - (n1/n2)^2 * (1 - cos^2(i)))) * normal
                for i in range(outside_dirs.shape[0]):
                    dir_scaled = ratio * outside_dirs[i]  # [3]
                    normal_scaled = coef[i] * outside_normals[i]  # [3]
                    refracted_dirs[i] = dir_scaled - normal_scaled
                
                # 归一化折射方向
                refracted_dirs = F.normalize(refracted_dirs, p=2, dim=-1)
                
                # 添加一个小的偏移量，避免数值精度问题导致折射光线起点正好位于表面
                epsilon = 1e-4
                refracted_origs = outside_points - epsilon * outside_normals
                
                # 更新有效交点位置
                refracted_origins[outside_indices] = refracted_origs
                refracted_directions[outside_indices] = refracted_dirs
                valid_refraction[outside_indices] = True
        
        return refracted_origins, refracted_directions, valid_refraction

    def compute_refractive_rays(self, ray_origins, ray_directions, n1=1.0, n2=1.5):
        """
        计算折射光线（调用批量版本）
        
        Args:
            ray_origins: 光线起点 [batch_size, 3]
            ray_directions: 光线方向 [batch_size, 3]
            n1: 介质1的折射率(空气)
            n2: 介质2的折射率(玻璃)
            
        Returns:
            refracted_origins: 折射光线起点 [batch_size, 3]
            refracted_directions: 折射光线方向 [batch_size, 3]
            valid_mask: 有效折射掩码 [batch_size]
        """
        return self.compute_batch_refractive_rays(ray_origins, ray_directions, n1, n2)
    
    def compute_batch_full_refractive_path(self, ray_origins, ray_directions, n1=1.0003, n2=1.51, max_batch_size=50000):
        """
        批量计算完整的折射光路（进入物体和离开物体），内存优化版本
        
        Args:
            ray_origins: 光线起点 [batch_size, 3]
            ray_directions: 光线方向 [batch_size, 3]
            n1: 空气的折射率
            n2: 玻璃的折射率
            max_batch_size: a每批处理的最大光线数量
            
        Returns:
            exit_origins: 出射光线起点 [batch_size, 3]
            exit_directions: 出射光线方向 [batch_size, 3]
            valid_mask: 有效折射掩码 [batch_size]
        """
        batch_size = ray_origins.shape[0]
        
        # 初始化结果张量
        exit_origins = ray_origins.clone()
        exit_directions = ray_directions.clone()
        valid_exit = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        
        # 分批处理入射光线以减少内存使用
        num_batches = (batch_size + max_batch_size - 1) // max_batch_size  # 向上取整
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * max_batch_size
            end_idx = min((batch_idx + 1) * max_batch_size, batch_size)
            
            # 当前批次的光线
            batch_ray_origins = ray_origins[start_idx:end_idx]
            batch_ray_directions = ray_directions[start_idx:end_idx]
            current_batch_size = batch_ray_origins.shape[0]
            
            # 计算第一次折射（空气->玻璃）
            refracted_origins, refracted_directions, valid_entry = self.compute_batch_refractive_rays(
                batch_ray_origins, batch_ray_directions, n1, n2
            )
            
            # 仅对第一次折射有效的点计算第二次折射
            if valid_entry.sum() > 0:
                # 获取有效的第一次折射的索引
                valid_indices = torch.where(valid_entry)[0]
                
                # 获取有效的第一次折射光线
                valid_refr_origins = refracted_origins[valid_indices]
                valid_refr_directions = refracted_directions[valid_indices]
                
                # 计算第二个交点（从玻璃离开到空气）- 使用批量版本，进一步分批处理
                # 将有效的折射光线分成更小的批次处理
                valid_batch_size = valid_indices.shape[0]
                valid_num_batches = (valid_batch_size + max_batch_size - 1) // max_batch_size
                
                # 用于收集所有有效的出射结果
                all_exit_points = []
                all_exit_normals = []
                all_has_exit = []
                all_exit_indices = []
                
                for valid_batch_idx in range(valid_num_batches):
                    v_start_idx = valid_batch_idx * max_batch_size
                    v_end_idx = min((valid_batch_idx + 1) * max_batch_size, valid_batch_size)
                    
                    # 当前有效批次的折射光线
                    curr_refr_origins = valid_refr_origins[v_start_idx:v_end_idx]
                    curr_refr_directions = valid_refr_directions[v_start_idx:v_end_idx]
                    curr_valid_indices = valid_indices[v_start_idx:v_end_idx]
                    
                    # 计算出射点
                    exit_points, exit_normals, has_exit = self.compute_batch_intersection(
                        curr_refr_origins, curr_refr_directions
                    )
                    
                    if has_exit.sum() > 0:
                        # 保存有效的出射结果
                        batch_exit_indices = torch.where(has_exit)[0]
                        all_exit_points.append(exit_points[batch_exit_indices])
                        all_exit_normals.append(exit_normals[batch_exit_indices])
                        all_has_exit.append(has_exit[batch_exit_indices])
                        # 映射回原始批次的索引
                        all_exit_indices.append(curr_valid_indices[batch_exit_indices])
                    
                    # 主动释放内存
                    torch.cuda.empty_cache()
                
                # 如果有任何出射点
                if all_exit_indices:
                    # 合并所有批次的结果
                    exit_points = torch.cat(all_exit_points, dim=0) if all_exit_points else torch.zeros((0, 3), device=self.device)
                    exit_normals = torch.cat(all_exit_normals, dim=0) if all_exit_normals else torch.zeros((0, 3), device=self.device)
                    exit_indices = torch.cat(all_exit_indices, dim=0) if all_exit_indices else torch.zeros((0,), dtype=torch.long, device=self.device)
                    
                    # 处理所有有效的出射点
                    if exit_indices.shape[0] > 0:
                        # 获取有效的出射点信息
                        valid_exit_points = exit_points
                        valid_exit_normals = exit_normals
                        valid_incident_dirs = refracted_directions[exit_indices - start_idx]
                        
                        # 计算入射角的余弦值（这次是从内到外）
                        cos_i = torch.sum(valid_incident_dirs * valid_exit_normals, dim=-1, keepdim=True)  # [n_exit, 1]
                        
                        # 处理从内部射向外部的情况（cos_i > 0）
                        from_inside = cos_i > 0
                        
                        if from_inside.any():
                            # 获取从内部射出的光线索引
                            inside_indices = torch.where(from_inside.squeeze(-1))[0]
                            # 映射回原始批次的索引
                            final_indices = exit_indices[inside_indices]
                            
                            # 获取相关数据
                            inside_exit_points = valid_exit_points[inside_indices]  # [n_inside, 3]
                            inside_exit_normals = valid_exit_normals[inside_indices]  # [n_inside, 3]
                            inside_incident_dirs = valid_incident_dirs[inside_indices]  # [n_inside, 3]
                            inside_cos_i = cos_i[inside_indices]  # [n_inside, 1]
                            
                            # 应用斯涅尔定律（从玻璃到空气，比率反转）
                            ratio = n2 / n1  # 标量
                            
                            # 计算折射系数的判别式
                            k = 1.0 - ratio * ratio * (1.0 - inside_cos_i * inside_cos_i)  # [n_inside, 1]
                            
                            # 判断是否发生全反射
                            total_internal_reflection = k < 0
                            k = torch.clamp(k, min=0.0)
                            
                            # 计算折射系数
                            sqrt_k = torch.sqrt(k)  # [n_inside, 1]
                            coef = ratio * inside_cos_i - sqrt_k  # [n_inside, 1]
                            
                            # 计算折射方向，使用循环确保维度正确
                            refracted_dirs = torch.zeros_like(inside_incident_dirs)  # [n_inside, 3]
                            
                            for i in range(inside_incident_dirs.shape[0]):
                                # 折射方向: n2/n1 * incident - (n2/n1 * cos(i) - sqrt(1 - (n2/n1)^2 * (1 - cos^2(i)))) * normal
                                dir_scaled = ratio * inside_incident_dirs[i]  # [3]
                                normal_scaled = coef[i] * inside_exit_normals[i]  # [3]
                                refracted_dirs[i] = dir_scaled - normal_scaled  # [3]
                            
                            # 处理全反射的情况
                            if total_internal_reflection.any():
                                # 计算反射方向
                                reflected_dirs = torch.zeros_like(inside_incident_dirs)  # [n_inside, 3]
                                
                                # 计算反射系数
                                refl_coef = 2.0 * inside_cos_i  # [n_inside, 1]
                                
                                # 使用循环计算反射方向
                                for i in range(inside_incident_dirs.shape[0]):
                                    if total_internal_reflection[i]:
                                        # 反射方向: incident - 2(incident·normal)normal
                                        normal_scaled = refl_coef[i] * inside_exit_normals[i]  # [3]
                                        reflected_dirs[i] = inside_incident_dirs[i] - normal_scaled  # [3]
                                
                                # 找到全反射的光线索引
                                total_refl_indices = torch.where(total_internal_reflection.squeeze(-1))[0]
                                
                                # 使用反射方向替换全反射情况
                                for i in range(len(total_refl_indices)):
                                    idx = total_refl_indices[i]
                                    refracted_dirs[idx] = reflected_dirs[idx]
                            
                            # 归一化方向
                            refracted_dirs = F.normalize(refracted_dirs, p=2, dim=-1)
                            
                            # 添加小偏移量，避免数值精度问题
                            epsilon = 1e-4
                            refracted_origs = inside_exit_points + epsilon * inside_exit_normals
                            
                            # 更新最终结果
                            exit_origins[final_indices] = refracted_origs
                            exit_directions[final_indices] = refracted_dirs
                            valid_exit[final_indices] = True
                        
                    # 主动释放内存
                    torch.cuda.empty_cache()
            
            # 主动释放当前批次的内存
            del refracted_origins, refracted_directions, valid_entry
            torch.cuda.empty_cache()
        
        return exit_origins, exit_directions, valid_exit

    def compute_full_refractive_path(self, ray_origins, ray_directions, n1=1.0003, n2=1.51):
        """
        计算完整的折射光路（调用批量版本）
        
        Args:
            ray_origins: 光线起点 [batch_size, 3]
            ray_directions: 光线方向 [batch_size, 3]
            n1: 空气的折射率
            n2: 玻璃的折射率
            
        Returns:
            exit_origins: 出射光线起点 [batch_size, 3]
            exit_directions: 出射光线方向 [batch_size, 3]
            valid_mask: 有效折射掩码 [batch_size]
        """
        return self.compute_batch_full_refractive_path(ray_origins, ray_directions, n1, n2) 