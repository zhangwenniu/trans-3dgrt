import os
import torch
from omegaconf import DictConfig, OmegaConf

# 将3DGRT/3DGUT相关路径添加到sys.path
import sys
sys.path.append(os.path.abspath(".."))

from threedgrut.model.model import MixtureOfGaussians

class GaussiansModelLoader:
    """加载预训练的3D高斯模型的工具类"""
    
    def __init__(self, checkpoint_path=None, config_path=None, device=None):
        """
        初始化3D高斯模型加载器
        
        Args:
            checkpoint_path: 模型checkpoint路径，必须提供
            config_path: 配置文件路径，如果为None则从checkpoint中加载
            device: 运行设备，默认为CUDA
        """
        self.device = device if device is not None else torch.device("cuda")
        
        # 确保提供了checkpoint路径
        if checkpoint_path is None:
            raise ValueError("必须提供checkpoint_path参数")
            
        self.checkpoint_path = checkpoint_path
        
        # 加载配置
        if config_path is not None:
            self.conf = OmegaConf.load(config_path)
        else:
            # 从checkpoint中加载配置
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.conf = checkpoint.get("config", None)
            
            if self.conf is None:
                raise ValueError("Checkpoint中不包含配置信息，请提供config_path")
        
        # 初始化模型
        self.model = None
        
        # 加载模型
        self._load_model()
        
    def _load_model(self):
        """加载3D高斯模型"""
        print(f"加载3D高斯模型checkpoint: {self.checkpoint_path}")
        
        # 初始化模型
        self.model = MixtureOfGaussians(self.conf)
        
        # 从checkpoint加载参数
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.init_from_checkpoint(checkpoint)
        
        # 构建加速结构
        self.model.build_acc()
        
        print("3D高斯模型加载完成")
    
    def render(self, batch):
        """
        使用3D高斯模型渲染图像
        
        Args:
            batch: 包含rays_ori, rays_dir等信息的batch数据
            
        Returns:
            outputs: 包含渲染结果的字典
        """
        with torch.no_grad():
            outputs = self.model(batch)
        return outputs
    
    def get_model(self):
        """获取3D高斯模型实例"""
        return self.model 