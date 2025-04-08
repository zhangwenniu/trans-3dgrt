# NeUS相关导入
import sys
import os
import torch
sys.path.append(os.path.abspath("./sdf/NeTO/Use3DGRUT"))
import torch.nn.functional as F
from models_silhouette.fields import SDFNetwork, SingleVarianceNetwork
from models_silhouette.renderer import NeuSRenderer
from pyhocon import ConfigFactory

# 使用新的推荐API设置默认张量类型和设备
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')

import matplotlib.pyplot as plt
import trimesh
import logging
from matplotlib.gridspec import GridSpec

class MeshExtract():
    def __init__(self):
        self.init_from_neus()
        pass

    def init_from_neus(self):
        device = torch.device('cuda')
        # Configuration
        conf_path = "/workspace/sdf/NeTO/Use3DGRUT/confs/silhouette.conf"
        case = "eiko_ball_masked"
        f = open(conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        conf = ConfigFactory.parse_string(conf_text)
        conf['dataset.data_dir'] = conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = conf['general.base_exp_dir']
        
        self.sdf_network = SDFNetwork(**conf['model.sdf_network']).to(device)
        self.deviation_network = SingleVarianceNetwork(**conf['model.variance_network']).to(device)
        
        checkpoint_path = "/workspace/outputs/eval/finetune/v1.7/checkpoints/ckpt_301000.pth"
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.iter_step = checkpoint['iter_step']
        self.renderer = NeuSRenderer(self.sdf_network, self.deviation_network, **conf['model.neus_renderer'])
        
        # 创建优化器
        params_to_train = list(self.sdf_network.parameters())
        params_to_train += list(self.deviation_network.parameters())
        
        self.optimizer = torch.optim.Adam(params_to_train, lr=0.0001)
        return self.renderer

    
    def validate_mesh(self, resolution=64, threshold=0.0):
        bound_min = torch.tensor([-1, -1, -1], dtype=torch.float32)
        bound_max = torch.tensor([1, 1, 1], dtype=torch.float32)

        mesh_path = "/workspace/outputs/eval/finetune/v1.7/meshes"
        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(mesh_path, exist_ok=True)
        
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(mesh_path, '{:0>8d}_resolution_{}.ply'.format(self.iter_step, resolution)))

        logging.info('End')


if __name__ == '__main__':
    MeshExtract().validate_mesh(resolution=512)