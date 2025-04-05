import argparse
import torch.multiprocessing as mp
import torch
from threedgrut.render_test_neto import Renderer

# 设置默认张量类型 - 使用新的推荐API
torch.set_default_dtype(torch.float32)
torch.set_default_device('cuda')

# 设置多进程启动方法为spawn，而非默认的fork
mp.set_start_method('spawn', force=True)



# 初始化NeuS模型

def init_neus():
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
    base_exp_dir = conf['general.base_exp_dir']
    
    sdf_network = SDFNetwork(**conf['model.sdf_network']).to(device)
    deviation_network = SingleVarianceNetwork(**conf['model.variance_network']).to(device)
    
    checkpoint_dir = os.path.join(base_exp_dir, 'checkpoints')
    model_list_raw = os.listdir(checkpoint_dir)
    model_list = []
    for model_name in model_list_raw:
        if model_name[-3:] == 'pth':
            model_list.append(model_name)
    model_list.sort()
    latest_model_name = model_list[-1]
    checkpoint = torch.load(os.path.join(checkpoint_dir, latest_model_name), map_location=device)
    sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
    deviation_network.load_state_dict(checkpoint['variance_network_fine'])
    renderer = NeuSRenderer(sdf_network,deviation_network,**conf['model.neus_renderer'])
    return renderer

# 初始化3DGURT模型

def init_3dgrut():
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="path to the pretrained checkpoint")
    parser.add_argument("--path", type=str, default="", help="Path to the training data, if not provided taken from ckpt")
    parser.add_argument("--out-dir", required=True, type=str, help="Output path")
    parser.add_argument("--save-gt", action="store_false", help="If set, the GT images will not be saved [True by default]")
    parser.add_argument("--compute-extra-metrics", action="store_false", help="If set, extra image metrics will not be computed [True by default]")
    args = parser.parse_args()

    renderer = Renderer.from_checkpoint(
                        checkpoint_path=args.checkpoint,
                        path=args.path,
                        out_dir=args.out_dir,
                        save_gt=args.save_gt,
                        computes_extra_metrics=args.compute_extra_metrics)

    
    
    return renderer

# 训练NeuS模型

def train_neus():
    for batch in dataloader:
        # 前向传播
        outputs = model(batch)
        # 计算损失
        loss = compute_loss(outputs, batch)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
    return None


def main():
    # 初始化数据集
    
    # 初始化NeuS模型
    
    # 初始化3DGURT模型
    
    # 训练NeuS模型

    # 渲染场景
    
    # 评估指标



if __name__ == '__main__':
    main()