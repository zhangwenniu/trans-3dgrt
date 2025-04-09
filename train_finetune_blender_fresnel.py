# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from threedgrut.trainer_finetune_blender_fresnel import Trainer
import multiprocessing
import torch

"""
Demo usage:

CUDA_VISIBLE_DEVICES=1 python train_finetune.py --checkpoint /workspace/runs/eiko_ball_masked_expanded_3dgrt/eiko_ball_masked_expanded-2703_050608/ckpt_last.pt --out-dir /workspace/outputs/eval/finetune/v1.4

CUDA_VISIBLE_DEVICES=1 python train_finetune_blender.py --sdf_checkpoint_path /workspace/runs/Dog_results_140/checkpoints/ckpt_030000.pth --gaussian_checkpoint_path /workspace/runs/empty_room_100_3dgut/empty_room_100-0804_145139/ckpt_last.pt --config_path /workspace/configs/apps/nerf_synthetic_with_mask_3dgrt.yaml --out_dir /workspace/runs/Dog_results_140_finetune

CUDA_VISIBLE_DEVICES=1 python train_finetune_blender.py --sdf_checkpoint_path /workspace/runs/Dog_results_140/checkpoints/ckpt_030000.pth --gaussian_checkpoint_path /workspace/runs/empty_room_100_3dgrt/empty_room_100-0804_225045/ours_1000/ckpt_1000.pt --out_dir /workspace/runs/Dog_results_140_finetune --data_dir /workspace/data/trans_synthetic/Dog_results_140 --experiment_name Dog_results_140_finetune

CUDA_VISIBLE_DEVICES=0 python train_finetune_blender.py --sdf_checkpoint_path /workspace/runs/Hand_results_140/checkpoints/ckpt_030000.pth --gaussian_checkpoint_path /workspace/runs/empty_room_100_3dgrt/empty_room_100-0804_225045/ours_1000/ckpt_1000.pt --out_dir /workspace/runs/Hand_results_140_finetune --data_dir /workspace/data/trans_synthetic/Hand_results_140 --experiment_name Hand_results_140_finetune

CUDA_VISIBLE_DEVICES=1 python train_finetune_blender_fresnel.py --sdf_checkpoint_path /workspace/runs/Hand_results_140/checkpoints/ckpt_030000.pth --gaussian_checkpoint_path /workspace/runs/empty_room_100_3dgrt/empty_room_100-0804_225045/ours_1000/ckpt_1000.pt --out_dir /workspace/runs/Hand_results_140_finetune/v.13 --data_dir /workspace/data/trans_synthetic/Hand_results_140 --experiment_name Hand_results_140_finetune

CUDA_VISIBLE_DEVICES=1 python train_finetune_blender_fresnel.py --sdf_checkpoint_path /workspace/runs/Hand_results_140/checkpoints/ckpt_030000.pth --gaussian_checkpoint_path /workspace/runs/Empty_box_results_140_3dgrt/Empty_box_results_140-0904_161534/ours_0/ckpt_0.pt --out_dir /workspace/runs/Hand_results_140_finetune/v.19 --data_dir /workspace/data/trans_synthetic/Hand_results_140 --experiment_name Hand_results_140_finetune   
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from threedgrut.utils.logger import logger
from threedgrut.utils.timer import timing_options

OmegaConf.register_new_resolver("int_list", lambda l: [int(x) for x in l])

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # 启用异常检测以帮助识别梯度问题
    torch.autograd.set_detect_anomaly(True)
    print("已启用自动梯度异常检测，这会降低训练速度但有助于检测梯度问题")
    
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdf_checkpoint_path", required=True, type=str, help="path to the pretrained sdf checkpoint")
    parser.add_argument("--gaussian_checkpoint_path", required=True, type=str, help="path to the pretrained sdf checkpoint")
    parser.add_argument("--out_dir", required=True, type=str, help="Output path")
    parser.add_argument("--data_dir", required=True, type=str, help="Data path")
    parser.add_argument("--experiment_name", required=True, type=str, help="experiment name")
    
    # parser.add_argument("--debug-grad", action="store_true", help="Keep anomaly detection on for the whole training")
    args = parser.parse_args()
    
    trainer = Trainer.from_checkpoint(
            sdf_checkpoint_path=args.sdf_checkpoint_path,
            gaussian_checkpoint_path=args.gaussian_checkpoint_path, 
            out_dir=args.out_dir, 
            data_dir=args.data_dir, 
            experiment_name=args.experiment_name)
    trainer.set_log_level("INFO")

    trainer.run_training()
    # trainer.run_render()