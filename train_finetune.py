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
from threedgrut.trainer_finetune import Trainer
import multiprocessing
import torch

"""
Demo usage:

CUDA_VISIBLE_DEVICES=1 python train_finetune.py --checkpoint /workspace/runs/eiko_ball_masked_expanded_3dgrt/eiko_ball_masked_expanded-2703_050608/ckpt_last.pt --out-dir /workspace/outputs/eval/finetune/v1.4
"""
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # 启用异常检测以帮助识别梯度问题
    torch.autograd.set_detect_anomaly(True)
    print("已启用自动梯度异常检测，这会降低训练速度但有助于检测梯度问题")
    
    # Set up command line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str, help="path to the pretrained checkpoint")
    parser.add_argument("--out-dir", required=True, type=str, help="Output path")
    parser.add_argument("--debug-grad", action="store_true", help="Keep anomaly detection on for the whole training")
    args = parser.parse_args()

    trainer = Trainer.from_checkpoint(
                        checkpoint_path=args.checkpoint,
                        out_dir=args.out_dir,
               )
    trainer.set_log_level("INFO")

    trainer.run_training()
    # trainer.run_render()


