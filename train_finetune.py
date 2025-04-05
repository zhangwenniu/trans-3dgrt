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

python train_finetune.py --checkpoint /workspace/runs/eiko_ball_masked_expanded_3dgrt/eiko_ball_masked_expanded-2703_050608/ckpt_last.pt --out-dir /workspace/outputs/eval/finetune
"""
if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    
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
    
    # 如果运行了50次迭代后没有梯度问题，关闭异常检测以提高性能
    # 除非用户使用了--debug-grad参数
    def disable_anomaly_detection():
        if not args.debug_grad:
            print("前50次迭代未发现梯度问题，正在关闭异常检测以提高性能")
            torch.autograd.set_detect_anomaly(False)
    
    # 注册回调函数到trainer
    if hasattr(trainer, 'register_callback'):
        trainer.register_callback(50, disable_anomaly_detection)
    
    trainer.run_training()
    # trainer.run_render()


