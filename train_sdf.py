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

from threedgrut.trainer_sdf import Trainer
import multiprocessing
import torch

"""

Demo usage:

CUDA_VISIBLE_DEVICES=0 python train_sdf.py --config-name apps/nerf_synthetic_with_mask_3dgrt.yaml path=data/trans_synthetic/Dog_results_140 out_dir=runs experiment_name=Dog_results_140
python train.py --config-name apps/nerf_synthetic_3dgrt.yaml path=data/nerf_synthetic/lego out_dir=runs experiment_name=lego_3dgrt

CUDA_VISIBLE_DEVICES=0 python train_sdf.py --config-name apps/nerf_synthetic_with_mask_3dgrt.yaml path=data/trans_synthetic/Dog_results_140 out_dir=runs experiment_name=Dog_results_140

CUDA_VISIBLE_DEVICES=1 python train_sdf.py --config-name apps/nerf_synthetic_with_mask_3dgrt.yaml path=data/trans_synthetic/Hand_results_140 out_dir=runs experiment_name=Hand_results_140

7000轮用了40分钟左右，纯用于训练的时间，大概是18:12秒（3用于训练:2用于渲染）。训练5.56 iter/s。一万轮是30分钟左右。
"""

import hydra
from omegaconf import DictConfig, OmegaConf
from threedgrut.utils.logger import logger
from threedgrut.utils.timer import timing_options

OmegaConf.register_new_resolver("int_list", lambda l: [int(x) for x in l])

# # Uncomment the following lines to enable debug timing
# timing_options.active = True
# timing_options.print_enabled = True


@hydra.main(config_path="configs", version_base=None)
def main(conf: DictConfig) -> None:
    trainer = Trainer(conf)
    trainer.set_log_level("INFO")

    trainer.run_training()
    

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # 启用异常检测以帮助识别梯度问题
    # torch.autograd.set_detect_anomaly(True)
    # print("已启用自动梯度异常检测，这会降低训练速度但有助于检测梯度问题")

    main()

