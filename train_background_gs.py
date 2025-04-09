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

import hydra
from omegaconf import DictConfig, OmegaConf
from threedgrut.utils.logger import logger
from threedgrut.utils.timer import timing_options

OmegaConf.register_new_resolver("int_list", lambda l: [int(x) for x in l])

# # Uncomment the following lines to enable debug timing
# timing_options.active = True
# timing_options.print_enabled = True

"""
使用方法：
CUDA_VISIBLE_DEVICES=1 python train_background_gs.py --config-name apps/nerf_synthetic_3dgrt.yaml path=data/trans_synthetic/empty_room_100 out_dir=runs experiment_name=empty_room_100_3dgrt checkpoint.iterations=[1000,3000,5000,10000]
"""


@hydra.main(config_path="configs", version_base=None)
def main(conf: DictConfig) -> None:
    logger.info(f"Compiling native code..")
    from threedgrut.trainer_background_gs import Trainer3DGRUT

    # # NOTE: It is also possible to directly instantiate a trainer from a checkpoint/INGP/PLY file
    # c = OmegaConf.load("example.yaml")
    # trainer = Trainer3DGRUT.create_from_ckpt("checkpoint.pt", DictConfig(c))
    # trainer = Trainer3DGRUT.create_from_ingp("export_last.ingp", DictConfig(c))
    # trainer = Trainer3DGRUT.create_from_ply("export_last.ply", DictConfig(c))

    # trainer = Trainer3DGRUT(conf)
    # trainer.run_training()
    # trainer = Trainer3DGRUT.create_from_ply("/workspace/data/gaussian_splatting/empty_room_100/point_cloud/iteration_30000/point_cloud.ply", conf)
    trainer = Trainer3DGRUT.create_from_ply("/workspace/data/gaussian_splatting/Empty_box_results_140/point_cloud/iteration_7000/point_cloud.ply", conf)

    trainer.run_training()


if __name__ == "__main__":
    main()



