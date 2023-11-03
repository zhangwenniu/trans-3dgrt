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

#!/bin/bash

# Exit on error
set -e


CONDA_ENV=${1:-"3dgrut"}

# Create and activate conda environment
eval "$(conda shell.bash hook)"
conda create -n $CONDA_ENV python=3.11 -y
conda activate $CONDA_ENV

# Make sure gcc is at most 11 for nvcc compatibility
gcc_version=$(gcc -dumpversion)
if (( $(echo "$gcc_version >= 12" | bc -l) )); then
    echo "Default gcc version $gcc_version is higher than 11, setting CC and CXX to gcc-11 and g++-11"
    # Install gcc-11
    conda install -c conda-forge gcc=11.2.0 gxx=11.2.0 -y
fi

# Install CUDA and PyTorch dependencies
conda install -y cuda-toolkit -c nvidia/label/cuda-11.8.0
conda install -y pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 "numpy<2.0" -c pytorch -c nvidia/label/cuda-11.8.0
conda install -y cmake ninja -c nvidia/label/cuda-11.8.0
# Install OpenGL headers for the playground
conda install -c conda-forge mesa-libgl-devel-cos7-x86_64 -y 

# Initialize git submodules and install Python requirements
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e .

echo "Setup completed successfully!"