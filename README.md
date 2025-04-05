<p align="center">
  <img height="100" src="assets/3dgrut_logo.png">
</p>

---
<p align="center">
  <img width="100%" src="assets/nvidia-hq-playground.gif">
</p>

This repository provides the official implementations of **3D Gaussian Ray Tracing (3DGRT)** and **3D Gaussian Unscented Transform (3DGUT)**. Unlike traditional methods that rely on splatting, 3DGRT performs ray tracing of volumetric Gaussian particles instead. This enables support for distorted cameras with complex, time-dependent effects such as rolling shutters, while also efficiently simulating secondary rays required for rendering phenomena like reflection, refraction, and shadows. However, 3DGRT requires dedicated ray-tracing hardware and remains slower than 3DGS.

To mitigate this limitation, we also propose 3DGUT, which enables support for distorted cameras with complex, time-dependent effects within a rasterization framework, maintaining the efficiency of rasterization methods. By aligning the rendering formulations of 3DGRT and 3DGUT, we introduce a hybrid approach called **3DGRUT**. This technique allows for rendering primary rays via rasterization and secondary rays via ray tracing, combining the strengths of both methods for improved performance and flexibility.


> __3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes__  
> [Nicolas Moenne-Loccoz*](https://www.linkedin.com/in/nicolas-moënne-loccoz-71040512/?original_referer=https%3A%2F%2Fwww%2Egoogle%2Ecom%2F&originalSubdomain=ca), [Ashkan Mirzaei*](https://ashmrz.github.io), [Or Perel](https://orperel.github.io/), [Riccardo De Lutio](https://riccardodelutio.github.io/), [Janick Martinez Esturo](https://jme.pub/),   
> [Gavriel State](https://www.linkedin.com/in/gavstate/?originalSubdomain=ca), [Sanja Fidler](https://www.cs.utoronto.ca/~fidler/), [Nicholas Sharp^](https://nmwsharp.com/), [Zan Gojcic^](https://zgojcic.github.io/) _(*,^ indicates equal contribution)_  
> _SIGGRAPH Asia 2024 (Journal Track)_  
> __[Project page](https://research.nvidia.com/labs/toronto-ai/3DGRT)&nbsp;/ [Paper](https://research.nvidia.com/labs/toronto-ai/3DGRT/res/3dgrt_compressed.pdf)&nbsp;/ [Video](https://research.nvidia.com/labs/toronto-ai/3DGRT/res/3dgrt_supplementary_video.mp4)&nbsp;/ [BibTeX](/-/raw/release/assets/3dgrt2024.bib)__

> __3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting__  
> [Qi Wu*](https://wilsoncernwq.github.io/), [Janick Martinez Esturo*](https://jme.pub/), [Ashkan Mirzaei](https://ashmrz.github.io),   
> [Nicolas Moenne-Loccoz](https://www.linkedin.com/in/nicolas-moënne-loccoz-71040512/?original_referer=https%3A%2F%2Fwww%2Egoogle%2Ecom%2F&originalSubdomain=ca), [Zan Gojcic](https://zgojcic.github.io/)  _(* indicates equal contribution)_  
> _CVPR 2025_  
> __[Project page](https://research.nvidia.com/labs/toronto-ai/3DGUT)&nbsp;/ [Paper](https://research.nvidia.com/labs/toronto-ai/3DGUT/res/3DGUT_ready_main.pdf)&nbsp;/ [Video](https://research.nvidia.com/labs/toronto-ai/3DGUT/#supp_video)&nbsp;/ [BibTeX](/-/raw/release/assets/3dgut2025.bib)__


## 🔥 News
- ✅[2025/03] Initial code release!
- ✅[2025/02] [3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/res/3DGUT_ready_main.pdf) was accepted to CVPR 2025!
- ✅[2024/08] [3DGRT](https://research.nvidia.com/labs/toronto-ai/3DGRT/res/3dgrt_compressed.pdf) was accepted to SIGGRAPH Asia 2024!

## Contents

- [🔥 News](#-news)
- [Contents](#contents)
- [🔧 1 Dependencies and Installation](#-1-dependencies-and-installation)
- [💻 2. Train 3DGRT or 3DGUT scenes](#-2-train-3dgrt-or-3dgut-scenes)
- [🎥 3. Rendering from Checkpoints](#-3-rendering-from-checkpoints)
  - [To visualize training progress interactively](#to-visualize-training-progress-interactively)
  - [To visualize a pre-trained checkpoint](#to-visualize-a-pre-trained-checkpoint)
- [📋 4. Evaluations](#-4-evaluations)
  - [3DGRT](#3dgrt)
    - [MipNeRF360 Dataset](#mipnerf360-dataset)
  - [3DGUT](#3dgut)
    - [MipNeRF360 Dataset](#mipnerf360-dataset-1)
    - [Scannet++ Dataset](#scannet-dataset)
- [🛝 5. Interactive Playground GUI](#-5-interactive-playground-gui)
- [🎓 6. Citations](#-6-citations)
- [🙏 7. Acknowledgements](#-7-acknowledgements)

## 🔧 1 Dependencies and Installation
- CUDA 11.8+ Compatible System
- For good performance with 3DGRT, we recommend using an NVIDIA GPU with Ray Tracing (RT) cores.
- Currently, only Linux environments are supported by the included install script (Windows support coming soon!)

<details> 
<summary> NOTE: gcc versions >11 (expand for details)</summary>

Currently the codebase requires gcc <= 11.  If your machine uses the compiler gcc-12 or newer (i.e., in Ubuntu 24.04), you may need to install and use gcc-11. 

First, install gcc 11:
```sh
sudo apt-get install gcc-11 g++-11
```

Then run the install script with the optional `WITH_GCC11` flag, which additionally configures the conda environment to use gcc-11:
```sh
./install_env.sh 3dgrut WITH_GCC11
```
</details>
</br>

To set up the environment using conda, first clone the repository and run `./install_env.sh` script as:

```bash
git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut

# You can install each component step by step following install_env.sh
chmod +x install_env.sh
./install_env.sh 3dgrut
conda activate 3dgrut
```

### Running with Docker

Build the docker image:
```bash
git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut
docker build . -t 3dgrut
````

Run it:
```bash
xhost +local:root
docker run -v --rm -it --gpus=all --net=host --ipc=host -v $PWD:/workspace --runtime=nvidia -e DISPLAY 3dgrut
```
> [!NOTE]
> Remember to set DISPLAY environment variable if you are running on a remote server from command line.

## 💻 2. Train 3DGRT or 3DGUT scenes

We provide different configurations for training using 3DGRT and 3DGUT models on common benchmark datasets. 
For example you can download [NeRF Synthetic dataset](https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset), 
[MipNeRF360 dataset](https://jonbarron.info/mipnerf360/) or [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/), 
and then run one of the following commands:

```bash
# Train Lego with 3DGRT & 3DGUT
python train.py --config-name apps/nerf_synthetic_3dgrt.yaml path=data/nerf_synthetic/lego out_dir=runs experiment_name=lego_3dgrt
python train.py --config-name apps/nerf_synthetic_3dgut.yaml path=data/nerf_synthetic/lego out_dir=runs experiment_name=lego_3dgut


# Train Dog and mask with 3DGRT & 3DGUT
python train.py --config-name apps/nerf_synthetic_with_mask_3dgrt.yaml path=data/trans_synthetic/Dog_results_140 out_dir=runs experiment_name=dog_3dgrt
python train.py --config-name apps/nerf_synthetic_with_mask_3dgut.yaml path=data/trans_synthetic/Dog_results_140 out_dir=runs experiment_name=dog_3dgut


# Train Dog with mask with 3DGRT & 3DGUT
python train_with_mask.py --config-name apps/nerf_synthetic_with_mask_3dgrt.yaml path=data/trans_synthetic/Dog_results_140 out_dir=runs experiment_name=dog_3dgrt_masked
python train_with_mask.py --config-name apps/nerf_synthetic_with_mask_3dgut.yaml path=data/trans_synthetic/Dog_results_140 out_dir=runs experiment_name=dog_3dgut_masked

# Train Bonsai
CUDA_VISIBLE_DEVICES=1 python train.py --config-name apps/colmap_3dgrt.yaml path=data/mipnerf360/bonsai out_dir=runs experiment_name=bonsai_3dgrt dataset.downsample_factor=2 
python train.py --config-name apps/colmap_3dgut.yaml path=data/mipnerf360/bonsai out_dir=runs experiment_name=bonsai_3dgut dataset.downsample_factor=2 

# Train Eiko_ball
python train.py --config-name apps/colmap_3dgrt.yaml path=data/trans/eiko_ball out_dir=runs experiment_name=eiko_ball_3dgrt 
python train.py --config-name apps/colmap_3dgut.yaml path=data/trans/eiko_ball out_dir=runs experiment_name=eiko_ball_3dgut

# Train Eiko_ball_masked
CUDA_VISIBLE_DEVICES=0 python train.py --config-name apps/colmap_3dgrt.yaml path=data/trans/eiko_ball_masked out_dir=runs experiment_name=eiko_ball_masked_3dgrt 
CUDA_VISIBLE_DEVICES=0 python train.py --config-name apps/colmap_3dgut.yaml path=data/trans/eiko_ball_masked out_dir=runs experiment_name=eiko_ball_masked_3dgut

# Train Eiko_ball_masked_expanded 
CUDA_VISIBLE_DEVICES=0 python train.py --config-name apps/colmap_3dgrt.yaml path=data/trans/eiko_ball_masked_expanded out_dir=runs experiment_name=eiko_ball_masked_expanded_3dgrt 
CUDA_VISIBLE_DEVICES=0 python train.py --config-name apps/colmap_3dgut.yaml path=data/trans/eiko_ball_masked_expanded out_dir=runs experiment_name=eiko_ball_masked_expanded_3dgut



# Train Scannet++
python train.py --config-name apps/scannetpp_3dgut.yaml path=data/scannetpp/0a5c013435/dslr out_dir=runs experiment_name=0a5c013435_3dgut
```

> [!Note] 
> For ScanNet++, we expect the dataset to be preprocessed following [FisheyeGS](https://github.com/zmliao/Fisheye-GS?tab=readme-ov-file#prepare-training-data-on-scannet-dataset)'s method.

> [!Note]  
> If you're running from PyCharm IDE, enable rich console through:
> Run Configuration > Modify Options > Emulate terminal in output console*

## 🎥 3. Rendering from Checkpoints
Evaluate Checkpoint with Splatting / OptiX Tracer / Torch
```bash
python render.py --checkpoint runs/lego/ckpt_last.pt --out-dir outputs/eval
```


### To visualize training progress interactively
```bash
python train.py --config-name apps/nerf_synthetic_3dgut.yaml path=data/nerf_synthetic/lego with_gui=True 
```

### To visualize a pre-trained checkpoint
```bash
python train.py --config-name apps/nerf_synthetic_3dgut.yaml path=data/nerf_synthetic/lego with_gui=True test_last=False export_ingp.enabled=False resume=runs/lego/ckpt_last.pt 
```
> [!NOTE]
> Remember to set DISPLAY environment variable if you are running on a remote server from command line.

On start up, you might see a black screen, but you can use the GUI to navigate to correct camera views:
<img src="assets/train_gui_initial.jpg" height="400"/> 
<img src="assets/render_lego.jpg" height="400"/>

## 📋 4. Evaluations

We provide scripts to reproduce results reported in publications.

```bash
# Training
bash ./benchmark/mipnerf360_3dgut.sh <config-yaml>
# Rendering
bash ./benchmark/mipnerf360_3dgut_render.sh <config-yaml>
```

### 3DGRT

#### MipNeRF360 Dataset

### 3DGUT

#### MipNeRF360 Dataset

```bash
bash ./benchmark/mipnerf360.sh paper/3dgut/unsorted_colmap.yaml
```

Results for unsorted 3DGUT (Produced on RTX 5090):

|          | Bicycle | Bonsai  | Counter | Flower  | Garden  | Kitchen | Room    | Stump   | Treehill | *Average* |
|----------|---------|---------|---------|---------|---------|---------|---------|---------|----------|-----------|
| train(s) | 835.93  | 476.00  | 468.86  | 681.15  | 665.59  | 502.89  | 447.66  | 682.58  | 669.83   | 608.28    |
| FPS      | 269.5   | 362.3   | 336.7   | 276.2   | 333.3   | 310.6   | 383.1   | 333.3   | 326.8    | 325.8     |
| PSNR     | 24.737  | 32.235  | 28.448  | 21.326  | 26.699  | 30.393  | 31.130  | 26.289  | 22.518   | 27.086    |


#### Scannet++ Dataset

```bash
bash ./benchmark/scannetpp.sh paper/3dgut/unsorted_scannetpp.yaml
```
> [!Note]
> We followed [FisheyeGS](https://github.com/zmliao/Fisheye-GS?tab=readme-ov-file#prepare-training-data-on-scannet-dataset)'s convention to prepare the dataset for fair comparisons

Results for unsorted 3DGUT (Produced on RTX 5090):

|          | 0a5c013435 | 8d563fc2cc | bb87c292ad | d415cc449b | e8ea9b4da8 | fe1733741f | *Average* |
|----------|------------|------------|------------|------------|------------|------------|-----------|
| train(s) | 298.09     | 295.64     | 330.89     | 405.95     | 288.53     | 371.71     |  331.80   |
| FPS      | 408.2      | 465.1      | 490.2      | 446.4      | 401.6      | 444.4      |  442.7    |
| PSNR     | 29.790     | 26.848     | 31.610     | 28.084     | 33.259     | 25.608     |  29.200   |

## 🛝 5. Interactive Playground GUI

The playground allows interactive exploration of pretrained scenes, with raytracing effects such as inserted objects, 
reflections, refractions, depth of field, and more.

Run the playground UI to visualize a pretrained scene with:
```
python playground.py --gs_object <ckpt_path>
```

See [Playground README](playground/README.md) for details.



## 🎓 6. Citations

```
@article{loccoz20243dgrt,
    author = {Nicolas Moenne-Loccoz and Ashkan Mirzaei and Or Perel and Riccardo de Lutio and Janick Martinez Esturo and Gavriel State and Sanja Fidler and Nicholas Sharp and Zan Gojcic},
    title = {3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes},
    journal = {ACM Transactions on Graphics and SIGGRAPH Asia},
    year = {2024},
}
```

```
@article{wu20253dgut,
    title={3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting},
    author={Wu, Qi and Martinez Esturo, Janick and Mirzaei, Ashkan and Moenne-Loccoz, Nicolas and Gojcic, Zan},
    journal = {Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2025}
}
```

## 🙏 7. Acknowledgements

We sincerely thank our colleagues for their valuable contributions to this project.

Hassan Abu Alhaija, Ronnie Sharif, Beau Perschall and Lars Fabiunke for assistance with assets.
Greg Muthler, Magnus Andersson, Maksim Eisenstein, Tanki Zhang, Nathan Morrical, Dietger van Antwerpen and John Burgess for performance feedback.
Thomas Müller, Merlin Nimier-David, and Carsten Kolve for inspiration and pointers. 
Ziyu Chen, Clement Fuji-Tsang, Masha Shugrina, and George Kopanas for technical & experiment assistance,
and to Ramana Kiran and Shailesh Mishra for typo fixes.



<p align="center">
  <img height="100" src="assets/3dgrut_logo.png">
</p>

---
<p align="center">
  <img width="100%" src="assets/nvidia-hq-playground.gif">
</p>

本仓库提供了**3D高斯光线追踪（3DGRT）**和**3D高斯无迹变换（3DGUT）**的官方实现。与依赖于喷溅的传统方法不同，3DGRT执行体积高斯粒子的光线追踪。这使得支持具有复杂、时间依赖效果（如滚动快门）的失真相机成为可能，同时有效模拟渲染现象（如反射、折射和阴影）所需的次级光线。然而，3DGRT需要专用的光线追踪硬件，并且速度仍然慢于3DGS。

为了减轻这一限制，我们还提出了3DGUT，它在光栅化框架内支持具有复杂、时间依赖效果的失真相机，同时保持光栅化方法的效率。通过对齐3DGRT和3DGUT的渲染公式，我们引入了一种称为**3DGRUT**的混合方法。这种技术允许通过光栅化渲染主光线，通过光线追踪渲染次级光线，结合两种方法的优点，以提高性能和灵活性。

> __3D高斯光线追踪：快速追踪粒子场景__  
> [Nicolas Moenne-Loccoz*](https://www.linkedin.com/in/nicolas-moënne-loccoz-71040512/?original_referer=https%3A%2F%2Fwww%2Egoogle%2Ecom%2F&originalSubdomain=ca), [Ashkan Mirzaei*](https://ashmrz.github.io), [Or Perel](https://orperel.github.io/), [Riccardo De Lutio](https://riccardodelutio.github.io/), [Janick Martinez Esturo](https://jme.pub/),   
> [Gavriel State](https://www.linkedin.com/in/gavstate/?originalSubdomain=ca), [Sanja Fidler](https://www.cs.utoronto.ca/~fidler/), [Nicholas Sharp^](https://nmwsharp.com/), [Zan Gojcic^](https://zgojcic.github.io/) _(*,^ 表示同等贡献)_  
> _SIGGRAPH Asia 2024（期刊轨道）_  
> __[项目页面](https://research.nvidia.com/labs/toronto-ai/3DGRT)&nbsp;/ [论文](https://research.nvidia.com/labs/toronto-ai/3DGRT/res/3dgrt_compressed.pdf)&nbsp;/ [视频](https://research.nvidia.com/labs/toronto-ai/3DGRT/res/3dgrt_supplementary_video.mp4)&nbsp;/ [BibTeX](/-/raw/release/assets/3dgrt2024.bib)__

> __3DGUT：在高斯喷溅中启用失真相机和次级光线__  
> [Qi Wu*](https://wilsoncernwq.github.io/), [Janick Martinez Esturo*](https://jme.pub/), [Ashkan Mirzaei](https://ashmrz.github.io),   
> [Nicolas Moenne-Loccoz](https://www.linkedin.com/in/nicolas-moënne-loccoz-71040512/?original_referer=https%3A%2F%2Fwww%2Egoogle%2Ecom%2F&originalSubdomain=ca), [Zan Gojcic](https://zgojcic.github.io/)  _(* 表示同等贡献)_  
> _CVPR 2025_  
> __[项目页面](https://research.nvidia.com/labs/toronto-ai/3DGUT)&nbsp;/ [论文](https://research.nvidia.com/labs/toronto-ai/3DGUT/res/3DGUT_ready_main.pdf)&nbsp;/ [视频](https://research.nvidia.com/labs/toronto-ai/3DGUT/#supp_video)&nbsp;/ [BibTeX](/-/raw/release/assets/3dgut2025.bib)__


## 🔥 新闻
- ✅[2025/03] 初始代码发布！
- ✅[2025/02] [3DGUT](https://research.nvidia.com/labs/toronto-ai/3DGUT/res/3DGUT_ready_main.pdf) 被接受到CVPR 2025！
- ✅[2024/08] [3DGRT](https://research.nvidia.com/labs/toronto-ai/3DGRT/res/3dgrt_compressed.pdf) 被接受到SIGGRAPH Asia 2024！

## 目录

- [🔥 新闻](#-新闻)
- [目录](#目录)
- [🔧 1 依赖项和安装](#-1-依赖项和安装)
- [💻 2. 训练3DGRT或3DGUT场景](#-2-训练3dgrt或3dgut场景)
- [🎥 3. 从检查点渲染](#-3-从检查点渲染)
  - [以交互方式可视化训练进度](#以交互方式可视化训练进度)
  - [可视化预训练检查点](#可视化预训练检查点)
- [📋 4. 评估](#-4-评估)
  - [3DGRT](#3dgrt)
    - [MipNeRF360 数据集](#mipnerf360-数据集)
  - [3DGUT](#3dgut)
    - [MipNeRF360 数据集](#mipnerf360-数据集-1)
    - [Scannet++ 数据集](#scannet-数据集)
- [🛝 5. 交互式游乐场GUI](#-5-交互式游乐场gui)
- [🎓 6. 引用](#-6-引用)
- [🙏 7. 致谢](#-7-致谢)

## 🔧 1 依赖项和安装
- CUDA 11.8+ 兼容系统
- 为了在3DGRT中获得良好的性能，我们建议使用具有光线追踪（RT）核心的NVIDIA GPU。
- 目前，仅支持Linux环境的安装脚本（Windows支持即将推出！）

<details> 
<summary> 注意：gcc版本 >11（展开以获取详细信息）</summary>

当前代码库要求gcc <= 11。如果您的机器使用的是gcc-12或更新版本的编译器（例如，在Ubuntu 24.04中），您可能需要安装并使用gcc-11。

首先，安装gcc 11：
```sh
sudo apt-get install gcc-11 g++-11
```

然后使用可选的`WITH_GCC11`标志运行安装脚本，该标志还会配置conda环境以使用gcc-11：
```sh
./install_env.sh 3dgrut WITH_GCC11
```
</details>
</br>

要使用conda设置环境，首先克隆该仓库并运行`./install_env.sh`脚本，如下所示：

```bash
git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut

# 您可以按照install_env.sh逐步安装每个组件
chmod +x install_env.sh
./install_env.sh 3dgrut
conda activate 3dgrut
```

### 使用Docker运行

构建docker镜像：
```bash
git clone --recursive https://github.com/nv-tlabs/3dgrut.git
cd 3dgrut
docker build . -t 3dgrut
````

运行它：
```bash
xhost +local:root
docker run -v --rm -it --gpus=all --net=host --ipc=host -v $PWD:/workspace --runtime=nvidia -e DISPLAY 3dgrut
```
> [!注意]
> 如果您从命令行在远程服务器上运行，请记得设置DISPLAY环境变量。

## 💻 2. 训练3DGRT或3DGUT场景

我们提供了不同的配置，用于在常见基准数据集上使用3DGRT和3DGUT模型进行训练。
例如，您可以下载[NeRF合成数据集](https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset)，
[MipNeRF360数据集](https://jonbarron.info/mipnerf360/)或[ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/)，
然后运行以下命令之一：

```bash
# 使用3DGRT和3DGUT训练乐高
python train.py --config-name apps/nerf_synthetic_3dgrt.yaml path=data/nerf_synthetic/lego out_dir=runs experiment_name=lego_3dgrt
python train.py --config-name apps/nerf_synthetic_3dgut.yaml path=data/nerf_synthetic/lego out_dir=runs experiment_name=lego_3dgut

# 训练盆栽
CUDA_VISIBLE_DEVICES=1 python train.py --config-name apps/colmap_3dgrt.yaml path=data/mipnerf360/bonsai out_dir=runs experiment_name=bonsai_3dgrt dataset.downsample_factor=2 
python train.py --config-name apps/colmap_3dgut.yaml path=data/mipnerf360/bonsai out_dir=runs experiment_name=bonsai_3dgut dataset.downsample_factor=2 

# 训练Eiko_ball
python train.py --config-name apps/colmap_3dgrt.yaml path=data/trans/eiko_ball out_dir=runs experiment_name=eiko_ball_3dgrt 
python train.py --config-name apps/colmap_3dgut.yaml path=data/trans/eiko_ball out_dir=runs experiment_name=eiko_ball_3dgut dataset.downsample_factor=1 

# 训练Scannet++
python train.py --config-name apps/scannetpp_3dgut.yaml path=data/scannetpp/0a5c013435/dslr out_dir=runs experiment_name=0a5c013435_3dgut
```

> [!注意] 
> 对于ScanNet++，我们期望数据集按照[FisheyeGS](https://github.com/zmliao/Fisheye-GS?tab=readme-ov-file#prepare-training-data-on-scannet-dataset)的方法进行预处理。

> [!注意]  
> 如果您从PyCharm IDE运行，请通过以下方式启用丰富控制台：
> 运行配置 > 修改选项 > 在输出控制台中模拟终端*

## 🎥 3. 从检查点渲染
使用喷溅/OptiX追踪器/Torch评估检查点
```bash
python render.py --checkpoint runs/lego/ckpt_last.pt --out-dir outputs/eval
```


### 以交互方式可视化训练进度
```bash
python train.py --config-name apps/nerf_synthetic_3dgut.yaml path=data/nerf_synthetic/lego with_gui=True 
```

### 可视化预训练检查点
```bash
python train.py --config-name apps/nerf_synthetic_3dgut.yaml path=data/nerf_synthetic/lego with_gui=True test_last=False export_ingp.enabled=False resume=runs/lego/ckpt_last.pt 
```
> [!注意]
> 如果您从命令行在远程服务器上运行，请记得设置DISPLAY环境变量。

启动时，您可能会看到黑屏，但您可以使用GUI导航到正确的相机视图：
<img src="assets/train_gui_initial.jpg" height="400"/> 
<img src="assets/render_lego.jpg" height="400"/>

## 📋 4. 评估

我们提供脚本以重现出版物中报告的结果。

```bash
# 训练
bash ./benchmark/mipnerf360_3dgut.sh <config-yaml>
# 渲染
bash ./benchmark/mipnerf360_3dgut_render.sh <config-yaml>
```

### 3DGRT

#### MipNeRF360 数据集

### 3DGUT

#### MipNeRF360 数据集

```bash
bash ./benchmark/mipnerf360.sh paper/3dgut/unsorted_colmap.yaml
```

对于未排序的3DGUT（在RTX 5090上生成的结果）：

|          | 自行车 | 盆栽  | 计数器 | 花卉  | 花园  | 厨房 | 房间    | 树桩   | 树丘   | *平均* |
|----------|---------|---------|---------|---------|---------|---------|---------|---------|----------|-----------|
| train(s) | 835.93  | 476.00  | 468.86  | 681.15  | 665.59  | 502.89  | 447.66  | 682.58  | 669.83   | 608.28    |
| FPS      | 269.5   | 362.3   | 336.7   | 276.2   | 333.3   | 310.6   | 383.1   | 333.3   | 326.8    | 325.8     |
| PSNR     | 24.737  | 32.235  | 28.448  | 21.326  | 26.699  | 30.393  | 31.130  | 26.289  | 22.518   | 27.086    |


#### Scannet++ 数据集

```bash
bash ./benchmark/scannetpp.sh paper/3dgut/unsorted_scannetpp.yaml
```
> [!注意]
> 我们遵循了[FisheyeGS](https://github.com/zmliao/Fisheye-GS?tab=readme-ov-file#prepare-training-data-on-scannet-dataset)的约定，以公平比较准备数据集

对于未排序的3DGUT（在RTX 5090上生成的结果）：

|          | 0a5c013435 | 8d563fc2cc | bb87c292ad | d415cc449b | e8ea9b4da8 | fe1733741f | *平均* |
|----------|------------|------------|------------|------------|------------|------------|-----------|
| train(s) | 298.09     | 295.64     | 330.89     | 405.95     | 288.53     | 371.71     |  331.80   |
| FPS      | 408.2      | 465.1      | 490.2      | 446.4      | 401.6      | 444.4      |  442.7    |
| PSNR     | 29.790     | 26.848     | 31.610     | 28.084     | 33.259     | 25.608     |  29.200   |

## 🛝 5. 交互式游乐场GUI

游乐场允许对预训练场景进行交互式探索，具有插入对象、反射、折射、景深等光线追踪效果。

运行游乐场UI以可视化预训练场景：
```
python playground.py --gs_object <ckpt_path>
```

有关详细信息，请参见[游乐场README](playground/README.md)。

## 🎓 6. 引用

```
@article{loccoz20243dgrt,
    author = {Nicolas Moenne-Loccoz and Ashkan Mirzaei and Or Perel and Riccardo de Lutio and Janick Martinez Esturo and Gavriel State and Sanja Fidler and Nicholas Sharp and Zan Gojcic},
    title = {3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes},
    journal = {ACM Transactions on Graphics and SIGGRAPH Asia},
    year = {2024},
}
```

```
@article{wu20253dgut,
    title={3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting},
    author={Wu, Qi and Martinez Esturo, Janick and Mirzaei, Ashkan and Moenne-Loccoz, Nicolas and Gojcic, Zan},
    journal = {Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2025}
}
```

## 🙏 7. 致谢

我们衷心感谢我们的同事对本项目的宝贵贡献。

Hassan Abu Alhaija、Ronnie Sharif、Beau Perschall和Lars Fabiunke在资产方面提供了帮助。
Greg Muthler、Magnus Andersson、Maksim Eisenstein、Tanki Zhang、Nathan Morrical、Dietger van Antwerpen和John Burgess提供了性能反馈。
Thomas Müller、Merlin Nimier-David和Carsten Kolve提供了灵感和建议。 
Ziyu Chen、Clement Fuji-Tsang、Masha Shugrina和George Kopanas提供了技术和实验支持，
并感谢Ramana Kiran和Shailesh Mishra的拼写错误修正。










