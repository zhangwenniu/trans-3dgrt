# COLMAP预处理流程说明

本文档详细说明了如何使用COLMAP对自定义数据进行预处理，以用于NeuS的训练。

## 环境要求

- Python 3.x
- COLMAP
- OpenCV
- NumPy
- Trimesh

## 数据准备

1. 准备图像数据
   - 将需要处理的图像放在目标目录下（例如：`${data_dir}/images/`）
   - 支持的图像格式：PNG
   - 建议图像分辨率适中，过高的分辨率会增加处理时间

2. 图像采集建议
   - 确保图像之间有足够的重叠区域（建议>50%）
   - 拍摄时保持相机参数（焦距、光圈等）一致
   - 避免使用广角镜头，以减少畸变
   - 拍摄时保持场景光照稳定

## 处理流程

### 1. 运行COLMAP SfM重建

```bash
python imgs2poses.py ${data_dir}
```

可选参数：
- `--match_type`: 选择特征匹配器类型
  - `exhaustive_matcher`: 全局匹配（默认）
  - `sequential_matcher`: 序列匹配

处理完成后，将在`${data_dir}`目录下生成：
- `sparse_points.ply`: 稀疏点云文件
- `poses.npy`: 相机位姿数据

### 2. 定义感兴趣区域

1. 使用MeshLab等工具打开`sparse_points.ply`
2. 清理点云数据，去除噪声点
3. 保存清理后的点云为`sparse_points_interest.ply`

### 3. 生成相机参数

```bash
python gen_cameras.py ${data_dir}
```

此步骤将：
1. 读取相机位姿和点云数据
2. 计算场景的包围球参数
3. 生成相机内参和外参矩阵
4. 创建预处理后的数据目录结构

## 输出数据说明

处理完成后，在`${data_dir}/preprocessed`目录下将生成：

### 目录结构
```
preprocessed/
├── image/          # 处理后的图像
│   ├── 000.png
│   ├── 001.png
│   └── ...
├── mask/           # 图像掩码
│   ├── 000.png
│   ├── 001.png
│   └── ...
└── cameras_sphere.npz  # 相机参数文件
```

### cameras_sphere.npz 文件内容
- `camera_mat_{i}`: 第i张图像的相机内参矩阵
- `camera_mat_inv_{i}`: 相机内参矩阵的逆
- `world_mat_{i}`: 第i张图像的世界坐标系变换矩阵
- `world_mat_inv_{i}`: 世界坐标系变换矩阵的逆
- `scale_mat_{i}`: 场景缩放矩阵
- `scale_mat_inv_{i}`: 场景缩放矩阵的逆

## 代码实现细节

### 1. COLMAP SfM重建流程

#### 1.1 特征提取和匹配
在`colmap_wrapper.py`中实现，主要包含以下步骤：

1. 特征提取
```python
feature_extractor_args = [
    'colmap', 'feature_extractor', 
    '--database_path', os.path.join(basedir, 'database.db'), 
    '--image_path', os.path.join(basedir, 'images'),
    '--ImageReader.single_camera', '1',
]
```
- 使用COLMAP的SIFT特征提取器
- 将所有图像视为使用同一相机拍摄
- 生成特征数据库文件

2. 特征匹配
```python
exhaustive_matcher_args = [
    'colmap', match_type, 
    '--database_path', os.path.join(basedir, 'database.db'), 
]
```
- 支持全局匹配和序列匹配两种模式
- 生成特征匹配结果

3. 稀疏重建
```python
mapper_args = [
    'colmap', 'mapper',
    '--database_path', os.path.join(basedir, 'database.db'),
    '--image_path', os.path.join(basedir, 'images'),
    '--output_path', os.path.join(basedir, 'sparse'),
    '--Mapper.num_threads', '16',
    '--Mapper.init_min_tri_angle', '4',
    '--Mapper.multiple_models', '0',
    '--Mapper.extract_colors', '0',
]
```
- 使用16线程进行并行重建
- 设置最小三角化角度为4度
- 禁用多模型重建
- 不提取点云颜色信息

### 2. 相机位姿处理

在`pose_utils.py`中实现，主要包含以下功能：

1. 加载COLMAP数据
```python
def load_colmap_data(realdir):
    # 读取相机参数
    camerasfile = os.path.join(realdir, 'sparse/0/cameras.bin')
    camdata = read_model.read_cameras_binary(camerasfile)
    
    # 读取图像位姿
    imagesfile = os.path.join(realdir, 'sparse/0/images.bin')
    imdata = read_model.read_images_binary(imagesfile)
    
    # 转换坐标系
    w2c_mats = []
    for k in imdata:
        im = imdata[k]
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3,1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)
```

2. 保存位姿数据
```python
def save_poses(basedir, poses, pts3d, perm):
    # 生成点云文件
    pts_arr = []
    for k in pts3d:
        pts_arr.append(pts3d[k].xyz)
    pts = np.stack(pts_arr, axis=0)
    pcd = trimesh.PointCloud(pts)
    pcd.export(os.path.join(basedir, 'sparse_points.ply'))
    
    # 保存相机位姿
    poses = np.moveaxis(poses, -1, 0)
    poses = poses[perm]
    np.save(os.path.join(basedir, 'poses.npy'), poses)
```

### 3. 相机参数生成

在`gen_cameras.py`中实现，主要包含以下步骤：

1. 读取位姿和点云数据
```python
poses_hwf = np.load(os.path.join(work_dir, 'poses.npy'))
poses_raw = poses_hwf[:, :, :4]
hwf = poses_hwf[:, :, 4]
```

2. 计算场景包围球
```python
pcd = trimesh.load(os.path.join(work_dir, 'sparse_points_interest.ply'))
vertices = pcd.vertices
bbox_max = np.max(vertices, axis=0)
bbox_min = np.min(vertices, axis=0)
center = (bbox_max + bbox_min) * 0.5
radius = np.linalg.norm(vertices - center, ord=2, axis=-1).max()
```

3. 生成相机参数
```python
for i in range(n_images):
    # 计算相机内参
    intrinsic = np.diag([f, f, 1.0, 1.0]).astype(np.float32)
    intrinsic[0, 2] = (w - 1) * 0.5
    intrinsic[1, 2] = (h - 1) * 0.5
    
    # 计算世界坐标系变换
    w2c = np.linalg.inv(pose)
    world_mat = intrinsic @ w2c
```

## 注意事项

1. 图像质量
   - 确保图像清晰，避免模糊
   - 避免过度曝光或曝光不足
   - 保持场景光照均匀

2. 点云处理
   - 仔细检查并清理点云数据
   - 确保感兴趣区域被完整覆盖
   - 去除离群点和噪声

3. 常见问题
   - 如果重建失败，检查图像质量和重叠度
   - 如果点云稀疏，可能需要增加图像数量
   - 如果相机参数异常，检查图像采集过程

## 故障排除

1. COLMAP重建失败
   - 检查图像质量
   - 增加图像数量
   - 调整特征匹配参数

2. 点云质量差
   - 检查图像采集角度
   - 确保场景特征丰富
   - 调整点云清理参数

3. 相机参数异常
   - 检查图像采集过程
   - 验证相机标定参数
   - 检查图像分辨率
