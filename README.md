# SAMesh 参数调优指南

## 使用方法

```bash
python mesh_samesh.py --filename data/mesh_face.ply --mode component --visualize --sam_mesh.repartition_iterations 0 --sam_mesh.smoothing_iterations 32
```

![DEMO](./assets/output.png)

## 安装说明

```bash
pip install -e .
```

如果遇到与ctypes相关的pyrenderer问题，请尝试安装`PyOpenGL==3.1.7`。

## 下载模型

默认配置（最小模型）

```bash
mkdir checkpoints && wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt -O checkpoints/sam2.1_hiera_tiny.pt
```

其他配置

```bash
mkdir checkpoints && wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt -O checkpoints/sam2.1_hiera_small.pt
```

```bash
mkdir checkpoints && wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt -O checkpoints/sam2.1_hiera_base_plus.pt
```

```bash
mkdir checkpoints && wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O checkpoints/sam2.1_hiera_large.pt
```

## 切换模型

默认模型（最小模型）

```bash
python mesh_samesh.py --sam.sam.checkpoint checkpoints/sam2.1_hiera_tiny.pt --sam.sam.model_config configs/sam2.1/sam2.1_hiera_t.yaml
```

其他模型

```bash
python mesh_samesh.py --sam.sam.checkpoint checkpoints/sam2.1_hiera_small.pt --sam.sam.model_config configs/sam2.1/sam2.1_hiera_s.yaml
```

```bash
python mesh_samesh.py --sam.sam.checkpoint checkpoints/sam2.1_hiera_base_plus.pt --sam.sam.model_config configs/sam2.1/sam2.1_hiera_b+.yaml
```

```bash
python mesh_samesh.py --sam.sam.checkpoint checkpoints/sam2.1_hiera_large.pt --sam.sam.model_config configs/sam2.1/sam2.1_hiera_l.yaml
```

## 参数调优参考

### SAM引擎配置 (`sam.engine_config`)

| 参数                  | 类型                   | 默认值 | 用途说明                                                                                                                                                                                                                                                                                                                                               | 调优建议 |
| --------------------- | ---------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| **points_per_side**   | integer                | 8      | **控制自动采样网格的密度。** 值为8时，在图像的每条边上放置8个采样点，总共产生8×8=64个点。降低此值可加快处理速度，但会增加遗漏小物体的风险。                                                                                                                                           | 速度优先：降低到4-6；精度优先：增加到16-32 |
| **crop_n_layers**     | integer                | 0      | **多尺度分割使用的层数。** 值为0时禁用多尺度处理，仅在原始图像上运行模型，达到最高速度。增加此值可在多个裁剪和缩放尺度上进行分割，改善小物体检测，但计算成本显著增加。                                                                                                                             | 快速处理：保持0；小物体检测：增加到2-3 |
| **pred_iou_thresh**   | float                  | 0.7    | **预测掩码质量的阈值。** 模型为每个点生成多个候选掩码；此参数基于预测的IoU（掩码-物体重叠）进行过滤。降低阈值会保留更多低质量候选，可能增加召回率但也会增加噪声。                                                                                                                               | 高精度：提高到0.8-0.9；高召回：降低到0.4-0.6 |
| **stability_score_thresh** | float              | 0.9    | **掩码稳定性的阈值。** 模型检查每个掩码在小扰动下的稳定性。降低此值会保留稳定性较差的掩码，可能引入粗糙或嘈杂的边缘。                                                                                                                                                                     | 边缘平滑：提高到0.95-1.0；保留细节：降低到0.7-0.8 |
| **stability_score_offset** | float              | 1.0    | **稳定性评分计算中使用的偏移量。** 标准用例通常不需要调整。                                                                                                                                                                                                                             | 保持默认值 |

### SAM网格参数 (`sam_mesh`)

| 参数                              | 类型   | 默认值  | 用途说明                                                                                                                                                                   | 调优建议 |
| --------------------------------- | ------ | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| `min_area`                        | int    | 32     | 连通组件大小阈值（以像素为单位），用于从二值掩码中移除小工件。像素计数小于此值的连通组件（孤岛或孔洞）将被移除。基于OpenCV的`connectedComponentsWithStats`。                                                                                          | 高噪声：增加到64-128；保留细节：降低到8-16 |
| `face2label_threshold`             | int    | 4       | 面ID在标签区域内必须出现的最小次数，然后该面-标签关联才被视为有效。                                                                                                       | 高精度：提高到8-12；高召回：降低到2-3 |
| `connections_threshold`            | int    | 4       | 两个标签（来自不同视图）之间的连接所需的最小共享面观察次数，以便保留在匹配图中。                                                                                         | 高置信度：增加到8-16；更多连接：降低到1-2 |
| `counter_lens_threshold_min`       | int    | 16      | 在将标签视为"过度连接"之前考虑的连接数下限。与基于百分位数的阈值一起使用以强制图稀疏性。                                                                                   | 更严格：增加到20-30；更宽松：降低到0 |
| `connections_bin_resolution`      | int    | 100     | 用于建模连接强度比率分布的直方图箱数（控制直方图粒度）。                                                                                                                   | 细粒度：增加到150-200；粗粒度：降低到50-80 |
| `connections_bin_threshold_percentage` | float | 0.0     | 用于选择自适应截止箱的直方图总面积分数；低于相应比率的连接将被丢弃。                                                                                                        | 更严格：增加到0.05-0.1；更宽松：保持0.0 |
| `smoothing_threshold_percentage_size`  | float | 0.10    | 基于面数移除小连通组件的分数大小阈值（相对于最大组件）。                                                                                                                   | 更积极：增加到0.15-0.2；保守：降低到0.05-0.08 |
| `smoothing_threshold_percentage_area`  | float | 0.10    | 基于表面积移除小连通组件的分数面积阈值（相对于最大组件）。                                                                                                                 | 更积极：增加到0.15-0.2；保守：降低到0.05-0.08 |
| `smoothing_iterations`             | int    | 32      | 平滑传递次数。在每次传递中，未标记的面采用其标记邻居中最常见的标签。                                                                                                       | 更多填充：增加到64-128；避免过度平滑：降低到8-16 |
| `repartition_lambda`               | float  | 1.0     | 在图割能量中平衡数据成本与平滑成本的权重(λ)：总成本 = 数据成本 + λ × 平滑成本。更高的λ有利于更平滑、更连续的片段。                                                         | 更平滑：增加到2-4；更多细节：降低到0.5-1.0 |
| `repartition_iterations`           | int    | 1       | 图割优化器执行的alpha扩展周期数。更多迭代可以改进分区，但计算成本增加。                                                                                                   | 更好收敛：增加到3-8；快速处理：保持1 |
| `use_modes`                        | list   | ['matte'] | 使用的渲染模式，可选'matte'（遮罩）和'norms'（法线）。                                                                                                                    | 需要法线信息：添加'norms'；仅遮罩：保持['matte'] |
| `color_res`                        | float  | 125     | 用于DFS初步分割的颜色容差。                                                                                                                                         | 初步分割更多面片：减少；初步分割更少面片：增加 |
| `repartition_cost`                  | int    | 1       | 重新分区的成本参数。                                                                                                                                                       | 根据需要调整 |

---

## 详细调优指南

### `min_area` - 最小面积阈值

**用途**
从二值掩码中过滤小的连通组件以去除噪声，同时保留有意义的细节。

**工作原理**
连通组件分析（CCA）找到掩码中的区域并计算其像素面积。面积 `< min_area` 的组件被移除。

**模式特定行为**
* `islands` 模式：小的孤立前景岛被删除（设置为背景）。
* `holes` 模式：被前景包围的小背景孔洞被填充。

**调优技巧**
默认值 `32` 是合理的起点。增加以更积极地对抗噪声；降低以保留更精细的特征。根据图像分辨率和物体尺度选择。

---

### `face2label_threshold` - 面-标签关联阈值

**用途**
通过要求面ID在标签区域内出现最小次数才被接受，防止虚假或不可靠的面-标签关联。

**工作原理**
对于每个标签区域和视图，算法计算面ID（例如通过`np.bincount`）。计数 `> face2label_threshold` 的面ID被保留为有效关联。

**调优技巧**
* 提高（→ 例如20-25）以提高精度（减少错误关联）。
* 降低（→ 例如8-10）以改善召回率，但有更多噪声的风险。
  默认值 `4` 在典型渲染/视图设置中平衡了精度和召回率。

---

### `connections_threshold` - 连接阈值

**用途**
通过要求最小数量的共享面观察来过滤跨视图的成对标签连接。

**工作原理**
当一个面在两个视图中被观察并映射到标签 `(label1, label2)` 时，该标签对的计数器递增。只有计数 `> connections_threshold` 的对被保留。

**调优技巧**
* 增加以获得更稀疏、更高置信度的匹配图（例如40-50）。
* 降低以接受更多、可能更弱的连接（例如10-20）。
  默认值 `4` 是鲁棒性的较低要求，适合大多数网格。

---

### `counter_lens_threshold_min` - 连接数下限

**用途**
通过防止动态百分位数阈值变得太小，为标签连接性强制基线稀疏性。

**工作原理**
计算每标签连接数的第95百分位数，然后取 `max(95th_percentile, counter_lens_threshold_min)` 作为有效异常值截止值。连接数高于此截止值的标签被视为"过度连接"，并从比率计算中移除。

**调优技巧**
* 增加以更严格地移除高度连接的标签（更多稀疏性）。
* 降低以允许更多连接的标签保留。
  默认值 `16` 建立了一个适中的下限。

---

### `connections_bin_resolution` & `connections_bin_threshold_percentage` - 连接直方图参数

**用途**
实现自适应、数据驱动的阈值，基于归一化强度过滤连接。

**工作原理**
1. 使用 `connections_bin_resolution` 个箱构建连接强度比率的直方图。
2. 从最弱的箱开始，累积计数直到累积计数超过 `connections_bin_threshold_percentage * total_count`。发生这种情况的箱确定连接比率截止值；弱于该截止值的连接被丢弃。

**调优技巧**
* `connections_bin_resolution`：更高的值给出更详细的直方图（更敏感），更低的值给出更粗略、更稳健的分箱。
* `connections_bin_threshold_percentage`：更大的分数保留更少的连接（更严格）；更小的分数保留更多（更宽松）。
  默认值 `0.0` 表示不过滤连接，保留所有连接。增加此值可以过滤弱连接。

---

### `smoothing_threshold_percentage_size`, `smoothing_threshold_percentage_area`, 和 `smoothing_iterations` - 平滑参数

**用途**
两阶段清理：(1) 移除小的噪声组件；(2) 将标签传播到未标记的面以填补间隙和平滑边界。

**工作原理**
1. **清理** — 识别每个标签的连通面组件并计算每个组件的面数和表面积。只有当组件小于两个阈值（相对于最大组件的大小分数和面积分数）时才被移除。这种双重条件减少了移除有效的小而密集或稀疏但大的组件的机会。
2. **平滑** — 对于 `smoothing_iterations` 次传递，每个未标记的面被分配其相邻面中最常见的标签。每次传递将标签向外传播一个面层。

**调优技巧**
* 增加分数阈值以更积极地移除微小碎片。
* 增加 `smoothing_iterations` 以填补更大的间隙；降低以避免标签渗透。
  默认值（阈值为0.10，32次迭代）对于中等强度的间隙填充是平衡的；对于高分辨率网格，可以增加迭代次数。

---

### `repartition_lambda` 和 `repartition_iterations` - 重新分区参数

**上下文**
当 `target_labels` 为 `None` 且方法执行能量最小化以改进分区时使用。

**能量公式**
`总成本 = 数据成本 + λ × 平滑成本`

* **数据成本：** 面的标签与初始分配（来自 `cost_data`）的一致程度。
* **平滑成本：** 基于邻接性（二面角）的惩罚；更大的二面角允许更便宜的标签更改。

**参数**
* `repartition_lambda` (λ)：控制对初始标签的保真度与平滑、连续段之间的权衡。更高的λ → 更平滑的分区；更低的λ → 更多对数据的保真度。
* `repartition_iterations`：alpha扩展周期的数量。每个周期迭代标签并解决二进制图割以决定哪些面采用当前标签。更多迭代允许更好地收敛到更低能量的解决方案，但计算成本更高。

**调优技巧**
* 如果分割太嘈杂 → 增加 `repartition_lambda`。
* 如果分割过度平滑（失去细节）→ 降低 `repartition_lambda`。
* 典型的 `repartition_iterations` 范围：大多数情况下3-8；如果计算受限，默认值1或几次迭代可能足够。

---

## 渲染器参数 (`renderer`)

| 参数                      | 默认值        | 用途说明                                                                                                                                                                                                                                       | 调优建议 |
| -------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- |
| **`target_dim`**           | `[512, 512]` | 渲染图像的输出分辨率。                                                                                                                                                                                                                         | 高质量：增加到1024x1024；快速处理：降低到256x256 |
| **`camera_generation_method`** | `edge`            | 用于采样相机视点的方法。有效选项：`tetrahedron`, `octohedron`, `cube`, `icosahedron`, `dodecahedron`, `standard`, `swirl`, `sphere`, `edge`。                                                                                        | 根据物体形状选择 |
| **`sampling_args`**        | `{radius: 2}`            | 相机采样方法使用的参数。<br>• 所有方法都需要 `radius`。<br>• `sphere`：需要 `n`（样本数）。<br>• `standard`：可选 `n` 和 `elevation`。<br>• `swirl`：可选 `n`, `cycles`, 和 `elevation_range`。 | 根据方法调整 |
| **`lighting_args`**        | `{}`         | 可选的照明配置（默认为空）。                                                                                                                                                                                                               | 根据需要添加 |
| **`camera_params.type`**   | `orth`       | 相机投影类型，可选`orth`（正交投影）或`pers`（透视投影）。                                                                                                                                                                                  | 正交投影：保持orth；透视效果：使用pers |

### `renderer.renderer_args`
| 参数               | 默认值 | 用途说明                                                             | 调优建议 |
| ------------------- | ------ | ------------------------------------------------------------------- | -------- |
| **`uv_map`**        | `True` | 启用使用UV坐标进行纹理采样。                                        | 有纹理：保持True；无纹理：可设为False |
| **`interpolate_norms`** | `True` | 启用三角形表面之间的法线插值（平滑着色）。                         | 平滑着色：保持True；平坦着色：设为False |

---

## 常用调优场景

### 场景1：快速预览
```bash
python mesh_samesh.py --filename data/mesh.ply \
  --sam.sam.engine_config.points_per_side 4 \
  --sam_mesh.smoothing_iterations 16 \
  --renderer.target_dim 256 256 \
  --sam_mesh.min_area 32 --visualize
```

### 场景2：高质量分割
```bash
python mesh_samesh.py --filename data/mesh.ply \
  --sam.sam.engine_config.points_per_side 32 \
  --sam.sam.engine_config.pred_iou_thresh 0.8 \
  --sam.sam.engine_config.stability_score_thresh 0.95 \
  --sam_mesh.min_area 64 \
  --sam_mesh.smoothing_iterations 64 \
  --sam_mesh.repartition_lambda 4.0 \
  --sam_mesh.repartition_iterations 3 \
  --renderer.target_dim 1024 1024 --visualize
```

### 场景3：小物体检测
```bash
python mesh_samesh.py --filename data/mesh.ply \
  --sam.sam.engine_config.points_per_side 32 \
  --sam.sam.engine_config.pred_iou_thresh 0.5 \
  --sam_mesh.min_area 8 \
  --sam_mesh.face2label_threshold 2 \
  --sam_mesh.connections_threshold 2 --visualize
```

### 场景4：噪声数据
```bash
python mesh_samesh.py --filename data/mesh.ply \
  --sam.sam.engine_config.pred_iou_thresh 0.9 \
  --sam.sam.engine_config.stability_score_thresh 0.95 \
  --sam_mesh.min_area 128 \
  --sam_mesh.face2label_threshold 12 \
  --sam_mesh.connections_threshold 12 \
  --sam_mesh.repartition_lambda 4.0 --visualize
```

---

## 性能优化建议

1. **速度优化**：降低 `points_per_side`，减少 `smoothing_iterations`，降低 `renderer.target_dim`
2. **质量提升**：增加 `points_per_side`，启用多尺度处理，增加平滑迭代次数

---

## 故障排除

### 常见问题及解决方案

1. **分割结果过于碎片化**
   - 增加 `min_area`
   - 增加 `smoothing_threshold_percentage_size` 和 `smoothing_threshold_percentage_area`
   - 增加 `repartition_lambda`

2. **丢失小物体**
   - 降低 `min_area`
   - 降低 `face2label_threshold`
   - 增加 `points_per_side`
   - 设置 `smoothing_iterations=0`
   - 设置 `repartition_iterations=0`

3. **边界过于粗糙**
   - 增加 `smoothing_iterations`
   - 降低 `stability_score_thresh`
   - 增加 `repartition_iterations`

4. **处理速度过慢**
   - 降低 `points_per_side`
   - 降低 `smoothing_iterations`
   - 降低 `renderer.target_dim`

---

希望这个中文调优指南能帮助工作人员更好地理解和使用SAMesh的各项参数！
