# ULTRALYTICS LANE ROBOT

基于 Ultralytics YOLO26 改造的机器人场景多线检测项目。

> 更新日期：2026-07-29 

本项目已经从原始的单线 Row-Anchor 检测，改造成**固定语义槽位的多线检测系统**。当前稳定主链路支持四种语义线，每种语义在一张图中最多对应一条曲线；训练、验证、预测、结果可视化和权重保存链路已经跑通。

当前工作重点已从“能否输出多条线”转向：完整数据集质量、训练参数基线、逐槽位指标、曲线连续性和端侧部署。

---

## 1. 项目目标

机器人需要在不使用 BEV 的前提下，直接从前视图像中识别具有固定语义的多条线，并将结果用于后续运动控制。

当前定义四个固定槽位：

| `lane_id` | 名称 | 语义 |
|---:|---|---|
| 0 | `lane_follow` | 跟随线 |
| 1 | `lead_lane` | 引导线 |
| 2 | `channel_left` | 黄色通道左边界 |
| 3 | `channel_right` | 黄色通道右边界 |

当前模型属于：

> 固定四种语义、每种最多一条曲线的多线检测模型。

它不是任意数量实例的曲线检测模型，也不是实例分割模型。

### 当前能力边界

支持：

- 每张图存在 0～4 条线。
- 某个槽位整张图缺失。
- 某条线只在部分纵向锚点可见。
- 四槽位联合训练、验证与预测。
- 分类网格、亚网格偏移、存在性、平滑性和曲率约束。

暂不支持：

- 同一张图中出现两条 `channel_left`。
- 任意数量、未知语义的曲线实例。
- Hungarian matching 或动态实例分配。
- 完整的实例级拓扑建模。
- 已完成部署验证的 polyline 输出分支。

---

## 2. 当前总体状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 单线改四槽位 | 已完成 | 数据、模型头和损失均支持四槽位 |
| 真实标签读取 | 已验证 | 可构造 `[B, 56, 4]` 标签张量 |
| 模型前向传播 | 已验证 | 分类与 offset 输出形状正确 |
| 六项损失 | 已验证 | 可计算并完成 `loss.backward()` |
| Trainer | 已验证 | 已完成真实数据 1 epoch 冒烟训练 |
| Validator | 已验证 | 可完成验证与权重保存 |
| Predictor | 已接通 | 可解码、画线、保存图片和 txt |
| 直接 Resize | 已完成 | 不使用 LetterBox 补黑边，训练/推理保持一致 |
| 多线水平翻转槽位交换 | 已实现/配置化 | `channel_left` 与 `channel_right` 需要同步交换 |
| 完整数据集正式训练 | 进行中 | 需要先完成全量标签检查和无增强基线 |
| 逐槽位指标 | 待完成 | 当前总体指标可能掩盖单槽位失败 |
| ONNX 端侧验证 | 部分完成 | 已进行导出/推理工作，运行环境仍需匹配 CUDA/cuDNN |
| Polyline head | 实验阶段 | 已执行接入步骤，仍需完整训练、导出和精度验证 |

---

## 3. 稳定主链路

```text
图片
  ↓
直接 Resize 到训练尺寸
  ↓
YOLO26 Backbone
  ↓
P4 + P5 多尺度特征融合
  ↓
LaneRobotV2 Head
  ├── cls:    [B, X+1, R, L]
  └── offset: [B, 1,   R, L]
  ↓
Row-Anchor 解码
  ↓
四个固定语义槽位的曲线坐标
```

默认参数：

```text
X = x_grids = 160
R = row_anchors = 56
L = num_lanes = 4
```

对应输出：

```text
cls:    [B, 161, 56, 4]
offset: [B,   1, 56, 4]
```

其中第 `160` 个分类位置是 `no-lane` 类。

---

## 4. 已完成的核心代码改造

### 4.1 新增 Lane 任务链路

围绕 Ultralytics 框架接入了独立的 `lane` 任务，包括：

```text
ultralytics/models/yolo/lane/
├── dataset.py
├── train.py
├── val.py
├── predict.py
├── plotting.py
└── __init__.py
```

同时修改了模型解析、任务注册、Loss 和配置系统，使 CLI/Python API 能识别：

```bash
task=lane
```

### 4.2 数据 YAML 控制多线维度

最新设计中，数据 YAML 是以下参数的主要来源：

```yaml
x_grids: 160
row_anchors: 56
num_lanes: 4
```

Trainer 构建模型时将这些数据配置同步到模型头，并检查数据维度与模型输出是否一致，避免直到损失计算阶段才发现形状错误。

### 4.3 LaneRobotV2 多尺度模型头

在原始 Row-Anchor 头基础上增加：

- P4 与 P5 多尺度特征融合。
- 横向网格分类输出。
- 亚网格 offset 回归输出。
- 自适应池化，允许实验不同输入尺寸。

模型配置位于：

```text
ultralytics/cfg/models/26/yolo26n-lane.yaml
ultralytics/cfg/models/26/yolo26s-lane.yaml
ultralytics/cfg/models/26/yolo26m-lane.yaml
ultralytics/cfg/models/26/yolo26l-lane.yaml
ultralytics/cfg/models/26/yolo26x-lane.yaml
```

### 4.4 六项训练损失

当前训练日志包含：

```text
lane_ce
lane_loc
lane_exist
lane_smooth
lane_curv
lane_offset
```

含义：

| Loss | 作用 |
|---|---|
| `lane_ce` | 每个 Row Anchor 的横向网格分类与 no-lane 分类 |
| `lane_loc` | soft-argmax 连续横坐标与标注坐标的定位误差 |
| `lane_exist` | 强化可见点与 no-lane 的区分 |
| `lane_smooth` | 约束相邻纵向锚点的一阶连续性 |
| `lane_curv` | 约束二阶变化，减少不合理折线和抖动 |
| `lane_offset` | 学习网格内亚像素/亚网格偏移 |

### 4.5 Predictor 与结果对象

预测链路已扩展为：

```text
模型原始输出
→ decode_lane()
→ [56, 4] 曲线坐标
→ LaneResults
→ 绘制结果
→ 保存图片
→ 保存预测 txt
```

结果对象支持的目标接口包括：

```python
result.lanes
result.row_y
result.active_lane_ids
result.verbose()
result.plot()
result.save()
result.save_txt()
```

### 4.6 训练与推理预处理对齐

当前使用**直接压缩/拉伸 Resize**，不使用 LetterBox 补黑边。

这样做的原因：

- Row Anchor 标签与图像归一化坐标直接对应。
- LetterBox 会改变有效画面区域和纵向坐标关系。
- 若训练和推理预处理不同，会造成解码位置系统性偏移。

因此训练、验证、PyTorch 推理和 ONNX 推理必须采用同一种 Resize 规则。

### 4.7 多线几何增强与水平翻转

几何增强不能只变换图像，还必须同步变换：

```text
lane_x
lane_y
有效性标记
lane_id 固定语义槽位
```

水平翻转时需要：

```text
x → 1 - x
channel_left ↔ channel_right
lane_id 2 ↔ lane_id 3
```

对应配置：

```yaml
flip_lane_pairs:
  - [2, 3]
```

该映射只在训练阶段实际触发水平翻转时生效，即：

```yaml
fliplr: 大于 0
```

若 `fliplr: 0.0`，即使保留 `flip_lane_pairs`，也不会执行槽位交换。

不配置映射却开启水平翻转，会导致图像左右反转，但标签仍保持原语义，从而污染 `channel_left` 和 `channel_right` 的监督信号。

### 4.8 Polyline head 实验分支

已经执行 polyline head 接入步骤，涉及：

```text
ultralytics/nn/modules/head.py
ultralytics/nn/modules/__init__.py
ultralytics/nn/tasks.py
ultralytics/cfg/models/26/yolo26n-lane-polyline.yaml
```

该方向用于直接回归或辅助表达更连续的曲线，解决通道拐角、横向边界和 Row Anchor 表达能力不足的问题。

目前应视为实验分支：

- 接入代码已开始。
- 尚不能替代稳定的 LaneRobotV2 主链路。
- 仍需 Dataset target、Loss、Trainer、Predictor、ONNX 导出和精度对比的完整闭环验证。

---

## 5. 数据集目录

当前本地项目根目录：

```text
/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT
```

推荐数据集结构：

```text
datasets/
├── images/
│   ├── train/
│   └── valid/
└── labels_corrected/
    ├── train/
    └── valid/
```

图片与标签必须同名：

```text
datasets/images/train/abc.jpg
datasets/labels_corrected/train/abc.txt
```

由于默认 Ultralytics 映射通常是 `images → labels`，使用 `labels_corrected` 时应在数据 YAML 中显式配置标签路径，不能依赖默认映射。

示例：

```yaml
path: /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/datasets

train: images/train
val: images/valid

train_labels: labels_corrected/train
val_labels: labels_corrected/valid

x_grids: 160
row_anchors: 56
num_lanes: 4

y_start: 0.333333
y_end: 1.0

channels: 3
nc: 4

names:
  0: lane_follow
  1: lead_lane
  2: channel_left
  3: channel_right

flip_lane_pairs:
  - [2, 3]
```

---

## 6. 标签格式

每一行表示一个固定槽位中的一条曲线：

```text
lane_id x1 y1 x2 y2 ... x56 y56
```

每行共有：

```text
1 + 56 × 2 = 113
```

个数值。

规则：

- `lane_id` 只能是 `0、1、2、3`。
- 同一个标签文件中，同一个 `lane_id` 最多出现一次。
- `x` 为归一化横坐标，正常范围为 `[0, 1]`。
- `x=-1` 表示该纵向锚点没有有效点。
- `y` 为归一化纵坐标。
- 某条线整张图不存在时，可不写该 `lane_id`。
- 某条线局部可见时，保留该行，不可见位置写 `x=-1`。
- 一张图片不要求同时存在四条线。

示例：某张图只有黄色通道左右边界：

```text
2 x1 y1 x2 y2 ... x56 y56
3 x1 y1 x2 y2 ... x56 y56
```

### Row Anchor 顺序

当前标签的 56 个纵向锚点按以下方向排列：

```text
1.000000 → 0.333333
```

即从图像底部向上。

空标签或自动生成默认 `row_y` 时，也必须保持同样顺序：

```python
np.linspace(y_end, y_start, row_anchors)
```

不能生成相反方向，否则预测曲线与图像纵向位置会错位。

---

## 7. `nc`、`num_lanes` 与模型 YAML

这是当前项目中最容易混淆的配置。

### 7.1 真正决定输出线槽位数量的是 `num_lanes`

```yaml
num_lanes: 4
```

最终模型输出最后一维应为 4。

### 7.2 `nc` 不是 Row-Anchor 横向分类数

横向分类数由：

```yaml
x_grids: 160
```

决定，模型内部会额外增加一个 no-lane 类，因此分类维度为 161。

### 7.3 为什么模型 YAML 中可能仍看到 `nc: 1`

Lane 任务沿用了部分 Ultralytics 通用模型配置字段，`nc` 在模型 YAML 中可能只是框架兼容占位值，不等于四个 Lane 槽位。

对于最新改造，应以数据 YAML 中的以下字段为准：

```yaml
num_lanes: 4
names:
  0: lane_follow
  1: lead_lane
  2: channel_left
  3: channel_right
```

判断是否真正生效，不要只看 YAML 文本，应查看模型构建日志：

```text
LaneRobotV2 [160, 56, 4, ...]
```

以及前向输出：

```text
cls:    [B, 161, 56, 4]
offset: [B, 1, 56, 4]
```

---

## 8. 配置来源与修改位置

训练参数可能来自多个位置，优先级必须明确。

### 8.1 推荐做法

将项目固定配置写入专用数据 YAML 和训练 YAML，训练命令只覆盖本次实验变化。

主要位置：

```text
ultralytics/cfg/default.yaml
ultralytics/cfg/datasets/lane-robot.yaml
ultralytics/cfg/models/26/yolo26n-lane.yaml
train.py
CLI 参数
```

一般情况下，显式 CLI 参数或 `model.train(...)` 参数会覆盖默认配置。

### 8.2 关闭所有训练增强

建立基线时建议显式设置：

```yaml
hsv_h: 0.0
hsv_s: 0.0
hsv_v: 0.0

degrees: 0.0
translate: 0.0
scale: 0.0
shear: 0.0
perspective: 0.0

flipud: 0.0
fliplr: 0.0

mosaic: 0.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
erasing: 0.0
```

只修改 `default.yaml` 并不保证一定生效；还需要检查：

- 根目录 `train.py` 是否传入覆盖参数。
- CLI 命令是否覆盖。
- Dataset 是否自行实现几何增强。
- 旧实验是否通过 `resume` 读取了原训练参数。

### 8.3 当前推荐无几何增强基线

考虑黄色通道与绿色外部区域的颜色语义，建议先用：

```yaml
degrees: 0.0
translate: 0.0
scale: 0.0
shear: 0.0
perspective: 0.0

fliplr: 0.0
flipud: 0.0

mosaic: 0.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0

hsv_h: 0.0
hsv_s: 0.05
hsv_v: 0.10
```

在基线稳定后，再逐项加入轻微平移、缩放和旋转，不能一次打开所有增强。

---

## 9. 训练

### 9.1 环境安装

建议在项目虚拟环境中以 editable 方式安装当前源码：

```bash
cd /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT
pip install -e .
```

确认当前导入的是本仓库，而不是系统中另一个 Ultralytics：

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
```

输出路径应位于：

```text
/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/ultralytics/
```

### 9.2 无增强基线训练

```bash
cd /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT

yolo task=lane mode=train \
  model=ultralytics/cfg/models/26/yolo26n-lane.yaml \
  data=ultralytics/cfg/datasets/lane-robot.yaml \
  epochs=100 \
  batch=8 \
  imgsz=320 \
  workers=4 \
  device=0 \
  degrees=0.0 \
  translate=0.0 \
  scale=0.0 \
  shear=0.0 \
  perspective=0.0 \
  fliplr=0.0 \
  flipud=0.0 \
  mosaic=0.0 \
  mixup=0.0 \
  cutmix=0.0 \
  copy_paste=0.0 \
  project=/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/runs/lane \
  name=baseline_no_aug
```

显存不足时优先降低：

```text
batch
imgsz
模型尺寸（m → s → n）
```

### 9.3 训练日志检查

启动训练后必须确认：

```text
LaneRobotV2 [160, 56, 4, ...]
```

以及日志中存在六项 Loss：

```text
lane_ce lane_loc lane_exist lane_smooth lane_curv lane_offset
```

如果模型日志仍显示 `num_lanes=1`，不要继续长时间训练，应先检查数据 YAML、Trainer 覆盖逻辑和实际导入的源码路径。

---

## 10. 验证与预测

### 10.1 验证

```bash
yolo task=lane mode=val \
  model=/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/runs/lane/baseline_no_aug/weights/best.pt \
  data=ultralytics/cfg/datasets/lane-robot.yaml \
  imgsz=320 \
  device=0
```

当前 Validator 主要提供整体指标，例如：

```text
MAE
MAE_px
Acc@1
Acc@3
Acc@5
Exist
```

后续应增加：

```text
lane_follow/MAE
lead_lane/MAE
channel_left/MAE
channel_right/MAE

lane_follow/Exist
lead_lane/Exist
channel_left/Exist
channel_right/Exist
```

### 10.2 PyTorch 预测

```bash
yolo task=lane mode=predict \
  model=/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/runs/lane/baseline_no_aug/weights/best.pt \
  source=/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/datasets/images/valid \
  imgsz=320 \
  device=0 \
  save=True \
  save_txt=True \
  project=/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/runs/lane \
  name=predict_baseline \
  exist_ok=True
```

预测 txt 与训练标签使用同一种固定槽位格式。

---

## 11. ONNX 导出与推理

项目已经开展 PT → ONNX 导出和 ONNX Runtime 推理适配工作。

关键要求：

- ONNX 输入预处理必须与训练保持一致，使用直接 Resize。
- 明确输出中 `cls` 与 `offset` 的拼接或多输出形式。
- 解码时使用相同的 `x_grids`、`row_anchors`、`num_lanes` 和存在性阈值。
- 保存结果时恢复到原图尺寸。

遇到以下错误：

```text
Failed to load library libonnxruntime_providers_cuda.so
libcudnn.so.9: cannot open shared object file
```

表示 ONNX Runtime GPU 包要求的 CUDA/cuDNN 版本与本机环境不匹配，不是模型结构本身报错。

临时验证可切换 CPU Provider；正式 GPU 部署应安装与当前 CUDA、cuDNN 对应的 `onnxruntime-gpu` 版本。

---

## 12. 已完成的辅助数据工具

项目过程中已经设计或生成以下类型的脚本：

- 标签格式检查。
- 图片与标签同名匹配检查。
- 从图片目录中筛选没有对应标签的图片。
- 两个或多个数据集的合并与移动。
- 按命名顺序执行 9:1 的 train/valid 划分。
- 从已有标签集合提取对应原图。
- 标签可视化并保存到 `labelview`。
- 56 Anchor 手工标注与修正工具。
- 无原始标签时从空白状态开始标注。
- 标注窗口、文字尺寸和控制点半径优化。
- PT 转 ONNX 与 ONNX 推理脚本。

这些工具的路径和版本需要在最终整理仓库时统一放入：

```text
scripts/
├── dataset/
├── annotation/
├── visualization/
└── deployment/
```

并补充统一命令说明。目前部分脚本可能位于项目外部或尚未包含在“仅代码”压缩包中。

---

## 13. 已完成验证

### 13.1 真实数据读取

已验证批次形状：

```text
batch img:    [2, 3, 256, 320]
batch lane:   [2, 56, 4]
batch lane_x: [2, 56, 4]
batch lane_y: [2, 56]
```

### 13.2 模型输出

已验证：

```text
cls:    [2, 161, 56, 4]
offset: [2, 1, 56, 4]
```

### 13.3 训练闭环

已完成：

```text
真实 YAML
→ 真实标签
→ LaneRobotDataset
→ 四槽位模型
→ 六项损失
→ backward
→ optimizer
→ Trainer
→ Validator
→ best.pt / last.pt
```

### 13.4 预测闭环

已完成：

```text
best.pt
→ yolo task=lane mode=predict
→ 多槽位解码
→ 可视化图片
→ 预测 txt
```

早期 1 epoch 或极少数据训练只用于验证代码链路，不代表模型已经具备实际精度。

---

## 14. 当前数据问题与训练风险

### 14.1 类别/槽位不平衡

早期测试数据曾出现：

```text
lane_follow:   0
lead_lane:     0
channel_left:  少量
channel_right: 少量
```

在这种数据上，代码可以正常训练，但模型不可能学会缺失的槽位。

正式训练前必须统计：

- 每个槽位出现的图片数。
- 每个槽位有效 Anchor 点数量。
- 每张图平均可见线数。
- 各场景、光照、转角和遮挡分布。

### 14.2 标签一致性

必须检查：

- 图片是否都有对应标签。
- 标签是否都有对应图片。
- `lane_id` 是否只在 `0～3`。
- 同一文件是否重复出现同一个 `lane_id`。
- 每行是否正好包含 56 对坐标。
- `x` 是否为 `-1` 或 `[0, 1]`。
- 所有标签是否使用相同的 56 个 `y`。
- `row_y` 顺序是否一致。
- 是否存在损坏图片、空文本和非法数值。

### 14.3 数据增强污染

本项目中的左右边界具有明确语义，几何增强错误会直接制造错误标签。

如果需要定位训练结果突然变差，应优先做以下对照：

1. 完全关闭所有增强。
2. 固定随机种子和数据划分。
3. 可视化 Dataset 送入模型前的图片与标签。
4. 比较直接 Resize 与 LetterBox 后的标签位置。
5. 单独开启一种增强，观察可视化结果。

---

## 15. 当前压缩包与配置注意事项

历史代码压缩包中可能仍保留旧配置，例如：

```text
/home/baater/ultralytics/...
num_lanes: 1
y_start: 0.67
```

这些值不应直接用于当前四槽位数据集。

当前本地环境应统一为：

```text
项目根目录：/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT
数据根目录：/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/datasets
num_lanes: 4
y_start: 0.333333
y_end: 1.0
```

每次训练前建议打印实际加载后的配置，而不是只查看某一个 YAML 文件。

---

## 16. 下一步工作顺序

### P0：冻结稳定主链路

- 不再频繁修改基础 LaneRobotV2 结构。
- 对当前可训练版本建立 Git tag 或明确提交。
- 将 polyline 分支与稳定分支分开。

### P1：完成数据闭环

- 合并所有已标注数据。
- 按命名顺序完成 train/valid 划分。
- 全量执行标签检查。
- 统计四槽位分布。
- 随机可视化并人工抽查。

### P2：训练无增强基线

- 关闭全部几何增强。
- 使用固定数据划分和固定随机种子。
- 保存配置、日志、权重和预测可视化。
- 不使用 1 epoch 结果判断精度。

### P3：增加逐槽位指标

重点防止总体指标掩盖：

- 某个槽位完全不输出。
- 左右边界混淆。
- 只学习样本最多的槽位。

### P4：逐项加入安全增强

建议顺序：

1. 轻微亮度变化。
2. 轻微平移。
3. 轻微缩放。
4. 极小角度旋转。
5. 最后测试带 `flip_lane_pairs` 的水平翻转。

### P5：评估 Polyline 分支

使用相同数据划分对比：

- LaneRobotV2 Row-Anchor。
- Row-Anchor + 后处理多项式平滑。
- Polyline head。

比较指标不仅包括点误差，还应包括：

- 曲线连续性。
- 转角处稳定性。
- 左右边界拓扑正确率。
- 推理速度。
- ONNX 导出复杂度。
- 端侧量化精度损失。

### P6：控制接口

模型输出线坐标后，还需要独立的控制层：

```text
线检测
→ 选取目标线/通道中心
→ 计算横向误差与航向误差
→ 滤波与异常处理
→ 速度/转角控制器
```

检测模型不应直接输出未经约束的电机控制命令。

---

## 17. Git 建议

建议后续提交按功能拆分：

```text
feat(lane): connect four-slot dataset to model head
feat(lane): add multi-lane prediction results and decoding
feat(lane): add synchronized geometric augmentation
fix(lane): keep resize preprocessing consistent
feat(lane): add per-slot validation metrics
feat(polyline): add experimental polyline head
chore(data): add dataset validation and split scripts
docs: update lane robot project README
```

不要将完整数据集、训练输出和临时推理结果提交到仓库。

推荐 `.gitignore` 至少包含：

```gitignore
datasets/
runs/
scripts/__pycache__/
test/
test_infer/
*.pt
*.onnx
```

是否忽略整个 `scripts/` 取决于其中是否包含需要版本管理的正式工具；通常正式脚本应提交，只忽略脚本输出目录。

---

## 18. 结论

当前已完成的主链路是：

```text
四槽位配置
→ 多线标签读取
→ P4+P5 LaneRobotV2
→ cls + offset 输出
→ 六项损失
→ 反向传播
→ Trainer
→ Validator
→ 权重保存
→ Predictor
→ 图片和 txt 输出
```

因此可以确认：

> 固定四类、每类最多一条曲线的多线检测主链路已经完成并经过冒烟验证。

当前尚不能确认的是正式精度、完整数据泛化能力、polyline 分支收益和端侧部署效果。下一阶段应优先完成全量数据检查与无增强基线训练，再进行模型结构扩展。

---

## 19. 上游项目与许可证

本项目基于 Ultralytics 源码修改。原始上游 README 已保存在：

```text
README.ultralytics.md
```

许可证见：

```text
LICENSE
```
