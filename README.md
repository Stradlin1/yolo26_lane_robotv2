# YOLO26 Lane Robot V2 — 固定语义多线检测

基于 Ultralytics YOLO26 与 UFLD/Row-Anchor 思路改造的机器人前视多线检测工程。

> 更新日期：2026-08-03  
> 当前稳定方案：四个固定语义槽位、每个槽位最多一条曲线  
> 当前部署目标：RDK X5 / OpenExplorer / NV12 Runtime

本项目已将原始单线 Lane Robot 改造成四槽位多线检测系统，训练、验证、PyTorch 推理、ONNX 导出与 ONNX Runtime 推理主链路已经跑通。RDK X5 Runtime BIN 已成功生成，但当前分类输出层仍因 BPU 维度限制回退到 CPU，属于 BPU + CPU 混合执行模型。

---

## 1. 模型解决什么问题

模型直接从机器人前视 RGB 图像中识别四种固定语义线，不依赖 BEV：

| `lane_id` | 名称 | 语义 |
|---:|---|---|
| 0 | `lane_follow` | 跟随线 |
| 1 | `lead_lane` | 引导线 |
| 2 | `channel_left` | 黄色通道左边界 |
| 3 | `channel_right` | 黄色通道右边界 |

模型属于：

> 固定四种语义、每种语义在一张图中最多一条曲线的多线检测模型。

支持：

- 每张图存在 0～4 条线。
- 某个槽位整张图缺失。
- 某条线仅局部可见。
- 用 `x=-1` 标记遮挡、断点或当前 Row Anchor 无有效点。
- 四槽位联合训练、验证、解码、可视化和标签导出。

暂不支持：

- 同一张图中出现两条相同语义线，例如两条 `channel_left`。
- 任意数量的未知曲线实例。
- Hungarian matching 或动态实例分配。
- 已完成闭环验证的 Polyline 实例头。

---

## 2. 当前工程状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 单线改四槽位 | 已完成 | Dataset、Head、Loss、Predictor 均支持四槽位 |
| 标签读取 | 已验证 | 目标张量形状为 `[B, 56, 4]` |
| 模型前向与反向 | 已验证 | 六项损失可计算并完成反向传播 |
| Trainer / Validator | 已验证 | 可训练、验证并保存 `best.pt` / `last.pt` |
| PyTorch Predictor | 已接通 | 可解码、绘图和保存预测 txt |
| ONNX 导出 | 已完成 | Opset 11，当前合并输出 `[B, 322, 56, 4]` |
| ONNX Runtime 推理 | 已完成 | 支持 CPU/CUDA Provider、Resize/LetterBox |
| RDK X5 Runtime BIN | 已生成 | NV12 输入，当前为 BPU + CPU 混合执行 |
| 逐槽位指标 | 待完成 | 当前总体指标可能掩盖单槽位失败 |
| 严格断点绘制 | 待修正 | 当前 ONNX 绘图仍可能跨缺失 Anchor 连线 |
| Polyline head | 实验阶段 | 尚未完成训练、导出和部署闭环 |

---

## 3. 稳定模型结构

```text
输入图像
  ↓
Direct Resize 或自定义 LetterBox
  ↓
YOLO26 Backbone
  ↓
P4 + P5 多尺度特征融合
  ↓
LaneRobotV2 Head
  ├── cls:    [B, X+1, R, L]
  └── offset: [B, 1,   R, L]
  ↓
Top-K soft-argmax + offset
  ↓
固定四槽位曲线坐标
```

当前部署配置：

```text
X = x_grids    = 320
R = row_anchors = 56
L = num_lanes   = 4
```

PyTorch Head 输出：

```text
cls:    [B, 321, 56, 4]
offset: [B,   1, 56, 4]
```

ONNX 合并输出：

```text
lane_output: [B, 322, 56, 4]
```

通道定义：

```text
0..319  : 320 个横向网格 logits
320     : no-lane logit
321     : 亚网格 offset
```

模型没有独立 confidence Head。点存在概率来自：

```text
existence = 1 - P(no-lane)
```

---

## 4. 仓库关键文件

```text
train_xhm.py                         正式训练入口
export_onnx_xhm.py                   PT → ONNX Opset 11
infer_onnx_xhm.py                    ONNX Runtime 图片推理
check_empty_labels.py                空标签检查

ultralytics/cfg/datasets/
└── lane-robot.yaml                  数据路径、槽位和预处理配置

ultralytics/cfg/models/26/
├── yolo26n-lane.yaml
├── yolo26s-lane.yaml
├── yolo26m-lane.yaml
└── yolo26x-lane.yaml

ultralytics/models/yolo/lane/
├── dataset.py
├── geometry.py
├── train.py
├── val.py
├── predict.py
└── plotting.py

Lane_Robot_RDK_X5_quantization_issues_and_solutions_2026-08-03.md
```

RDK X5 的完整量化记录、YAML 配置、错误分析和拆 Head 方案见：

[Lane Robot RDK X5 量化问题与解决思路](Lane_Robot_RDK_X5_quantization_issues_and_solutions_2026-08-03.md)

---

## 5. 数据集结构

默认本地项目路径：

```text
/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT
```

数据目录：

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

当前 `ultralytics/cfg/datasets/lane-robot.yaml` 已显式配置：

```yaml
path: /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/datasets
train: images/train
val: images/valid

train_labels: labels_corrected/train
val_labels: labels_corrected/valid

x_grids: 320
row_anchors: 56
num_lanes: 4
y_start: 0.333333
y_end: 1.0

letterbox: false
letterbox_color: [0, 0, 0]
letterbox_bottom_align: true

nc: 4
names:
  0: lane_follow
  1: lead_lane
  2: channel_left
  3: channel_right

flip_lane_pairs:
  - [2, 3]
```

`num_lanes` 决定 Lane Head 的固定槽位数量；`x_grids` 决定横向分类网格数量。模型 YAML 中可能存在用于框架兼容的 `nc` 字段，判断实际 Lane 输出时应查看：

```text
LaneRobotV2 [320, 56, 4, ...]
```

以及实际输出形状，而不是只看某一个 YAML 字段。

---

## 6. 标签格式

每一行表示一个固定语义槽位：

```text
lane_id x1 y1 x2 y2 ... x56 y56
```

每行应有：

```text
1 + 56 × 2 = 113
```

个数值。

规则：

- `lane_id` 只能是 `0、1、2、3`。
- 同一标签文件中，同一个 `lane_id` 最多出现一次。
- `x` 为归一化横坐标，正常范围 `[0, 1]`。
- `x=-1` 表示该 Row Anchor 没有有效点。
- `y` 为归一化纵坐标。
- 某条线整张图不存在时，可省略该 `lane_id` 行。
- 某条线中间被遮挡时，遮挡区对应 Anchor 的 `x` 写成 `-1`，前后可见部分继续保留坐标。

例如一张图只有左右通道边界：

```text
2 x1 y1 x2 y2 ... x56 y56
3 x1 y1 x2 y2 ... x56 y56
```

56 个纵向锚点顺序为：

```text
1.000000 → 0.333333
```

即从图像底部向上。默认 Anchor 生成必须使用：

```python
np.linspace(y_end, y_start, row_anchors)
```

---

## 7. 遮挡、断点与绘制现状

训练标签可以通过 `x=-1` 表达真实断点。解码时，no-lane 概率超过阈值的点也会被恢复为 `-1`。

但当前 `infer_onnx_xhm.py` 的绘图逻辑会：

1. 跳过无效 Anchor；
2. 收集该槽位所有有效点；
3. 对全部有效点调用一次 `cv2.polylines()`。

因此，当一条线中间存在一段 `-1` 时，当前可视化仍可能把遮挡前后的两段跨空白连接起来。该问题属于绘图/后处理表达问题，不等于模型一定没有学到 no-lane。

严格保留断点时，应采用以下任一方式：

- 仅绘制点，不调用 `cv2.polylines()`；
- 按连续有效 Anchor 分段，每段分别画折线；
- 当相邻有效 Anchor 的索引差或像素距离超过阈值时强制断开。

部署控制层也不应把跨越大段无效 Anchor 的点直接拟合成一条连续曲线。

---

## 8. 环境安装

```bash
conda activate lane_robot
cd /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT
pip install -e .
```

确认导入的是当前仓库：

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
```

输出路径应位于：

```text
/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/ultralytics/
```

---

## 9. 训练

### 9.1 当前 `train_xhm.py` 的真实默认值

当前仓库脚本默认：

```text
model       = yolo26s-lane.yaml
imgsz       = [256, 320]
epochs      = 1000
batch       = 8
optimizer   = AdamW
lr0         = 3e-4
```

并且 `AUGMENTATION_CONFIG` 当前不是“全关闭”：

```text
hsv_h    = 0.005
hsv_s    = 0.15
hsv_v    = 0.15
degrees  = 2.0
fliplr   = 0.3
```

其中水平翻转会通过：

```yaml
flip_lane_pairs:
  - [2, 3]
```

同步交换 `channel_left` 和 `channel_right`。

如果要建立严格无增强基线，需要先修改 `train_xhm.py` 中的 `AUGMENTATION_CONFIG`，将颜色和几何增强显式设为 0。当前脚本没有为每个增强项单独提供 CLI 参数。

### 9.2 直接运行当前训练配置

```bash
python train_xhm.py \
  --name lane_s_current \
  --epochs 1000 \
  --patience 100 \
  --batch 8
```

### 9.3 面向当前 320×320 ONNX / RDK 部署的训练

当前 ONNX 导出和 RDK X5 量化基线使用静态 `320×320` 输入，因此建议训练时也明确指定：

```bash
python train_xhm.py \
  --img-height 320 \
  --img-width 320 \
  --name lane_s_320 \
  --epochs 1000 \
  --patience 100 \
  --batch 8
```

必须保持以下环节一致：

```text
训练预处理
→ PyTorch 验证/推理
→ ONNX 导出输入尺寸
→ ONNX Runtime 预处理
→ 量化校准预处理
→ RDK 板端预处理
```

不要用 `256×320` 训练权重，在未验证自适应池化和坐标映射影响的情况下直接按 `320×320` 量化部署。

### 9.4 训练日志检查

训练启动后应确认：

```text
LaneRobotV2 [320, 56, 4, ...]
```

并看到六项损失：

```text
lane_ce
lane_loc
lane_exist
lane_smooth
lane_curv
lane_offset
```

---

## 10. 验证与 PyTorch 推理

验证：

```bash
yolo task=lane mode=val \
  model=runs/lane/lane_s_320/weights/best.pt \
  data=ultralytics/cfg/datasets/lane-robot.yaml \
  imgsz=320 \
  device=0
```

预测：

```bash
yolo task=lane mode=predict \
  model=runs/lane/lane_s_320/weights/best.pt \
  source=datasets/images/valid \
  imgsz=320 \
  device=0 \
  save=True \
  save_txt=True \
  project=runs/lane \
  name=predict_lane_s_320 \
  exist_ok=True
```

当前 Validator 主要输出总体指标，例如：

```text
MAE
MAE_px
Acc@1
Acc@3
Acc@5
Exist
```

后续应增加每个槽位独立的 MAE、Exist、漏检率和左右边界混淆统计。

---

## 11. ONNX 导出

`export_onnx_xhm.py` 默认：

```text
input        = float32 [1, 3, 320, 320]
output       = float32 [1, 322, 56, 4]
opset        = 11
static batch = 1
```

导出并执行 ONNX Runtime 校验：

```bash
conda run -n lane_robot python export_onnx_xhm.py \
  --weights runs/lane/lane_s_320/weights/best.pt \
  --output runs/lane/lane_s_320/weights/best.onnx \
  --imgsz 320 320 \
  --verify-runtime
```

脚本默认要求 `x_grids=320`，用于防止误把旧的 160-grid 权重当成当前模型导出。只有明确处理旧模型时才使用：

```bash
--allow-legacy-x-grids
```

---

## 12. ONNX Runtime 推理

```bash
conda run -n lane_robot python infer_onnx_xhm.py \
  --model runs/lane/lane_s_320/weights/best.onnx \
  --source test \
  --output test_infer \
  --save-txt \
  --overwrite
```

主要后处理参数：

```text
exist_thr   = 0.5
topk        = 5
poly_degree = 2
poly_blend  = 0.5
offset clip = [-0.5, 0.5]
```

默认预处理：

```text
EXIF 修正
→ RGB
→ INTER_LINEAR 直接 Resize
→ float32 / 255
→ HWC 转 CHW
→ 增加 Batch
```

只有使用 LetterBox 训练的权重才应在推理时添加：

```bash
--letterbox
```

若出现：

```text
Failed to load library libonnxruntime_providers_cuda.so
libcudnn.so.9: cannot open shared object file
```

说明 ONNX Runtime GPU 包与本机 CUDA/cuDNN 不匹配，不是模型输出结构错误。可先使用：

```bash
--device cpu
```

验证模型和后处理。

---

## 13. RDK X5 量化与部署状态

当前量化环境：

```text
OpenExplorer
hb_mapper 1.24.3
hbdk 3.49.15
march = bayes-e
Runtime input = NV12
```

已经确认的 ONNX 输入输出：

```text
images      [1, 3, 320, 320] float32 NCHW
lane_output [1, 322, 56, 4] float32
```

量化训练输入语义：

```yaml
input_type_train: rgb
input_layout_train: NCHW
norm_type: data_scale
scale_value: 0.003921568627451
```

NV12 Runtime 输入不能配置为普通 DDR 输入源。删除 `input_source` 让工具链自动推导，或显式配置为 pyramid。

### 13.1 已解决

- ONNX Opset 11 模型可被工具链读取。
- NV12 Runtime 输入配置通过。
- 注意力 Softmax 可通过 `node_info` 指定到 BPU。
- Runtime BIN 已成功生成。
- offset 分支可运行在 BPU INT8。

### 13.2 当前核心限制

分类层：

```text
/model/model.16/cls_fc2/Gemm
```

一次性输出：

```text
321 × 56 × 4 = 71904
```

当前 BPU 相关维度上限为：

```text
65536
```

因此：

```text
71904 > 65536
```

该大 `Gemm` 无法进入 BPU，只能回退到 CPU float。

当前执行结构近似为：

```text
NV12 输入
→ Backbone：BPU INT8
→ Attention Softmax：BPU
→ cls_fc2：CPU float
→ offset_fc：BPU INT8
→ Reshape / Concat：CPU
→ lane_output [1, 322, 56, 4]
```

### 13.3 当前推荐路线

第一阶段先使用混合模型完成：

- 板端加载与输出检查。
- C++ 后处理。
- 精度对比。
- `hb_perf` 和真实端到端 FPS 测试。
- CPU 占用与 BPU/CPU 数据搬运开销测试。

如果分类 Head 成为明显瓶颈，再把一个大 Linear 真正拆成两个：

```text
cls_fc2_01: 321 × 56 × 2 = 35952
cls_fc2_23: 321 × 56 × 2 = 35952
offset_fc : 1 × 56 × 4   = 224
```

推荐新 ONNX 输出：

```text
cls_01 [1, 321, 56, 2]
cls_23 [1, 321, 56, 2]
offset [1,   1, 56, 4]
```

只在旧大 `Gemm` 后增加 `Split` 无效，必须在 PyTorch Head 中创建两个较小的 Linear/Gemm，并正确迁移旧权重。

---

## 14. 数据增强注意事项

通道左右边界具有固定语义，几何增强必须同时变换：

```text
图像
lane_x
lane_y
有效性标记
固定槽位 lane_id
```

水平翻转必须执行：

```text
x → 1 - x
channel_left ↔ channel_right
lane_id 2 ↔ lane_id 3
```

Mosaic、MixUp、CutMix、Copy-Paste 会破坏连续通道结构，当前训练脚本保持关闭。

颜色对黄色通道与绿色背景具有语义，HSV 增强不应过强。正式实验应保存每次运行的完整参数，不要只依赖默认值。

---

## 15. 正式训练前的数据检查

至少检查：

- 图片是否都有对应标签。
- 标签是否都有对应图片。
- `lane_id` 是否只在 `0～3`。
- 同一文件是否重复出现同一 `lane_id`。
- 每行是否正好包含 56 对坐标。
- `x` 是否为 `-1` 或 `[0, 1]`。
- 所有标签是否使用一致的 56 个 `y`。
- Row Anchor 顺序是否为底部到上方。
- 是否存在空文件、损坏图片、NaN 或非法文本。
- 四个槽位的图片数和有效 Anchor 数是否严重失衡。

代码能正常训练，不代表缺失槽位或极少样本槽位能被模型学会。

---

## 16. 当前已知配置风险

1. `train_xhm.py` 当前默认输入为 `256×320`，而 ONNX/RDK 基线为 `320×320`。
2. `train_xhm.py` 当前启用了轻微 HSV、旋转和水平翻转，并非无增强基线。
3. `infer_onnx_xhm.py` 的默认权重路径仍指向特定历史实验目录，正式使用应显式传 `--model`。
4. 当前 ONNX 绘图会把同一槽位所有有效点一次性连成折线，可能跨遮挡区连接。
5. 当前 RDK BIN 不是全 BPU 模型，分类大 `Gemm` 位于 CPU。
6. 当前总体验证指标不足以确认每个固定语义槽位都学会。

---

## 17. 下一步工作顺序

### P0：冻结可复现基线

- 固定输入尺寸和预处理策略。
- 固定数据划分和随机种子。
- 保存完整训练参数、代码提交、日志与权重。

### P1：数据质量与逐槽位统计

- 全量标签检查。
- 统计四槽位样本数和有效点数。
- 可视化训练输入经过增强后的真实标签。

### P2：增加逐槽位指标

至少增加：

```text
lane_follow/MAE
lead_lane/MAE
channel_left/MAE
channel_right/MAE

每槽位 Exist / Miss / False Positive
左右边界混淆率
```

### P3：修正断点绘制与控制输入

- 推理可视化改为点模式或连续段模式。
- 控制层禁止跨大段 no-lane 拟合。
- 对遮挡前后段分别评估稳定性。

### P4：RDK 板端实测

- 完成 C++ Softmax、Top-K soft-argmax、offset 和坐标恢复。
- 测试 Runtime BIN 的精度、FPS、CPU 和 BPU 占用。
- 再决定是否拆分分类 Head。

### P5：实验 Polyline Head

在相同数据和预处理条件下，对比：

- 当前 Row-Anchor Head。
- Row-Anchor + 分段后处理。
- Polyline Head。

重点比较转角、横向边界、遮挡断点、量化损失和端侧延迟。

---

## 18. 检测输出到机器人控制

模型只负责输出语义线坐标。控制层建议独立实现：

```text
线检测
→ 选择目标线或计算通道中心
→ 横向误差与航向误差
→ 时序滤波与异常检测
→ 速度/转角控制器
```

不应把单帧、未滤波、可能存在断点的预测坐标直接映射为电机命令。

---

## 19. 上游与许可证

本项目基于 Ultralytics 源码和原始 Lane Robot 项目继续修改。

原始 Ultralytics README：

```text
README.ultralytics.md
```

许可证：

```text
LICENSE
```

提交数据集、训练权重或第三方代码前，请分别确认数据授权、模型许可证和上游项目许可证要求。
