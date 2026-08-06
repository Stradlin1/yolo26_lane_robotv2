# YOLO26 Lane Robot — 固定语义多线检测

基于 Ultralytics YOLO26 与 UFLD / Row-Anchor 思路改造的机器人前视多线检测工程。

> 更新日期：2026-08-06  
> 当前开发分支：`zbn`  
> 当前训练方案：`LaneRobotV3B + CausalRowConv`，输入 `256×448`（非正方形）  
> 部署状态：V3B+causal 正式训练完成，ONNX 与 RDK X5 int8 量化已生成（待部署实测）

本项目将原始单线 Lane Robot 改造成四个固定语义槽位的多线检测系统。`zbn` 分支新增了 V3-B Row-Anchor Head、独立训练入口、逐槽位验证指标和可移植的数据集路径配置。

需要明确区分：

- **V2**：训练、验证、PyTorch 推理、ONNX 导出、ONNX Runtime 推理和 RDK X5 Runtime BIN 已跑通，是当前部署基线。
- **V3B + Causal**：模型结构、训练管线、正式训练、ONNX 导出和 RDK X5 量化均已完成，输出协议保持兼容。

---

## 1. 任务定义

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
- 某个语义槽位整张图缺失。
- 某条线仅局部可见。
- 使用 `x=-1` 标记遮挡、断点或当前 Row Anchor 无有效点。
- 四槽位联合训练、验证、解码、可视化和标签导出。

暂不支持：

- 同一张图中出现两条相同语义线，例如两条 `channel_left`。
- 任意数量的未知曲线实例。
- Hungarian matching 或动态实例分配。
- 已完成训练、导出和部署闭环的 Polyline 实例头。

---

## 2. 当前工程状态

| 模块 | V2 | V3B / `zbn` |
|---|---|---|
| 四槽位 Dataset / Loss | 已验证 | 复用并修正 offset / soft-label 目标 |
| 模型前向与反向 | 已验证 | 代码已接入，需正式训练验证 |
| Trainer | 已验证 | 新增 `train_v3b.py` |
| Validator | 仅总体指标 | 已增加逐槽位指标 |
| PyTorch Predictor | 已接通 | 输出协议相同，需端到端复测 |
| ONNX 导出 | 已完成，`320×320` | 尚未完成 V3B `256×448` 闭环 |
| ONNX Runtime 推理 | 已完成 | 尚未复测 |
| RDK X5 Runtime BIN | 已生成，混合执行 | 尚未量化 |
| 数据集路径 | YAML 中固定路径 | 支持 `LANE_ROBOT_DATASETS` 覆盖 |
| 严格断点绘制 | 仍需注意 | 仍需注意 |

---

## 3. V3B 新模型结构

V3B 保留 V2 的 YOLO26 Backbone 和 P4 + P5 融合部分，替换最终 Lane Head。

```text
输入 RGB 图像 [B, 3, 256, 448]
  ↓
YOLO26 Backbone
  ↓
P4 + P5 多尺度融合，stride = 16
  ↓
融合特征 [B, 256, 16, 28]
  ↓
Conv1x1: 256 → 16
  ↓
固定双线性 Row Sampling：56 个纵向锚点
  ↓
每行特征：16×28 展平 + 16 维行均值 = 464
  ↓
Causal Row Conv：2×Conv2d(kernel=(1,3))，近处行信息向远处行传播
  ↓
共享投影：464 → 512 + ReLU
  ↓
加可学习 Row Embedding
  ↓
4 个 lane-specific 分类器：512 → 321
4 个 lane-specific offset 回归器：512 → 1
  ↓
cls    [B, 321, 56, 4]
offset [B,   1, 56, 4]
```

V3B 的关键变化：

- 使用固定双线性采样，将特征图映射到 56 个 Row Anchors。
- 分类器参数在 56 个 Row Anchors 之间共享。
- 四个语义槽位使用独立分类器和 offset 回归器。
- 使用可学习 Row Embedding 保留不同纵向位置的信息。
- 行向量经过 Causal Row Conv（r=0 底部 → r=55 顶部单向传播），远行可以显式参考近处行信息，缓解弱特征远行的孤立决策与跳变。
- offset 经 `tanh` 限制到 `[-0.5, 0.5]`。
- 不再使用 V2 中一次性输出 `321×56×4` 的单个大分类 Linear。

模型 YAML：

```text
ultralytics/cfg/models/26/yolo26s-lane-v3b.yaml
```

核心配置：

```yaml
x_grids: 320
row_anchors: 56
num_lanes: 4
reduce_channels: 16
hidden_dim: 512
feat_h: 16
feat_w: 28
y_start: 0.333333
y_end: 1.0
causal_kernel: 3
causal_layers: 2
```

目标输入为：

```text
height = 256
width  = 448
aspect = 1.75
```

宽高都能被 32 整除，比例接近原始 16:9 相机画面。

---

## 4. 输出协议

V2 与 V3B 的 PyTorch 输出协议保持一致：

```text
cls:    [B, X+1, R, L]
offset: [B, 1,   R, L]
```

当前配置：

```text
X = x_grids     = 320
R = row_anchors = 56
L = num_lanes   = 4
```

因此：

```text
cls:    [B, 321, 56, 4]
offset: [B,   1, 56, 4]
```

分类通道定义：

```text
0..319 : 320 个横向网格位置
320    : no-lane 类别
```

点存在概率来自：

```text
existence = 1 - P(no-lane)
```

连续横向坐标由局部 Top-K soft-argmax 与 offset 共同得到。

V2 现有 ONNX 导出脚本会把两部分合并成：

```text
lane_output [B, 322, 56, 4]
```

V3B 虽然保持相同逻辑输出，但必须重新验证导出脚本对 `256×448` 输入和 V3B Head 的兼容性，不能直接把 V2 的部署结论套用到 V3B。

---

## 5. 仓库关键文件

```text
train_v3b.py                         V3B 正式训练入口
train_xhm.py                         V2 训练入口
export_onnx_xhm.py                   V2 PT → ONNX Opset 11 基线
infer_onnx_xhm.py                    ONNX Runtime 图片推理
check_empty_labels.py                空标签检查

ultralytics/cfg/datasets/
└── lane-robot.yaml                  数据路径、槽位和预处理配置

ultralytics/cfg/models/26/
├── yolo26s-lane-v3b.yaml            V3B 模型配置
├── yolo26n-lane.yaml                V2 配置
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

ultralytics/nn/modules/head.py        LaneRobot / V2 / V3B Head
ultralytics/utils/loss.py             LaneRobot 六项损失

Lane_Robot_RDK_X5_quantization_issues_and_solutions_2026-08-03.md
```

RDK X5 的 V2 量化记录、YAML 配置、错误分析和拆 Head 思路见：

[Lane Robot RDK X5 量化问题与解决思路](Lane_Robot_RDK_X5_quantization_issues_and_solutions_2026-08-03.md)

---

## 6. 数据集结构

默认目录结构：

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

`ultralytics/cfg/datasets/lane-robot.yaml`：

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

### 6.1 使用环境变量覆盖数据集根目录

`zbn` 分支支持：

```bash
export LANE_ROBOT_DATASETS=/absolute/path/to/datasets
```

Trainer 和显式标签路径会优先使用该环境变量，不需要在不同机器上反复修改 YAML 中的 `path`。

例如：

```bash
export LANE_ROBOT_DATASETS=/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/datasets
python train_v3b.py --weights runs/lane/lane_n_baseline/weights/last.pt
```

---

## 7. 标签格式

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
- 某条线中间被遮挡时，遮挡区对应 Anchor 的 `x` 写为 `-1`，前后可见部分继续保留坐标。

例如一张图只有左右通道边界：

```text
2 x1 y1 x2 y2 ... x56 y56
3 x1 y1 x2 y2 ... x56 y56
```

56 个纵向锚点按以下顺序保存：

```text
1.000000 → 0.333333
```

即从图像底部向上。默认锚点生成应使用：

```python
np.linspace(y_end, y_start, row_anchors)
```

---

## 8. 环境安装

```bash
conda activate lane_robot
cd /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT
pip install -e .
```

确认 Python 导入的是当前仓库：

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
```

---

## 9. V3B 训练

> ⚠️ **训练必须使用非正方形 imgsz `[256, 448]`**。Ultralytics 官方 Trainer 会把 train/val imgsz 强制为正方形
> （`check_imgsz(max_dim=1)` 会把 `[256, 448]` 改写成 `448`），导致训练输入变成 `448×448`，
> 与导出/推理的 `256×448` 不一致——历史版本曾因此出现训练/推理口径错位。本仓库已在
> `LaneRobotTrainer` / `LaneRobotValidator` 中修复为保留矩形 imgsz，但使用时必须遵守：
>
> 1. 通过 `train_v3b.py`（或 lane 分支的 Trainer）启动，不要直接调用通用 `YOLO.train(imgsz=448)`；
> 2. 启动日志必须出现 `Image sizes [256, 448] train, [256, 448] val`；
> 3. 若看到 `Image sizes 448 train, 448 val`（单值）或 `updating to 'imgsz=448'` 以外的强制提示，说明走了官方正方形逻辑，立即停止。

### 9.1 默认训练配置

`train_v3b.py` 的主要默认值：

```text
model        = yolo26s-lane-v3b.yaml
imgsz        = [256, 448]
epochs       = 500
patience     = 100
batch        = -1       # Ultralytics Autobatch
workers      = 8
optimizer    = AdamW
lr0          = 3e-4
lrf          = 0.01
weight_decay = 0.01
warmup       = 3 epochs
cos_lr       = true
```

增强配置：

```text
hsv_h       = 0.002
hsv_s       = 0.05
hsv_v       = 0.05
degrees     = 2.0
translate   = 0.03
scale       = 0.05
fliplr      = 0.5
flipud      = 0.0
shear       = 0.0
perspective = 0.0
mosaic      = 0.0
mixup       = 0.0
cutmix      = 0.0
copy_paste  = 0.0
erasing     = 0.0
```

推荐正式训练命令（从上一版 V3B 权重续训；新增的 causal 层会自动随机初始化，
其余 backbone/neck/head 全部继承，启动日志会显示 `Transferred 283/285 items`）：

```bash
python train_v3b.py \
  --weights runs/lane/lane_v3b/weights/best.pt \
  --name lane_v3b_causal \
  --epochs 200 \
  --patience 60 \
  --batch -1
```

`--batch -1` 使用 Ultralytics Autobatch（RTX 4090 上约为 107）；`--patience 60` 配合
续训通常会在 100 轮以内早停（本仓库实测 best epoch 39、99 轮早停，总耗时约 2.7 小时）。

### 9.2 从 V2 权重迁移

推荐命令：

```bash
python train_v3b.py \
  --weights runs/lane/lane_n_baseline/weights/last.pt \
  --name lane_v3b
```

V2 与 V3B 的 Backbone / Neck 保持相同，兼容层可迁移；V3B Head 结构不同，应从随机初始化开始训练。启动日志中应检查实际 transferred 参数数量，不能只根据命令假设 Head 已正确排除。

从已有 V3B checkpoint 续训同理：`--weights runs/lane/lane_v3b/weights/best.pt`。
若旧 checkpoint 不含 causal 权重，Ultralytics 会非严格加载，仅 causal 层随机初始化，无需手动处理。

常用覆盖：

```bash
python train_v3b.py \
  --weights runs/lane/lane_n_baseline/weights/last.pt \
  --epochs 300 \
  --patience 60 \
  --batch 16 \
  --name lane_v3b_e300
```

### 9.3 从随机初始化训练

```bash
python train_v3b.py --name lane_v3b_scratch
```

### 9.4 中断后续训

```bash
python train_v3b.py \
  --resume runs/lane/lane_v3b/weights/last.pt
```

`--resume` 与 `--weights` 不是同一用途：

- `--weights`：创建新的 V3B 实验并迁移兼容权重。
- `--resume`：恢复同一个已中断实验的模型、优化器和训练状态。

### 9.5 启动后检查

模型摘要中应看到：

```text
LaneRobotV3B
```

并确认：

```text
input  = [B, 3, 256, 448]
cls    = [B, 321, 56, 4]
offset = [B,   1, 56, 4]
causal = CausalRowConv(kernel=3, layers=2)
```

同时确认日志出现 `Image sizes [256, 448] train, [256, 448] val`（非正方形已生效）。

训练日志仍包含六项损失：

```text
lane_ce
lane_loc
lane_exist
lane_smooth
lane_curv
lane_offset
```

---

## 10. 损失函数修正

`zbn` 分支统一了 soft-label 分类中心和 offset 目标的整数基准：

```text
base = round(target_x)
offset_target = target_x - base
```

soft-label 高斯中心同样使用：

```text
round(target_x)
```

这样分类网格中心与 offset 残差使用相同基准，避免一个使用 floor、另一个使用 round 造成目标不一致。

### 10.1 Offset 监督（Plan-A）

offset 头**不再有独立的 SmoothL1 目标**（`lane_offset` 参数保留但已停用），统一由
`lane_loc` 对 `soft-argmax(cls) + offset` 的整体位置进行监督。这样 offset 学到的是
“soft-argmax 解码偏差的补偿”，训练目标与推理行为完全一致，避免弱特征行上两个损失互相拉扯。

### 10.2 远端行加权已拉平

`lane_end_weight` / `lane_end_weight_tail` / `lane_end_no_lane_weight` 默认均为 `1.0`。
历史版本曾对 r25~34 行 CE/loc 加权 3→6、并将 no-lane 监督降权到 0.3，实测导致远端
误检翻倍。causal 行耦合已从结构上补偿远端信息，不再需要 loss 加权补丁；如需复现旧行为，
可显式传入 `--lane-end-weight 3.0 --lane-end-weight-tail 6.0 --lane-end-no-lane-weight 0.3`。

---

## 11. 验证指标

总体指标仍包括：

```text
metrics/lane_mae
metrics/lane_mae_px
metrics/lane_acc_valid_tol1
metrics/lane_acc_valid_tol3
metrics/lane_acc_valid_tol5
metrics/lane_exist_acc
```

`zbn` 分支新增每个槽位独立指标：

```text
metrics/lane0_mae
metrics/lane0_acc_valid_tol1
metrics/lane0_acc_valid_tol3
metrics/lane0_acc_valid_tol5
metrics/lane0_exist_acc

...

metrics/lane3_mae
metrics/lane3_acc_valid_tol1
metrics/lane3_acc_valid_tol3
metrics/lane3_acc_valid_tol5
metrics/lane3_exist_acc
```

槽位对应关系：

```text
lane0 = lane_follow
lane1 = lead_lane
lane2 = channel_left
lane3 = channel_right
```

这些指标可以发现总体平均值掩盖的单槽位漏检、偏移或类别不平衡问题。

当前仍建议后续补充：

- 每槽位 False Positive / False Negative。
- 左右边界交换率。
- 按近端 / 远端 Row Anchor 分段的误差。
- 按遮挡与非遮挡样本分组的指标。

---

## 12. 正式训练前的数据 EDA

在启用 V3B 默认 `fliplr=0.5` 前，至少完成以下检查：

1. 每个槽位的图片级出现率。
2. 每个槽位的有效 Anchor 数量和 `x` 分布。
3. `lead_lane` 在水平翻转后是否仍保持同一语义。
4. 远端 Row Anchors 的有效点密度。
5. 标签叠加到原图后的可视化抽检。
6. Train / Valid 是否存在同帧、近重复帧或时间序列泄漏。
7. 空标签、坏图、NaN、非法行和重复 `lane_id`。

若某槽位的 no-lane 比例极高，应根据统计结果再决定是否需要调整 no-lane bias、采样策略或 loss 权重，不应直接凭经验修改。

---

## 13. 数据增强注意事项

通道左右边界具有固定语义，几何增强必须同步处理：

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

当前 YAML 已配置：

```yaml
flip_lane_pairs:
  - [2, 3]
```

但 `lane_follow` 和 `lead_lane` 是否允许保持原槽位，必须由真实任务语义和 EDA 结果确认。

Mosaic、MixUp、CutMix、Copy-Paste 和随机擦除会破坏连续车道结构，V3B 训练脚本保持关闭。

---

## 14. 遮挡、断点与绘制

训练标签可以通过 `x=-1` 表达真实断点，解码时 no-lane 概率超过阈值的点也会恢复为 `-1`。

绘制和控制层必须避免把遮挡前后的有效点强制连接。可采用：

- 仅绘制点。
- 按连续有效 Anchor 分段绘制折线。
- 相邻有效 Anchor 的索引差或像素距离超过阈值时强制断开。

控制层不应把跨越大段无效 Anchor 的点直接拟合为一条连续曲线。

---

## 15. V2 ONNX 与 RDK X5 部署基线

以下内容仍对应 **V2 320×320 模型**，不代表 V3B 已完成部署验证。

现有 V2 ONNX：

```text
input  images      [1, 3, 320, 320] float32 NCHW
output lane_output [1, 322, 56, 4] float32
opset  11
```

现有 RDK X5 量化环境：

```text
OpenExplorer
hb_mapper 1.24.3
hbdk 3.49.15
march = bayes-e
Runtime input = NV12
```

V2 已生成 Runtime BIN，但分类层：

```text
/model/model.16/cls_fc2/Gemm
```

一次性输出：

```text
321 × 56 × 4 = 71904
```

超过当前 BPU 相关维度限制 `65536`，因此分类 Head 回退到 CPU float，形成 BPU + CPU 混合执行。

V3B 已将分类器拆为四个 lane-specific Linear，每个分类器单次输出：

```text
321 × 56 = 17976
```

从结构上规避了 V2 单个 `71904` 大 Gemm，但能否完整进入 RDK X5 BPU 仍需重新导出 ONNX、量化并查看工具链节点分配，README 不预先宣称全 BPU。

---

## 16. V3B 导出与部署待办

在 V3B 正式部署前，需要依次完成：

1. 使用固定输入 `256×448` 完成 PyTorch 前向和验证。
2. 更新或确认 `export_onnx_xhm.py` 支持非方形输入。
3. 比较 PyTorch 与 ONNX Runtime 的 `cls`、`offset` 和最终解码坐标。
4. 确认 ONNX 输入输出 shape。
5. 用与训练一致的 direct resize / RGB / `/255` 预处理重新生成校准数据。
6. 在 OpenExplorer 中检查四个分类 Gemm、offset Gemm、Row Sampling 和 Softmax 的节点分配。
7. 完成板端 C++ Softmax、Top-K soft-argmax、offset 和坐标恢复。
8. 对比 V2、V3B FP32、V3B ONNX 与 V3B INT8 的逐槽位误差。
9. 测试端到端 FPS、CPU 占用、BPU 占用和数据搬运开销。

训练、导出、量化校准和板端预处理必须统一为：

```text
256×448
Direct Resize（当前 lane-robot.yaml 中 letterbox=false）
RGB
float32 / 255
```

---

## 17. 已知风险

1. V3B 代码已接入，但分支中尚无正式训练结果，不能仅凭前向 shape 判断精度。
2. `fliplr=0.5` 依赖槽位语义正确交换，尤其要验证 `lead_lane` 的翻转语义。
3. V2 权重迁移必须检查实际 transferred 参数，避免误加载或漏加载。
4. V3B 输入从 V2 的 `320×320` 改为 `256×448`，旧 ONNX、量化校准数据和板端坐标恢复不能直接复用。
5. 当前逐槽位指标仍未包含近端 / 远端分段、左右混淆和遮挡分组统计。
6. 断点信息可能在不正确的绘图或控制拟合中被重新连接。
7. `LANE_ROBOT_DATASETS` 只覆盖数据集根目录，模型、权重和输出路径仍由各自 CLI 参数控制。
8. V3B 小 Gemm 是否全部落在 BPU 取决于实际 ONNX 图和工具链约束，必须以量化日志和 `hb_perf` 为准。

---

## 18. 建议工作顺序

### P0：数据 EDA

- 逐槽位出现率与有效 Anchor 数。
- `lead_lane` 翻转语义检查。
- 远端密度和标签叠图。
- Train / Valid 泄漏检查。

### P1：V3B 训练冒烟测试

- 单 batch 前向、六项损失和 backward。
- 1 个 epoch Trainer / Validator。
- 检查逐槽位指标是否写入结果文件。

### P2：正式 V3B 基线

- 固定数据划分、随机种子、输入和增强。
- 保存完整配置、提交 SHA、日志和权重。
- 与 V2 使用相同验证集对比。

### P3：V3B ONNX

- 支持 `256×448` 静态输入。
- PyTorch / ONNX 数值等价验证。
- 推理可视化和 txt 输出复测。

### P4：RDK X5 量化

- 重建校准集。
- 检查四个分类 Linear 的 BPU 分配。
- 测试精度、FPS、CPU / BPU 占用。

### P5：控制闭环

```text
线检测
→ 选择目标线或计算通道中心
→ 横向误差与航向误差
→ 时序滤波与异常检测
→ 速度 / 转角控制器
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
