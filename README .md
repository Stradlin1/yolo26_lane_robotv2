# Lane Robot：固定语义多线检测

基于 Ultralytics YOLO26 和 Row-Anchor 表示改造的多线检测项目。项目由原始单线任务扩展为 **4 个固定语义槽位**，支持在单张图像中同时检测 0～4 条具有明确业务语义的曲线。

> 当前模型不是任意数量的实例曲线检测器。  
> 它采用固定槽位设计，每种语义在一张图像中最多出现一条曲线。


## 1. 项目来源与归属

本仓库是以下项目的派生开发版本：

| 项目 | 信息 |
|---|---|
| 原始项目 | [TarochLee/ULTRALYTICS_LANE_ROBOT](https://github.com/TarochLee/ULTRALYTICS_LANE_ROBOT) |
| 原始项目维护者 | [TarochLee](https://github.com/TarochLee) |
| 上游默认分支 | `master` |
| 当前派生仓库 | [Stradlin1/yolo26_lane_robotv2](https://github.com/Stradlin1/yolo26_lane_robotv2) |
| 当前版本维护者 | [Stradlin1](https://github.com/Stradlin1) |
| 开源许可证 | GNU Affero General Public License v3.0（AGPL-3.0） |

本项目保留原始项目的 Git 提交历史、许可证和相关版权信息。在原始 Lane Robot 单线检测实现基础上，当前派生版本主要完成了以下改造：

- 将单线输出扩展为 4 个固定语义槽位；
- 接通多线 Dataset、Trainer、Validator 和 Predictor；
- 增加多线解码、可视化与标签导出；
- 对齐训练和预测阶段的图像预处理；
- 增加固定形状导出协议，为 ONNX 和端侧部署提供基础；
- 增加项目 README、技术报告和数据检查流程。

本仓库中的新增实现、实验结论和文档由当前维护者负责，不代表原始项目维护者对本派生版本的功能、精度或部署结果作出认可或担保。



## 2. 当前任务定义

| `lane_id` | 名称 | 语义 |
|---:|---|---|
| 0 | `lane_follow` | 跟随线 |
| 1 | `lead_lane` | 引导线 |
| 2 | `channel_left` | 黄色通道左边界 |
| 3 | `channel_right` | 黄色通道右边界 |

模型支持：

- 每张图像存在 0～4 条线；
- 每条线可以局部缺失；
- 每个语义槽位最多一条曲线；
- 固定 56 个纵向 Row Anchor；
- 横向分类与亚网格偏移联合预测；
- 训练、验证、预测、可视化和标签保存；
- 固定形状输出，便于 ONNX 和端侧部署。

## 3. 已完成的改造

```text
数据 YAML
→ 多线标签读取
→ 四槽位模型构建
→ 分类与偏移输出
→ 六项联合损失
→ 反向传播
→ Trainer
→ Validator
→ 权重保存
→ Predictor
→ 预测可视化与标签保存
```

核心修改包括：

- 数据 YAML 统一控制 `x_grids`、`row_anchors` 和 `num_lanes`；
- Dataset 输出固定四槽位标签张量；
- `LaneRobotV2` 输出分类张量和亚网格偏移张量；
- Predictor 使用与训练一致的直接 resize；
- 预测结果可保存为与训练标签兼容的格式；
- 模型构建阶段检查数据配置与检测头结构是否一致。

## 4. 模型输出

当前默认配置：

```yaml
x_grids: 160
row_anchors: 56
num_lanes: 4
```

训练和 PyTorch 推理阶段输出：

```text
cls:    [B, 161, 56, 4]
offset: [B,   1, 56, 4]
```

维度含义：

- `B`：批大小；
- `161`：160 个横向网格类别，加 1 个 `no-lane` 类别；
- `56`：纵向 Row Anchor 数量；
- `4`：固定语义槽位数量。

导出模式下，分类和偏移会拼接为：

```text
output: [B, 162, 56, 4]
```

其中：

```text
output[:, :161, :, :]   → 分类输出
output[:, 161:162, :, :] → 偏移输出
```

ONNX 推理代码必须按此协议拆分输出。

## 5. 项目结构

```text
ULTRALYTICS_LANE_ROBOT/
├── ultralytics/
│   ├── cfg/
│   │   ├── datasets/lane-robot.yaml
│   │   └── models/26/yolo26n-lane.yaml
│   ├── models/yolo/lane/
│   │   ├── dataset.py
│   │   ├── train.py
│   │   ├── val.py
│   │   ├── predict.py
│   │   └── plotting.py
│   ├── nn/modules/head.py
│   └── utils/loss.py
├── datasets/
│   ├── images/
│   │   ├── train/
│   │   └── valid/
│   ├── labels/
│   │   ├── train/
│   │   └── valid/
│   └── labels_corrected/
│       ├── train/
│       └── valid/
├── scripts/
├── runs/
├── docs/
├── pyproject.toml
└── README.md
```

## 6. 环境安装

建议在独立 Python 环境中运行。仓库中的 Lane Robot 修改必须通过源码安装，不能只安装官方 PyPI 版本后直接运行。

### 5.1 创建环境

```bash
conda create -n lane_robot python=3.10 -y
conda activate lane_robot
```

### 5.2 安装 PyTorch

根据本机 CUDA 版本安装匹配的 PyTorch。确认 GPU 环境：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

### 5.3 安装当前仓库

```bash
cd /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT
pip install -e .
```

验证本地代码是否生效：

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
yolo checks
```

`ultralytics.__file__` 应指向当前项目目录，而不是其他 Python 环境中的官方安装目录。

## 7. 数据集准备

### 6.1 目录格式

```text
datasets/
├── images/
│   ├── train/
│   └── valid/
└── labels/
    ├── train/
    └── valid/
```

图片与标签必须同名：

```text
datasets/images/train/000001.jpg
datasets/labels/train/000001.txt
```

### 6.2 标签格式

每一行表示一个固定语义槽位：

```text
lane_id x1 y1 x2 y2 ... x56 y56
```

每行应包含：

```text
1 + 56 × 2 = 113 个数值
```

规则：

- `lane_id` 只能是 `0、1、2、3`；
- 同一个标签文件中，同一 `lane_id` 最多出现一次；
- `x` 为归一化横坐标，范围为 `[0, 1]`；
- `x=-1` 表示该 Row Anchor 没有有效曲线点；
- `y` 为归一化纵坐标；
- 整条线不存在时，可以不写对应行；
- 局部不可见时，保留该行，并将不可见位置写为 `x=-1`。

例如，图像中只有黄色通道左右边界：

```text
2 x1 y1 x2 y2 ... x56 y56
3 x1 y1 x2 y2 ... x56 y56
```

### 6.3 Row Anchor 顺序

当前使用 56 个纵向锚点，顺序为从图像底部向上：

```text
1.000000 → 0.333333
```

对应生成逻辑：

```python
np.linspace(y_end, y_start, row_anchors)
```

不要将其改成相反顺序，否则训练标签、预测解码和可视化会发生纵向错位。

## 8. 数据配置

配置文件：

```text
ultralytics/cfg/datasets/lane-robot.yaml
```

推荐内容：

```yaml
path: /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/datasets

train: images/train
val: images/valid

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
```

### 使用修正后的标签

人工修正脚本默认将结果保存到：

```text
datasets/labels_corrected/train
datasets/labels_corrected/valid
```

训练前必须确保模型读取的是修正标签。可以在数据 YAML 中显式添加：

```yaml
train_labels: labels_corrected/train
val_labels: labels_corrected/valid
```

或者在检查无误后，将修正标签同步到 `datasets/labels/`。

不要出现“修正标签已经保存，但训练仍读取旧标签”的情况。

## 9. 训练

### 8.1 推荐无几何增强基线

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
  hsv_h=0.0 \
  hsv_s=0.05 \
  hsv_v=0.10 \
  project=runs/lane \
  name=baseline_no_aug
```

显存不足时，优先降低：

```text
batch=8 → batch=4 → batch=2
```

训练结果默认保存到：

```text
runs/lane/baseline_no_aug/
├── weights/
│   ├── best.pt
│   └── last.pt
└── ...
```

### 8.2 恢复训练

```bash
yolo task=lane mode=train \
  model=runs/lane/baseline_no_aug/weights/last.pt \
  resume=True
```

## 10. 验证

```bash
yolo task=lane mode=val \
  model=runs/lane/baseline_no_aug/weights/best.pt \
  data=ultralytics/cfg/datasets/lane-robot.yaml \
  imgsz=320 \
  batch=8 \
  device=0
```

当前 Validator 提供整体指标：

| 指标 | 含义 |
|---|---|
| `lane_mae` | 有效点横向网格平均绝对误差 |
| `lane_mae_px` | 近似像素平均绝对误差 |
| `lane_acc_valid_tol1` | 误差不超过 1 个网格的比例 |
| `lane_acc_valid_tol3` | 误差不超过 3 个网格的比例 |
| `lane_acc_valid_tol5` | 误差不超过 5 个网格的比例 |
| `lane_exist_acc` | 所有槽位与锚点的存在性准确率 |

当前尚未完成四个语义槽位的独立指标统计。整体指标可能掩盖某个槽位完全没有学会的问题。

## 11. PyTorch 推理

```bash
yolo task=lane mode=predict \
  model=runs/lane/baseline_no_aug/weights/best.pt \
  source=datasets/images/valid \
  imgsz=320 \
  device=0 \
  save=True \
  save_txt=True \
  project=runs/lane \
  name=predict_valid \
  exist_ok=True
```

预测结果包括：

- 绘制多条曲线的图片；
- 活动语义槽位信息；
- 与训练标签格式兼容的预测 `.txt` 文件。

预测链路：

```text
模型原始输出
→ decode_lane()
→ [56, 4] 横向网格坐标
→ LaneResults
→ 绘制
→ 保存图片
→ 保存标签
```

## 12. 标签人工修正工具

推荐位置：

```text
scripts/manual_fix_56anchors_v8_relative_dataset.py
```

运行训练集：

```bash
python scripts/manual_fix_56anchors_v8_relative_dataset.py --split train
```

运行验证集：

```bash
python scripts/manual_fix_56anchors_v8_relative_dataset.py --split valid
```

工具默认数据流：

```text
datasets/images/<split>
datasets/labels/<split>
          ↓ 人工修正
datasets/labels_corrected/<split>
```

主要操作：

| 操作 | 按键 |
|---|---|
| 选择类别 | `0`～`3` |
| 添加或替换控制点 | 鼠标左键 |
| 删除最近控制点 | 鼠标右键 |
| 生成固定锚点曲线 | `Enter` |
| 撤销 | `Z` |
| 清空当前类别手动点 | `C` |
| 删除当前类别整条线 | `X` |
| 保存 | `S` |
| 下一张 | `D` |
| 上一张 | `A` |
| 重新加载 | `R` |
| 退出并保存 | `Q` |
| 退出且不保存当前修改 | `Esc` |

注意：当前模型只接受类别 `0～3`。标注工具如果仍允许类别 `4`，必须先将其限制为 `0～3`，否则类别 `4` 不会进入当前四槽位训练。

## 13. 数据增强约束

黄色通道左右边界具有固定左右语义，因此不能直接使用普通水平翻转。

未来若启用水平翻转，必须同步执行：

```text
x → 1 - x
lane_id 2 ↔ lane_id 3
channel_left ↔ channel_right
```

还需要单独确认：

```text
lane_follow
lead_lane
```

翻转后是否保持原语义。

在该逻辑实现前必须保持：

```yaml
fliplr: 0.0
```

以下增强也应在基线阶段关闭：

```yaml
flipud: 0.0
mosaic: 0.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0
degrees: 0.0
shear: 0.0
perspective: 0.0
```

## 14. 当前验证状态

已完成的工程验证：

- 真实数据集可以读取；
- Batch 标签形状正确；
- 四线模型前向输出形状正确；
- 六项损失可以正常计算；
- `loss.backward()` 可以产生有限梯度；
- Trainer 和 Validator 可以完成运行；
- `best.pt` 与 `last.pt` 可以正常保存；
- Predictor 可以保存预测图片和标签。

已验证的关键形状：

```text
batch img:    [2, 3, 256, 320]
batch lane:   [2, 56, 4]
batch lane_x: [2, 56, 4]
batch lane_y: [2, 56]

cls:          [2, 161, 56, 4]
offset:       [2,   1, 56, 4]
```

早期少量数据和单 epoch 测试仅证明工程链路可运行，不代表模型已经具备实际检测精度。

## 15. 当前限制

当前不支持：

- 同一图像中出现两条相同语义的曲线；
- 任意数量的未知语义曲线；
- 动态槽位分配；
- Hungarian Matching；
- 实例分割；
- BEV 处理；
- 逐语义槽位独立验证指标；
- 语义感知水平翻转；
- 完整 ONNX 数值一致性报告；
- INT8 或目标 NPU 量化精度验证；
- 实车闭环控制稳定性验证。

## 16. 已知风险

| 风险 | 影响 | 处理建议 |
|---|---|---|
| 四类数据分布不均衡 | 某些槽位无法学会 | 统计每类图像数、有效点数和缺失率 |
| 标签中出现 class 4 | 标签被当前 Dataset 跳过 | 将标注工具限制为 `0～3` |
| 修正标签未接入训练 | 模型继续读取旧标签 | 配置 `train_labels` 和 `val_labels` |
| 缺少逐槽位指标 | 整体指标掩盖单类失败 | 增加 per-lane metrics |
| 错误开启水平翻转 | 左右边界监督错误 | 保持 `fliplr=0` |
| 部署使用 LetterBox | 锚点坐标系统性偏移 | 与训练一致地直接 resize |
| ONNX 输出拆分错误 | 偏移通道被当成分类通道 | 按 `161 + 1` 通道拆分 |
| 数据量过少 | 冒烟测试结果无精度意义 | 使用完整数据训练正式基线 |

## 17. 后续计划

优先级建议：

1. 完成全量标签检查；
2. 将标注工具类别范围限制为 `0～3`；
3. 固化 `labels_corrected` 到训练链路；
4. 建立无几何增强正式基线；
5. 增加逐语义槽位验证指标；
6. 实现安全的语义感知数据增强；
7. 完成 PyTorch 与 ONNX 一致性测试；
8. 完成目标端侧平台精度和性能验证；
9. 根据曲线输出构建车辆控制接口。

## 18. 控制接口建议

模型输出稳定后，不建议仅使用单个 Row Anchor 直接控制车辆。控制模块应综合计算：

- 近场横向偏差；
- 曲线方向角；
- 曲率变化；
- 通道中心线；
- 左右边界有效状态；
- 缺线和异常线降级状态。

当左右通道边界同时有效时，可在共同有效的 Row Anchor 上计算：

```text
channel_center = (channel_left + channel_right) / 2
```

仅检测到单侧边界时，应结合通道宽度先验进行降级估计，不能直接把单侧边界当作车辆跟踪中心。

## 19. 关键文件

| 文件 | 作用 |
|---|---|
| `ultralytics/cfg/datasets/lane-robot.yaml` | 数据路径、锚点和语义槽位配置 |
| `ultralytics/cfg/default.yaml` | Lane 默认参数和损失权重 |
| `ultralytics/cfg/models/26/yolo26n-lane.yaml` | YOLO26n Lane 模型结构 |
| `ultralytics/models/yolo/lane/dataset.py` | 多线标签读取与预处理 |
| `ultralytics/models/yolo/lane/train.py` | 数据配置解析与训练器 |
| `ultralytics/models/yolo/lane/val.py` | Lane 验证器 |
| `ultralytics/models/yolo/lane/predict.py` | 多线预测与结果封装 |
| `ultralytics/models/yolo/lane/plotting.py` | 解码和可视化 |
| `ultralytics/nn/modules/head.py` | `LaneRobotV2` 检测头 |
| `ultralytics/utils/loss.py` | 六项联合损失 |
| `scripts/manual_fix_56anchors_v8_relative_dataset.py` | 56 锚点人工修正工具 |

更完整的设计说明和风险分析见：

```text
docs/lane_robot_multiline_technical_report.md
```

## 20. 许可证与致谢

原始仓库 `TarochLee/ULTRALYTICS_LANE_ROBOT` 的 `LICENSE` 文件采用 **GNU Affero General Public License v3.0（AGPL-3.0）**。本派生仓库保留该许可证，并应按照 AGPL-3.0 的要求使用、修改和分发。

使用本项目时应注意：

- 不要删除原始许可证、版权说明和 Git 提交历史；
- 对外分发修改版本时，应同时提供相应源代码和许可证；
- 将修改版本作为网络服务向用户提供时，应特别检查 AGPL-3.0 对源代码提供义务的要求；
- 商业使用或闭源集成前，应自行核对 Ultralytics 及其他第三方组件的许可条件。

完整法律条款以项目根目录中的 [`LICENSE`](LICENSE) 文件为准。本 README 仅提供工程层面的许可证提示，不构成法律意见。

感谢以下项目和维护者：

- [TarochLee/ULTRALYTICS_LANE_ROBOT](https://github.com/TarochLee/ULTRALYTICS_LANE_ROBOT)：原始 Lane Robot 项目；
- [TarochLee](https://github.com/TarochLee)：原始项目维护者；
- [Ultralytics](https://github.com/ultralytics/ultralytics)：底层 YOLO 训练与推理框架。
