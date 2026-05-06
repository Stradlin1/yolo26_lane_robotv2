# YOLO26 LaneRobot 单任务补丁

## 输出定义

默认输出：

```text
[B, 641, 56, 2]
```

通用形式：

```text
[B, lane_x_grids + 1, lane_row_anchors, lane_num_lanes]
```

- `lane_x_grids=640` 表示 x 方向 640 个分类位置。
- 额外的 `+1` 是 no-lane 类。
- `lane_row_anchors=56` 表示 y 方向 56 行。
- `lane_num_lanes` 是车道线数量，必须与你标注里的 lane_id 数量一致。

例如 `lane_num_lanes=4` 时，输出为：

```text
[B, 641, 56, 4]
```

## 标签格式

每张图对应一个 `.txt`，路径默认按 YOLO 规则映射：

```text
images/train/xxx.jpg -> labels/train/xxx.txt
```

每一行是一条车道线：

```text
lane_id x1 y1 x2 y2 ... x56 y56
```

你的样例格式已支持：

```text
0 -1.000000 0.670000 ... 0.557121 1.000000
```

规则：

- 第一个值是 `lane_id`。
- 后面是从上到下 56 行的 `x y`。
- `x=-1` 表示该行没有车道线点。
- `x` 和 `y` 均按 0 到 1 归一化。
- `lane_id` 范围必须是 `0 <= lane_id < lane_num_lanes`。

## 完全复现的 head

新增 `LaneRobot` head，结构与上面 YOLOP 的 `model.34` 一致：

```text
stride-32 deep feature
  -> Conv1x1 C -> 8
  -> Flatten 8*8*10 = 640
  -> Linear 640 -> 2048
  -> ReLU
  -> Linear 2048 -> (lane_x_grids + 1) * lane_row_anchors * lane_num_lanes
  -> Reshape [B, lane_x_grids + 1, lane_row_anchors, lane_num_lanes]
```

YOLO26 取 backbone 最后的 P5/32 深层特征，对应原 YOLOP 图里的 `/model.9/cv4/act/Div_output_0` 这类深层语义特征。

## 训练命令

```bash
yolo task=lane mode=train \
  model=ultralytics/cfg/models/26/yolo26-lane.yaml \
  data=ultralytics/cfg/datasets/lane-robot.yaml \
  imgsz=256,320 \
  epochs=100 batch=16 plots=True
```

修改车道线数量：

```bash
yolo task=lane mode=train \
  model=ultralytics/cfg/models/26/yolo26-lane.yaml \
  data=your_lane.yaml \
  lane_x_grids=640 lane_row_anchors=56 lane_num_lanes=4 \
  imgsz=256,320 plots=True
```

## Loss

实现的 loss：

```text
loss = lane_ce * CE
     + lane_loc * SmoothL1(expected_x, target_x)
     + lane_exist * BCE(existence)
     + lane_smooth * adjacent_row_smoothness
```

默认权重：

```yaml
lane_ce: 1.0
lane_loc: 0.25
lane_exist: 0.10
lane_smooth: 0.05
lane_label_smoothing: 0.0
```

`CE` 负责 x 分类和 no-lane 分类；`lane_loc` 让可见点的 soft-argmax x 更接近标签；`lane_exist` 强化有点和无点区分；`lane_smooth` 对相邻 y 行提供轻量连续性约束。

## runs 输出

`plots=True` 时，训练结束或 early stopping 后会在 runs 目录中输出：

```text
train_batch*_lane.jpg
val_batch*_labels.jpg
val_batch*_pred.jpg
lane_final_predictions.jpg
results.csv
weights/best.pt
weights/last.pt
```
