# 160-column independent four-task lane model

This variant preserves the original single-task `LaneRobotV2` computation and repeats the complete prediction head four times after the shared YOLO backbone and P4/P5 fusion.

## Model contract

- `x_grids = 160`
- classification states per row: `161` (`0..159` positions plus `160` no-lane)
- `row_anchors = 56`
- `num_lanes = 4`
- PyTorch / default ONNX outputs:
  - `cls: [B, 161, 56, 4]`
  - `offset: [B, 1, 56, 4]`
- Head `export=True` concatenated output: `[B, 162, 56, 4]`

The final dimension contains four independent fixed tasks. Softmax is only applied along the 161-position dimension.

## Files added

- `ultralytics/nn/modules/head.py`
  - `SingleLaneRobotV2Branch`
  - `LaneRobotV2Independent`
- `ultralytics/cfg/models/26/yolo26*-lane-independent.yaml`
- `ultralytics/cfg/datasets/lane-robot-4tasks.yaml`
- `init_independent_lane_from_single_models.py`
- `verify_independent_lane.py`

The original single-task `LaneRobotV2` and `yolo26*-lane.yaml` files remain available.

## Labels

Each image label file contains one line per present task:

```text
lane_id x1 y1 x2 y2 ... x56 y56
```

`lane_id` is `0..3`. A missing task line means the whole task is no-lane. A missing point inside an existing task uses `x=-1`.

## Training

The default configuration now points to:

```text
model: yolo26m-lane-independent.yaml
data: ultralytics/cfg/datasets/lane-robot-4tasks.yaml
```

Edit the dataset path and task names before training, then run:

```bash
python train.py
```

The loss reproduces the original single-task loss independently for each branch and combines the four task totals using normalized `lane_task_weights`.

## Initialize from four successful single-task checkpoints

```bash
python init_independent_lane_from_single_models.py \
  --model ultralytics/cfg/models/26/yolo26n-lane-independent.yaml \
  --base task0_best.pt \
  --task-weights task0_best.pt task1_best.pt task2_best.pt task3_best.pt \
  --output independent_4task_init.pt
```

The base checkpoint supplies the shared backbone, neck and fusion weights. Each single-task checkpoint supplies its own `conv_1x1`, `cls_fc1`, `cls_fc2` and `offset_fc` parameters to the matching branch.

Because the four source backbones may differ, initialize from the most generally reliable source model and jointly fine-tune the combined model. A practical schedule is:

1. Freeze most of the backbone for several epochs and train the four branches plus late fusion layers.
2. Unfreeze the backbone and fine-tune using a lower backbone learning rate.

## Verification

```bash
PYTHONPATH=. python verify_independent_lane.py
```

The script checks exact numerical equivalence to the old single-task head, output shapes, distinct parameters, gradient isolation, joint loss, four-task labels and checkpoint migration. ONNX export is tested when the `onnx` package is installed.
