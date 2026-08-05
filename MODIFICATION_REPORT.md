# Independent four-task lane modification report

## Scope

The project was modified from the uploaded 160-column single-task baseline. The original single-task `LaneRobotV2` implementation and original model YAML files remain unchanged and usable.

The new architecture shares the YOLO backbone, neck, and P4/P5 fusion, then runs four complete independent copies of the proven single-task prediction head:

```text
shared backbone / neck / P4+P5 fusion
  ├─ independent branch 0: Conv1x1 → Pool → FC1 → cls/offset
  ├─ independent branch 1: Conv1x1 → Pool → FC1 → cls/offset
  ├─ independent branch 2: Conv1x1 → Pool → FC1 → cls/offset
  └─ independent branch 3: Conv1x1 → Pool → FC1 → cls/offset
```

## Frozen tensor protocol

```text
x_grids:       160
no-lane index: 160
row_anchors:   56
num_tasks:     4
cls:           [B, 161, 56, 4]
offset:        [B,   1, 56, 4]
export concat: [B, 162, 56, 4]
```

No softmax is applied across the final four-task dimension.

## Main code changes

- Added `SingleLaneRobotV2Branch` and `LaneRobotV2Independent` in `ultralytics/nn/modules/head.py`.
- Registered the new head in module exports, model parsing, task guessing, and validator head discovery.
- Refactored `LaneRobotLoss` to execute the original loss independently for each task, then combine normalized task totals.
- Added configurable `lane_task_weights` while preserving the existing six aggregate loss items.
- Added four-task model and dataset YAML files without deleting the original single-task files.
- Added checkpoint migration from four successful single-task models.
- Updated ONNX wrapper and inference scripts for four outputs and separate visualization colors.

## Verification results

```text
Gate 1: PASSED — a copied independent branch is numerically identical to LaneRobotV2(num_lanes=1)
Gate 2: PASSED — cls [1,161,56,4], offset [1,1,56,4]
Gate 3: PASSED — branch modules and Parameter storage are distinct
Gate 4: PASSED — task-0 loss gives non-zero gradient only to branch 0; shared backbone receives gradient
Gate 5: PASSED — joint per-task loss is finite and all four branches receive gradients
Gate 6: PASSED — four lane_id rows load into lane/lane_x [56,4]
Gate 7: PASSED — four synthetic single-task checkpoints map exactly to branches 0..3
Gate 8: SKIPPED — ONNX Python packages are not installed in the execution container
```

Additional checks:

```text
YOLO wrapper model construction: PASSED
Ultralytics checkpoint save/reload: PASSED
Python compileall: PASSED
export=True concatenated shape [1,162,56,4]: PASSED
single-task refactored loss parity: PASSED
```

## Parameter count

For the `n` scale used by the verification script:

```text
Original single model:      6,521,176
Original single lane head:  4,983,160
Independent four model:    21,470,656
Independent four head:     19,932,640
```

The four-task head intentionally contains four complete copies of the successful single-task head. This favors feature isolation and expected accuracy over minimal parameter count.

## Remaining validation

Real-data training and accuracy comparison could not be run because the uploaded archive does not include the four-task dataset or trained single-task checkpoints. ONNX export should be rerun in the deployment environment after installing `onnx` and `onnxruntime`.
