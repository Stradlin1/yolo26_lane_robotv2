#!/usr/bin/env python3
"""Run Lane Robot ONNX inference for the RDK X5 split-head output contract.

Required outputs:
    cls_01 [B, 321, 56, 2]
    cls_23 [B, 321, 56, 2]
    offset [B,   1, 56, 4]

The shared inference implementation concatenates cls_01 and cls_23 on axis 3
before applying the unchanged Softmax, no-lane test, Top-K soft-argmax, offset,
optional smoothing, drawing, and label serialization steps.
"""

from pathlib import Path

from infer_onnx_xhm import PROJECT_ROOT, main


DEFAULT_MODEL = Path(PROJECT_ROOT) / "runs/lane/lane_n_baseline-3/weights/best_newhead.onnx"


if __name__ == "__main__":
    main(default_model=DEFAULT_MODEL, require_split_cls=True)
