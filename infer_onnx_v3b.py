#!/usr/bin/env python3
"""Run batch inference with the zbn LaneRobotV3B ONNX model.

This entrypoint reuses the image scanning, decoding, visualization, and txt
export implementation in ``infer_onnx_xhm.py`` while enforcing the ONNX
contract produced by ``export_onnx_v3b.py``.

Standard contract
-----------------
Input:
    images       float32 [1, 3, 256, 448]
Output:
    lane_output  float32 [1, 322, 56, 4]

Output channel layout:
    0..320: classification logits; channel 320 is the no-lane class
    321:    sub-grid offset in [-0.5, 0.5]

Preprocessing defaults to direct RGB resize and division by 255. Use
``--letterbox`` only when the checkpoint was trained with the same top-padded,
bottom-aligned letterbox policy.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import infer_onnx_xhm as base  # noqa: E402


V3B_INPUT_NAME = "images"
V3B_OUTPUT_NAME = "lane_output"
V3B_INPUT_HW = (256, 448)
V3B_X_GRIDS = 320
V3B_ROW_ANCHORS = 56
V3B_NUM_LANES = 4
V3B_OUTPUT_CHANNELS = V3B_X_GRIDS + 2
V3B_OUTPUT_SHAPE = (
    1,
    V3B_OUTPUT_CHANNELS,
    V3B_ROW_ANCHORS,
    V3B_NUM_LANES,
)

# Keep all defaults relative to the checked-out repository instead of relying
# on one developer machine's absolute path.
base.PROJECT_ROOT = PROJECT_ROOT
base.DEFAULT_MODEL = PROJECT_ROOT / "runs/lane/lane_v3b/weights/best.onnx"
base.DEFAULT_SOURCE = PROJECT_ROOT / "test"
base.DEFAULT_OUTPUT = PROJECT_ROOT / "test_infer_v3b"
base.CURRENT_X_GRIDS = V3B_X_GRIDS


def _is_dynamic_dimension(value: Any) -> bool:
    return not isinstance(value, int)


def get_v3b_input_hw(session: Any) -> tuple[int, int]:
    """Validate the V3B ONNX input and return its static spatial size."""
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(
            f"LaneRobotV3B ONNX must expose one input, found {len(inputs)}"
        )

    info = inputs[0]
    if info.name != V3B_INPUT_NAME:
        raise RuntimeError(
            f"Unexpected ONNX input name {info.name!r}; expected {V3B_INPUT_NAME!r}. "
            "Re-export with export_onnx_v3b.py."
        )
    if info.type != "tensor(float)":
        raise RuntimeError(
            f"V3B input must be float32, got ONNX Runtime type {info.type!r}"
        )

    shape = list(info.shape)
    if len(shape) != 4:
        raise RuntimeError(f"V3B input must be BCHW, got {shape}")

    batch, channels, height, width = shape
    if isinstance(batch, int) and batch != 1:
        raise RuntimeError(
            "This script processes one image at a time, but the model has "
            f"static batch={batch}. Export with --batch 1 or --dynamic-batch."
        )
    if _is_dynamic_dimension(channels) or int(channels) != 3:
        raise RuntimeError(f"V3B input channel dimension must be static 3, got {shape}")
    if _is_dynamic_dimension(height) or _is_dynamic_dimension(width):
        raise RuntimeError(
            "V3B spatial dimensions must be static. Re-export without dynamic "
            "height/width; only --dynamic-batch is supported."
        )

    input_hw = (int(height), int(width))
    if input_hw != V3B_INPUT_HW:
        raise RuntimeError(
            "This is not the canonical zbn LaneRobotV3B input size.\n"
            f"Model input: {input_hw[0]}x{input_hw[1]}\n"
            f"Expected:    {V3B_INPUT_HW[0]}x{V3B_INPUT_HW[1]}\n"
            "Re-export the checkpoint with export_onnx_v3b.py."
        )

    return input_hw


def inspect_v3b_output_layout(
    output_infos: list[Any],
    expected_x_grids: int | None,
) -> tuple[int, str]:
    """Enforce the single merged output produced by export_onnx_v3b.py."""
    if len(output_infos) != 1:
        shapes = [tuple(info.shape) for info in output_infos]
        raise RuntimeError(
            "LaneRobotV3B ONNX must expose one merged lane_output tensor; "
            f"found {len(output_infos)} outputs with shapes {shapes}"
        )

    info = output_infos[0]
    if info.name != V3B_OUTPUT_NAME:
        raise RuntimeError(
            f"Unexpected ONNX output name {info.name!r}; expected "
            f"{V3B_OUTPUT_NAME!r}. Re-export with export_onnx_v3b.py."
        )
    if info.type != "tensor(float)":
        raise RuntimeError(
            f"V3B output must be float32, got ONNX Runtime type {info.type!r}"
        )

    shape = list(info.shape)
    if len(shape) != 4:
        raise RuntimeError(f"lane_output must be rank 4, got {shape}")

    batch, channels, rows, lanes = shape
    if isinstance(batch, int) and batch != 1:
        raise RuntimeError(
            f"lane_output static batch must be 1, got shape {shape}"
        )

    expected_tail = (
        V3B_OUTPUT_CHANNELS,
        V3B_ROW_ANCHORS,
        V3B_NUM_LANES,
    )
    actual_tail = (channels, rows, lanes)
    if any(_is_dynamic_dimension(value) for value in actual_tail):
        raise RuntimeError(
            "V3B output channel, row, and lane dimensions must be static, "
            f"got {shape}"
        )
    if tuple(map(int, actual_tail)) != expected_tail:
        raise RuntimeError(
            "This is not the standard zbn LaneRobotV3B output.\n"
            f"Model output: {shape}\n"
            f"Expected:     [B, {expected_tail[0]}, {expected_tail[1]}, "
            f"{expected_tail[2]}]"
        )

    if expected_x_grids is not None and int(expected_x_grids) != V3B_X_GRIDS:
        raise RuntimeError(
            f"infer_onnx_v3b.py requires x_grids={V3B_X_GRIDS}, "
            f"but --expected-x-grids={expected_x_grids} was supplied"
        )

    return V3B_X_GRIDS, "merged cls+offset (LaneRobotV3B)"


def split_v3b_output(
    outputs: list[np.ndarray],
    expected_x_grids: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Split lane_output into cls logits and offset with strict shape checks."""
    if len(outputs) != 1:
        raise RuntimeError(
            f"Expected one lane_output array, received {len(outputs)} arrays"
        )

    output = np.asarray(outputs[0])
    if output.shape != V3B_OUTPUT_SHAPE:
        raise RuntimeError(
            f"Unexpected runtime lane_output shape {output.shape}; "
            f"expected {V3B_OUTPUT_SHAPE}"
        )
    if output.dtype != np.float32:
        raise RuntimeError(
            f"lane_output must be float32, got dtype {output.dtype}"
        )
    if not np.isfinite(output).all():
        raise RuntimeError("lane_output contains NaN or Inf")

    if expected_x_grids is not None and int(expected_x_grids) != V3B_X_GRIDS:
        raise RuntimeError(
            f"Expected x_grids={V3B_X_GRIDS}, got {expected_x_grids}"
        )

    cls_logits = output[:, : V3B_X_GRIDS + 1, :, :]
    offset = output[:, V3B_X_GRIDS + 1 :, :, :]
    return cls_logits, offset


# base.main resolves these functions through module globals at runtime, so the
# strict V3B validators replace the generic V2-compatible implementations.
base.get_input_hw = get_v3b_input_hw
base.inspect_output_layout = inspect_v3b_output_layout
base.split_outputs = split_v3b_output


if __name__ == "__main__":
    base.main()
