#!/usr/bin/env python3
"""LaneRobotV3B ONNX batch inference entrypoint.

This script reuses the tested image scanning, preprocessing, decoding,
visualization, and txt export implementation in infer_onnx_xhm.py, while
changing the defaults and shape checks to the zbn LaneRobotV3B contract.

Default paths follow the existing ONNX inference script:
    model:
        /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/
        runs/lane/lane_v3b/weights/best.onnx
    source:
        /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/test
    output:
        /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/test_infer_v3b

Expected model contract:
    images      float32 [1, 3, 256, 448]
    lane_output float32 [1, 322, 56, 4]
"""

from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

import infer_onnx_xhm as base  # noqa: E402


LOCAL_PROJECT_ROOT = Path("/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT")
V3B_INPUT_HW = (256, 448)
V3B_X_GRIDS = 320
V3B_ROW_ANCHORS = 56
V3B_NUM_LANES = 4

# Keep the same default source convention as infer_onnx_xhm.py.
base.PROJECT_ROOT = LOCAL_PROJECT_ROOT
base.DEFAULT_MODEL = LOCAL_PROJECT_ROOT / "runs/lane/lane_v3b/weights/best.onnx"
base.DEFAULT_SOURCE = LOCAL_PROJECT_ROOT / "test"
base.DEFAULT_OUTPUT = LOCAL_PROJECT_ROOT / "test_infer_v3b"
base.CURRENT_X_GRIDS = V3B_X_GRIDS


_original_inspect_output_layout = base.inspect_output_layout


def get_v3b_input_hw(session) -> tuple[int, int]:
    """Require the canonical static V3B input, or use it for dynamic H/W."""
    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise RuntimeError(f"V3B ONNX should have one input, found {len(inputs)}")

    shape = list(inputs[0].shape)
    if len(shape) != 4:
        raise RuntimeError(f"V3B input must be BCHW, got: {shape}")

    batch, channels, height, width = shape
    if isinstance(batch, int) and batch != 1:
        raise RuntimeError(
            "infer_onnx_v3b.py processes one image at a time; "
            f"the ONNX input batch is {batch}."
        )
    if isinstance(channels, int) and channels != 3:
        raise RuntimeError(f"V3B input must have 3 RGB channels, got: {shape}")

    if isinstance(height, int) and isinstance(width, int):
        input_hw = (int(height), int(width))
        if input_hw != V3B_INPUT_HW:
            raise RuntimeError(
                "This is not the canonical zbn LaneRobotV3B ONNX input.\n"
                f"Model input: {input_hw[0]}x{input_hw[1]}\n"
                f"Expected:    {V3B_INPUT_HW[0]}x{V3B_INPUT_HW[1]}\n"
                "Export the checkpoint with export_onnx_v3b.py."
            )
        return input_hw

    return V3B_INPUT_HW


def inspect_v3b_output_layout(output_infos, expected_x_grids):
    """Reuse the base layout parser, then enforce 56 rows and 4 lanes."""
    x_grids, layout = _original_inspect_output_layout(
        output_infos,
        expected_x_grids,
    )
    shapes = [tuple(info.shape) for info in output_infos]

    lane_shapes = [
        shape
        for shape in shapes
        if len(shape) == 4
        and isinstance(shape[1], int)
        and shape[1] > 1
    ]
    if len(lane_shapes) != 1:
        raise RuntimeError(
            "Could not identify exactly one V3B classification/merged output: "
            f"{shapes}"
        )

    lane_shape = lane_shapes[0]
    rows = lane_shape[2]
    lanes = lane_shape[3]
    if rows != V3B_ROW_ANCHORS or lanes != V3B_NUM_LANES:
        raise RuntimeError(
            "This is not the standard zbn LaneRobotV3B output.\n"
            f"Model outputs: {shapes}\n"
            f"Expected rows={V3B_ROW_ANCHORS}, lanes={V3B_NUM_LANES}."
        )

    return x_grids, layout


base.get_input_hw = get_v3b_input_hw
base.inspect_output_layout = inspect_v3b_output_layout


if __name__ == "__main__":
    base.main()
