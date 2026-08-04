#!/usr/bin/env python3
"""Batch-infer an external image dataset with the split-head Lane Robot ONNX model.

Edit SOURCE_PATH to the absolute path of the external image directory.
The script recursively scans supported images and preserves the source directory
structure under OUTPUT_PATH.

Required ONNX outputs:
    cls_01 [B, 321, 56, 2]
    cls_23 [B, 321, 56, 2]
    offset [B,   1, 56, 4]
"""

from __future__ import annotations

import sys
from pathlib import Path

from infer_onnx_xhm import main


# Repository path on the local machine.
PROJECT_ROOT = Path("/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT")

# Split-head ONNX model exported by export_onnx_newhead.py.
MODEL_PATH = (
    PROJECT_ROOT
    / "runs/lane/lane_n_baseline-3/weights/best_newhead.onnx"
)

# Absolute path of the external dataset to infer.
# Change this path to the actual image directory or to one image file.
SOURCE_PATH = Path("/home/xhm/Desktop/external_lane_dataset/images")

# Absolute output directory. The source subdirectory structure is preserved.
OUTPUT_PATH = PROJECT_ROOT / "external_dataset_newhead_infer"


def has_cli_option(option: str) -> bool:
    """Return True for both '--name value' and '--name=value' forms."""
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in sys.argv[1:]
    )


def append_default_option(option: str, value: Path | str) -> None:
    """Append a default CLI option while still allowing an explicit override."""
    if not has_cli_option(option):
        sys.argv.extend((option, str(value)))


def run() -> None:
    if not SOURCE_PATH.is_absolute():
        raise ValueError(f"SOURCE_PATH 必须是绝对路径：{SOURCE_PATH}")
    if not OUTPUT_PATH.is_absolute():
        raise ValueError(f"OUTPUT_PATH 必须是绝对路径：{OUTPUT_PATH}")
    if not MODEL_PATH.is_absolute():
        raise ValueError(f"MODEL_PATH 必须是绝对路径：{MODEL_PATH}")

    append_default_option("--model", MODEL_PATH)
    append_default_option("--source", SOURCE_PATH)
    append_default_option("--output", OUTPUT_PATH)

    # Save prediction labels by default. Remove this block if only images are needed.
    if "--save-txt" not in sys.argv[1:]:
        sys.argv.append("--save-txt")

    # require_split_cls=True rejects old merged-output ONNX models.
    main(default_model=MODEL_PATH, require_split_cls=True)


if __name__ == "__main__":
    run()
