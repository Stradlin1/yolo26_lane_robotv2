#!/usr/bin/env python3
"""Export a LaneRobotV3B checkpoint to ONNX opset 11.

Default contract:
    input:
        images      float32 [1, 3, 256, 448]
    output:
        lane_output float32 [1, 322, 56, 4]

Output channels:
    0..320: classification logits (320 is the no-lane class)
    321:    sub-grid offset in [-0.5, 0.5]

The canonical V3B input is 256x448. It produces the expected 16x28 fused
feature map before LaneRobotV3B. This exporter checks that shape explicitly so
the head does not silently enter its adaptive-pooling fallback during export.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


OPSET = 11
DEFAULT_IMGSZ = (256, 448)
EXPECTED_X_GRIDS = 320
EXPECTED_ROW_ANCHORS = 56
EXPECTED_NUM_LANES = 4
EXPECTED_FEAT_HW = (16, 28)


class LaneV3BOnnxWrapper(nn.Module):
    """Normalize the custom lane model output to one ONNX tensor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def _from_dict(output: dict) -> torch.Tensor:
        if "cls" not in output or "offset" not in output:
            raise KeyError("Lane output dict must contain 'cls' and 'offset'.")
        return torch.cat((output["cls"], output["offset"]), dim=1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images)

        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            return self._from_dict(output)
        if isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, torch.Tensor) and item.ndim == 4:
                    return item
                if isinstance(item, dict) and "cls" in item and "offset" in item:
                    return self._from_dict(item)

        raise TypeError(
            "Unsupported LaneRobotV3B model output during ONNX export: "
            f"{type(output).__name__}"
        )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Export LaneRobotV3B to ONNX with fixed opset 11."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=project_root / "runs/lane/lane_v3b/weights/best.pt",
        help="V3B checkpoint path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ONNX path. Default: beside the checkpoint with .onnx suffix.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=DEFAULT_IMGSZ,
        help="Static input height and width. Default: 256 448.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Static export batch size. Default: 1.",
    )
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Make only the batch dimension dynamic.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Device used for tracing. CPU is the safest default.",
    )
    parser.add_argument(
        "--verify-runtime",
        action="store_true",
        help="Compare PyTorch and ONNX Runtime outputs after export.",
    )
    parser.add_argument(
        "--allow-nonstandard-head",
        action="store_true",
        help="Allow head dimensions other than the zbn V3B defaults.",
    )
    return parser.parse_args()


def find_v3b_head(model: nn.Module) -> nn.Module:
    from ultralytics.nn.modules.head import LaneRobotV3B

    heads = [module for module in model.modules() if isinstance(module, LaneRobotV3B)]
    if len(heads) != 1:
        names = [type(module).__name__ for module in heads]
        raise RuntimeError(
            "Expected exactly one LaneRobotV3B head, "
            f"but found {len(heads)}: {names}"
        )
    head = heads[0]
    head.export = True
    return head


def tensor_shape_text(value) -> str:
    dims = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            dims.append(str(dim.dim_value))
        else:
            dims.append("?")
    return "[" + ", ".join(dims) + "]"


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root))

    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {weights}")

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else weights.with_suffix(".onnx")
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    if args.batch < 1:
        raise ValueError("--batch must be at least 1")

    height, width = map(int, args.imgsz)
    if height <= 0 or width <= 0:
        raise ValueError("--imgsz values must be positive")
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(
            f"V3B export size must be divisible by 32, got {height}x{width}"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "The 'onnx' package is required. Install it with:\n"
            "  python -m pip install 'onnx>=1.12,<2'"
        ) from exc

    from ultralytics import YOLO

    print("=" * 72)
    print("LaneRobotV3B ONNX export")
    print(f"weights       : {weights}")
    print(f"output        : {output}")
    print(f"opset         : {OPSET}")
    print(f"input size    : {height} x {width}")
    print(f"batch         : {args.batch}")
    print(f"dynamic batch : {args.dynamic_batch}")
    print(f"device        : {args.device}")
    print("=" * 72)

    yolo = YOLO(str(weights), task="lane")
    torch_model = yolo.model.to(args.device).float().eval()
    for parameter in torch_model.parameters():
        parameter.requires_grad_(False)

    head = find_v3b_head(torch_model)
    x_grids = int(head.x_grids)
    row_anchors = int(head.row_anchors)
    num_lanes = int(head.num_lanes)
    feat_hw = (int(head.feat_h), int(head.feat_w))

    actual_head = (x_grids, row_anchors, num_lanes, *feat_hw)
    expected_head = (
        EXPECTED_X_GRIDS,
        EXPECTED_ROW_ANCHORS,
        EXPECTED_NUM_LANES,
        *EXPECTED_FEAT_HW,
    )
    if actual_head != expected_head and not args.allow_nonstandard_head:
        raise RuntimeError(
            "Checkpoint is not the standard zbn LaneRobotV3B configuration.\n"
            f"Checkpoint: x_grids={x_grids}, row_anchors={row_anchors}, "
            f"num_lanes={num_lanes}, feat_h={feat_hw[0]}, feat_w={feat_hw[1]}\n"
            f"Expected:   x_grids={EXPECTED_X_GRIDS}, "
            f"row_anchors={EXPECTED_ROW_ANCHORS}, "
            f"num_lanes={EXPECTED_NUM_LANES}, "
            f"feat_h={EXPECTED_FEAT_HW[0]}, feat_w={EXPECTED_FEAT_HW[1]}\n"
            "Use --allow-nonstandard-head only for an intentional custom model."
        )

    wrapper = LaneV3BOnnxWrapper(torch_model).to(args.device).float().eval()

    torch.manual_seed(0)
    dummy = torch.randn(
        args.batch,
        3,
        height,
        width,
        dtype=torch.float32,
        device=args.device,
    )

    captured = {}

    def capture_head_input(_module, inputs) -> None:
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise RuntimeError("LaneRobotV3B did not receive a tensor input")
        captured["shape"] = tuple(inputs[0].shape)

    hook = head.register_forward_pre_hook(capture_head_input)
    try:
        with torch.inference_mode():
            torch_output = wrapper(dummy)
    finally:
        hook.remove()

    head_input_shape = captured.get("shape")
    if head_input_shape is None:
        raise RuntimeError("Could not capture the LaneRobotV3B input feature shape")
    if tuple(head_input_shape[-2:]) != feat_hw:
        raise RuntimeError(
            "Input size would trigger LaneRobotV3B adaptive-pooling fallback, "
            "which is intentionally forbidden for this opset-11 export.\n"
            f"Head input feature: {head_input_shape}\n"
            f"Head expects:        [B, C, {feat_hw[0]}, {feat_hw[1]}]\n"
            "Use the canonical --imgsz 256 448 for the zbn model."
        )

    expected_shape = (args.batch, x_grids + 2, row_anchors, num_lanes)
    actual_shape = tuple(torch_output.shape)
    if actual_shape != expected_shape:
        raise RuntimeError(
            "Unexpected model output shape before export.\n"
            f"Expected: {expected_shape}\n"
            f"Actual:   {actual_shape}"
        )
    if not torch.isfinite(torch_output).all():
        raise RuntimeError("PyTorch output contains NaN or Inf before export")

    print(f"Lane head     : {type(head).__name__}")
    print(f"head feature : {head_input_shape}")
    print(f"x_grids      : {x_grids}")
    print(f"row_anchors  : {row_anchors}")
    print(f"num_lanes    : {num_lanes}")
    print(f"output shape : {actual_shape}")

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch"},
            "lane_output": {0: "batch"},
        }

    export_kwargs = dict(
        model=wrapper,
        args=dummy,
        f=str(output),
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["lane_output"],
        dynamic_axes=dynamic_axes,
    )

    try:
        torch.onnx.export(**export_kwargs, dynamo=False)
    except TypeError:
        torch.onnx.export(**export_kwargs)

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)

    print("\nONNX checker  : PASS")
    print(f"IR version    : {onnx_model.ir_version}")
    print(f"graph nodes   : {len(onnx_model.graph.node)}")
    print("ONNX inputs:")
    for value in onnx_model.graph.input:
        print(f"  {value.name}: {tensor_shape_text(value)}")
    print("ONNX outputs:")
    for value in onnx_model.graph.output:
        print(f"  {value.name}: {tensor_shape_text(value)}")

    default_opsets = [
        item.version
        for item in onnx_model.opset_import
        if item.domain in ("", "ai.onnx")
    ]
    if OPSET not in default_opsets:
        raise RuntimeError(
            f"Exported model does not declare ONNX opset {OPSET}; "
            f"found {default_opsets}"
        )

    if args.verify_runtime:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "--verify-runtime requires onnxruntime. Install it with:\n"
                "  python -m pip install onnxruntime"
            ) from exc

        session = ort.InferenceSession(
            str(output),
            providers=["CPUExecutionProvider"],
        )
        ort_output = session.run(
            ["lane_output"],
            {"images": dummy.detach().cpu().numpy()},
        )[0]
        reference = torch_output.detach().cpu().numpy()
        max_abs_error = float(np.max(np.abs(reference - ort_output)))
        mean_abs_error = float(np.mean(np.abs(reference - ort_output)))

        if ort_output.shape != reference.shape:
            raise RuntimeError(
                f"ONNX Runtime shape mismatch: {ort_output.shape} vs {reference.shape}"
            )
        if not np.isfinite(ort_output).all():
            raise RuntimeError("ONNX Runtime output contains NaN or Inf")
        if not np.allclose(reference, ort_output, rtol=1e-3, atol=1e-4):
            raise RuntimeError(
                "ONNX Runtime output differs from PyTorch output.\n"
                f"max_abs_error={max_abs_error:.8g}, "
                f"mean_abs_error={mean_abs_error:.8g}"
            )

        print("\nONNX Runtime  : PASS")
        print(f"max abs error : {max_abs_error:.8g}")
        print(f"mean abs error: {mean_abs_error:.8g}")

    size_mb = output.stat().st_size / (1024 * 1024)
    print("\nExport complete")
    print(f"file          : {output}")
    print(f"size          : {size_mb:.2f} MB")
    print(f"opset         : {OPSET}")
    print(
        f"output layout : cls [0:{x_grids + 1}], "
        f"offset [{x_grids + 1}:{x_grids + 2}]"
    )


if __name__ == "__main__":
    main()
