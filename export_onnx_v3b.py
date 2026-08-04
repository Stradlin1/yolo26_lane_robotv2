#!/usr/bin/env python3
"""Export the zbn LaneRobotV3B model to ONNX opset 11.

Default contract
----------------
Input:
    images       float32 [1, 3, 256, 448]
Output:
    lane_output  float32 [1, 322, 56, 4]

Output channel layout for x_grids=320:
    lane_output[:, 0:321, :, :]   classification logits
        0..319: horizontal grid classes
        320:    no-lane class
    lane_output[:, 321:322, :, :] sub-grid offset in [-0.5, 0.5]

The canonical 256x448 input produces a 16x28 feature map at LaneRobotV3B.
This script rejects an input size that would trigger the head's adaptive-pool
fallback, because fixed-shape export is safer for ONNX opset 11 and RDK X5.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn


OPSET = 11
DEFAULT_IMGSZ = (256, 448)
STANDARD_HEAD = {
    "x_grids": 320,
    "row_anchors": 56,
    "num_lanes": 4,
    "feat_h": 16,
    "feat_w": 28,
}


class LaneV3BOnnxWrapper(nn.Module):
    """Normalize Ultralytics custom lane outputs to one ONNX tensor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    @staticmethod
    def _merge_dict(output: dict[str, torch.Tensor]) -> torch.Tensor:
        try:
            cls = output["cls"]
            offset = output["offset"]
        except KeyError as exc:
            raise KeyError("Lane output dict must contain 'cls' and 'offset'.") from exc

        if cls.ndim != 4 or offset.ndim != 4:
            raise RuntimeError(
                f"Unexpected lane tensor ranks: cls={tuple(cls.shape)}, "
                f"offset={tuple(offset.shape)}"
            )
        return torch.cat((cls, offset), dim=1)

    def _normalize(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, dict):
            return self._merge_dict(output)

        if isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, dict) and "cls" in item and "offset" in item:
                    return self._merge_dict(item)
                if isinstance(item, torch.Tensor) and item.ndim == 4:
                    return item

        raise TypeError(
            "Unsupported LaneRobotV3B output type during export: "
            f"{type(output).__name__}"
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self._normalize(self.model(images))


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Export the zbn LaneRobotV3B checkpoint to ONNX opset 11."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=root / "runs/lane/lane_v3b/weights/best.pt",
        help="Path to the trained V3B checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ONNX path. Default: checkpoint path with .onnx suffix.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=DEFAULT_IMGSZ,
        help="Static input size. Standard zbn V3B size: 256 448.",
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Make only dimension 0 dynamic. Spatial dimensions stay fixed.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Tracing device. CPU is the safest default.",
    )
    parser.add_argument(
        "--verify-runtime",
        action="store_true",
        help="Compare PyTorch and ONNX Runtime outputs after export.",
    )
    parser.add_argument(
        "--allow-nonstandard-head",
        action="store_true",
        help="Allow checkpoint head dimensions other than the zbn defaults.",
    )
    return parser.parse_args()


def find_v3b_head(model: nn.Module) -> nn.Module:
    """Find exactly one LaneRobotV3B head and enable tensor export mode."""
    from ultralytics.nn.modules.head import LaneRobotV3B

    heads = [module for module in model.modules() if isinstance(module, LaneRobotV3B)]
    if len(heads) != 1:
        raise RuntimeError(
            "Expected exactly one LaneRobotV3B head, "
            f"but found {len(heads)}: {[type(x).__name__ for x in heads]}"
        )

    head = heads[0]
    head.export = True
    return head


def shape_text(value: Any) -> str:
    dims: list[str] = []
    for dim in value.type.tensor_type.shape.dim:
        if dim.dim_param:
            dims.append(dim.dim_param)
        elif dim.HasField("dim_value"):
            dims.append(str(dim.dim_value))
        else:
            dims.append("?")
    return "[" + ", ".join(dims) + "]"


def validate_args(args: argparse.Namespace) -> tuple[int, int]:
    if args.batch < 1:
        raise ValueError("--batch must be at least 1")

    height, width = map(int, args.imgsz)
    if height <= 0 or width <= 0:
        raise ValueError("--imgsz values must be positive")
    if height % 32 != 0 or width % 32 != 0:
        raise ValueError(
            f"V3B export dimensions must be divisible by 32, got {height}x{width}"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")

    return height, width


def check_standard_head(head: nn.Module, allow_nonstandard: bool) -> None:
    actual = {
        key: int(getattr(head, key))
        for key in ("x_grids", "row_anchors", "num_lanes", "feat_h", "feat_w")
    }
    if actual != STANDARD_HEAD and not allow_nonstandard:
        details = "\n".join(
            f"  {key}: checkpoint={actual[key]}, expected={STANDARD_HEAD[key]}"
            for key in STANDARD_HEAD
        )
        raise RuntimeError(
            "Checkpoint is not the standard zbn LaneRobotV3B configuration:\n"
            f"{details}\n"
            "Use --allow-nonstandard-head only for an intentional custom model."
        )


def export_model(args: argparse.Namespace) -> Path:
    height, width = validate_args(args)

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

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "The onnx package is required. Install it with:\n"
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
    check_standard_head(head, args.allow_nonstandard_head)

    x_grids = int(head.x_grids)
    row_anchors = int(head.row_anchors)
    num_lanes = int(head.num_lanes)
    feat_hw = (int(head.feat_h), int(head.feat_w))

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

    captured: dict[str, tuple[int, ...]] = {}

    def capture_head_input(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
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
            "The selected input size triggers LaneRobotV3B adaptive pooling.\n"
            f"Actual head input: {head_input_shape}\n"
            f"Required spatial shape: {feat_hw}\n"
            "Use --imgsz 256 448 for the standard zbn checkpoint."
        )

    expected_shape = (args.batch, x_grids + 2, row_anchors, num_lanes)
    if tuple(torch_output.shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected output shape: {tuple(torch_output.shape)}, "
            f"expected {expected_shape}"
        )
    if not torch.isfinite(torch_output).all():
        raise RuntimeError("PyTorch output contains NaN or Inf")

    print(f"Lane head     : {type(head).__name__}")
    print(f"head feature : {head_input_shape}")
    print(f"x_grids      : {x_grids}")
    print(f"row_anchors  : {row_anchors}")
    print(f"num_lanes    : {num_lanes}")
    print(f"output shape : {tuple(torch_output.shape)}")

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch"},
            "lane_output": {0: "batch"},
        }

    kwargs = dict(
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
        torch.onnx.export(**kwargs, dynamo=False)
    except TypeError:
        torch.onnx.export(**kwargs)

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)

    declared_opsets = [
        item.version
        for item in onnx_model.opset_import
        if item.domain in ("", "ai.onnx")
    ]
    if OPSET not in declared_opsets:
        raise RuntimeError(
            f"Expected ONNX opset {OPSET}, but graph declares {declared_opsets}"
        )

    adaptive_nodes = [
        node.name or f"{node.op_type}@{index}"
        for index, node in enumerate(onnx_model.graph.node)
        if node.op_type in {"AdaptiveAveragePool", "AdaptiveMaxPool"}
    ]
    if adaptive_nodes:
        raise RuntimeError(
            "Unexpected adaptive-pooling operators remain in ONNX: "
            + ", ".join(adaptive_nodes)
        )

    print("\nONNX checker  : PASS")
    print(f"IR version    : {onnx_model.ir_version}")
    print(f"graph nodes   : {len(onnx_model.graph.node)}")
    print("ONNX inputs:")
    for value in onnx_model.graph.input:
        print(f"  {value.name}: {shape_text(value)}")
    print("ONNX outputs:")
    for value in onnx_model.graph.output:
        print(f"  {value.name}: {shape_text(value)}")

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
            str(output), providers=["CPUExecutionProvider"]
        )
        ort_output = session.run(
            ["lane_output"], {"images": dummy.detach().cpu().numpy()}
        )[0]
        reference = torch_output.detach().cpu().numpy()

        if ort_output.shape != reference.shape:
            raise RuntimeError(
                f"ONNX Runtime shape mismatch: {ort_output.shape} vs {reference.shape}"
            )
        if not np.isfinite(ort_output).all():
            raise RuntimeError("ONNX Runtime output contains NaN or Inf")

        max_abs_error = float(np.max(np.abs(reference - ort_output)))
        mean_abs_error = float(np.mean(np.abs(reference - ort_output)))
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
    print(f"output layout : cls [0:{x_grids + 1}], offset [{x_grids + 1}:{x_grids + 2}]")
    return output


def main() -> None:
    export_model(parse_args())


if __name__ == "__main__":
    main()
