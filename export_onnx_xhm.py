#!/usr/bin/env python3
"""
Export the trained Lane Robot model to ONNX opset 11.

The ONNX model has:
    input:
        images      float32 [B, 3, 320, 320]

    output:
        lane_output float32 [B, x_grids + 2, row_anchors, num_lanes]
        The new x_grids=320 configuration produces [B, 322, 56, 4].

Output channel layout for x_grids=320:
    lane_output[:, 0:321, :, :]  -> classification logits
        channels 0..319: horizontal grid logits
        channel 320: no-lane logit

    lane_output[:, 321:322, :, :] -> sub-grid offset in [-0.5, 0.5]

The checkpoint head is the source of truth for x_grids. Legacy x_grids=160
checkpoints still export 162 channels. The default export is static batch=1;
use --dynamic-batch only when the deployment runtime needs a variable batch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


DEFAULT_WEIGHTS = Path(
    "/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/"
    "runs/lane/lane_n_baseline-2/weights/best.pt"
)
DEFAULT_IMGSZ = (320, 320)
OPSET = 11


class LaneOnnxWrapper(nn.Module):
    """Convert the custom Lane model output into one ONNX-friendly tensor."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        output = self.model(images)

        if isinstance(output, torch.Tensor):
            return output

        if isinstance(output, dict):
            cls = output["cls"]
            offset = output["offset"]
            return torch.cat((cls, offset), dim=1)

        if isinstance(output, (tuple, list)):
            if len(output) == 1:
                output = output[0]

            if isinstance(output, torch.Tensor):
                return output

            if isinstance(output, dict):
                cls = output["cls"]
                offset = output["offset"]
                return torch.cat((cls, offset), dim=1)

        raise TypeError(
            "Unsupported Lane model output type during ONNX export: "
            f"{type(output).__name__}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Lane Robot best.pt to ONNX opset 11."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help=f"PyTorch checkpoint path. Default: {DEFAULT_WEIGHTS}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ONNX path. Default: best.onnx beside the checkpoint.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=DEFAULT_IMGSZ,
        help="Export input size. Default: 320 320.",
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
        help="Device used while exporting. CPU is the safest default.",
    )
    parser.add_argument(
        "--verify-runtime",
        action="store_true",
        help="Also compare PyTorch and ONNX Runtime outputs when onnxruntime is installed.",
    )
    return parser.parse_args()


def find_and_enable_lane_export(model: nn.Module) -> nn.Module:
    """Find LaneRobotV2-like head and force tensor output mode."""
    candidates = []

    for module in model.modules():
        required = (
            "x_grids",
            "row_anchors",
            "num_lanes",
            "cls_fc2",
            "offset_fc",
        )
        if all(hasattr(module, name) for name in required):
            candidates.append(module)

    if len(candidates) != 1:
        names = [type(module).__name__ for module in candidates]
        raise RuntimeError(
            "Expected exactly one Lane head, "
            f"but found {len(candidates)}: {names}"
        )

    head = candidates[0]
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

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "The 'onnx' package is required. Install it in the lane_robot "
            "environment with:\n"
            "  python -m pip install 'onnx>=1.12,<2'"
        ) from exc

    # Import after adding the local project root, so the customized Lane branch
    # is used instead of an unrelated PyPI installation.
    from ultralytics import YOLO

    print("=" * 72)
    print("Lane Robot ONNX export")
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

    head = find_and_enable_lane_export(torch_model)

    x_grids = int(head.x_grids)
    row_anchors = int(head.row_anchors)
    num_lanes = int(head.num_lanes)
    expected_channels = x_grids + 2

    wrapper = LaneOnnxWrapper(torch_model).to(args.device).float().eval()
    dummy = torch.zeros(
        args.batch,
        3,
        height,
        width,
        dtype=torch.float32,
        device=args.device,
    )

    with torch.inference_mode():
        torch_output = wrapper(dummy)

    expected_shape = (
        args.batch,
        expected_channels,
        row_anchors,
        num_lanes,
    )
    actual_shape = tuple(torch_output.shape)

    if actual_shape != expected_shape:
        raise RuntimeError(
            "Unexpected model output shape before export.\n"
            f"Expected: {expected_shape}\n"
            f"Actual:   {actual_shape}"
        )

    print(f"Lane head     : {type(head).__name__}")
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

    # PyTorch 2.x defaults may use the new dynamo exporter. Explicitly request
    # the legacy exporter because opset 11 and dynamic_axes are more predictable
    # through this path for the current custom model.
    try:
        torch.onnx.export(**export_kwargs, dynamo=False)
    except TypeError:
        # Compatibility with older PyTorch versions that do not expose dynamo.
        torch.onnx.export(**export_kwargs)

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)

    print("\nONNX checker  : PASS")
    print("ONNX inputs:")
    for value in onnx_model.graph.input:
        print(f"  {value.name}: {tensor_shape_text(value)}")

    print("ONNX outputs:")
    for value in onnx_model.graph.output:
        print(f"  {value.name}: {tensor_shape_text(value)}")

    if args.verify_runtime:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError:
            print(
                "\nONNX Runtime verification skipped. Install with:\n"
                "  python -m pip install onnxruntime"
            )
        else:
            session = ort.InferenceSession(
                str(output),
                providers=["CPUExecutionProvider"],
            )
            test_input = dummy.detach().cpu().numpy()
            ort_output = session.run(
                ["lane_output"],
                {"images": test_input},
            )[0]

            reference = torch_output.detach().cpu().numpy()
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
    print(
        f"output layout : cls logits [0:{x_grids + 1}], "
        f"offset [{x_grids + 1}:{x_grids + 2}]"
    )


if __name__ == "__main__":
    main()
