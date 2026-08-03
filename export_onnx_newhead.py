#!/usr/bin/env python3
"""Export the four-lane Lane Robot split classification head to ONNX opset 11.

Output contract:
    cls_01  float32 [B, X+1, R, 2]  lanes 0 and 1
    cls_23  float32 [B, X+1, R, 2]  lanes 2 and 3
    offset  float32 [B,   1, R, 4]

For X=320 and R=56, each classification Gemm has 321*56*2=35952
outputs, below the RDK X5 limit of 65536. Legacy checkpoints containing one
71904-output cls_fc2 are migrated exactly by LaneRobotV2 while loading.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn

from export_onnx_xhm import replace_unsupported_adaptive_pool, tensor_shape_text


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = PROJECT_ROOT / "runs/lane/lane_n_baseline-3/weights/best.pt"
DEFAULT_IMGSZ = (320, 320)
CURRENT_X_GRIDS = 320
OPSET = 11
BPU_GEMM_OUTPUT_LIMIT = 65536
OUTPUT_NAMES = ("cls_01", "cls_23", "offset")


class LaneSplitOnnxWrapper(nn.Module):
    """Expose the split classification tensors as three stable ONNX outputs."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.model(images)
        if isinstance(output, (tuple, list)) and len(output) == 3:
            cls_01, cls_23, offset = output
            if all(isinstance(item, torch.Tensor) for item in (cls_01, cls_23, offset)):
                return cls_01, cls_23, offset
        if isinstance(output, dict):
            if "cls_01" in output and "cls_23" in output:
                return output["cls_01"], output["cls_23"], output["offset"]
            cls = output["cls"]
            if cls.shape[-1] == 4:
                return cls[..., :2], cls[..., 2:], output["offset"]
        raise TypeError(
            "Expected Lane Robot split outputs (cls_01, cls_23, offset), "
            f"got {type(output).__name__}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the RDK X5 split-head Lane Robot ONNX model.")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help=f"Checkpoint. Default: {DEFAULT_WEIGHTS}")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output ONNX. Default: <weights stem>_newhead.onnx beside the checkpoint.",
    )
    parser.add_argument(
        "--imgsz", type=int, nargs=2, metavar=("HEIGHT", "WIDTH"), default=DEFAULT_IMGSZ, help="Default: 320 320"
    )
    parser.add_argument("--batch", type=int, default=1, help="Static export batch. Default: 1")
    parser.add_argument("--dynamic-batch", action="store_true", help="Make only the batch dimension dynamic.")
    parser.add_argument("--expected-x-grids", type=int, default=CURRENT_X_GRIDS, help="Default: 320")
    parser.add_argument(
        "--allow-legacy-x-grids",
        action="store_true",
        help="Allow an intentional non-320-grid checkpoint; this does not allow a non-four-lane head.",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Export device. Default: cpu")
    parser.add_argument(
        "--verify-runtime", action="store_true", help="Compare all three PyTorch outputs with ONNX Runtime CPU."
    )
    return parser.parse_args()


def find_and_enable_split_export(model: nn.Module) -> nn.Module:
    candidates = []
    for module in model.modules():
        required = ("x_grids", "row_anchors", "num_lanes", "offset_fc")
        if all(hasattr(module, name) for name in required) and (
            hasattr(module, "cls_fc2") or (hasattr(module, "cls_fc2_01") and hasattr(module, "cls_fc2_23"))
        ):
            candidates.append(module)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected exactly one LaneRobotV2 head, found {[type(item).__name__ for item in candidates]}.")

    head = candidates[0]
    if int(head.num_lanes) != 4:
        raise RuntimeError(f"Solution A requires num_lanes=4, got {head.num_lanes}.")
    if hasattr(head, "cls_fc2"):
        migrate = getattr(head, "migrate_legacy_classification_head", None)
        if migrate is None:
            raise RuntimeError("The checkpoint has legacy cls_fc2 but this code cannot migrate it.")
        migrate()
    if not all(hasattr(head, name) for name in ("cls_fc2_01", "cls_fc2_23")):
        raise RuntimeError("Lane head does not contain both split classification projections.")

    head.export = True
    head.export_split_outputs = True
    return head


def compare_output_tuples(reference, candidate, *, rtol: float, atol: float, context: str) -> list[float]:
    if len(reference) != len(OUTPUT_NAMES) or len(candidate) != len(OUTPUT_NAMES):
        raise RuntimeError(f"{context}: expected three outputs.")
    errors = []
    for name, ref, actual in zip(OUTPUT_NAMES, reference, candidate):
        error = float(torch.max(torch.abs(ref - actual)).item())
        errors.append(error)
        if not torch.allclose(ref, actual, rtol=rtol, atol=atol):
            raise RuntimeError(f"{context}: {name} differs, max_abs_error={error:.8g}.")
    return errors


def validate_split_gemms(onnx_model) -> None:
    """Make the deployment-critical split visible and fail if the old large Gemm survived."""
    gemm_names = [node.name for node in onnx_model.graph.node if node.op_type == "Gemm"]
    split_01 = [name for name in gemm_names if "cls_fc2_01" in name]
    split_23 = [name for name in gemm_names if "cls_fc2_23" in name]
    legacy = [name for name in gemm_names if "cls_fc2/Gemm" in name]
    if len(split_01) != 1 or len(split_23) != 1 or legacy:
        raise RuntimeError(
            "ONNX graph does not contain exactly the two expected split classification Gemms. "
            f"cls_01={split_01}, cls_23={split_23}, legacy={legacy}"
        )


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(PROJECT_ROOT))

    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {weights}")
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else weights.with_name(f"{weights.stem}_newhead.onnx")
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    height, width = map(int, args.imgsz)
    if args.batch < 1 or height < 1 or width < 1 or args.expected_x_grids < 1:
        raise ValueError("--batch, --imgsz, and --expected-x-grids must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable.")

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("Install onnx in the lane_robot environment before exporting.") from exc

    from ultralytics import YOLO

    yolo = YOLO(str(weights), task="lane")
    torch_model = yolo.model.to(args.device).float().eval()
    head = find_and_enable_split_export(torch_model)
    for parameter in torch_model.parameters():
        parameter.requires_grad_(False)

    x_grids = int(head.x_grids)
    row_anchors = int(head.row_anchors)
    num_lanes = int(head.num_lanes)
    if x_grids != args.expected_x_grids and not args.allow_legacy_x_grids:
        raise RuntimeError(f"Checkpoint uses x_grids={x_grids}; expected {args.expected_x_grids}.")

    per_head_outputs = (x_grids + 1) * row_anchors * 2
    if per_head_outputs > BPU_GEMM_OUTPUT_LIMIT:
        raise RuntimeError(
            f"Each classification Gemm would output {per_head_outputs}, above the RDK X5 limit {BPU_GEMM_OUTPUT_LIMIT}."
        )
    if head.cls_fc2_01.out_features != per_head_outputs or head.cls_fc2_23.out_features != per_head_outputs:
        raise RuntimeError("Split Linear out_features do not match the declared output contract.")

    wrapper = LaneSplitOnnxWrapper(torch_model).to(args.device).float().eval()
    dummy = torch.zeros(args.batch, 3, height, width, dtype=torch.float32, device=args.device)
    with torch.inference_mode():
        before_pool_rewrite = wrapper(dummy)
    replace_unsupported_adaptive_pool(head)
    with torch.inference_mode():
        torch_outputs = wrapper(dummy)
    pool_errors = compare_output_tuples(
        before_pool_rewrite, torch_outputs, rtol=1e-5, atol=1e-5, context="ONNX-compatible pool rewrite"
    )

    expected_shapes = (
        (args.batch, x_grids + 1, row_anchors, 2),
        (args.batch, x_grids + 1, row_anchors, 2),
        (args.batch, 1, row_anchors, num_lanes),
    )
    actual_shapes = tuple(tuple(item.shape) for item in torch_outputs)
    if actual_shapes != expected_shapes:
        raise RuntimeError(f"Unexpected split output shapes: expected {expected_shapes}, got {actual_shapes}.")

    print("=" * 72)
    print("Lane Robot split-head ONNX export")
    print(f"weights          : {weights}")
    print(f"output           : {output}")
    print(f"input            : [{args.batch}, 3, {height}, {width}]")
    print(f"outputs          : {dict(zip(OUTPUT_NAMES, actual_shapes))}")
    print(f"Gemm outputs     : {per_head_outputs} each (limit {BPU_GEMM_OUTPUT_LIMIT})")
    print(f"pool rewrite     : PASS (max errors {pool_errors})")
    migration_status = (
        "PASS (legacy cls_fc2 reconstructed bit-exactly)"
        if getattr(head, "legacy_cls_migrated", False)
        else "not needed (checkpoint already uses split heads)"
    )
    print(f"legacy migration : {migration_status}")
    print("=" * 72)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"images": {0: "batch"}, **{name: {0: "batch"} for name in OUTPUT_NAMES}}
    export_kwargs = dict(
        model=wrapper,
        args=dummy,
        f=str(output),
        export_params=True,
        opset_version=OPSET,
        do_constant_folding=True,
        input_names=["images"],
        output_names=list(OUTPUT_NAMES),
        dynamic_axes=dynamic_axes,
    )
    try:
        torch.onnx.export(**export_kwargs, dynamo=False)
    except TypeError:
        torch.onnx.export(**export_kwargs)

    onnx_model = onnx.load(str(output))
    onnx.checker.check_model(onnx_model)
    validate_split_gemms(onnx_model)
    print("\nONNX checker     : PASS")
    for value in onnx_model.graph.output:
        print(f"  {value.name}: {tensor_shape_text(value)}")
    print("split Gemm graph : PASS (one cls_fc2_01/Gemm + one cls_fc2_23/Gemm, no legacy cls_fc2/Gemm)")

    if args.verify_runtime:
        try:
            import numpy as np
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("--verify-runtime requires onnxruntime and numpy.") from exc
        session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        ort_outputs = session.run(list(OUTPUT_NAMES), {"images": dummy.detach().cpu().numpy()})
        runtime_errors = []
        for name, reference, actual in zip(OUTPUT_NAMES, torch_outputs, ort_outputs):
            reference = reference.detach().cpu().numpy()
            max_error = float(np.max(np.abs(reference - actual)))
            runtime_errors.append(max_error)
            if not np.allclose(reference, actual, rtol=1e-3, atol=1e-4):
                raise RuntimeError(f"ONNX Runtime {name} differs from PyTorch, max_abs_error={max_error:.8g}.")
        print(f"ONNX Runtime     : PASS (max errors {dict(zip(OUTPUT_NAMES, runtime_errors))})")

    print(f"\nExport complete: {output} ({output.stat().st_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()
