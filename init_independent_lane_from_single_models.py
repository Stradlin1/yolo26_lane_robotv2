"""Initialize a four-task independent LaneRobotV2 model from proven single-task checkpoints."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ultralytics.nn.tasks import LaneRobotModel


HEAD_SUFFIXES = (
    "conv_1x1.weight",
    "conv_1x1.bias",
    "cls_fc1.weight",
    "cls_fc1.bias",
    "cls_fc2.weight",
    "cls_fc2.bias",
    "offset_fc.weight",
    "offset_fc.bias",
)


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Load a state dictionary from an Ultralytics checkpoint, raw module, or raw state-dict file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint: Any = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict):
        candidate = checkpoint.get("ema") or checkpoint.get("model") or checkpoint.get("state_dict") or checkpoint
    else:
        candidate = checkpoint

    if isinstance(candidate, nn.Module):
        state = candidate.float().state_dict()
    elif isinstance(candidate, dict) and all(isinstance(k, str) for k in candidate):
        state = candidate
    else:
        raise TypeError(f"Unsupported checkpoint content in {path}: {type(candidate)}")

    normalized = {}
    for key, value in state.items():
        key = key.removeprefix("_orig_mod.")
        if torch.is_tensor(value):
            normalized[key] = value.detach().cpu()
    return normalized


def find_single_head_prefix(state: dict[str, torch.Tensor]) -> str:
    """Locate the prefix before the original single-task LaneRobotV2 layer names."""
    candidates = [key[: -len("conv_1x1.weight")] for key in state if key.endswith("conv_1x1.weight")]
    candidates = [prefix for prefix in candidates if "task_branches." not in prefix]
    if not candidates:
        raise KeyError("Could not find a single-task LaneRobotV2 'conv_1x1.weight' in checkpoint")
    # The lane head is normally the last model layer; prefer the longest/numerically latest prefix.
    return sorted(candidates, key=lambda x: (x.count("."), x))[-1]


def find_independent_head_prefix(state: dict[str, torch.Tensor]) -> str:
    marker = "task_branches.0.conv_1x1.weight"
    matches = [key[: -len(marker)] for key in state if key.endswith(marker)]
    if len(matches) != 1:
        raise KeyError(f"Expected one independent lane head prefix, found {matches}")
    return matches[0]


def copy_matching_shared_weights(
    destination: dict[str, torch.Tensor], source: dict[str, torch.Tensor], independent_head_prefix: str
) -> tuple[int, list[str]]:
    """Copy exact-shape shared backbone/neck weights and skip all prediction-head parameters."""
    copied = 0
    mismatched = []
    for key, value in source.items():
        if key.startswith(independent_head_prefix) or key not in destination:
            continue
        if destination[key].shape != value.shape:
            mismatched.append(f"{key}: source={tuple(value.shape)} destination={tuple(destination[key].shape)}")
            continue
        destination[key] = value.to(dtype=destination[key].dtype)
        copied += 1
    return copied, mismatched


def copy_single_head_to_branch(
    destination: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    source_prefix: str,
    destination_prefix: str,
    task_id: int,
) -> tuple[int, list[str], list[str]]:
    """Copy one single-task head into one independent task branch by value, never by Parameter reference."""
    copied = 0
    missing = []
    mismatched = []
    for suffix in HEAD_SUFFIXES:
        src_key = source_prefix + suffix
        dst_key = f"{destination_prefix}task_branches.{task_id}.{suffix}"
        if src_key not in source:
            missing.append(src_key)
            continue
        if dst_key not in destination:
            missing.append(dst_key)
            continue
        if source[src_key].shape != destination[dst_key].shape:
            mismatched.append(
                f"{src_key} -> {dst_key}: source={tuple(source[src_key].shape)} "
                f"destination={tuple(destination[dst_key].shape)}"
            )
            continue
        destination[dst_key] = source[src_key].to(dtype=destination[dst_key].dtype).clone()
        copied += 1
    return copied, missing, mismatched


def initialize_model(
    model_yaml: str | Path,
    base_checkpoint: str | Path,
    task_checkpoints: list[str | Path],
) -> tuple[LaneRobotModel, dict[str, Any]]:
    """Build and initialize an independent model, returning the model and a detailed migration report."""
    model = LaneRobotModel(str(model_yaml), ch=3, verbose=False)
    if len(task_checkpoints) != model.model[-1].num_lanes:
        raise ValueError(
            f"Expected {model.model[-1].num_lanes} task checkpoints, received {len(task_checkpoints)}"
        )

    destination = model.state_dict()
    independent_prefix = find_independent_head_prefix(destination)
    base_state = load_state_dict(base_checkpoint)
    shared_count, shared_mismatched = copy_matching_shared_weights(destination, base_state, independent_prefix)

    task_reports = []
    for task_id, task_checkpoint in enumerate(task_checkpoints):
        task_state = load_state_dict(task_checkpoint)
        source_prefix = find_single_head_prefix(task_state)
        copied, missing, mismatched = copy_single_head_to_branch(
            destination, task_state, source_prefix, independent_prefix, task_id
        )
        task_reports.append(
            {
                "task_id": task_id,
                "checkpoint": str(task_checkpoint),
                "source_prefix": source_prefix,
                "copied": copied,
                "missing": missing,
                "mismatched": mismatched,
            }
        )

    model.load_state_dict(destination, strict=True)
    report = {
        "independent_head_prefix": independent_prefix,
        "shared_parameters_copied": shared_count,
        "shared_mismatched": shared_mismatched,
        "tasks": task_reports,
    }
    return model, report


def save_ultralytics_checkpoint(
    model: LaneRobotModel,
    output: str | Path,
    model_yaml: str | Path,
    base_checkpoint: str | Path,
    task_checkpoints: list[str | Path],
    report: dict[str, Any],
) -> None:
    """Save an Ultralytics-loadable initialization checkpoint."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": -1,
        "best_fitness": None,
        "model": deepcopy(model).half().cpu(),
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "train_args": {
            "task": "lane",
            "model": str(model_yaml),
            "base_checkpoint": str(base_checkpoint),
            "task_checkpoints": [str(x) for x in task_checkpoints],
        },
        "independent_lane_migration": report,
    }
    torch.save(checkpoint, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="ultralytics/cfg/models/26/yolo26n-lane-independent.yaml",
        help="Independent four-task model YAML.",
    )
    parser.add_argument("--base", required=True, help="Checkpoint used for shared backbone/neck/fusion weights.")
    parser.add_argument(
        "--task-weights",
        nargs="+",
        required=True,
        help="Single-task checkpoints ordered by destination task id.",
    )
    parser.add_argument("--output", required=True, help="Output Ultralytics .pt initialization checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, report = initialize_model(args.model, args.base, args.task_weights)

    print(f"Independent head prefix: {report['independent_head_prefix']}")
    print(f"Shared parameters copied: {report['shared_parameters_copied']}")
    for item in report["tasks"]:
        print(
            f"Task {item['task_id']}: copied={item['copied']} source={item['source_prefix']} "
            f"missing={len(item['missing'])} mismatched={len(item['mismatched'])}"
        )
        for message in item["missing"]:
            print(f"  missing: {message}")
        for message in item["mismatched"]:
            print(f"  mismatched: {message}")

    errors = report["shared_mismatched"] or any(
        item["missing"] or item["mismatched"] or item["copied"] != len(HEAD_SUFFIXES)
        for item in report["tasks"]
    )
    if errors:
        raise RuntimeError("Checkpoint migration failed validation; no output checkpoint was written.")

    save_ultralytics_checkpoint(model, args.output, args.model, args.base, args.task_weights, report)
    print(f"Saved independent initialization checkpoint: {args.output}")


if __name__ == "__main__":
    main()
