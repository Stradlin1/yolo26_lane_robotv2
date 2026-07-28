#!/usr/bin/env python3
"""Verify the real LaneRobot multi-lane training pipeline.

Run from the repository root, for example:
    python verify_lane_pipeline.py \
        --data ultralytics/cfg/datasets/lane-robot.yaml \
        --model ultralytics/cfg/models/26/yolo26n-lane.yaml
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ultralytics.models.yolo.lane.dataset import LaneRobotDataset
from ultralytics.models.yolo.lane.train import LaneRobotTrainer


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify LaneRobot dataset, model shapes, loss, and backward pass.")
    parser.add_argument(
        "--data",
        default="ultralytics/cfg/datasets/lane-robot.yaml",
        help="Path to the LaneRobot dataset YAML.",
    )
    parser.add_argument(
        "--model",
        default="ultralytics/cfg/models/26/yolo26n-lane.yaml",
        help="Path to the LaneRobot model YAML.",
    )
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--samples", type=int, default=2, help="Number of real images used for forward/backward.")
    parser.add_argument("--imgsz", nargs=2, type=int, default=(256, 320), metavar=("H", "W"))
    parser.add_argument("--device", default="cpu", help="cpu, cuda, cuda:0, etc.")
    parser.add_argument(
        "--max-errors",
        type=int,
        default=20,
        help="Maximum number of detailed label errors to print.",
    )
    return parser.parse_args()


def make_trainer(data_yaml: Path, imgsz: tuple[int, int]) -> LaneRobotTrainer:
    trainer = object.__new__(LaneRobotTrainer)
    trainer.args = SimpleNamespace(
        data=str(data_yaml),
        imgsz=list(imgsz),
        workers=0,
        lane_x_grids=160,
        lane_row_anchors=56,
        lane_num_lanes=1,
        lane_y_start=0.67,
        lane_y_end=1.0,
    )
    trainer.data = trainer.get_dataset()
    return trainer


def collect_images(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() == ".txt":
        base = path.parent
        result = []
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            p = Path(raw)
            result.append(p if p.is_absolute() else base / p)
        return result
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def infer_label_path(image_path: Path, split: str, data: dict) -> Path:
    explicit = data.get(f"{split}_labels") or data.get("labels")
    if explicit:
        label_root = Path(explicit)
        if not label_root.is_absolute():
            label_root = Path(data.get("path", ".")) / label_root
        return label_root / image_path.with_suffix(".txt").name

    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def validate_labels(images: list[Path], split: str, data: dict, max_errors: int) -> tuple[list[str], list[str], Counter]:
    row_anchors = int(data["row_anchors"])
    num_lanes = int(data["num_lanes"])
    expected_tokens = 1 + 2 * row_anchors
    errors: list[str] = []
    warnings: list[str] = []
    lane_counts: Counter = Counter()
    missing_labels = 0
    empty_labels = 0
    y_reference: np.ndarray | None = None
    y_direction: str | None = None

    def add_error(message: str) -> None:
        if len(errors) < max_errors:
            errors.append(message)

    for image_path in images:
        label_path = infer_label_path(image_path, split, data)
        if not label_path.exists():
            missing_labels += 1
            add_error(f"missing label: {label_path} (image: {image_path})")
            continue

        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            empty_labels += 1
            continue

        seen_ids: set[int] = set()
        for line_no, raw in enumerate(text.splitlines(), start=1):
            raw = raw.strip()
            if not raw:
                continue
            fields = raw.replace(",", " ").split()
            if len(fields) != expected_tokens:
                add_error(
                    f"wrong token count: {label_path}:{line_no}, got {len(fields)}, expected {expected_tokens}"
                )
                continue

            try:
                values = [float(v) for v in fields]
            except ValueError as exc:
                add_error(f"non-numeric token: {label_path}:{line_no}: {exc}")
                continue

            lane_value = values[0]
            lane_id = int(lane_value)
            if lane_value != lane_id:
                add_error(f"lane_id is not an integer: {label_path}:{line_no}: {lane_value}")
                continue
            if not 0 <= lane_id < num_lanes:
                add_error(
                    f"lane_id out of range: {label_path}:{line_no}: {lane_id}, expected 0..{num_lanes - 1}"
                )
                continue
            if lane_id in seen_ids:
                add_error(f"duplicate lane_id {lane_id}: {label_path}:{line_no}")
                continue
            seen_ids.add(lane_id)
            lane_counts[lane_id] += 1

            xs = np.asarray(values[1::2], dtype=np.float64)
            ys = np.asarray(values[2::2], dtype=np.float64)

            invalid_x = xs[(xs < 0) & (~np.isclose(xs, -1.0))]
            invalid_x = np.concatenate((invalid_x, xs[xs > 1.0]))
            if invalid_x.size:
                add_error(
                    f"x must be -1 or normalized [0,1]: {label_path}:{line_no}; examples={invalid_x[:5].tolist()}"
                )

            if np.any((ys < 0.0) | (ys > 1.0)):
                bad = ys[(ys < 0.0) | (ys > 1.0)]
                add_error(f"y outside [0,1]: {label_path}:{line_no}; examples={bad[:5].tolist()}")

            if y_reference is None:
                y_reference = ys.copy()
                if np.all(np.diff(ys) < 0):
                    y_direction = "descending"
                elif np.all(np.diff(ys) > 0):
                    y_direction = "ascending"
                else:
                    y_direction = "non-monotonic"
            elif not np.allclose(ys, y_reference, atol=1e-5, rtol=0.0):
                add_error(f"y anchors differ from dataset reference: {label_path}:{line_no}")

    if missing_labels:
        warnings.append(f"missing label files: {missing_labels}")
    if empty_labels:
        warnings.append(f"empty label files (all lanes absent): {empty_labels}")

    if y_reference is not None:
        warnings.append(
            "label y anchors: "
            f"first={y_reference[0]:.6f}, last={y_reference[-1]:.6f}, order={y_direction}"
        )
        yaml_start = float(data["y_start"])
        yaml_end = float(data["y_end"])
        low = float(np.min(y_reference))
        high = float(np.max(y_reference))
        if not (math.isclose(low, min(yaml_start, yaml_end), abs_tol=1e-5) and math.isclose(high, max(yaml_start, yaml_end), abs_tol=1e-5)):
            warnings.append(
                "YAML y range differs from labels: "
                f"yaml=({yaml_start:.6f},{yaml_end:.6f}), labels=({low:.6f},{high:.6f})"
            )

    return errors, warnings, lane_counts


def validate_dataset_samples(dataset: LaneRobotDataset, sample_count: int) -> dict:
    if len(dataset) == 0:
        raise RuntimeError("Dataset contains no images.")

    count = min(max(sample_count, 1), len(dataset))
    samples = [dataset[i] for i in range(count)]
    expected_lane = (dataset.row_anchors, dataset.num_lanes)
    expected_y = (dataset.row_anchors,)

    for i, sample in enumerate(samples):
        if tuple(sample["lane"].shape) != expected_lane:
            raise AssertionError(f"sample {i} lane shape={tuple(sample['lane'].shape)}, expected={expected_lane}")
        if tuple(sample["lane_x"].shape) != expected_lane:
            raise AssertionError(f"sample {i} lane_x shape={tuple(sample['lane_x'].shape)}, expected={expected_lane}")
        if tuple(sample["lane_y"].shape) != expected_y:
            raise AssertionError(f"sample {i} lane_y shape={tuple(sample['lane_y'].shape)}, expected={expected_y}")

        active = (sample["lane_x"] >= 0).any(dim=0).nonzero(as_tuple=False).flatten().tolist()
        print(f"sample[{i}] {sample['im_file']}")
        print(f"  img={tuple(sample['img'].shape)}, active lane IDs={active}")

    return LaneRobotDataset.collate_fn(samples)


def validate_model_and_loss(trainer: LaneRobotTrainer, model_yaml: Path, batch: dict, device: str) -> None:
    model = trainer.get_model(str(model_yaml), verbose=False)
    model.names = dict(trainer.data["names"])
    model.nc = int(trainer.data["nc"])

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.train()

    train_batch = {
        "img": batch["img"].to(torch_device).float() / 255.0,
        "lane": batch["lane"].to(torch_device).long(),
        "lane_x": batch["lane_x"].to(torch_device).float(),
        "lane_y": batch["lane_y"].to(torch_device).float(),
        "cls": batch["cls"].to(torch_device),
    }

    model.zero_grad(set_to_none=True)
    output = model(train_batch["img"])
    head = model.model[-1]

    expected_cls = (
        train_batch["img"].shape[0],
        int(trainer.data["x_grids"]) + 1,
        int(trainer.data["row_anchors"]),
        int(trainer.data["num_lanes"]),
    )
    expected_offset = (
        train_batch["img"].shape[0],
        1,
        int(trainer.data["row_anchors"]),
        int(trainer.data["num_lanes"]),
    )

    print(f"model head: x_grids={head.x_grids}, row_anchors={head.row_anchors}, num_lanes={head.num_lanes}")
    print(f"cls shape: {tuple(output['cls'].shape)}")
    print(f"offset shape: {tuple(output['offset'].shape)}")

    if tuple(output["cls"].shape) != expected_cls:
        raise AssertionError(f"cls shape mismatch: got {tuple(output['cls'].shape)}, expected {expected_cls}")
    if tuple(output["offset"].shape) != expected_offset:
        raise AssertionError(f"offset shape mismatch: got {tuple(output['offset'].shape)}, expected {expected_offset}")

    criterion = model.init_criterion()
    loss, items = criterion(output, train_batch)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"loss is not finite: {loss.item()}")

    loss.backward()
    grad_ok = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if not grad_ok:
        raise RuntimeError("No finite gradients were produced by backward().")

    names = ("ce", "loc", "exist", "smooth", "curv", "offset")
    print(f"total loss: {loss.item():.6f}")
    print("loss items:")
    for name, value in zip(names, items.detach().cpu().tolist()):
        print(f"  {name}: {value:.6f}")
    print("backward: PASS (finite gradients found)")


def main() -> int:
    args = parse_args()
    data_yaml = Path(args.data).expanduser().resolve()
    model_yaml = Path(args.model).expanduser().resolve()

    if not data_yaml.exists():
        print(f"ERROR: data YAML not found: {data_yaml}", file=sys.stderr)
        return 2
    if not model_yaml.exists():
        print(f"ERROR: model YAML not found: {model_yaml}", file=sys.stderr)
        return 2

    print("=== 1. Resolve real dataset YAML ===")
    trainer = make_trainer(data_yaml, tuple(args.imgsz))
    data = trainer.data
    for key in ("path", args.split, "x_grids", "row_anchors", "num_lanes", "y_start", "y_end", "nc", "names"):
        print(f"{key}: {data.get(key)}")

    split_path_raw = data.get(args.split)
    if not split_path_raw:
        print(f"ERROR: split '{args.split}' is missing from data YAML.", file=sys.stderr)
        return 2
    split_path = Path(split_path_raw)
    images = collect_images(split_path)
    print(f"resolved {args.split} path: {split_path}")
    print(f"image count: {len(images)}")
    if not images:
        print("ERROR: no images found.", file=sys.stderr)
        return 2

    print("\n=== 2. Validate all label files ===")
    errors, warnings, lane_counts = validate_labels(images, args.split, data, args.max_errors)
    for warning in warnings:
        print(f"WARNING: {warning}")
    for lane_id in range(int(data["num_lanes"])):
        name = data["names"].get(lane_id, data["names"].get(str(lane_id), f"lane_{lane_id}"))
        print(f"lane {lane_id} ({name}): present in {lane_counts[lane_id]} label files")

    if errors:
        print(f"\nLABEL CHECK FAILED: {len(errors)} error(s) shown (limited by --max-errors):")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("label structure: PASS")

    print("\n=== 3. Load real samples through LaneRobotDataset ===")
    dataset = LaneRobotDataset(split_path, trainer.args, data, mode=args.split)
    batch = validate_dataset_samples(dataset, args.samples)
    print(f"batch img shape: {tuple(batch['img'].shape)}")
    print(f"batch lane shape: {tuple(batch['lane'].shape)}")
    print(f"batch lane_x shape: {tuple(batch['lane_x'].shape)}")
    print(f"batch lane_y shape: {tuple(batch['lane_y'].shape)}")
    print("dataset loading: PASS")

    print("\n=== 4. Real forward, loss, and backward ===")
    validate_model_and_loss(trainer, model_yaml, batch, args.device)

    print("\n=== FINAL RESULT ===")
    print("PASS: real YAML -> labels -> Dataset -> 4-lane model -> loss -> backward is connected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
