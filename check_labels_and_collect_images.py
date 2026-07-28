#!/usr/bin/env python3
"""检查 Lane Robot 多线标签，并收集对应图片。

默认目录结构：
    /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/
    ├── datasets/
    │   ├── labels_corrected/train/*.txt
    │   └── images/train/*
    └── waited/

标签行格式：
    lane_id x1 y1 x2 y2 ... x56 y56

严格要求：
- 每行正好 113 个数值（1 + 56 * 2）。
- lane_id 是 0、1、2、3，且同一文件内不能重复。
- x 必须为 -1，或位于 [0, 1]。
- y 必须位于 [0, 1]。
- 每条线的 56 个 y 锚点顺序默认应从大到小。
- 所有非空标签行应使用相同的 56 个 y 锚点。

默认会复制“所有有对应图片的标签”的图片；使用 --only-valid 可只复制格式正确标签的图片。
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path("/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT")
DEFAULT_LABEL_DIR = PROJECT_ROOT / "datasets/labels_corrected/train"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "datasets/images/train"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "waited"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LANE_NAMES = {
    0: "lane_follow",
    1: "lead_lane",
    2: "channel_left",
    3: "channel_right",
}


@dataclass
class LabelResult:
    path: Path
    valid: bool = True
    empty: bool = False
    lane_ids: list[int] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    y_reference: tuple[float, ...] | None = None

    def add_error(self, message: str) -> None:
        self.valid = False
        self.errors.append(message)


@dataclass
class ImageMatch:
    label_path: Path
    image_path: Path | None
    status: str  # matched, missing, ambiguous
    candidates: list[Path] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查 Lane Robot 标签格式，并把对应图片复制到 waited 文件夹。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR, help="标签目录")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR, help="图片目录")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="图片输出目录")
    parser.add_argument("--row-anchors", type=int, default=56, help="每条线的采样点数量")
    parser.add_argument("--num-lanes", type=int, default=4, help="固定语义槽位数量")
    parser.add_argument(
        "--y-order",
        choices=("descending", "ascending", "any"),
        default="descending",
        help="y 锚点顺序；当前 corrected 标签应使用 descending（约 1.0 到 0.333333）",
    )
    parser.add_argument("--tolerance", type=float, default=1e-6, help="浮点比较容差")
    parser.add_argument(
        "--only-valid",
        action="store_true",
        help="只复制格式检查通过的标签所对应的图片；默认复制所有匹配图片",
    )
    parser.add_argument("--clear-output", action="store_true", help="运行前清空 output-dir 中已有内容")
    parser.add_argument("--dry-run", action="store_true", help="只检查和生成报告，不复制图片")
    return parser.parse_args()


def split_tokens(line: str) -> list[str]:
    return line.replace(",", " ").split()


def is_monotonic(values: list[float], order: str, tolerance: float) -> bool:
    if order == "any":
        return True
    if order == "descending":
        return all(values[i] <= values[i - 1] + tolerance for i in range(1, len(values)))
    return all(values[i] + tolerance >= values[i - 1] for i in range(1, len(values)))


def same_float_sequence(a: Iterable[float], b: Iterable[float], tolerance: float) -> bool:
    a_list = list(a)
    b_list = list(b)
    return len(a_list) == len(b_list) and all(abs(x - y) <= tolerance for x, y in zip(a_list, b_list))


def check_label_file(
    label_path: Path,
    row_anchors: int,
    num_lanes: int,
    y_order: str,
    tolerance: float,
) -> LabelResult:
    result = LabelResult(path=label_path)
    expected_columns = 1 + row_anchors * 2

    try:
        text = label_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as exc:
        result.add_error(f"无法读取文件：{exc}")
        return result

    nonempty_lines = [(number, line.strip()) for number, line in enumerate(text.splitlines(), start=1) if line.strip()]
    if not nonempty_lines:
        result.empty = True
        result.warnings.append("空标签文件：按‘图中没有任何车道线’处理")
        return result

    seen_lane_ids: set[int] = set()
    file_y_reference: tuple[float, ...] | None = None

    for line_number, line in nonempty_lines:
        tokens = split_tokens(line)
        if len(tokens) != expected_columns:
            result.add_error(
                f"第 {line_number} 行列数错误：实际 {len(tokens)}，应为 {expected_columns} "
                f"(lane_id + {row_anchors} 对 x/y)"
            )
            continue

        try:
            values = [float(token) for token in tokens]
        except ValueError as exc:
            result.add_error(f"第 {line_number} 行包含非数值字段：{exc}")
            continue

        if not all(math.isfinite(value) for value in values):
            result.add_error(f"第 {line_number} 行包含 NaN 或 Inf")
            continue

        lane_id_value = values[0]
        if not lane_id_value.is_integer():
            result.add_error(f"第 {line_number} 行 lane_id 不是整数：{tokens[0]}")
            continue

        lane_id = int(lane_id_value)
        if not 0 <= lane_id < num_lanes:
            result.add_error(f"第 {line_number} 行 lane_id={lane_id}，允许范围为 0~{num_lanes - 1}")
            continue

        result.lane_ids.append(lane_id)
        if lane_id in seen_lane_ids:
            result.add_error(f"第 {line_number} 行 lane_id={lane_id} 重复；同一文件每个槽位最多一条线")
        seen_lane_ids.add(lane_id)

        xs = values[1::2]
        ys = values[2::2]

        for index, x in enumerate(xs, start=1):
            if not (abs(x + 1.0) <= tolerance or -tolerance <= x <= 1.0 + tolerance):
                result.add_error(
                    f"第 {line_number} 行第 {index} 个 x={x:g} 非法；x 只能为 -1 或位于 [0, 1]"
                )

        for index, y in enumerate(ys, start=1):
            if not -tolerance <= y <= 1.0 + tolerance:
                result.add_error(f"第 {line_number} 行第 {index} 个 y={y:g} 超出 [0, 1]")

        if not is_monotonic(ys, y_order, tolerance):
            direction = "从大到小" if y_order == "descending" else "从小到大"
            result.add_error(f"第 {line_number} 行 y 锚点顺序错误；应当{direction}排列")

        y_tuple = tuple(ys)
        if file_y_reference is None:
            file_y_reference = y_tuple
        elif not same_float_sequence(file_y_reference, y_tuple, tolerance):
            result.add_error(f"第 {line_number} 行的 y 锚点与本文件第一条线不一致")

    result.y_reference = file_y_reference
    return result


def scan_images(image_dir: Path) -> tuple[list[Path], dict[str, list[Path]]]:
    images = sorted(
        path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for image_path in images:
        by_stem[image_path.stem].append(image_path)
    return images, by_stem


def find_image_for_label(
    label_path: Path,
    label_dir: Path,
    image_dir: Path,
    images_by_stem: dict[str, list[Path]],
) -> ImageMatch:
    relative_label = label_path.relative_to(label_dir)
    relative_without_suffix = relative_label.with_suffix("")

    exact_candidates = [
        (image_dir / relative_without_suffix).with_suffix(suffix)
        for suffix in sorted(IMAGE_SUFFIXES)
    ]
    exact_matches = [path for path in exact_candidates if path.is_file()]

    if len(exact_matches) == 1:
        return ImageMatch(label_path, exact_matches[0], "matched", exact_matches)
    if len(exact_matches) > 1:
        return ImageMatch(label_path, None, "ambiguous", exact_matches)

    stem_matches = images_by_stem.get(label_path.stem, [])
    if len(stem_matches) == 1:
        return ImageMatch(label_path, stem_matches[0], "matched", stem_matches)
    if len(stem_matches) > 1:
        return ImageMatch(label_path, None, "ambiguous", stem_matches)
    return ImageMatch(label_path, None, "missing", [])


def clear_output_dir(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    forbidden = {Path("/").resolve(), PROJECT_ROOT.resolve(), (PROJECT_ROOT / "datasets").resolve()}
    if resolved in forbidden or len(resolved.parts) < 3:
        raise ValueError(f"拒绝清空危险目录：{resolved}")
    if output_dir.exists():
        shutil.rmtree(output_dir)


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(lines)
    path.write_text(content + ("\n" if content else ""), encoding="utf-8")


def format_label_details(result: LabelResult, label_dir: Path) -> list[str]:
    relative = result.path.relative_to(label_dir)
    lines = [f"[{relative}]"]
    if result.empty:
        lines.append("  WARNING: 空标签文件")
    for warning in result.warnings:
        lines.append(f"  WARNING: {warning}")
    for error in result.errors:
        lines.append(f"  ERROR: {error}")
    return lines


def main() -> int:
    args = parse_args()
    label_dir = args.label_dir.expanduser().resolve()
    image_dir = args.image_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not label_dir.is_dir():
        print(f"错误：标签目录不存在：{label_dir}", file=sys.stderr)
        return 2
    if not image_dir.is_dir():
        print(f"错误：图片目录不存在：{image_dir}", file=sys.stderr)
        return 2
    if args.row_anchors <= 0 or args.num_lanes <= 0:
        print("错误：row-anchors 和 num-lanes 必须大于 0", file=sys.stderr)
        return 2

    label_files = sorted(path for path in label_dir.rglob("*.txt") if path.is_file())
    if not label_files:
        print(f"错误：没有在 {label_dir} 中找到 .txt 标签", file=sys.stderr)
        return 2

    if args.clear_output and not args.dry_run:
        clear_output_dir(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / "_reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"标签目录：{label_dir}")
    print(f"图片目录：{image_dir}")
    print(f"输出目录：{output_dir}")
    print(f"找到标签：{len(label_files)} 个")

    results: list[LabelResult] = []
    global_y_reference: tuple[float, ...] | None = None
    global_y_source: Path | None = None

    for label_path in label_files:
        result = check_label_file(
            label_path=label_path,
            row_anchors=args.row_anchors,
            num_lanes=args.num_lanes,
            y_order=args.y_order,
            tolerance=args.tolerance,
        )
        if result.y_reference is not None:
            if global_y_reference is None:
                global_y_reference = result.y_reference
                global_y_source = label_path
            elif not same_float_sequence(global_y_reference, result.y_reference, args.tolerance):
                source_name = global_y_source.relative_to(label_dir) if global_y_source else "参考文件"
                result.add_error(f"本文件 y 锚点与全局参考文件 {source_name} 不一致")
        results.append(result)

    valid_results = [result for result in results if result.valid]
    invalid_results = [result for result in results if not result.valid]
    empty_results = [result for result in results if result.empty]

    lane_file_counts = {lane_id: 0 for lane_id in range(args.num_lanes)}
    for result in results:
        for lane_id in set(result.lane_ids):
            lane_file_counts[lane_id] += 1

    images, images_by_stem = scan_images(image_dir)
    matches = [find_image_for_label(result.path, label_dir, image_dir, images_by_stem) for result in results]
    matched = [match for match in matches if match.status == "matched"]
    missing = [match for match in matches if match.status == "missing"]
    ambiguous = [match for match in matches if match.status == "ambiguous"]

    result_by_path = {result.path: result for result in results}
    copied: list[tuple[Path, Path]] = []
    skipped_invalid: list[Path] = []

    for match in matched:
        assert match.image_path is not None
        label_result = result_by_path[match.label_path]
        if args.only_valid and not label_result.valid:
            skipped_invalid.append(match.label_path)
            continue

        relative_image = match.image_path.relative_to(image_dir)
        destination = output_dir / relative_image
        if not args.dry_run:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(match.image_path, destination)
        copied.append((match.image_path, destination))

    summary_lines = [
        "Lane Robot 标签检查与图片收集报告",
        "=" * 44,
        f"标签目录: {label_dir}",
        f"图片目录: {image_dir}",
        f"输出目录: {output_dir}",
        "",
        f"标签文件总数: {len(results)}",
        f"格式正确: {len(valid_results)}",
        f"格式错误: {len(invalid_results)}",
        f"空标签文件: {len(empty_results)}",
        f"扫描到的图片总数: {len(images)}",
        f"成功匹配图片: {len(matched)}",
        f"缺少对应图片: {len(missing)}",
        f"同名图片不唯一: {len(ambiguous)}",
        f"复制图片: {len(copied)}" + ("（dry-run，未实际复制）" if args.dry_run else ""),
        f"因 --only-valid 跳过: {len(skipped_invalid)}",
        "",
        "各槽位出现于多少个标签文件：",
    ]
    for lane_id in range(args.num_lanes):
        summary_lines.append(f"  {lane_id} ({LANE_NAMES.get(lane_id, 'unknown')}): {lane_file_counts[lane_id]}")
    if global_y_reference:
        summary_lines.extend(
            [
                "",
                f"全局 y 锚点数量: {len(global_y_reference)}",
                f"全局 y 起点/终点: {global_y_reference[0]:.6f} -> {global_y_reference[-1]:.6f}",
            ]
        )

    invalid_detail_lines: list[str] = []
    for result in invalid_results:
        invalid_detail_lines.extend(format_label_details(result, label_dir))
        invalid_detail_lines.append("")

    warning_detail_lines: list[str] = []
    for result in results:
        if result.warnings:
            warning_detail_lines.extend(format_label_details(result, label_dir))
            warning_detail_lines.append("")

    write_lines(report_dir / "summary.txt", summary_lines)
    write_lines(
        report_dir / "valid_labels.txt",
        [str(result.path.relative_to(label_dir)) for result in valid_results],
    )
    write_lines(report_dir / "invalid_labels.txt", invalid_detail_lines)
    write_lines(report_dir / "warnings.txt", warning_detail_lines)
    write_lines(
        report_dir / "missing_images.txt",
        [str(match.label_path.relative_to(label_dir)) for match in missing],
    )
    write_lines(
        report_dir / "ambiguous_images.txt",
        [
            f"{match.label_path.relative_to(label_dir)} -> "
            + " | ".join(str(path.relative_to(image_dir)) for path in match.candidates)
            for match in ambiguous
        ],
    )
    write_lines(
        report_dir / "copied_images.txt",
        [f"{source.relative_to(image_dir)} -> {destination.relative_to(output_dir)}" for source, destination in copied],
    )
    write_lines(
        report_dir / "skipped_invalid_labels.txt",
        [str(path.relative_to(label_dir)) for path in skipped_invalid],
    )

    print("\n" + "\n".join(summary_lines[5:]))
    print(f"\n详细报告：{report_dir}")

    if invalid_results or missing or ambiguous:
        print("检查完成，但存在格式错误、缺图或同名图片冲突。", file=sys.stderr)
        return 1

    print("检查通过：未发现格式错误、缺图或同名图片冲突。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
