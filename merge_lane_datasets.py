#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将两个 Lane Robot 数据集合并到统一目录。

源目录：
1. /home/xhm/Desktop/lane./datasets
2. /home/xhm/Desktop/channel-labeled/datasets

目标目录：
/home/xhm/Desktop/all/datasets
├── images/
│   ├── train/
│   └── valid/
└── labels_corrected/
    ├── train/
    └── valid/

工作流程：
1. 检查源目录结构。
2. 检查每个 split 中图片与标签是否按 stem 一一对应。
3. 检查两个源数据集之间是否存在同名样本冲突。
4. 检查目标目录中是否已有同名样本。
5. 所有检查全部通过后，才使用 shutil.move() 移动文件。

注意：
- 本脚本执行的是真实移动，不是复制。
- 空标签文件允许存在。
- 图片支持常见扩展名：jpg/jpeg/png/bmp/tif/tiff/webp。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SOURCE_DATASETS = [
    Path("/home/xhm/Desktop/lane./datasets"),
    Path("/home/xhm/Desktop/channel-labeled/datasets"),
]

DEST_DATASET = Path("/home/xhm/Desktop/all/datasets")
SPLITS = ("train", "valid")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    source_root: Path
    split: str
    stem: str
    image_path: Path
    label_path: Path


class DatasetCheckError(RuntimeError):
    """数据集预检查失败。"""


def list_image_files(directory: Path) -> list[Path]:
    """返回目录下所有受支持的图片文件，不递归。"""
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda p: p.name,
    )


def list_label_files(directory: Path) -> list[Path]:
    """返回目录下所有 txt 标签文件，不递归。"""
    return sorted(
        (path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".txt"),
        key=lambda p: p.name,
    )


def index_by_stem(paths: Iterable[Path], kind: str, directory: Path) -> dict[str, Path]:
    """按文件 stem 建索引，并检查同目录内 stem 是否重复。"""
    index: dict[str, Path] = {}
    duplicates: defaultdict[str, list[Path]] = defaultdict(list)

    for path in paths:
        if path.stem in index:
            duplicates[path.stem].append(index[path.stem])
            duplicates[path.stem].append(path)
        else:
            index[path.stem] = path

    if duplicates:
        lines = [f"{directory} 中存在重复 {kind} stem："]
        for stem, files in sorted(duplicates.items()):
            unique_files = sorted({str(p) for p in files})
            lines.append(f"  - {stem}: {', '.join(unique_files)}")
        raise DatasetCheckError("\n".join(lines))

    return index


def ensure_required_directories(source_root: Path) -> None:
    """检查源数据集的标准目录结构。"""
    required = []
    for split in SPLITS:
        required.extend(
            [
                source_root / "images" / split,
                source_root / "labels_corrected" / split,
            ]
        )

    missing = [path for path in required if not path.is_dir()]
    if missing:
        message = [f"源数据集目录结构不完整：{source_root}"]
        message.extend(f"  - 缺少目录：{path}" for path in missing)
        raise DatasetCheckError("\n".join(message))


def collect_source_samples(source_root: Path) -> list[Sample]:
    """检查单个源数据集并收集合法样本。"""
    ensure_required_directories(source_root)
    samples: list[Sample] = []

    for split in SPLITS:
        image_dir = source_root / "images" / split
        label_dir = source_root / "labels_corrected" / split

        image_index = index_by_stem(list_image_files(image_dir), "图片", image_dir)
        label_index = index_by_stem(list_label_files(label_dir), "标签", label_dir)

        image_stems = set(image_index)
        label_stems = set(label_index)

        missing_labels = sorted(image_stems - label_stems)
        missing_images = sorted(label_stems - image_stems)

        if missing_labels or missing_images:
            message = [f"图片和标签不匹配：{source_root} / {split}"]
            if missing_labels:
                message.append(f"  - 有图片但无标签，共 {len(missing_labels)} 个：")
                message.extend(f"      {stem}" for stem in missing_labels[:30])
                if len(missing_labels) > 30:
                    message.append(f"      ... 其余 {len(missing_labels) - 30} 个省略")
            if missing_images:
                message.append(f"  - 有标签但无图片，共 {len(missing_images)} 个：")
                message.extend(f"      {stem}" for stem in missing_images[:30])
                if len(missing_images) > 30:
                    message.append(f"      ... 其余 {len(missing_images) - 30} 个省略")
            raise DatasetCheckError("\n".join(message))

        for stem in sorted(image_stems):
            image_path = image_index[stem]
            label_path = label_index[stem]

            if image_path.stat().st_size == 0:
                raise DatasetCheckError(f"发现空图片文件：{image_path}")

            samples.append(
                Sample(
                    source_root=source_root,
                    split=split,
                    stem=stem,
                    image_path=image_path,
                    label_path=label_path,
                )
            )

    return samples


def check_cross_source_conflicts(samples: list[Sample]) -> None:
    """检查不同源数据集在同一 split 中是否存在同名样本。"""
    owners: dict[tuple[str, str], Sample] = {}
    conflicts: list[tuple[Sample, Sample]] = []

    for sample in samples:
        key = (sample.split, sample.stem)
        if key in owners:
            conflicts.append((owners[key], sample))
        else:
            owners[key] = sample

    if conflicts:
        message = ["两个源数据集之间存在同名样本，无法安全合并："]
        for first, second in conflicts[:50]:
            message.append(
                f"  - split={first.split}, stem={first.stem}\n"
                f"      {first.image_path}\n"
                f"      {second.image_path}"
            )
        if len(conflicts) > 50:
            message.append(f"  ... 其余 {len(conflicts) - 50} 个冲突省略")
        raise DatasetCheckError("\n".join(message))


def check_destination_conflicts(samples: list[Sample], dest_root: Path) -> None:
    """检查目标目录是否已有可能冲突的同 stem 文件。"""
    conflicts: list[str] = []

    for sample in samples:
        dest_image_dir = dest_root / "images" / sample.split
        dest_label_dir = dest_root / "labels_corrected" / sample.split

        # 标签固定为 stem.txt。
        dest_label = dest_label_dir / f"{sample.stem}.txt"
        if dest_label.exists():
            conflicts.append(str(dest_label))

        # 图片扩展名可能不同，所以检查所有受支持图片扩展名的同 stem 文件。
        for extension in IMAGE_EXTENSIONS:
            candidate = dest_image_dir / f"{sample.stem}{extension}"
            if candidate.exists():
                conflicts.append(str(candidate))

    if conflicts:
        unique_conflicts = sorted(set(conflicts))
        message = ["目标数据集中已存在同名文件，脚本不会覆盖："]
        message.extend(f"  - {path}" for path in unique_conflicts[:100])
        if len(unique_conflicts) > 100:
            message.append(f"  ... 其余 {len(unique_conflicts) - 100} 个冲突省略")
        raise DatasetCheckError("\n".join(message))


def ensure_destination_directories(dest_root: Path) -> None:
    """创建目标数据集目录结构。"""
    for split in SPLITS:
        (dest_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest_root / "labels_corrected" / split).mkdir(parents=True, exist_ok=True)


def print_summary(samples: list[Sample]) -> None:
    """打印移动前统计。"""
    counts: defaultdict[tuple[Path, str], int] = defaultdict(int)
    split_totals: defaultdict[str, int] = defaultdict(int)

    for sample in samples:
        counts[(sample.source_root, sample.split)] += 1
        split_totals[sample.split] += 1

    print("\n预检查通过，待移动样本统计：")
    for source_root in SOURCE_DATASETS:
        print(f"\n源数据集：{source_root}")
        for split in SPLITS:
            print(f"  {split}: {counts[(source_root, split)]} 对图片/标签")

    print("\n合并后新增：")
    for split in SPLITS:
        print(f"  {split}: {split_totals[split]} 对图片/标签")
    print(f"  总计: {len(samples)} 对图片/标签")


def move_samples(samples: list[Sample], dest_root: Path) -> None:
    """执行真实移动。预检查必须在调用该函数前全部完成。"""
    ensure_destination_directories(dest_root)

    moved_pairs = 0
    try:
        for sample in samples:
            dest_image = dest_root / "images" / sample.split / sample.image_path.name
            dest_label = dest_root / "labels_corrected" / sample.split / sample.label_path.name

            shutil.move(str(sample.image_path), str(dest_image))
            try:
                shutil.move(str(sample.label_path), str(dest_label))
            except Exception:
                # 尽可能回滚当前样本的图片，避免出现只有图片没有标签的半移动状态。
                if dest_image.exists() and not sample.image_path.exists():
                    shutil.move(str(dest_image), str(sample.image_path))
                raise

            moved_pairs += 1
            if moved_pairs % 100 == 0 or moved_pairs == len(samples):
                print(f"已移动 {moved_pairs}/{len(samples)} 对样本")

    except Exception as exc:
        raise RuntimeError(
            f"移动过程中发生错误。已完成 {moved_pairs}/{len(samples)} 对样本。\n"
            f"错误：{exc}\n"
            "此前已经成功移动的样本不会自动整体回滚，请根据终端输出检查目标目录。"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查并移动合并两个 Lane Robot 数据集")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="只执行检查和统计，不移动任何文件",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("源数据集：")
    for source in SOURCE_DATASETS:
        print(f"  - {source}")
    print(f"目标数据集：\n  - {DEST_DATASET}")

    try:
        all_samples: list[Sample] = []
        for source_root in SOURCE_DATASETS:
            print(f"\n正在检查：{source_root}")
            source_samples = collect_source_samples(source_root)
            all_samples.extend(source_samples)
            print(f"检查通过：{len(source_samples)} 对图片/标签")

        check_cross_source_conflicts(all_samples)
        check_destination_conflicts(all_samples, DEST_DATASET)
        print_summary(all_samples)

        if args.check_only:
            print("\n当前为 --check-only 模式，没有移动任何文件。")
            return 0

        if not all_samples:
            print("\n没有可移动的样本。")
            return 0

        print("\n所有检查均已通过，开始移动文件……")
        move_samples(all_samples, DEST_DATASET)
        print("\n数据集整合完成。")
        return 0

    except DatasetCheckError as exc:
        print(f"\n预检查失败，未移动任何文件：\n{exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"\n执行失败：\n{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
