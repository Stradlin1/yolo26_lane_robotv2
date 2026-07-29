#!/usr/bin/env python3

from pathlib import Path
import argparse


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}


def main():
    parser = argparse.ArgumentParser(
        description="删除没有对应图片的 TXT 标签"
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/datasets/images"),
        help="图片根目录",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path("/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/datasets/labels_corrected"),
        help="标签根目录",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="实际删除；不加该参数时只预览",
    )
    args = parser.parse_args()

    images_root = args.images.resolve()
    labels_root = args.labels.resolve()

    if not images_root.is_dir():
        raise RuntimeError(f"图片目录不存在: {images_root}")

    if not labels_root.is_dir():
        raise RuntimeError(f"标签目录不存在: {labels_root}")

    label_files = sorted(labels_root.rglob("*.txt"))

    orphan_labels = []
    matched_count = 0

    for label_path in label_files:
        relative_path = label_path.relative_to(labels_root)

        # 标签：
        # labels_corrected/train/example.txt
        #
        # 对应图片：
        # images/train/example.jpg/png/...
        image_parent = images_root / relative_path.parent
        image_stem = relative_path.stem

        image_exists = any(
            (image_parent / f"{image_stem}{ext}").is_file()
            for ext in IMAGE_EXTENSIONS
        )

        # 同时兼容大写扩展名，例如 JPG、PNG
        if not image_exists and image_parent.is_dir():
            image_exists = any(
                p.is_file()
                and p.stem == image_stem
                and p.suffix.lower() in IMAGE_EXTENSIONS
                for p in image_parent.iterdir()
            )

        if image_exists:
            matched_count += 1
        else:
            orphan_labels.append(label_path)

    print("=" * 70)
    print(f"图片目录: {images_root}")
    print(f"标签目录: {labels_root}")
    print(f"TXT 标签总数: {len(label_files)}")
    print(f"有对应图片: {matched_count}")
    print(f"没有对应图片: {len(orphan_labels)}")
    print("=" * 70)

    if not orphan_labels:
        print("没有发现需要删除的标签。")
        return

    for path in orphan_labels:
        relative = path.relative_to(labels_root)

        if args.delete:
            path.unlink()
            print(f"[已删除] {relative}")
        else:
            print(f"[待删除] {relative}")

    if args.delete:
        # 删除因标签删除而留下的空文件夹
        directories = sorted(
            (p for p in labels_root.rglob("*") if p.is_dir()),
            key=lambda p: len(p.parts),
            reverse=True,
        )

        for directory in directories:
            try:
                directory.rmdir()
                print(f"[删除空目录] {directory.relative_to(labels_root)}")
            except OSError:
                pass

        print(f"\n处理完成，共删除 {len(orphan_labels)} 个标签。")
    else:
        print("\n当前仅为预览，没有删除文件。")
        print("确认无误后增加 --delete 参数执行实际删除。")


if __name__ == "__main__":
    main()
