from pathlib import Path


DATASETS_DIR = Path("/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/datasets")
REPORT_PATH = DATASETS_DIR / "empty_labels_report.txt"


def find_label_dirs(datasets_dir: Path) -> list[Path]:
    """自动查找可能存在的标签目录。"""
    candidates = [
        datasets_dir / "labels_corrected" / "train",
        datasets_dir / "labels_corrected" / "valid",
        datasets_dir / "labels" / "train",
        datasets_dir / "labels" / "valid",
    ]
    return [path for path in candidates if path.is_dir()]


def is_empty_label(label_path: Path) -> bool:
    """
    空标签定义：
    1. 文件大小为 0；
    2. 文件只包含空格、制表符或换行。
    """
    try:
        content = label_path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        content = label_path.read_text(
            encoding="utf-8-sig",
            errors="replace",
        )

    return not content.strip()


def main() -> None:
    if not DATASETS_DIR.is_dir():
        raise SystemExit(f"错误：数据集目录不存在：{DATASETS_DIR}")

    label_dirs = find_label_dirs(DATASETS_DIR)

    if not label_dirs:
        raise SystemExit(
            "错误：没有找到标签目录。\n"
            "检查过：datasets/labels_corrected/{train,valid} "
            "和 datasets/labels/{train,valid}"
        )

    all_labels: list[Path] = []
    empty_labels: list[Path] = []

    print("=" * 80)
    print("标签空文件检查")
    print(f"数据集目录：{DATASETS_DIR}")
    print("=" * 80)

    for label_dir in label_dirs:
        labels = sorted(
            path
            for path in label_dir.rglob("*.txt")
            if path.is_file()
        )

        current_empty = [
            label_path
            for label_path in labels
            if is_empty_label(label_path)
        ]

        all_labels.extend(labels)
        empty_labels.extend(current_empty)

        print(f"\n检查目录：{label_dir}")
        print(f"标签总数：{len(labels)}")
        print(f"空标签数：{len(current_empty)}")

        for label_path in current_empty:
            print(f"  空标签：{label_path}")

    report_lines = [
        "Lane Robot 空标签检查报告",
        f"数据集目录：{DATASETS_DIR}",
        f"检查标签总数：{len(all_labels)}",
        f"空标签总数：{len(empty_labels)}",
        "",
    ]

    if empty_labels:
        report_lines.append("空标签文件：")
        report_lines.extend(str(path) for path in empty_labels)
    else:
        report_lines.append("未发现空标签。")

    REPORT_PATH.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 80)
    print(f"检查标签总数：{len(all_labels)}")
    print(f"空标签总数：{len(empty_labels)}")
    print(f"报告已保存：{REPORT_PATH}")
    print("=" * 80)

    if not empty_labels:
        print("结果：没有发现空标签。")


if __name__ == "__main__":
    main()
