from __future__ import annotations

import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


# ============================================================
# 配置
# ============================================================

DATASET_ROOT = Path(
    r"D:\download\ULTRALYTICS_LANE_ROBOT-independent\datasets"
)

LABEL_DIRS = [
    DATASET_ROOT / "labels" / "train",
    DATASET_ROOT / "labels" / "valid",
]

# 你的模型配置
X_GRIDS = 160
ROW_ANCHORS = 56

# 目标 y 范围：从 0.67 到 1.0
# 输出顺序：
# row 0  -> y=0.670000
# row 55 -> y=1.000000
Y_START = 0.67
Y_END = 1.0

# 输出小数位数
PRECISION = 6

# True：备份后覆盖原标签
# False：写入 labels_067_100 目录，不覆盖原标签
OVERWRITE_SOURCE = True


Point = Tuple[float, float]
OptionalPoint = Optional[Point]


def linspace(start: float, end: float, count: int) -> List[float]:
    """不依赖 numpy 的 linspace。"""
    if count <= 0:
        raise ValueError("count 必须大于 0")

    if count == 1:
        return [float(start)]

    step = (end - start) / float(count - 1)
    return [start + step * index for index in range(count)]


TARGET_Y_ROWS = linspace(
    start=Y_START,
    end=Y_END,
    count=ROW_ANCHORS,
)


def parse_lane_line(
    line: str,
    *,
    file_path: Path,
    line_number: int,
) -> Tuple[int, List[OptionalPoint]]:
    """
    输入格式：

        lane_id x1 y1 x2 y2 ...

    x < 0 表示该点不存在，并作为折线断点。
    """
    fields = line.replace(",", " ").split()

    if len(fields) < 3:
        raise ValueError(
            f"{file_path} 第 {line_number} 行字段不足：{line}"
        )

    try:
        lane_id = int(float(fields[0]))
    except ValueError as exc:
        raise ValueError(
            f"{file_path} 第 {line_number} 行 lane_id 无效：{fields[0]}"
        ) from exc

    try:
        values = [float(value) for value in fields[1:]]
    except ValueError as exc:
        raise ValueError(
            f"{file_path} 第 {line_number} 行存在非数字字段"
        ) from exc

    if len(values) % 2 != 0:
        raise ValueError(
            f"{file_path} 第 {line_number} 行坐标字段数量为奇数："
            f"{len(values)}"
        )

    points: List[OptionalPoint] = []

    for index in range(0, len(values), 2):
        x = values[index]
        y = values[index + 1]

        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError(
                f"{file_path} 第 {line_number} 行第 "
                f"{index // 2 + 1} 个点包含 NaN 或 Inf"
            )

        # x=-1 或其他负值表示无效点/断点
        if x < 0:
            points.append(None)
            continue

        if not 0.0 <= x <= 1.0:
            raise ValueError(
                f"{file_path} 第 {line_number} 行第 "
                f"{index // 2 + 1} 个点 x 越界：{x}"
            )

        if not 0.0 <= y <= 1.0:
            raise ValueError(
                f"{file_path} 第 {line_number} 行第 "
                f"{index // 2 + 1} 个点 y 越界：{y}"
            )

        points.append((x, y))

    return lane_id, points


def split_segments(
    points: Sequence[OptionalPoint],
) -> List[List[Point]]:
    """
    按 None 拆分折线。

    不允许跨过 x=-1 的缺失区域连接车道线。
    """
    segments: List[List[Point]] = []
    current: List[Point] = []

    for point in points:
        if point is None:
            if current:
                segments.append(current)
                current = []
            continue

        current.append(point)

    if current:
        segments.append(current)

    return segments


def deduplicate_x(
    values: Sequence[float],
    epsilon: float = 1e-9,
) -> List[float]:
    result: List[float] = []

    for value in values:
        if not any(abs(value - existing) <= epsilon for existing in result):
            result.append(value)

    return result


def find_horizontal_intersections(
    segments: Sequence[Sequence[Point]],
    target_y: float,
    epsilon: float = 1e-9,
) -> List[float]:
    """
    求车道折线和水平线 y=target_y 的所有交点横坐标。
    """
    intersections: List[float] = []

    for segment in segments:
        if not segment:
            continue

        # 单独一个点
        if len(segment) == 1:
            x, y = segment[0]

            if abs(y - target_y) <= epsilon:
                intersections.append(x)

            continue

        for point1, point2 in zip(segment[:-1], segment[1:]):
            x1, y1 = point1
            x2, y2 = point2

            min_y = min(y1, y2) - epsilon
            max_y = max(y1, y2) + epsilon

            if target_y < min_y or target_y > max_y:
                continue

            delta_y = y2 - y1

            # 原折线中出现水平线段
            if abs(delta_y) <= epsilon:
                if abs(target_y - y1) <= epsilon:
                    intersections.append((x1 + x2) * 0.5)

                continue

            ratio = (target_y - y1) / delta_y

            if -epsilon <= ratio <= 1.0 + epsilon:
                ratio = max(0.0, min(1.0, ratio))
                x = x1 + ratio * (x2 - x1)

                if 0.0 <= x <= 1.0:
                    intersections.append(x)

    return deduplicate_x(intersections)


def select_intersection(
    candidates: Sequence[float],
    previous_x: Optional[float],
) -> Optional[float]:
    """
    正常车道每个 y 只有一个交点。

    如果出现多个交点：
    - 优先选择最接近上一行交点的位置，保持曲线连续；
    - 第一行没有历史位置时，选择中间位置。
    """
    if not candidates:
        return None

    if previous_x is not None:
        return min(
            candidates,
            key=lambda value: abs(value - previous_x),
        )

    sorted_candidates = sorted(candidates)
    middle = len(sorted_candidates) // 2

    if len(sorted_candidates) % 2 == 1:
        return sorted_candidates[middle]

    return (
        sorted_candidates[middle - 1]
        + sorted_candidates[middle]
    ) * 0.5


def snap_x_to_classification_grid(
    x_normalized: float,
) -> Tuple[int, float]:
    """
    吸附到 160 个有效横坐标分类点。

    有效分类：
        0～159

    no-lane 分类：
        160

    TXT 中不写 no-lane=160，而是继续使用 x=-1。
    """
    grid_id = int(round(x_normalized * (X_GRIDS - 1)))
    grid_id = max(0, min(X_GRIDS - 1, grid_id))

    snapped_x = grid_id / float(X_GRIDS - 1)

    return grid_id, snapped_x


def convert_lane_points(
    source_points: Sequence[OptionalPoint],
) -> List[OptionalPoint]:
    """
    将原始折线转换到 y=0.67～1.0 的固定 56 行，
    并把 x 吸附到 160 个分类位置。
    """
    segments = split_segments(source_points)

    converted_points: List[OptionalPoint] = []
    previous_raw_x: Optional[float] = None

    for target_y in TARGET_Y_ROWS:
        candidates = find_horizontal_intersections(
            segments=segments,
            target_y=target_y,
        )

        raw_x = select_intersection(
            candidates=candidates,
            previous_x=previous_raw_x,
        )

        if raw_x is None:
            converted_points.append(None)
            continue

        _, snapped_x = snap_x_to_classification_grid(raw_x)

        converted_points.append(
            (
                snapped_x,
                target_y,
            )
        )

        previous_raw_x = raw_x

    return converted_points


def format_lane_line(
    lane_id: int,
    points: Sequence[OptionalPoint],
) -> str:
    if len(points) != ROW_ANCHORS:
        raise ValueError(
            f"lane_id={lane_id} 转换后应有 {ROW_ANCHORS} 个点，"
            f"实际为 {len(points)}"
        )

    output_fields = [str(lane_id)]

    for row_index, point in enumerate(points):
        target_y = TARGET_Y_ROWS[row_index]

        if point is None:
            output_fields.extend(
                [
                    "-1",
                    f"{target_y:.{PRECISION}f}",
                ]
            )
            continue

        x, y = point

        output_fields.extend(
            [
                f"{x:.{PRECISION}f}",
                f"{y:.{PRECISION}f}",
            ]
        )

    return " ".join(output_fields)


def convert_label_file(
    source_path: Path,
    destination_path: Path,
) -> Tuple[int, int]:
    """
    返回：
        lane 数量
        有效交点总数
    """
    raw_lines = source_path.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    output_lines: List[str] = []
    lane_ids = set()
    valid_point_count = 0

    for line_number, raw_line in enumerate(raw_lines, start=1):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        lane_id, source_points = parse_lane_line(
            line,
            file_path=source_path,
            line_number=line_number,
        )

        if lane_id in lane_ids:
            raise ValueError(
                f"{source_path} 中 lane_id={lane_id} 重复"
            )

        lane_ids.add(lane_id)

        converted_points = convert_lane_points(source_points)

        valid_point_count += sum(
            point is not None
            for point in converted_points
        )

        output_lines.append(
            format_lane_line(
                lane_id=lane_id,
                points=converted_points,
            )
        )

    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 先写临时文件，再原子替换，避免中断时损坏标签
    temporary_path = destination_path.with_suffix(
        destination_path.suffix + ".tmp"
    )

    temporary_path.write_text(
        "\n".join(output_lines)
        + ("\n" if output_lines else ""),
        encoding="utf-8",
    )

    temporary_path.replace(destination_path)

    return len(output_lines), valid_point_count


def prepare_backup() -> Optional[Path]:
    if not OVERWRITE_SOURCE:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    backup_root = DATASET_ROOT / (
        f"labels_backup_before_067_100_{timestamp}"
    )

    for label_dir in LABEL_DIRS:
        if not label_dir.exists():
            raise FileNotFoundError(
                f"标签目录不存在：{label_dir}"
            )

        split_name = label_dir.name
        backup_dir = backup_root / split_name

        shutil.copytree(
            label_dir,
            backup_dir,
        )

    return backup_root


def get_destination_dir(
    source_dir: Path,
) -> Path:
    if OVERWRITE_SOURCE:
        return source_dir

    return (
        DATASET_ROOT
        / "labels_067_100"
        / source_dir.name
    )


def main() -> None:
    print("=" * 70)
    print("车道线标签转换")
    print(f"目标 y：{Y_START} -> {Y_END}")
    print(f"目标行数：{ROW_ANCHORS}")
    print(f"横向分类点：{X_GRIDS}")
    print(f"覆盖原标签：{OVERWRITE_SOURCE}")
    print("=" * 70)

    backup_root = prepare_backup()

    if backup_root is not None:
        print(f"原始标签已备份到：{backup_root}")

    total_files = 0
    total_lanes = 0
    total_valid_points = 0
    failed_files: List[Tuple[Path, str]] = []

    for source_dir in LABEL_DIRS:
        if not source_dir.exists():
            raise FileNotFoundError(
                f"目录不存在：{source_dir}"
            )

        destination_dir = get_destination_dir(source_dir)

        label_files = sorted(source_dir.glob("*.txt"))

        print()
        print(
            f"处理 {source_dir.name}："
            f"{len(label_files)} 个 TXT"
        )

        for index, source_path in enumerate(label_files, start=1):
            destination_path = (
                destination_dir / source_path.name
            )

            try:
                lane_count, valid_count = convert_label_file(
                    source_path=source_path,
                    destination_path=destination_path,
                )

                total_files += 1
                total_lanes += lane_count
                total_valid_points += valid_count

                print(
                    f"[{index:05d}/{len(label_files):05d}] "
                    f"成功：{source_path.name}，"
                    f"lane={lane_count}，"
                    f"有效点={valid_count}"
                )

            except Exception as exc:
                failed_files.append(
                    (
                        source_path,
                        str(exc),
                    )
                )

                print(
                    f"[{index:05d}/{len(label_files):05d}] "
                    f"失败：{source_path.name}：{exc}"
                )

    print()
    print("=" * 70)
    print("转换完成")
    print(f"成功文件：{total_files}")
    print(f"车道总数：{total_lanes}")
    print(f"有效交点：{total_valid_points}")
    print(f"失败文件：{len(failed_files)}")

    if backup_root is not None:
        print(f"备份目录：{backup_root}")

    if not OVERWRITE_SOURCE:
        print(
            "输出目录："
            f"{DATASET_ROOT / 'labels_067_100'}"
        )

    if failed_files:
        print()
        print("失败详情：")

        for file_path, error in failed_files:
            print(f"- {file_path}: {error}")

        raise SystemExit(1)

    print("=" * 70)


if __name__ == "__main__":
    main()