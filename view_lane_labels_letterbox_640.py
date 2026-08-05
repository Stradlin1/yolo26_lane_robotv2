from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# ============================================================
# 默认路径
# ============================================================

PROJECT_ROOT = Path(
    r"D:\download\ULTRALYTICS_LANE_ROBOT-independent"
)

DEFAULT_SPLIT = "valid"

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# OpenCV 使用 BGR。
LANE_COLORS = [
    (0, 255, 0),      # lane_id 0
    (0, 255, 255),    # lane_id 1
    (255, 128, 0),    # lane_id 2
    (0, 0, 255),      # lane_id 3
    (255, 0, 255),
    (255, 255, 0),
]

# 模型/标签参数
PANEL_WIDTH = 640
PANEL_HEIGHT = 640

X_GRIDS = 160
ROW_ANCHORS = 56
Y_START = 0.67
Y_END = 1.0

# 标准 Ultralytics LetterBox 常用填充值。
LETTERBOX_COLOR = (114, 114, 114)


@dataclass(frozen=True)
class LetterBoxInfo:
    scale: float
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int


@dataclass(frozen=True)
class LanePoint:
    x_norm: float
    y_norm: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "显示 640x640 LetterBox 图像，并同步绘制车道线标签。"
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="项目根目录。",
    )
    parser.add_argument(
        "--split",
        choices=("train", "valid"),
        default=DEFAULT_SPLIT,
        help="查看 train 或 valid。",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="自定义图片目录；提供后覆盖 --root/--split。",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="自定义标签目录；提供后覆盖 --root/--split。",
    )
    parser.add_argument(
        "--show-anchors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="显示 y=0.67~1.0 的 56 条行锚点。",
    )
    parser.add_argument(
        "--show-grid",
        action="store_true",
        help="显示 160 列分类网格；默认只每 10 列画一条。",
    )
    parser.add_argument(
        "--point-radius",
        type=int,
        default=4,
        help="标签点半径。",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="车道线宽度。",
    )

    return parser.parse_args()


def resolve_directories(
    args: argparse.Namespace,
) -> tuple[Path, Path]:
    if args.images is not None:
        image_dir = args.images
    else:
        image_dir = (
            args.root
            / "datasets"
            / "images"
            / args.split
        )

    if args.labels is not None:
        label_dir = args.labels
    else:
        label_dir = (
            args.root
            / "datasets"
            / "labels"
            / args.split
        )

    return image_dir, label_dir


def collect_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(
            f"图片目录不存在：{image_dir}"
        )

    images = sorted(
        path
        for path in image_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if not images:
        raise RuntimeError(
            f"目录中没有找到图片：{image_dir}"
        )

    return images


def read_image(path: Path) -> np.ndarray:
    """
    使用 imdecode，兼容 Windows 中文路径。
    """
    buffer = np.fromfile(
        str(path),
        dtype=np.uint8,
    )

    image = cv2.imdecode(
        buffer,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"图片读取失败：{path}"
        )

    return image


def letterbox_640(
    image: np.ndarray,
) -> tuple[np.ndarray, LetterBoxInfo]:
    """
    标准居中 LetterBox：

        原图按比例缩放
        → 居中填充到 640x640

    对 1920x1080：
        resize = 640x360
        pad_top = 140
        pad_bottom = 140
    """
    source_height, source_width = image.shape[:2]

    scale = min(
        PANEL_WIDTH / source_width,
        PANEL_HEIGHT / source_height,
    )

    resized_width = int(
        round(source_width * scale)
    )
    resized_height = int(
        round(source_height * scale)
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    total_pad_width = PANEL_WIDTH - resized_width
    total_pad_height = PANEL_HEIGHT - resized_height

    pad_left = total_pad_width // 2
    pad_right = total_pad_width - pad_left
    pad_top = total_pad_height // 2
    pad_bottom = total_pad_height - pad_top

    canvas = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=LETTERBOX_COLOR,
    )

    if canvas.shape[:2] != (
        PANEL_HEIGHT,
        PANEL_WIDTH,
    ):
        raise RuntimeError(
            f"LetterBox 输出尺寸错误："
            f"{canvas.shape[:2]}"
        )

    info = LetterBoxInfo(
        scale=scale,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )

    return canvas, info


def parse_label_file(
    label_path: Path,
) -> tuple[
    dict[int, list[LanePoint | None]],
    list[str],
]:
    """
    标签格式：

        lane_id x1 y1 x2 y2 ... x56 y56

    x < 0 表示无效点。
    """
    lanes: dict[int, list[LanePoint | None]] = {}
    warnings: list[str] = []

    if not label_path.exists():
        warnings.append(
            f"缺少标签：{label_path.name}"
        )
        return lanes, warnings

    raw_lines = label_path.read_text(
        encoding="utf-8-sig"
    ).splitlines()

    for line_number, raw_line in enumerate(
        raw_lines,
        start=1,
    ):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        fields = line.replace(",", " ").split()

        try:
            lane_id = int(float(fields[0]))
        except (ValueError, IndexError):
            warnings.append(
                f"第 {line_number} 行 lane_id 无效"
            )
            continue

        try:
            values = [
                float(value)
                for value in fields[1:]
            ]
        except ValueError:
            warnings.append(
                f"第 {line_number} 行包含非数字字段"
            )
            continue

        if len(values) % 2 != 0:
            warnings.append(
                f"第 {line_number} 行坐标字段为奇数"
            )
            values = values[:-1]

        point_count = len(values) // 2

        if point_count != ROW_ANCHORS:
            warnings.append(
                f"lane_id={lane_id} 有 {point_count} 个点，"
                f"不是 {ROW_ANCHORS}"
            )

        points: list[LanePoint | None] = []

        for point_index in range(point_count):
            x_norm = values[point_index * 2]
            y_norm = values[point_index * 2 + 1]

            if x_norm < 0:
                points.append(None)
                continue

            if not (
                0.0 <= x_norm <= 1.0
                and 0.0 <= y_norm <= 1.0
            ):
                warnings.append(
                    f"lane_id={lane_id} "
                    f"第 {point_index + 1} 点越界："
                    f"x={x_norm:.6f}, "
                    f"y={y_norm:.6f}"
                )
                points.append(None)
                continue

            points.append(
                LanePoint(
                    x_norm=x_norm,
                    y_norm=y_norm,
                )
            )

        if lane_id in lanes:
            warnings.append(
                f"lane_id={lane_id} 重复，后者覆盖前者"
            )

        lanes[lane_id] = points

    return lanes, warnings


def normalized_point_to_letterbox(
    point: LanePoint,
    source_width: int,
    source_height: int,
    info: LetterBoxInfo,
) -> tuple[int, int]:
    """
    标签坐标是相对原图归一化坐标。

    先还原到原图像素，再执行 LetterBox 的缩放和平移。
    """
    source_x = (
        point.x_norm
        * (source_width - 1)
    )
    source_y = (
        point.y_norm
        * (source_height - 1)
    )

    letterbox_x = (
        source_x * info.scale
        + info.pad_left
    )
    letterbox_y = (
        source_y * info.scale
        + info.pad_top
    )

    x = int(round(letterbox_x))
    y = int(round(letterbox_y))

    return (
        int(np.clip(x, 0, PANEL_WIDTH - 1)),
        int(np.clip(y, 0, PANEL_HEIGHT - 1)),
    )


def y_norm_to_letterbox_y(
    y_norm: float,
    source_height: int,
    info: LetterBoxInfo,
) -> int:
    source_y = y_norm * (source_height - 1)

    letterbox_y = (
        source_y * info.scale
        + info.pad_top
    )

    return int(
        np.clip(
            round(letterbox_y),
            0,
            PANEL_HEIGHT - 1,
        )
    )


def x_norm_to_letterbox_x(
    x_norm: float,
    source_width: int,
    info: LetterBoxInfo,
) -> int:
    source_x = x_norm * (source_width - 1)

    letterbox_x = (
        source_x * info.scale
        + info.pad_left
    )

    return int(
        np.clip(
            round(letterbox_x),
            0,
            PANEL_WIDTH - 1,
        )
    )


def iter_valid_segments(
    points: Iterable[LanePoint | None],
) -> Iterable[list[LanePoint]]:
    current: list[LanePoint] = []

    for point in points:
        if point is None:
            if current:
                yield current
                current = []
            continue

        current.append(point)

    if current:
        yield current


def draw_content_boundary(
    panel: np.ndarray,
    info: LetterBoxInfo,
) -> None:
    x1 = info.pad_left
    y1 = info.pad_top
    x2 = (
        info.pad_left
        + info.resized_width
        - 1
    )
    y2 = (
        info.pad_top
        + info.resized_height
        - 1
    )

    cv2.rectangle(
        panel,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_row_anchors(
    panel: np.ndarray,
    source_height: int,
    info: LetterBoxInfo,
) -> None:
    rows = np.linspace(
        Y_START,
        Y_END,
        ROW_ANCHORS,
    )

    content_x1 = info.pad_left
    content_x2 = (
        info.pad_left
        + info.resized_width
        - 1
    )

    for row_id, y_norm in enumerate(rows):
        y = y_norm_to_letterbox_y(
            float(y_norm),
            source_height,
            info,
        )

        if row_id % 5 == 0:
            color = (125, 125, 125)
        else:
            color = (80, 80, 80)

        cv2.line(
            panel,
            (content_x1, y),
            (content_x2, y),
            color,
            1,
            cv2.LINE_AA,
        )


def draw_classification_grid(
    panel: np.ndarray,
    source_width: int,
    info: LetterBoxInfo,
) -> None:
    """
    为避免过度拥挤，只绘制每 10 个分类点中的一条。
    标签点仍然位于完整 160 分类网格上。
    """
    content_y1 = info.pad_top
    content_y2 = (
        info.pad_top
        + info.resized_height
        - 1
    )

    for grid_id in range(0, X_GRIDS, 10):
        x_norm = grid_id / float(X_GRIDS - 1)

        x = x_norm_to_letterbox_x(
            x_norm,
            source_width,
            info,
        )

        cv2.line(
            panel,
            (x, content_y1),
            (x, content_y2),
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )


def draw_lanes(
    panel: np.ndarray,
    lanes: dict[int, list[LanePoint | None]],
    source_width: int,
    source_height: int,
    info: LetterBoxInfo,
    point_radius: int,
    line_thickness: int,
) -> None:
    for lane_id in sorted(lanes):
        color = LANE_COLORS[
            lane_id % len(LANE_COLORS)
        ]

        points = lanes[lane_id]

        for segment in iter_valid_segments(points):
            pixels = [
                normalized_point_to_letterbox(
                    point,
                    source_width,
                    source_height,
                    info,
                )
                for point in segment
            ]

            if len(pixels) >= 2:
                polyline = np.asarray(
                    pixels,
                    dtype=np.int32,
                ).reshape(-1, 1, 2)

                cv2.polylines(
                    panel,
                    [polyline],
                    isClosed=False,
                    color=color,
                    thickness=line_thickness,
                    lineType=cv2.LINE_AA,
                )

            for pixel in pixels:
                cv2.circle(
                    panel,
                    pixel,
                    point_radius,
                    color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

                cv2.circle(
                    panel,
                    pixel,
                    point_radius + 1,
                    (0, 0, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        first_valid = next(
            (
                point
                for point in points
                if point is not None
            ),
            None,
        )

        if first_valid is not None:
            x, y = normalized_point_to_letterbox(
                first_valid,
                source_width,
                source_height,
                info,
            )

            cv2.putText(
                panel,
                f"lane {lane_id}",
                (
                    min(x + 8, PANEL_WIDTH - 100),
                    max(y - 8, 24),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )


def draw_status_panel(
    panel: np.ndarray,
    *,
    image_path: Path,
    label_path: Path,
    index: int,
    total: int,
    source_width: int,
    source_height: int,
    info: LetterBoxInfo,
    lanes: dict[int, list[LanePoint | None]],
    warnings: list[str],
) -> None:
    lines = [
        f"{index + 1}/{total}  {image_path.name}",
        (
            f"source={source_width}x{source_height}  "
            f"letterbox=640x640"
        ),
        (
            f"resize={info.resized_width}x"
            f"{info.resized_height}  "
            f"pad LTRB={info.pad_left},"
            f"{info.pad_top},"
            f"{info.pad_right},"
            f"{info.pad_bottom}"
        ),
        (
            f"label={label_path.name}  "
            f"lane_ids={sorted(lanes.keys())}"
        ),
        "A: previous   D: next   Q/ESC: quit",
    ]

    for lane_id in sorted(lanes):
        valid_count = sum(
            point is not None
            for point in lanes[lane_id]
        )

        lines.append(
            f"lane {lane_id}: "
            f"{valid_count}/{len(lanes[lane_id])} valid"
        )

    lines.extend(
        f"WARNING: {warning}"
        for warning in warnings[:3]
    )

    line_height = 20
    panel_height = (
        10 + line_height * len(lines)
    )

    overlay = panel.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (PANEL_WIDTH - 1, panel_height),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.addWeighted(
        overlay,
        0.68,
        panel,
        0.32,
        0,
        panel,
    )

    for line_id, text in enumerate(lines):
        color = (
            (0, 220, 255)
            if text.startswith("WARNING")
            else (255, 255, 255)
        )

        cv2.putText(
            panel,
            text,
            (10, 19 + line_id * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )


def build_panel(
    image_path: Path,
    label_dir: Path,
    index: int,
    total: int,
    *,
    show_anchors: bool,
    show_grid: bool,
    point_radius: int,
    line_thickness: int,
) -> np.ndarray:
    source_image = read_image(image_path)
    source_height, source_width = source_image.shape[:2]

    panel, info = letterbox_640(source_image)

    label_path = (
        label_dir / f"{image_path.stem}.txt"
    )

    lanes, warnings = parse_label_file(
        label_path
    )

    draw_content_boundary(
        panel,
        info,
    )

    if show_anchors:
        draw_row_anchors(
            panel,
            source_height,
            info,
        )

    if show_grid:
        draw_classification_grid(
            panel,
            source_width,
            info,
        )

    draw_lanes(
        panel,
        lanes,
        source_width,
        source_height,
        info,
        point_radius,
        line_thickness,
    )

    draw_status_panel(
        panel,
        image_path=image_path,
        label_path=label_path,
        index=index,
        total=total,
        source_width=source_width,
        source_height=source_height,
        info=info,
        lanes=lanes,
        warnings=warnings,
    )

    return panel


def main() -> None:
    args = parse_args()

    image_dir, label_dir = resolve_directories(
        args
    )

    if not label_dir.exists():
        raise FileNotFoundError(
            f"标签目录不存在：{label_dir}"
        )

    images = collect_images(image_dir)

    print(f"图片目录：{image_dir}")
    print(f"标签目录：{label_dir}")
    print(f"图片数量：{len(images)}")
    print("按键：A 上一张，D 下一张，Q/Esc 退出")

    window_name = "Lane Labels - 640x640 LetterBox"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_AUTOSIZE,
    )

    index = 0

    while True:
        try:
            panel = build_panel(
                image_path=images[index],
                label_dir=label_dir,
                index=index,
                total=len(images),
                show_anchors=args.show_anchors,
                show_grid=args.show_grid,
                point_radius=args.point_radius,
                line_thickness=args.line_thickness,
            )
        except Exception as exc:
            panel = np.zeros(
                (
                    PANEL_HEIGHT,
                    PANEL_WIDTH,
                    3,
                ),
                dtype=np.uint8,
            )

            cv2.putText(
                panel,
                f"ERROR: {images[index].name}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                panel,
                str(exc)[:90],
                (20, 125),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.imshow(
            window_name,
            panel,
        )

        key = cv2.waitKeyEx(0)

        if key in (
            ord("a"),
            ord("A"),
        ):
            index = (
                index - 1
            ) % len(images)

        elif key in (
            ord("d"),
            ord("D"),
        ):
            index = (
                index + 1
            ) % len(images)

        elif key in (
            ord("q"),
            ord("Q"),
            27,
        ):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
