#!/usr/bin/env python3
"""
使用 ONNX Runtime 批量推理 Lane Robot 四线模型。

默认路径：
    ONNX:
      /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/
      runs/lane/lane_n_baseline-3/weights/best.onnx

    输入图片目录:
      /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/test

    输出图片目录:
      /home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT/test_infer

支持的图片格式：
    .jpg .jpeg .png .bmp .webp

模型输入：
    RGB, float32, BCHW, [0, 1]
    默认尺寸为 320x320；若 ONNX 输入形状是静态的，会自动读取。

模型输出支持：
    1. 单输出 [B, X+2, R, L]；当前 X=320 时为 [B, 322, 56, 4]
       - 0:321 为分类 logits（含索引 320 的 no-lane 类）
       - 321:322 为 offset
    2. 双输出
       - cls    [B, 321, 56, 4]
       - offset [B,   1, 56, 4]
    3. RDK X5 三输出（方案 A）
       - cls_01 [B, 321, 56, 2]
       - cls_23 [B, 321, 56, 2]
       - offset [B,   1, 56, 4]

默认要求 X=320，避免误用旧的 X=160 模型。仅在明确需要推理旧模型时使用
--allow-legacy-x-grids。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from ultralytics.models.yolo.lane.geometry import letterbox_lane_image, restore_lanes_from_letterbox


PROJECT_ROOT = Path("/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT")
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "runs/lane/lane_n_baseline-2/weights/best.onnx"
)
DEFAULT_SOURCE = PROJECT_ROOT / "test"
DEFAULT_OUTPUT = PROJECT_ROOT / "test_infer"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CURRENT_X_GRIDS = 320

LANE_NAMES = {
    0: "lane_follow",
    1: "lead_lane",
    2: "channel_left",
    3: "channel_right",
}

# OpenCV 使用 BGR。
LANE_COLORS = {
    0: (0, 0, 255),       # 红色
    1: (0, 255, 0),       # 绿色
    2: (255, 128, 0),     # 蓝橙色
    3: (0, 255, 255),     # 黄色
}


def parse_args(default_model: Path = DEFAULT_MODEL) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量使用 ONNX 推理 Lane Robot 图片并仅绘制预测锚点。"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model,
        help=f"ONNX 模型路径，默认：{default_model}",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"输入图片目录或单张图片，默认：{DEFAULT_SOURCE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"输出目录，默认：{DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="ONNX Runtime 执行设备，默认 auto。",
    )
    parser.add_argument(
        "--letterbox",
        action="store_true",
        help="保持宽高比，纵向黑边全部补在顶部，横向黑边在左右居中；必须与训练设置一致。",
    )
    parser.add_argument(
        "--expected-x-grids",
        type=int,
        default=CURRENT_X_GRIDS,
        help=f"模型应使用的有效横向网格数，默认 {CURRENT_X_GRIDS}。",
    )
    parser.add_argument(
        "--allow-legacy-x-grids",
        action="store_true",
        help="允许推理 x_grids 不是 320 的旧 ONNX 模型。",
    )
    parser.add_argument(
        "--exist-thr",
        type=float,
        default=0.5,
        help="no-lane 概率阈值，默认 0.5。",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=5,
        help="横向网格 soft-argmax 使用的 Top-K，默认 5。",
    )
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="关闭二次多项式曲线平滑。",
    )
    parser.add_argument(
        "--poly-degree",
        type=int,
        default=2,
        help="曲线平滑多项式阶数，默认 2。",
    )
    parser.add_argument(
        "--poly-blend",
        type=float,
        default=0.5,
        help="原预测与拟合曲线的混合比例，默认 0.5。",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=0,
        help="兼容旧参数：现在用于控制预测点半径；0 表示根据图片尺寸自动选择。",
    )
    parser.add_argument(
        "--save-txt",
        action="store_true",
        help="同时保存与训练标签格式一致的预测 txt。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="允许覆盖已存在的输出图片。",
    )
    return parser.parse_args()


def scan_images(source: Path) -> list[Path]:
    source = source.expanduser().resolve()

    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"不支持的图片格式：{source}")
        return [source]

    if not source.is_dir():
        raise FileNotFoundError(f"输入路径不存在：{source}")

    images = sorted(
        path
        for path in source.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if not images:
        raise FileNotFoundError(f"目录中没有找到图片：{source}")

    return images


def choose_providers(
    ort_module,
    device: str,
) -> list[str]:
    available = ort_module.get_available_providers()

    if device == "cpu":
        return ["CPUExecutionProvider"]

    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "当前 onnxruntime 没有 CUDAExecutionProvider。\n"
                "请安装 GPU 版本：python -m pip install onnxruntime-gpu\n"
                f"当前可用 providers：{available}"
            )
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]

    return ["CPUExecutionProvider"]


def get_input_hw(session) -> tuple[int, int]:
    shape = session.get_inputs()[0].shape

    if len(shape) != 4:
        raise RuntimeError(
            f"模型输入必须是 BCHW 四维张量，实际输入形状：{shape}"
        )

    height = shape[2]
    width = shape[3]

    if isinstance(height, int) and isinstance(width, int):
        return int(height), int(width)

    # 动态空间尺寸时使用训练尺寸。
    return 320, 320


def load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return np.asarray(image).copy()


def preprocess_with_policy(
    rgb_image: np.ndarray,
    input_hw: tuple[int, int],
    *,
    letterbox: bool,
) -> tuple[np.ndarray, dict | None]:
    height, width = input_hw
    if letterbox:
        resized, meta = letterbox_lane_image(
            rgb_image,
            (height, width),
            color=(0, 0, 0),
            bottom_align=True,
        )
    else:
        resized = cv2.resize(
            rgb_image,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        meta = None

    tensor = resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    return np.ascontiguousarray(tensor), meta


def preprocess(
    rgb_image: np.ndarray,
    input_hw: tuple[int, int],
) -> np.ndarray:
    """Backward-compatible direct-resize preprocessing helper."""
    return preprocess_with_policy(rgb_image, input_hw, letterbox=False)[0]


def split_outputs(
    outputs: list[np.ndarray],
    expected_x_grids: int | None = None,
    output_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    返回：
        cls_logits: [B, X+1, R, L]
        offset:     [B, 1, R, L] 或 None
    """
    if not outputs:
        raise RuntimeError("ONNX Runtime 没有返回任何输出。")

    arrays = [np.asarray(item) for item in outputs]
    if output_names is not None and len(output_names) != len(arrays):
        raise RuntimeError(
            f"输出名称数量 {len(output_names)} 与输出张量数量 {len(arrays)} 不一致。"
        )

    # export_onnx_xhm.py 生成的合并输出。
    if len(arrays) == 1:
        output = arrays[0]
        if output.ndim != 4:
            raise RuntimeError(
                f"ONNX 输出必须为四维，实际形状：{output.shape}"
            )

        if output.shape[1] >= 3:
            # 合并输出约定为 (X+1) 个分类通道 + 1 个 offset 通道。
            actual_x_grids = int(output.shape[1] - 2)
            if (
                expected_x_grids is not None
                and actual_x_grids != expected_x_grids
            ):
                raise RuntimeError(
                    "ONNX 合并输出与当前 x_grids 配置不一致："
                    f"输出形状 {output.shape} 表示 x_grids={actual_x_grids}，"
                    f"但期望 x_grids={expected_x_grids}。"
                )
            cls_logits = output[:, :-1, :, :]
            offset = output[:, -1:, :, :]
            return cls_logits, offset

        raise RuntimeError(
            "无法从单输出中拆分分类和 offset："
            f"{output.shape}"
        )

    # 兼容 cls/offset 双输出，以及方案 A 的 cls_01/cls_23/offset 三输出。
    cls_parts = []
    offset = None

    for index, output in enumerate(arrays):
        output_name = output_names[index] if output_names is not None else None
        if output.ndim != 4:
            continue
        if output.shape[1] == 1:
            if offset is not None:
                raise RuntimeError("ONNX 返回了多个无法区分的 offset 输出。")
            offset = output
        elif output.shape[1] > 1:
            cls_parts.append((output_name, output))

    if not cls_parts:
        shapes = [tuple(item.shape) for item in arrays]
        raise RuntimeError(
            f"无法在 ONNX 输出中找到分类 logits，输出形状：{shapes}"
        )

    if len(cls_parts) == 1:
        cls_logits = cls_parts[0][1]
    elif len(cls_parts) == 2:
        named_parts = {name: value for name, value in cls_parts if name is not None}
        if output_names is not None:
            if set(named_parts) != {"cls_01", "cls_23"}:
                raise RuntimeError(
                    "方案 A 的两个分类输出必须命名为 cls_01 和 cls_23，"
                    f"实际：{[name for name, _ in cls_parts]}"
                )
            left, right = named_parts["cls_01"], named_parts["cls_23"]
        else:
            left, right = cls_parts[0][1], cls_parts[1][1]
        if left.shape[:3] != right.shape[:3]:
            raise RuntimeError(
                "cls_01 与 cls_23 的 [B,X+1,R] 必须一致，"
                f"实际：{left.shape} 和 {right.shape}"
            )
        if left.shape[3] != 2 or right.shape[3] != 2:
            raise RuntimeError(
                "方案 A 要求 cls_01/cls_23 各包含 2 条车道，"
                f"实际：{left.shape} 和 {right.shape}"
            )
        cls_logits = np.concatenate((left, right), axis=3)
    else:
        raise RuntimeError(
            "分类输出数量无效；仅支持一个完整 cls 或 cls_01/cls_23 两个分头，"
            f"实际形状：{[tuple(item.shape) for _, item in cls_parts]}"
        )

    if offset is not None and (
        offset.shape[0] != cls_logits.shape[0]
        or offset.shape[2] != cls_logits.shape[2]
        or offset.shape[3] != cls_logits.shape[3]
    ):
        raise RuntimeError(
            "offset 的 [B,R,L] 必须与拼接后的分类输出一致，"
            f"实际 cls={cls_logits.shape}, offset={offset.shape}"
        )

    actual_x_grids = int(cls_logits.shape[1] - 1)
    if expected_x_grids is not None and actual_x_grids != expected_x_grids:
        raise RuntimeError(
            "ONNX 分类输出与当前 x_grids 配置不一致："
            f"输出形状 {cls_logits.shape} 表示 x_grids={actual_x_grids}，"
            f"但期望 x_grids={expected_x_grids}。"
        )

    return cls_logits, offset


def inspect_output_layout(
    output_infos,
    expected_x_grids: int | None,
) -> tuple[int, str]:
    """Validate static ONNX output metadata before processing any images."""
    shapes = [tuple(info.shape) for info in output_infos]

    cls_shapes = [shape for shape in shapes if len(shape) == 4 and isinstance(shape[1], int) and shape[1] > 1]
    if len(shapes) == 1 and len(cls_shapes) == 1:
        x_grids = cls_shapes[0][1] - 2
        layout = "merged cls+offset"
    elif len(cls_shapes) == 1:
        x_grids = cls_shapes[0][1] - 1
        layout = "separate cls/offset"
    elif len(cls_shapes) == 2:
        cls_names = {
            info.name
            for info in output_infos
            if len(tuple(info.shape)) == 4 and isinstance(info.shape[1], int) and info.shape[1] > 1
        }
        if cls_names != {"cls_01", "cls_23"}:
            raise RuntimeError(f"方案 A 分类输出名称必须为 cls_01/cls_23，实际：{sorted(cls_names)}")
        left, right = cls_shapes
        if left[1] != right[1] or left[3] != 2 or right[3] != 2:
            raise RuntimeError(f"方案 A 分类输出元数据无效：{cls_shapes}")
        offset_infos = [
            info
            for info in output_infos
            if len(tuple(info.shape)) == 4 and isinstance(info.shape[1], int) and info.shape[1] == 1
        ]
        if len(offset_infos) != 1 or offset_infos[0].name != "offset":
            raise RuntimeError("方案 A 必须包含唯一的 offset 输出。")
        offset_shape = tuple(offset_infos[0].shape)
        if offset_shape[2] != left[2] or offset_shape[3] != 4:
            raise RuntimeError(f"方案 A offset 应为 [B,1,R,4]，实际：{offset_shape}")
        x_grids = left[1] - 1
        layout = "split cls_01/cls_23 + offset"
    else:
        raise RuntimeError(
            "无法从 ONNX 输出元数据识别 Lane Robot 输出布局："
            f"{shapes}"
        )

    if expected_x_grids is not None and x_grids != expected_x_grids:
        raise RuntimeError(
            "ONNX 模型不是当前要求的 x_grids 版本。\n"
            f"模型输出：{shapes}\n"
            f"解析得到：x_grids={x_grids}\n"
            f"当前要求：x_grids={expected_x_grids}\n"
            "请使用 320-grid ONNX 模型；若明确要运行旧模型，添加 "
            "--allow-legacy-x-grids。"
        )
    return int(x_grids), layout


def softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.maximum(
        np.sum(exp_values, axis=axis, keepdims=True),
        1e-12,
    )


def poly_smooth_1d(
    xs: np.ndarray,
    valid: np.ndarray,
    degree: int,
    blend: float,
) -> np.ndarray:
    result = xs.astype(np.float32).copy()
    valid = valid.astype(bool)

    if int(valid.sum()) < degree + 1:
        return result

    ys = np.arange(result.shape[0], dtype=np.float32)

    try:
        coefficients = np.polyfit(
            ys[valid],
            result[valid],
            int(degree),
        )
        fitted = np.polyval(coefficients, ys).astype(np.float32)
        result[valid] = (
            (1.0 - float(blend)) * result[valid]
            + float(blend) * fitted[valid]
        )
    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
        pass

    return result


def decode_lanes(
    cls_logits: np.ndarray,
    offset: np.ndarray | None,
    topk: int,
    exist_thr: float,
    smooth: bool,
    poly_degree: int,
    poly_blend: float,
) -> tuple[np.ndarray, int]:
    """
    返回：
        lanes: [B, R, L]，x-grid 坐标；-1 表示该点不存在。
        x_grids: 有效横向网格数量。
    """
    if cls_logits.ndim != 4:
        raise RuntimeError(
            f"分类输出必须为 [B,X+1,R,L]，实际：{cls_logits.shape}"
        )

    x_grids = int(cls_logits.shape[1] - 1)
    if x_grids <= 0:
        raise RuntimeError(
            f"分类输出通道数量无效：{cls_logits.shape}"
        )

    all_probs = softmax(cls_logits, axis=1)
    grid_probs = all_probs[:, :x_grids, :, :]

    k = max(1, min(int(topk), x_grids))

    # 先取 Top-K 索引，再按概率从大到小排列。
    top_indices = np.argpartition(
        grid_probs,
        kth=x_grids - k,
        axis=1,
    )[:, -k:, :, :]

    top_values = np.take_along_axis(
        grid_probs,
        top_indices,
        axis=1,
    )

    order = np.argsort(top_values, axis=1)[:, ::-1, :, :]
    top_indices = np.take_along_axis(
        top_indices,
        order,
        axis=1,
    )
    top_values = np.take_along_axis(
        top_values,
        order,
        axis=1,
    )

    numerator = np.sum(
        top_values * top_indices.astype(np.float32),
        axis=1,
    )
    denominator = np.maximum(
        np.sum(top_values, axis=1),
        1e-6,
    )

    pred_x = numerator / denominator

    if offset is not None:
        if offset.ndim != 4 or offset.shape[1] != 1:
            raise RuntimeError(
                f"offset 应为 [B,1,R,L]，实际：{offset.shape}"
            )
        pred_x = pred_x + np.clip(
            offset[:, 0, :, :],
            -0.5,
            0.5,
        )

    no_lane_prob = all_probs[:, x_grids, :, :]
    valid = no_lane_prob < float(exist_thr)
    pred_x = np.where(valid, pred_x, -1.0).astype(np.float32)

    if smooth:
        for batch_index in range(pred_x.shape[0]):
            for lane_id in range(pred_x.shape[2]):
                lane = pred_x[batch_index, :, lane_id]
                lane_valid = lane >= 0

                smoothed = poly_smooth_1d(
                    lane,
                    lane_valid,
                    degree=poly_degree,
                    blend=poly_blend,
                )
                smoothed[~lane_valid] = -1.0
                pred_x[batch_index, :, lane_id] = smoothed

    return pred_x, x_grids


def draw_lanes(
    rgb_image: np.ndarray,
    lanes: np.ndarray,
    x_grids: int,
    y_start: float = 0.333333,
    y_end: float = 1.0,
    line_width: int = 0,
    row_y: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    """
    lanes:
        [R, L]，行顺序与训练标签一致：y_end -> y_start。

    只绘制有效锚点；x=-1 的位置不画点，也不会在相邻有效点之间连线。
    """
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    output = np.ascontiguousarray(bgr_image.copy())

    height, width = output.shape[:2]
    row_anchors, num_lanes = lanes.shape

    # 仅绘制锚点，不绘制任何折线。为兼容既有命令，继续接受
    # --line-width 参数，但现在将它作为点半径使用。
    radius = (
        int(line_width)
        if int(line_width) > 0
        else max(round((height + width) / 700), 3)
    )
    text_thickness = max(radius // 2, 1)

    if row_y is None:
        row_y = np.linspace(float(y_end), float(y_start), row_anchors, dtype=np.float32)
    else:
        row_y = np.asarray(row_y, dtype=np.float32).reshape(-1)
        if row_y.size != row_anchors:
            raise ValueError(f"row_y must contain {row_anchors} values, got {row_y.size}")
    y_pixels = np.clip(row_y, 0.0, 1.0) * max(height - 1, 1)

    active_lane_ids: list[int] = []

    for lane_id in range(num_lanes):
        color = LANE_COLORS.get(
            lane_id,
            (255, 0, 255),
        )
        points: list[tuple[int, int]] = []

        for row_index, y_pixel in enumerate(y_pixels):
            x_grid = float(lanes[row_index, lane_id])

            if x_grid < 0 or x_grid >= x_grids:
                continue

            x_pixel = int(
                round(
                    x_grid
                    / max(x_grids - 1, 1)
                    * max(width - 1, 1)
                )
            )
            point = (x_pixel, int(round(y_pixel)))
            points.append(point)

            cv2.circle(
                output,
                point,
                radius,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )

        if not points:
            continue

        active_lane_ids.append(lane_id)

        lane_name = LANE_NAMES.get(
            lane_id,
            f"lane_{lane_id}",
        )
        text_origin = (
            points[0][0] + 4,
            max(points[0][1] - 6, 16),
        )

        cv2.putText(
            output,
            lane_name,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            text_thickness,
            cv2.LINE_AA,
        )

    return output, active_lane_ids


def save_prediction_txt(
    txt_path: Path,
    lanes: np.ndarray,
    x_grids: int,
    y_start: float = 0.333333,
    y_end: float = 1.0,
    row_y: np.ndarray | None = None,
) -> None:
    row_anchors, num_lanes = lanes.shape
    if row_y is None:
        row_y = np.linspace(float(y_end), float(y_start), row_anchors, dtype=np.float32)
    else:
        row_y = np.asarray(row_y, dtype=np.float32).reshape(-1)
        if row_y.size != row_anchors:
            raise ValueError(f"row_y must contain {row_anchors} values, got {row_y.size}")
    denominator = max(x_grids - 1, 1)

    lines: list[str] = []

    for lane_id in range(num_lanes):
        if not np.any(lanes[:, lane_id] >= 0):
            continue

        values = [str(lane_id)]

        for x_grid, y_value in zip(
            lanes[:, lane_id],
            row_y,
        ):
            x_value = (
                -1.0
                if x_grid < 0
                else float(x_grid) / denominator
            )
            values.extend(
                (
                    f"{x_value:.6f}",
                    f"{float(y_value):.6f}",
                )
            )

        lines.append(" ".join(values))

    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def relative_output_path(
    image_path: Path,
    source: Path,
    output_root: Path,
) -> Path:
    if source.is_file():
        return output_root / image_path.name

    return output_root / image_path.relative_to(source)


def main(default_model: Path = DEFAULT_MODEL, require_split_cls: bool = False) -> None:
    args = parse_args(default_model=default_model)

    model_path = args.model.expanduser().resolve()
    source = args.source.expanduser().resolve()
    output_root = args.output.expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX 模型不存在：{model_path}")

    if not 0.0 <= args.exist_thr <= 1.0:
        raise ValueError("--exist-thr 必须位于 [0, 1]。")

    if args.topk < 1:
        raise ValueError("--topk 必须至少为 1。")

    if args.expected_x_grids < 1:
        raise ValueError("--expected-x-grids 必须至少为 1。")

    if not 0.0 <= args.poly_blend <= 1.0:
        raise ValueError("--poly-blend 必须位于 [0, 1]。")

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "缺少 onnxruntime。\n"
            "CPU 版本：python -m pip install onnxruntime\n"
            "GPU 版本：python -m pip install onnxruntime-gpu"
        ) from exc

    image_paths = scan_images(source)
    providers = choose_providers(ort, args.device)

    session = ort.InferenceSession(
        str(model_path),
        providers=providers,
    )

    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_hw = get_input_hw(session)
    output_infos = session.get_outputs()
    output_names = [output.name for output in output_infos]
    required_x_grids = (
        None
        if args.allow_legacy_x_grids
        else args.expected_x_grids
    )
    model_x_grids, output_layout = inspect_output_layout(
        output_infos,
        required_x_grids,
    )
    if require_split_cls and output_layout != "split cls_01/cls_23 + offset":
        raise RuntimeError(
            "infer_onnx_newhead.py 只接受方案 A 三输出模型，"
            f"当前输出布局：{output_layout}。"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Lane Robot ONNX batch inference")
    print(f"model       : {model_path}")
    print(f"source      : {source}")
    print(f"output      : {output_root}")
    print(f"images      : {len(image_paths)}")
    print(f"input name  : {input_name}")
    print(f"input size  : {input_hw[0]} x {input_hw[1]}")
    print(f"outputs     : {output_names}")
    print(f"layout      : {output_layout}")
    print(f"x_grids     : {model_x_grids}")
    print(f"providers   : {session.get_providers()}")
    print(f"exist_thr   : {args.exist_thr}")
    print(f"topk        : {args.topk}")
    print(f"letterbox   : {args.letterbox} (top/left/right black padding)")
    print(f"smoothing   : {not args.no_smooth}")
    print("=" * 72)

    completed = 0
    skipped = 0
    failed = 0

    for index, image_path in enumerate(image_paths, start=1):
        output_path = relative_output_path(
            image_path,
            source,
            output_root,
        )

        if output_path.exists() and not args.overwrite:
            print(
                f"[{index}/{len(image_paths)}] SKIP "
                f"{image_path.name} -> 已存在"
            )
            skipped += 1
            continue

        try:
            rgb_image = load_image_rgb(image_path)
            input_tensor, letterbox_meta = preprocess_with_policy(
                rgb_image,
                input_hw,
                letterbox=args.letterbox,
            )

            outputs = session.run(
                output_names,
                {input_name: input_tensor},
            )

            cls_logits, offset = split_outputs(
                outputs,
                expected_x_grids=required_x_grids,
                output_names=output_names,
            )

            lanes_batch, x_grids = decode_lanes(
                cls_logits=cls_logits,
                offset=offset,
                topk=args.topk,
                exist_thr=args.exist_thr,
                smooth=not args.no_smooth,
                poly_degree=args.poly_degree,
                poly_blend=args.poly_blend,
            )

            lanes = lanes_batch[0]
            result_row_y = None
            if letterbox_meta is not None:
                model_row_y = np.linspace(
                    1.0,
                    0.333333,
                    lanes.shape[0],
                    dtype=np.float32,
                )
                lanes, result_row_y = restore_lanes_from_letterbox(
                    lanes,
                    model_row_y,
                    x_grids,
                    letterbox_meta,
                )

            drawn, active_lane_ids = draw_lanes(
                rgb_image=rgb_image,
                lanes=lanes,
                x_grids=x_grids,
                line_width=args.line_width,
                row_y=result_row_y,
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if not cv2.imwrite(str(output_path), drawn):
                raise RuntimeError(
                    f"OpenCV 无法写入输出图片：{output_path}"
                )

            if args.save_txt:
                txt_path = (
                    output_root
                    / "labels"
                    / output_path.relative_to(output_root)
                ).with_suffix(".txt")

                save_prediction_txt(
                    txt_path=txt_path,
                    lanes=lanes,
                    x_grids=x_grids,
                    row_y=result_row_y,
                )

            lane_text = (
                ", ".join(
                    LANE_NAMES.get(
                        lane_id,
                        f"lane_{lane_id}",
                    )
                    for lane_id in active_lane_ids
                )
                if active_lane_ids
                else "no lanes"
            )

            print(
                f"[{index}/{len(image_paths)}] OK   "
                f"{image_path.name} -> {output_path.name} "
                f"({lane_text})"
            )
            completed += 1

        except Exception as exc:
            print(
                f"[{index}/{len(image_paths)}] FAIL "
                f"{image_path}: {exc}",
                file=sys.stderr,
            )
            failed += 1

    print("\n" + "=" * 72)
    print("Inference complete")
    print(f"completed : {completed}")
    print(f"skipped   : {skipped}")
    print(f"failed    : {failed}")
    print(f"output    : {output_root}")
    print("=" * 72)

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
