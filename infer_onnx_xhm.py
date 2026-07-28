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
    默认尺寸为 256x320；若 ONNX 输入形状是静态的，会自动读取。

模型输出支持：
    1. 单输出 [B, 162, 56, 4]
       - 0:161 为分类 logits
       - 161:162 为 offset
    2. 双输出
       - cls    [B, 161, 56, 4]
       - offset [B,   1, 56, 4]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path("/home/xhm/Desktop/ULTRALYTICS_LANE_ROBOT")
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "runs/lane/lane_n_baseline-6/weights/best.onnx"
)
DEFAULT_SOURCE = PROJECT_ROOT / "test"
DEFAULT_OUTPUT = PROJECT_ROOT / "test_infer"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量使用 ONNX 推理 Lane Robot 图片并绘制四线结果。"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"ONNX 模型路径，默认：{DEFAULT_MODEL}",
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
        help="绘制线宽；0 表示根据图片尺寸自动选择。",
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
    return 256, 320


def load_image_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        return np.asarray(image).copy()


def preprocess(
    rgb_image: np.ndarray,
    input_hw: tuple[int, int],
) -> np.ndarray:
    height, width = input_hw

    resized = cv2.resize(
        rgb_image,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    tensor = resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)
    return np.ascontiguousarray(tensor)


def split_outputs(
    outputs: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray | None]:
    """
    返回：
        cls_logits: [B, X+1, R, L]
        offset:     [B, 1, R, L] 或 None
    """
    if not outputs:
        raise RuntimeError("ONNX Runtime 没有返回任何输出。")

    arrays = [np.asarray(item) for item in outputs]

    # export_onnx_xhm.py 生成的合并输出。
    if len(arrays) == 1:
        output = arrays[0]
        if output.ndim != 4:
            raise RuntimeError(
                f"ONNX 输出必须为四维，实际形状：{output.shape}"
            )

        if output.shape[1] >= 3:
            # 当前模型为 162 = 161 cls + 1 offset。
            cls_logits = output[:, :-1, :, :]
            offset = output[:, -1:, :, :]
            return cls_logits, offset

        raise RuntimeError(
            "无法从单输出中拆分分类和 offset："
            f"{output.shape}"
        )

    # 兼容 cls/offset 两个独立输出，按通道数量识别。
    cls_logits = None
    offset = None

    for output in arrays:
        if output.ndim != 4:
            continue
        if output.shape[1] == 1:
            offset = output
        elif output.shape[1] > 1:
            cls_logits = output

    if cls_logits is None:
        shapes = [tuple(item.shape) for item in arrays]
        raise RuntimeError(
            f"无法在 ONNX 输出中找到分类 logits，输出形状：{shapes}"
        )

    return cls_logits, offset


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
) -> tuple[np.ndarray, list[int]]:
    """
    lanes:
        [R, L]，行顺序与训练标签一致：y_end -> y_start。
    """
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    output = np.ascontiguousarray(bgr_image.copy())

    height, width = output.shape[:2]
    row_anchors, num_lanes = lanes.shape

    thickness = (
        int(line_width)
        if int(line_width) > 0
        else max(round((height + width) / 700), 2)
    )
    radius = max(thickness + 1, 3)

    row_y = np.linspace(
        float(y_end),
        float(y_start),
        row_anchors,
        dtype=np.float32,
    )
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

        if len(points) >= 2:
            cv2.polylines(
                output,
                [np.asarray(points, dtype=np.int32)],
                isClosed=False,
                color=color,
                thickness=thickness,
                lineType=cv2.LINE_AA,
            )

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
            max(thickness - 1, 1),
            cv2.LINE_AA,
        )

    return output, active_lane_ids


def save_prediction_txt(
    txt_path: Path,
    lanes: np.ndarray,
    x_grids: int,
    y_start: float = 0.333333,
    y_end: float = 1.0,
) -> None:
    row_anchors, num_lanes = lanes.shape
    row_y = np.linspace(
        float(y_end),
        float(y_start),
        row_anchors,
        dtype=np.float32,
    )
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


def main() -> None:
    args = parse_args()

    model_path = args.model.expanduser().resolve()
    source = args.source.expanduser().resolve()
    output_root = args.output.expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX 模型不存在：{model_path}")

    if not 0.0 <= args.exist_thr <= 1.0:
        raise ValueError("--exist-thr 必须位于 [0, 1]。")

    if args.topk < 1:
        raise ValueError("--topk 必须至少为 1。")

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
    output_names = [
        output.name
        for output in session.get_outputs()
    ]

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
    print(f"providers   : {session.get_providers()}")
    print(f"exist_thr   : {args.exist_thr}")
    print(f"topk        : {args.topk}")
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
            input_tensor = preprocess(rgb_image, input_hw)

            outputs = session.run(
                output_names,
                {input_name: input_tensor},
            )

            cls_logits, offset = split_outputs(outputs)

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

            drawn, active_lane_ids = draw_lanes(
                rgb_image=rgb_image,
                lanes=lanes,
                x_grids=x_grids,
                line_width=args.line_width,
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
