#!/usr/bin/env python3
"""Lane Robot new-head ONNX camera inference.

Expected ONNX outputs:
    cls_01 [B, 321, 56, 2]
    cls_23 [B, 321, 56, 2]
    offset [B,   1, 56, 4]

Run this script from the ULTRALYTICS_LANE_ROBOT repository root so it can
import infer_onnx_xhm.py and the local ultralytics package.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from infer_onnx_xhm import (
    CURRENT_X_GRIDS,
    LANE_COLORS,
    LANE_NAMES,
    PROJECT_ROOT,
    choose_providers,
    decode_lanes,
    get_input_hw,
    inspect_output_layout,
    preprocess_with_policy,
    split_outputs,
)
from ultralytics.models.yolo.lane.geometry import restore_lanes_from_letterbox


DEFAULT_MODEL = (
    Path(PROJECT_ROOT)
    / "runs/lane/lane_n_baseline-3/weights/best_newhead.onnx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use the split-head Lane Robot ONNX model on a live camera stream."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help=f"new-head ONNX path. Default: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="OpenCV camera index. Default: 0",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "v4l2"),
        default="auto",
        help="Camera backend. Linux USB cameras may work better with v4l2.",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=720, help="Requested camera height.")
    parser.add_argument("--camera-fps", type=float, default=30.0, help="Requested camera FPS.")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="ONNX Runtime execution provider.",
    )
    parser.add_argument(
        "--letterbox",
        action="store_true",
        help="Use the repository's top-padding, bottom-aligned letterbox preprocessing.",
    )
    parser.add_argument(
        "--expected-x-grids",
        type=int,
        default=CURRENT_X_GRIDS,
        help=f"Expected valid horizontal grids. Default: {CURRENT_X_GRIDS}",
    )
    parser.add_argument(
        "--exist-thr",
        type=float,
        default=0.5,
        help="A point is valid when P(no-lane) is below this value.",
    )
    parser.add_argument("--topk", type=int, default=5, help="Top-K soft-argmax size.")
    parser.add_argument(
        "--no-smooth",
        action="store_true",
        help="Disable polynomial smoothing.",
    )
    parser.add_argument("--poly-degree", type=int, default=2)
    parser.add_argument("--poly-blend", type=float, default=0.5)
    parser.add_argument(
        "--draw-lines",
        action="store_true",
        help="Draw line segments in addition to points. Segments never cross invalid anchors.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=0,
        help="0 selects width automatically from frame size.",
    )
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Optional output video path, for example runs/camera_newhead.mp4",
    )
    parser.add_argument(
        "--window-name",
        default="Lane Robot new-head ONNX camera",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.model.expanduser().is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {args.model.expanduser()}")
    if args.width < 1 or args.height < 1:
        raise ValueError("--width and --height must be positive.")
    if args.camera_fps <= 0:
        raise ValueError("--camera-fps must be positive.")
    if not 0.0 <= args.exist_thr <= 1.0:
        raise ValueError("--exist-thr must be in [0, 1].")
    if args.topk < 1:
        raise ValueError("--topk must be at least 1.")
    if args.expected_x_grids < 1:
        raise ValueError("--expected-x-grids must be at least 1.")
    if not 0.0 <= args.poly_blend <= 1.0:
        raise ValueError("--poly-blend must be in [0, 1].")


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    backend = cv2.CAP_V4L2 if args.backend == "v4l2" else cv2.CAP_ANY
    capture = cv2.VideoCapture(args.camera, backend)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"Cannot open camera index {args.camera}. "
            "Try --camera 1 or --backend v4l2."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(args.width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(args.height))
    capture.set(cv2.CAP_PROP_FPS, float(args.camera_fps))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def flush_segment(
    image: np.ndarray,
    segment: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    if len(segment) < 2:
        return
    cv2.polylines(
        image,
        [np.asarray(segment, dtype=np.int32)],
        isClosed=False,
        color=color,
        thickness=thickness,
        lineType=cv2.LINE_AA,
    )


def draw_lane_points(
    bgr_frame: np.ndarray,
    lanes: np.ndarray,
    x_grids: int,
    *,
    row_y: np.ndarray | None,
    draw_lines: bool,
    line_width: int,
) -> tuple[np.ndarray, list[int]]:
    """Draw points and optionally gap-aware line segments on a BGR frame."""
    output = np.ascontiguousarray(bgr_frame.copy())
    height, width = output.shape[:2]
    row_anchors, num_lanes = lanes.shape

    if row_y is None:
        row_y = np.linspace(1.0, 0.333333, row_anchors, dtype=np.float32)
    else:
        row_y = np.asarray(row_y, dtype=np.float32).reshape(-1)
        if row_y.size != row_anchors:
            raise ValueError(
                f"row_y must contain {row_anchors} values, got {row_y.size}"
            )

    y_pixels = np.clip(row_y, 0.0, 1.0) * max(height - 1, 1)
    thickness = (
        int(line_width)
        if int(line_width) > 0
        else max(round((height + width) / 700), 2)
    )
    radius = max(thickness + 1, 3)
    active_lane_ids: list[int] = []

    for lane_id in range(num_lanes):
        color = LANE_COLORS.get(lane_id, (255, 0, 255))
        first_point: tuple[int, int] | None = None
        segment: list[tuple[int, int]] = []
        valid_count = 0

        for row_index, y_pixel in enumerate(y_pixels):
            x_grid = float(lanes[row_index, lane_id])
            valid = 0.0 <= x_grid < x_grids

            if not valid:
                if draw_lines:
                    flush_segment(output, segment, color, thickness)
                segment = []
                continue

            x_pixel = int(
                round(
                    x_grid
                    / max(x_grids - 1, 1)
                    * max(width - 1, 1)
                )
            )
            point = (x_pixel, int(round(y_pixel)))
            if first_point is None:
                first_point = point
            valid_count += 1
            segment.append(point)

            cv2.circle(
                output,
                point,
                radius,
                color,
                -1,
                lineType=cv2.LINE_AA,
            )

        if draw_lines:
            flush_segment(output, segment, color, thickness)

        if valid_count == 0 or first_point is None:
            continue

        active_lane_ids.append(lane_id)
        lane_name = LANE_NAMES.get(lane_id, f"lane_{lane_id}")
        text_origin = (
            min(first_point[0] + 5, max(width - 180, 0)),
            max(first_point[1] - 8, 18),
        )
        cv2.putText(
            output,
            lane_name,
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            max(thickness - 1, 1),
            cv2.LINE_AA,
        )

    return output, active_lane_ids


def create_video_writer(
    path: Path,
    frame_size: tuple[int, int],
    fps: float,
) -> cv2.VideoWriter:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        frame_size,
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"Cannot create output video: {path}")
    return writer


def main() -> None:
    args = parse_args()
    args.model = args.model.expanduser().resolve()
    validate_args(args)

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is not installed. Install onnxruntime or onnxruntime-gpu."
        ) from exc

    providers = choose_providers(ort, args.device)
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        str(args.model),
        sess_options=session_options,
        providers=providers,
    )

    input_name = session.get_inputs()[0].name
    input_hw = get_input_hw(session)
    output_infos = session.get_outputs()
    output_names = [item.name for item in output_infos]
    model_x_grids, output_layout = inspect_output_layout(
        output_infos,
        args.expected_x_grids,
    )

    required_layout = "split cls_01/cls_23 + offset"
    if output_layout != required_layout:
        raise RuntimeError(
            "This script only accepts the new-head three-output model.\n"
            f"Required layout: {required_layout}\n"
            f"Actual layout:   {output_layout}\n"
            f"Outputs:         {[(item.name, item.shape) for item in output_infos]}"
        )

    capture = open_camera(args)
    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))

    print("=" * 78)
    print("Lane Robot new-head ONNX camera inference")
    print(f"model          : {args.model}")
    print(f"camera         : {args.camera} ({args.backend})")
    print(f"camera size    : {actual_width} x {actual_height}")
    print(f"camera fps     : {actual_fps:.2f}")
    print(f"model input    : {input_hw[1]} x {input_hw[0]}")
    print(f"outputs        : {[(item.name, item.shape) for item in output_infos]}")
    print(f"layout         : {output_layout}")
    print(f"x_grids        : {model_x_grids}")
    print(f"providers      : {session.get_providers()}")
    print(f"letterbox      : {args.letterbox}")
    print(f"draw lines     : {args.draw_lines}")
    print("keys           : q or ESC to quit")
    print("=" * 78)

    writer: cv2.VideoWriter | None = None
    fps_ema = 0.0
    frame_count = 0

    try:
        while True:
            ok, bgr_frame = capture.read()
            if not ok or bgr_frame is None:
                raise RuntimeError("Camera returned an empty frame.")

            frame_start = time.perf_counter()
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            input_tensor, letterbox_meta = preprocess_with_policy(
                rgb_frame,
                input_hw,
                letterbox=args.letterbox,
            )

            inference_start = time.perf_counter()
            outputs = session.run(
                output_names,
                {input_name: input_tensor},
            )
            inference_ms = (time.perf_counter() - inference_start) * 1000.0

            cls_logits, offset = split_outputs(
                outputs,
                expected_x_grids=args.expected_x_grids,
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

            rendered, active_lane_ids = draw_lane_points(
                bgr_frame,
                lanes,
                x_grids,
                row_y=result_row_y,
                draw_lines=args.draw_lines,
                line_width=args.line_width,
            )

            total_ms = (time.perf_counter() - frame_start) * 1000.0
            instantaneous_fps = 1000.0 / max(total_ms, 1e-6)
            fps_ema = (
                instantaneous_fps
                if frame_count == 0
                else 0.90 * fps_ema + 0.10 * instantaneous_fps
            )
            frame_count += 1

            lane_text = (
                ", ".join(LANE_NAMES.get(i, f"lane_{i}") for i in active_lane_ids)
                if active_lane_ids
                else "no lanes"
            )
            cv2.putText(
                rendered,
                f"FPS {fps_ema:.1f} | infer {inference_ms:.1f} ms | {lane_text}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if args.save_video is not None:
                if writer is None:
                    output_height, output_width = rendered.shape[:2]
                    writer = create_video_writer(
                        args.save_video,
                        (output_width, output_height),
                        actual_fps if actual_fps > 1.0 else args.camera_fps,
                    )
                    print(f"save video     : {args.save_video.expanduser().resolve()}")
                writer.write(rendered)

            cv2.imshow(args.window_name, rendered)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
