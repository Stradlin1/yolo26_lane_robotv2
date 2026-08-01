# Ultralytics 🚀 AGPL-3.0 License

from __future__ import annotations

import cv2
import numpy as np


def parse_letterbox_color(value) -> tuple[int, int, int]:
    """Validate and normalize a three-channel padding color."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("letterbox color must contain three values, for example [0, 0, 0]")
    return tuple(int(np.clip(channel, 0, 255)) for channel in value)


def compute_lane_letterbox_meta(
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    *,
    bottom_align: bool = True,
) -> dict:
    """Compute aspect-ratio-preserving resize and padding geometry for lane images.

    Horizontal padding is always split between the left and right sides. With
    ``bottom_align=True``, all vertical padding is placed above the resized image,
    which keeps the road/image bottom fixed at the bottom of the model input.
    """
    source_height, source_width = (int(source_shape[0]), int(source_shape[1]))
    target_height, target_width = (int(target_shape[0]), int(target_shape[1]))
    if min(source_height, source_width, target_height, target_width) <= 0:
        raise ValueError(
            f"source and target dimensions must be positive, got source={source_shape}, target={target_shape}"
        )

    nominal_scale = min(target_width / source_width, target_height / source_height)
    resized_width = max(1, min(target_width, int(round(source_width * nominal_scale))))
    resized_height = max(1, min(target_height, int(round(source_height * nominal_scale))))

    horizontal_padding = target_width - resized_width
    vertical_padding = target_height - resized_height
    pad_left = horizontal_padding // 2
    pad_right = horizontal_padding - pad_left
    if bottom_align:
        pad_top = vertical_padding
        pad_bottom = 0
    else:
        pad_top = vertical_padding // 2
        pad_bottom = vertical_padding - pad_top

    # Endpoint-aligned scales match the normalized coordinate convention used by
    # lane labels: pixel 0 maps to 0 and pixel (size - 1) maps to 1.
    scale_x = (resized_width - 1) / max(source_width - 1, 1)
    scale_y = (resized_height - 1) / max(source_height - 1, 1)
    matrix = np.array(
        [
            [scale_x, 0.0, float(pad_left)],
            [0.0, scale_y, float(pad_top)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return {
        "source_shape": (source_height, source_width),
        "target_shape": (target_height, target_width),
        "resized_shape": (resized_height, resized_width),
        "pad": (pad_left, pad_top, pad_right, pad_bottom),
        "scale": float(nominal_scale),
        "scale_xy": (float(scale_x), float(scale_y)),
        "matrix": matrix,
        "bottom_align": bool(bottom_align),
    }


def letterbox_lane_image(
    image: np.ndarray,
    target_shape: tuple[int, int],
    *,
    color: tuple[int, int, int] = (0, 0, 0),
    bottom_align: bool = True,
) -> tuple[np.ndarray, dict]:
    """Resize an image without distortion and pad it using lane letterbox geometry."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"lane letterbox expects an HWC three-channel image, got {image.shape}")

    meta = compute_lane_letterbox_meta(image.shape[:2], target_shape, bottom_align=bottom_align)
    resized_height, resized_width = meta["resized_shape"]
    pad_left, pad_top, _, _ = meta["pad"]
    target_height, target_width = meta["target_shape"]

    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_height, target_width, 3), parse_letterbox_color(color), dtype=image.dtype)
    canvas[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width] = resized
    return canvas, meta


def restore_lanes_from_letterbox(
    lanes: np.ndarray,
    row_y: np.ndarray,
    x_grids: int,
    meta: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Map decoded lane points from letterboxed input coordinates back to the source image.

    The returned x values remain in the model's x-grid coordinate convention so
    existing LaneResults plotting and txt export can consume them unchanged.
    Rows or points that fall in padded pixels are marked absent with x=-1.
    """
    restored = np.asarray(lanes, dtype=np.float32).copy()
    restored_row_y = np.asarray(row_y, dtype=np.float32).reshape(-1).copy()
    if restored.ndim != 2 or restored.shape[0] != restored_row_y.size:
        raise ValueError(
            f"expected lanes [R,L] and row_y [R], got lanes={restored.shape}, row_y={restored_row_y.shape}"
        )
    if int(x_grids) < 2:
        raise ValueError(f"x_grids must be >= 2, got {x_grids}")

    source_height, source_width = meta["source_shape"]
    target_height, target_width = meta["target_shape"]
    pad_left, pad_top, _, _ = meta["pad"]
    scale_x, scale_y = meta["scale_xy"]
    if scale_x <= 0.0 or scale_y <= 0.0:
        raise ValueError(f"invalid letterbox scale {meta['scale_xy']}")

    target_y = restored_row_y * max(target_height - 1, 1)
    source_y = (target_y - float(pad_top)) / scale_y
    valid_rows = (source_y >= 0.0) & (source_y <= source_height - 1.0)
    restored_row_y = np.clip(source_y / max(source_height - 1, 1), 0.0, 1.0).astype(np.float32)

    valid_points = restored >= 0.0
    target_x = restored / max(int(x_grids) - 1, 1) * max(target_width - 1, 1)
    source_x = (target_x - float(pad_left)) / scale_x
    valid_points &= valid_rows[:, None]
    valid_points &= (source_x >= 0.0) & (source_x <= source_width - 1.0)

    restored.fill(-1.0)
    restored[valid_points] = source_x[valid_points] / max(source_width - 1, 1) * max(int(x_grids) - 1, 1)
    return restored, restored_row_y
