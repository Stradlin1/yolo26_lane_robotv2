# Ultralytics 🚀 AGPL-3.0 License

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from ultralytics.models.yolo.lane.geometry import letterbox_lane_image, parse_letterbox_color

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _imgsz_to_hw(imgsz):
    """Convert an Ultralytics imgsz value to (height, width)."""
    if isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 2:
            return int(imgsz[0]), int(imgsz[1])
        if len(imgsz) == 1:
            return int(imgsz[0]), int(imgsz[0])
    return int(imgsz), int(imgsz)


def _float_arg(args, name: str, default: float = 0.0) -> float:
    value = getattr(args, name, default)
    return default if value is None else float(value)


class LaneRobotDataset(Dataset):
    """Dataset for the fixed-slot Lane Robot label format.

    Label line format:
        lane_id x1 y1 x2 y2 ... xN yN

    Supported preprocessing and augmentation:
    - Optional aspect-ratio-preserving Letterbox resize.
    - Horizontal centering and optional bottom alignment for Letterbox.
    - HSV/BGR color augmentation.
    - Synchronized rotation, translation, scaling, shear, perspective and flips.
    - Re-sampling of transformed curves at the fixed row anchors.
    - Horizontal-flip semantic slot swapping, e.g. channel_left 2 <-> channel_right 3.

    Unsupported for the current fixed-slot representation:
    - Mosaic, MixUp, CutMix, Copy-Paste and random erasing.
    """

    def __init__(self, img_path, args, data, mode="train"):
        self.img_path = Path(img_path)
        self.args = args
        self.data = data
        self.mode = mode
        self.is_train = mode == "train"

        self.row_anchors = int(
            data.get("row_anchors", data.get("lane_row_anchors", getattr(args, "lane_row_anchors", 56)))
        )
        self.x_grids = int(
            data.get("x_grids", data.get("lane_x_grids", getattr(args, "lane_x_grids", 320)))
        )
        self.num_lanes = int(
            data.get("num_lanes", data.get("lane_num_lanes", getattr(args, "lane_num_lanes", 2)))
        )
        self.y_start = float(
            data.get("y_start", data.get("lane_y_start", getattr(args, "lane_y_start", 0.67)))
        )
        self.y_end = float(
            data.get("y_end", data.get("lane_y_end", getattr(args, "lane_y_end", 1.0)))
        )
        self.imgsz = _imgsz_to_hw(getattr(args, "imgsz", data.get("imgsz", [320, 320])))

        # Letterbox configuration from lane-robot.yaml.
        # Direct resize is the lane pipeline default and matches LaneRobotPredictor.
        # Letterbox is opt-in and applies the same geometry to images and lane labels.
        self.letterbox = bool(data.get("letterbox", getattr(args, "lane_letterbox", False)))

        raw_letterbox_color = data.get(
            "letterbox_color", getattr(args, "lane_letterbox_color", [0, 0, 0])
        )
        self.letterbox_color = parse_letterbox_color(raw_letterbox_color)

        # True: all vertical padding goes above the image, preserving the image bottom.
        # False: vertical padding is split between top and bottom.
        self.letterbox_bottom_align = bool(
            data.get("letterbox_bottom_align", getattr(args, "lane_letterbox_bottom_align", True))
        )

        # Ultralytics-style augmentation parameters passed by model.train(...).
        self.hsv_h = max(0.0, _float_arg(args, "hsv_h"))
        self.hsv_s = max(0.0, _float_arg(args, "hsv_s"))
        self.hsv_v = max(0.0, _float_arg(args, "hsv_v"))
        self.bgr = float(np.clip(_float_arg(args, "bgr"), 0.0, 1.0))

        self.degrees = max(0.0, _float_arg(args, "degrees"))
        self.translate = max(0.0, _float_arg(args, "translate"))
        self.scale = max(0.0, _float_arg(args, "scale"))
        self.shear = max(0.0, _float_arg(args, "shear"))
        self.perspective = max(0.0, _float_arg(args, "perspective"))
        self.flipud = float(np.clip(_float_arg(args, "flipud"), 0.0, 1.0))
        self.fliplr = float(np.clip(_float_arg(args, "fliplr"), 0.0, 1.0))

        self.mosaic = max(0.0, _float_arg(args, "mosaic"))
        self.mixup = max(0.0, _float_arg(args, "mixup"))
        self.cutmix = max(0.0, _float_arg(args, "cutmix"))
        self.copy_paste = max(0.0, _float_arg(args, "copy_paste"))
        self.erasing = max(0.0, _float_arg(args, "erasing"))

        # Example in lane-robot.yaml: flip_lane_pairs: [[2, 3]]
        self.flip_lane_pairs = self._parse_flip_lane_pairs(data.get("flip_lane_pairs", [[2, 3]]))
        self._validate_augmentation_config()

        self.label_dir = data.get(f"{mode}_labels") or data.get("labels") or None
        if self.label_dir:
            self.label_dir = Path(self.label_dir)
            if not self.label_dir.is_absolute():
                root = Path(os.environ.get("LANE_ROBOT_DATASETS", data.get("path", ".")))
                self.label_dir = root.expanduser() / self.label_dir

        self.im_files = self._scan_images(self.img_path)
        if not self.im_files:
            raise FileNotFoundError(f"No lane images found in {self.img_path}")
        self.labels = [self._label_path(path) for path in self.im_files]
        # Ultralytics-style dataset fraction (train only; useful for smoke tests on CPU).
        if self.is_train:
            fraction = float(getattr(args, "fraction", 1.0))
            if 0.0 < fraction < 1.0:
                n = max(1, int(round(len(self.im_files) * fraction)))
                self.im_files = self.im_files[:n]
                self.labels = self.labels[:n]

    def _parse_flip_lane_pairs(self, raw_pairs):
        if raw_pairs is None:
            return []

        pairs = []
        used_ids = set()
        for pair in raw_pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise ValueError(f"Each flip_lane_pairs entry must contain two lane IDs, got: {pair!r}")

            first, second = int(pair[0]), int(pair[1])
            if not (0 <= first < self.num_lanes and 0 <= second < self.num_lanes):
                raise ValueError(
                    f"flip_lane_pairs contains out-of-range IDs {(first, second)} "
                    f"for num_lanes={self.num_lanes}"
                )
            if first == second:
                raise ValueError(f"flip_lane_pairs cannot swap a lane with itself: {(first, second)}")
            if first in used_ids or second in used_ids:
                raise ValueError(f"A lane ID may appear in only one flip pair, got: {(first, second)}")

            used_ids.update((first, second))
            pairs.append((first, second))

        return pairs

    def _validate_augmentation_config(self):
        if not self.is_train:
            return

        unsupported = {
            "mosaic": self.mosaic,
            "mixup": self.mixup,
            "cutmix": self.cutmix,
            "copy_paste": self.copy_paste,
            "erasing": self.erasing,
        }
        enabled = {name: value for name, value in unsupported.items() if value > 0.0}
        if enabled:
            formatted = ", ".join(f"{name}={value}" for name, value in enabled.items())
            raise ValueError(
                "LaneRobotDataset does not support sample-composition/erasing augmentation "
                f"({formatted}). Set mosaic, mixup, cutmix, copy_paste and erasing to 0.0."
            )

        if self.translate > 1.0:
            raise ValueError(f"translate must be in [0, 1], got {self.translate}")
        if self.scale > 1.0:
            raise ValueError(f"scale must be in [0, 1], got {self.scale}")
        if self.perspective > 0.01:
            raise ValueError(
                f"perspective={self.perspective} is unusually large; use a small value such as 0.0001-0.001"
            )

    @staticmethod
    def _scan_images(path: Path):
        if path.is_file():
            if path.suffix.lower() == ".txt":
                base = path.parent
                return [
                    Path(item.strip()) if Path(item.strip()).is_absolute() else base / item.strip()
                    for item in path.read_text().splitlines()
                    if item.strip()
                ]
            return [path]
        return sorted(item for item in path.rglob("*") if item.suffix.lower() in IMG_SUFFIXES)

    def _label_path(self, image_path: Path):
        if self.label_dir is not None:
            txt_path = self.label_dir / image_path.with_suffix(".txt").name
            if txt_path.exists():
                return txt_path
            return self.label_dir / image_path.with_suffix(".npy").name

        parts = list(image_path.parts)
        if "images" in parts:
            index = len(parts) - 1 - parts[::-1].index("images")
            parts[index] = "labels"
            txt_path = Path(*parts).with_suffix(".txt")
            if txt_path.exists():
                return txt_path
            return txt_path.with_suffix(".npy")

        txt_path = image_path.with_suffix(".txt")
        return txt_path if txt_path.exists() else image_path.with_suffix(".npy")

    def _default_row_y(self):
        return np.linspace(self.y_end, self.y_start, self.row_anchors, dtype=np.float32)

    def _load_label(self, label_path: Path):
        lane = np.full((self.row_anchors, self.num_lanes), self.x_grids, dtype=np.int64)
        lane_x = np.full((self.row_anchors, self.num_lanes), -1.0, dtype=np.float32)
        row_y = self._default_row_y()

        if not label_path.exists():
            return lane, lane_x, row_y

        if label_path.suffix.lower() == ".npy":
            array = np.load(label_path)
            if array.shape == (self.num_lanes, self.row_anchors):
                array = array.T
            if array.shape != (self.row_anchors, self.num_lanes):
                raise ValueError(
                    f"Lane npy {label_path} shape {array.shape}; "
                    f"expected {(self.row_anchors, self.num_lanes)}"
                )

            array = array.astype(np.float32)
            valid = array >= 0
            lane_x[valid] = np.clip(array[valid], 0, self.x_grids - 1)
            lane[valid] = np.clip(np.rint(lane_x[valid]), 0, self.x_grids - 1).astype(np.int64)
            return lane, lane_x, row_y

        text = label_path.read_text().strip()
        if not text:
            return lane, lane_x, row_y

        for line in text.splitlines():
            if not line.strip():
                continue

            values = [float(value) for value in line.replace(",", " ").split()]
            if len(values) < 3:
                continue

            lane_id = int(values[0])
            if lane_id < 0 or lane_id >= self.num_lanes:
                continue

            pair_count = min(self.row_anchors, (len(values) - 1) // 2)
            for row_index in range(pair_count):
                x_value = values[1 + 2 * row_index]
                y_value = values[2 + 2 * row_index]

                if 0.0 <= y_value <= 1.0:
                    row_y[row_index] = y_value

                if x_value < 0.0:
                    continue

                if 0.0 <= x_value <= 1.0:
                    x_grid = float(np.clip(x_value * (self.x_grids - 1), 0, self.x_grids - 1))
                else:
                    x_grid = float(np.clip(x_value, 0, self.x_grids - 1))

                lane_x[row_index, lane_id] = x_grid
                lane[row_index, lane_id] = int(np.clip(round(x_grid), 0, self.x_grids - 1))

        return lane, lane_x, row_y

    @staticmethod
    def _contiguous_runs(indices: np.ndarray):
        if indices.size == 0:
            return []
        split_positions = np.where(np.diff(indices) > 1)[0] + 1
        return np.split(indices, split_positions)

    @staticmethod
    def _transform_points(points_xy: np.ndarray, matrix: np.ndarray):
        if points_xy.size == 0:
            return points_xy.astype(np.float32)

        homogeneous = np.concatenate(
            [points_xy.astype(np.float32), np.ones((len(points_xy), 1), dtype=np.float32)],
            axis=1,
        )
        transformed = homogeneous @ matrix.T
        denominator = transformed[:, 2]
        valid = np.abs(denominator) > 1e-6

        output = np.full((len(points_xy), 2), np.nan, dtype=np.float32)
        output[valid] = transformed[valid, :2] / denominator[valid, None]
        return output

    def _resample_lane_x(
        self,
        lane_x: np.ndarray,
        source_row_y: np.ndarray,
        matrix: np.ndarray,
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
        target_row_y: np.ndarray | None = None,
    ):
        """Transform lane curves and sample them at the target fixed row anchors."""
        if target_row_y is None:
            target_row_y = self._default_row_y()

        output = np.full_like(lane_x, -1.0, dtype=np.float32)
        source_y_pixels = source_row_y.astype(np.float32) * max(source_height - 1, 1)
        target_y_pixels = target_row_y.astype(np.float32) * max(target_height - 1, 1)

        x_grid_denominator = max(self.x_grids - 1, 1)
        source_pixel_width = max(source_width - 1, 1)
        target_pixel_width = max(target_width - 1, 1)

        for lane_id in range(self.num_lanes):
            valid_indices = np.flatnonzero(lane_x[:, lane_id] >= 0.0)
            if valid_indices.size == 0:
                continue

            row_candidates = [[] for _ in range(self.row_anchors)]

            for run in self._contiguous_runs(valid_indices):
                x_pixels = lane_x[run, lane_id] / x_grid_denominator * source_pixel_width
                y_pixels = source_y_pixels[run]
                source_points = np.column_stack((x_pixels, y_pixels))
                transformed_points = self._transform_points(source_points, matrix)

                finite = np.isfinite(transformed_points).all(axis=1)
                transformed_points = transformed_points[finite]
                if transformed_points.shape[0] == 0:
                    continue

                if transformed_points.shape[0] == 1:
                    x_single, y_single = transformed_points[0]
                    nearest_row = int(np.argmin(np.abs(target_y_pixels - y_single)))
                    anchor_spacing = target_height / max(self.row_anchors - 1, 1)
                    if abs(target_y_pixels[nearest_row] - y_single) <= anchor_spacing * 0.6:
                        row_candidates[nearest_row].append(float(x_single))
                    continue

                for point_index in range(transformed_points.shape[0] - 1):
                    x0, y0 = transformed_points[point_index]
                    x1, y1 = transformed_points[point_index + 1]
                    delta_y = y1 - y0
                    if abs(delta_y) < 1e-6:
                        continue

                    lower_y = min(y0, y1) - 1e-4
                    upper_y = max(y0, y1) + 1e-4
                    rows = np.flatnonzero((target_y_pixels >= lower_y) & (target_y_pixels <= upper_y))
                    if rows.size == 0:
                        continue

                    interpolation = (target_y_pixels[rows] - y0) / delta_y
                    x_intersections = x0 + interpolation * (x1 - x0)

                    for row_index, x_intersection in zip(rows.tolist(), x_intersections.tolist()):
                        if np.isfinite(x_intersection):
                            row_candidates[row_index].append(float(x_intersection))

            for row_index, candidates in enumerate(row_candidates):
                if not candidates:
                    continue

                x_pixel = float(np.median(candidates))
                if 0.0 <= x_pixel <= target_width - 1.0:
                    output[row_index, lane_id] = (
                        x_pixel / target_pixel_width * x_grid_denominator
                    )

        return output

    def _letterbox_image_and_lanes(
        self,
        image: np.ndarray,
        lane_x: np.ndarray,
        source_row_y: np.ndarray,
    ):
        """Resize without distortion, pad unused area and transform lane labels."""
        source_height, source_width = image.shape[:2]
        target_height, target_width = self.imgsz
        canvas, letterbox_meta = letterbox_lane_image(
            image,
            self.imgsz,
            color=self.letterbox_color,
            bottom_align=self.letterbox_bottom_align,
        )

        target_row_y = self._default_row_y()
        transformed_lane_x = self._resample_lane_x(
            lane_x=lane_x,
            source_row_y=source_row_y,
            matrix=letterbox_meta["matrix"],
            source_width=source_width,
            source_height=source_height,
            target_width=target_width,
            target_height=target_height,
            target_row_y=target_row_y,
        )

        return canvas, transformed_lane_x, target_row_y, letterbox_meta

    def _direct_resize_image_and_lanes(
        self,
        image: np.ndarray,
        lane_x: np.ndarray,
        source_row_y: np.ndarray,
    ):
        """Direct resize compatibility path. Normalized lane coordinates remain aligned."""
        target_height, target_width = self.imgsz
        resized = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        return resized, lane_x.astype(np.float32, copy=True), source_row_y.astype(np.float32, copy=True)

    def _random_transform_matrix(self, width: int, height: int):
        """Create one forward homography mapping current pixels to augmented pixels."""
        center = np.eye(3, dtype=np.float32)
        center[0, 2] = -width / 2.0
        center[1, 2] = -height / 2.0

        perspective_matrix = np.eye(3, dtype=np.float32)
        perspective_matrix[2, 0] = np.random.uniform(-self.perspective, self.perspective)
        perspective_matrix[2, 1] = np.random.uniform(-self.perspective, self.perspective)

        angle = np.random.uniform(-self.degrees, self.degrees)
        scale_factor = np.random.uniform(1.0 - self.scale, 1.0 + self.scale)
        scale_factor = max(scale_factor, 0.05)

        rotation = np.eye(3, dtype=np.float32)
        rotation[:2] = cv2.getRotationMatrix2D((0.0, 0.0), angle, scale_factor)

        shear_matrix = np.eye(3, dtype=np.float32)
        shear_matrix[0, 1] = np.tan(np.deg2rad(np.random.uniform(-self.shear, self.shear)))
        shear_matrix[1, 0] = np.tan(np.deg2rad(np.random.uniform(-self.shear, self.shear)))

        translation = np.eye(3, dtype=np.float32)
        translation[0, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * width
        translation[1, 2] = np.random.uniform(0.5 - self.translate, 0.5 + self.translate) * height

        matrix = translation @ shear_matrix @ rotation @ perspective_matrix @ center

        did_fliplr = self.fliplr > 0.0 and np.random.random() < self.fliplr
        did_flipud = self.flipud > 0.0 and np.random.random() < self.flipud

        if did_fliplr:
            horizontal_flip = np.array(
                [
                    [-1.0, 0.0, width - 1.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            matrix = horizontal_flip @ matrix

        if did_flipud:
            vertical_flip = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, -1.0, height - 1.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            matrix = vertical_flip @ matrix

        return matrix, did_fliplr

    def _apply_geometric_augmentation(
        self,
        image: np.ndarray,
        lane_x: np.ndarray,
        row_y: np.ndarray,
    ):
        height, width = image.shape[:2]
        matrix, did_fliplr = self._random_transform_matrix(width, height)

        if self.perspective > 0.0:
            image = cv2.warpPerspective(
                image,
                matrix,
                dsize=(width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.letterbox_color,
            )
        else:
            image = cv2.warpAffine(
                image,
                matrix[:2],
                dsize=(width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=self.letterbox_color,
            )

        lane_x = self._resample_lane_x(
            lane_x=lane_x,
            source_row_y=row_y,
            matrix=matrix,
            source_width=width,
            source_height=height,
            target_width=width,
            target_height=height,
            target_row_y=row_y,
        )

        if did_fliplr:
            for first, second in self.flip_lane_pairs:
                lane_x[:, [first, second]] = lane_x[:, [second, first]]

        return image, lane_x

    def _apply_color_augmentation(self, image: np.ndarray):
        if self.hsv_h > 0.0 or self.hsv_s > 0.0 or self.hsv_v > 0.0:
            gains = np.random.uniform(-1.0, 1.0, 3) * np.array(
                [self.hsv_h, self.hsv_s, self.hsv_v],
                dtype=np.float32,
            )

            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + gains[0] * 180.0) % 180.0
            hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + gains[1]), 0.0, 255.0)
            hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + gains[2]), 0.0, 255.0)
            image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        if self.bgr > 0.0 and np.random.random() < self.bgr:
            image = image[..., ::-1]

        return np.ascontiguousarray(image)

    def _has_geometric_augmentation(self):
        return any(
            value > 0.0
            for value in (
                self.degrees,
                self.translate,
                self.scale,
                self.shear,
                self.perspective,
                self.flipud,
                self.fliplr,
            )
        )

    def _build_lane_classes(self, lane_x: np.ndarray):
        lane = np.full((self.row_anchors, self.num_lanes), self.x_grids, dtype=np.int64)
        valid = lane_x >= 0.0
        lane[valid] = np.clip(np.rint(lane_x[valid]), 0, self.x_grids - 1).astype(np.int64)
        return lane

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        image_file = self.im_files[index]

        with Image.open(image_file) as pil_image:
            pil_image = pil_image.convert("RGB")
            pil_image = ImageOps.exif_transpose(pil_image)
            original_shape = (pil_image.height, pil_image.width)
            image = np.asarray(pil_image).copy()

        _, lane_x, source_row_y = self._load_label(self.labels[index])

        if self.letterbox:
            image, lane_x, lane_y, _letterbox_meta = self._letterbox_image_and_lanes(
                image,
                lane_x,
                source_row_y,
            )
        else:
            image, lane_x, lane_y = self._direct_resize_image_and_lanes(
                image,
                lane_x,
                source_row_y,
            )

        if self.is_train:
            if self._has_geometric_augmentation():
                image, lane_x = self._apply_geometric_augmentation(image, lane_x, lane_y)
            image = self._apply_color_augmentation(image)

        lane_x = lane_x.astype(np.float32, copy=False)
        lane = self._build_lane_classes(lane_x)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))

        return {
            "img": image_tensor,
            "lane": torch.from_numpy(lane),
            "lane_x": torch.from_numpy(lane_x),
            "lane_y": torch.from_numpy(lane_y.astype(np.float32, copy=False)),
            "cls": torch.zeros((1, 1)),
            "im_file": str(image_file),
            "ori_shape": original_shape,
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "img": torch.stack([item["img"] for item in batch], 0),
            "lane": torch.stack([item["lane"] for item in batch], 0),
            "lane_x": torch.stack([item["lane_x"] for item in batch], 0),
            "lane_y": torch.stack([item["lane_y"] for item in batch], 0),
            "cls": torch.zeros((len(batch), 1)),
            "im_file": [item["im_file"] for item in batch],
            "ori_shape": [item["ori_shape"] for item in batch],
        }
