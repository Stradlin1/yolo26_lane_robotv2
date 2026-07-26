# Ultralytics 🚀 AGPL-3.0 License

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _imgsz_to_hw(imgsz):
    if isinstance(imgsz, (list, tuple)):
        if len(imgsz) == 2:
            return int(imgsz[0]), int(imgsz[1])
        if len(imgsz) == 1:
            return int(imgsz[0]), int(imgsz[0])
    return int(imgsz), int(imgsz)


class LaneRobotDataset(Dataset):
    """Dataset for your lane label format.

    Label line format:
        lane_id x1 y1 x2 y2 ... xN yN

    - lane_id maps to output lane dimension [0, num_lanes - 1].
    - N is row_anchors, default 56.
    - x is normalized [0, 1]; x=-1 means absent and maps to no-lane class x_grids.
    - y is normalized [0, 1] and is stored for visualization. If absent, y_start/y_end are used.
    """

    def __init__(self, img_path, args, data, mode="train"):
        self.img_path = Path(img_path)
        self.args = args
        self.data = data
        self.mode = mode
        self.row_anchors = int(data.get("row_anchors", data.get("lane_row_anchors", getattr(args, "lane_row_anchors", 56))))
        self.x_grids = int(data.get("x_grids", data.get("lane_x_grids", getattr(args, "lane_x_grids", 640))))
        self.num_lanes = int(data.get("num_lanes", data.get("lane_num_lanes", getattr(args, "lane_num_lanes", 2))))
        self.y_start = float(data.get("y_start", data.get("lane_y_start", getattr(args, "lane_y_start", 0.67))))
        self.y_end = float(data.get("y_end", data.get("lane_y_end", getattr(args, "lane_y_end", 1.0))))
        self.imgsz = _imgsz_to_hw(getattr(args, "imgsz", data.get("imgsz", [256, 320])))
        self.label_dir = data.get(f"{mode}_labels") or data.get("labels") or None
        if self.label_dir:
            self.label_dir = Path(self.label_dir)
            if not self.label_dir.is_absolute():
                self.label_dir = Path(data.get("path", ".")) / self.label_dir
        self.im_files = self._scan_images(self.img_path)
        if not self.im_files:
            raise FileNotFoundError(f"No lane images found in {self.img_path}")
        self.labels = [self._label_path(p) for p in self.im_files]

    @staticmethod
    def _scan_images(path: Path):
        if path.is_file():
            if path.suffix.lower() == ".txt":
                base = path.parent
                return [Path(x.strip()) if Path(x.strip()).is_absolute() else base / x.strip() for x in path.read_text().splitlines() if x.strip()]
            return [path]
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMG_SUFFIXES)

    def _label_path(self, image_path: Path):
        if self.label_dir is not None:
            base = self.label_dir / image_path.with_suffix(".txt").name
            if base.exists():
                return base
            return self.label_dir / image_path.with_suffix(".npy").name
        parts = list(image_path.parts)
        if "images" in parts:
            idx = len(parts) - 1 - parts[::-1].index("images")
            parts[idx] = "labels"
            p = Path(*parts).with_suffix(".txt")
            if p.exists():
                return p
            return p.with_suffix(".npy")
        p = image_path.with_suffix(".txt")
        return p if p.exists() else image_path.with_suffix(".npy")

    def _load_label(self, label_path: Path):
        lane = np.full((self.row_anchors, self.num_lanes), self.x_grids, dtype=np.int64)
        lane_x = np.full((self.row_anchors, self.num_lanes), -1.0, dtype=np.float32)
        row_y = np.linspace(self.y_end, self.y_start, self.row_anchors, dtype=np.float32)
        if not label_path.exists():
            return lane, lane_x, row_y
        if label_path.suffix.lower() == ".npy":
            arr = np.load(label_path)
            if arr.shape == (self.num_lanes, self.row_anchors):
                arr = arr.T
            if arr.shape != (self.row_anchors, self.num_lanes):
                raise ValueError(f"Lane npy {label_path} shape {arr.shape}; expected {(self.row_anchors, self.num_lanes)}")
            arr = arr.astype(np.float32)
            valid = arr >= 0
            lane_x[valid] = np.clip(arr[valid], 0, self.x_grids - 1)
            lane[valid] = np.clip(np.rint(lane_x[valid]), 0, self.x_grids - 1).astype(np.int64)
            lane[~valid] = self.x_grids
            return lane.astype(np.int64), lane_x.astype(np.float32), row_y

        text = label_path.read_text().strip()
        if not text:
            return lane, lane_x, row_y
        for line in text.splitlines():
            if not line.strip():
                continue
            vals = [float(x) for x in line.replace(',', ' ').split()]
            if len(vals) < 3:
                continue
            lane_id = int(vals[0])
            if lane_id < 0 or lane_id >= self.num_lanes:
                continue
            pairs = min(self.row_anchors, (len(vals) - 1) // 2)
            for r in range(pairs):
                x = vals[1 + 2 * r]
                y = vals[2 + 2 * r]
                if 0.0 <= y <= 1.0:
                    row_y[r] = y
                if x < 0:
                    lane[r, lane_id] = self.x_grids
                    lane_x[r, lane_id] = -1.0
                elif 0.0 <= x <= 1.0:
                    xf = float(np.clip(x * (self.x_grids - 1), 0, self.x_grids - 1))
                    lane_x[r, lane_id] = xf
                    lane[r, lane_id] = int(np.clip(round(xf), 0, self.x_grids - 1))
                else:
                    # Already an x-grid index is also accepted for convenience.
                    xf = float(np.clip(x, 0, self.x_grids - 1))
                    lane_x[r, lane_id] = xf
                    lane[r, lane_id] = int(np.clip(round(xf), 0, self.x_grids - 1))
        return lane, lane_x, row_y

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, i):
        im_file = self.im_files[i]
        im = Image.open(im_file).convert("RGB")
        im = ImageOps.exif_transpose(im)
        ori_shape = (im.height, im.width)
        im = im.resize((self.imgsz[1], self.imgsz[0]), Image.BILINEAR)
        img = torch.from_numpy(np.asarray(im).transpose(2, 0, 1).copy())
        lane, lane_x, lane_y = self._load_label(self.labels[i])
        return {
            "img": img,
            "lane": torch.from_numpy(lane),
            "lane_x": torch.from_numpy(lane_x),
            "lane_y": torch.from_numpy(lane_y),
            "cls": torch.zeros((1, 1)),
            "im_file": str(im_file),
            "ori_shape": ori_shape,
        }

    @staticmethod
    def collate_fn(batch):
        return {
            "img": torch.stack([b["img"] for b in batch], 0),
            "lane": torch.stack([b["lane"] for b in batch], 0),
            "lane_x": torch.stack([b["lane_x"] for b in batch], 0),
            "lane_y": torch.stack([b["lane_y"] for b in batch], 0),
            "cls": torch.zeros((len(batch), 1)),
            "im_file": [b["im_file"] for b in batch],
            "ori_shape": [b["ori_shape"] for b in batch],
        }
