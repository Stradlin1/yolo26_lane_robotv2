# Ultralytics 🚀 AGPL-3.0 License

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.lane.geometry import (
    compute_lane_letterbox_meta,
    letterbox_lane_image,
    parse_letterbox_color,
    restore_lanes_from_letterbox,
)
from ultralytics.models.yolo.lane.plotting import decode_lane
from ultralytics.utils import ops


class LaneResults(Results):
    """Prediction results for fixed-slot row-anchor lane detection.

    Attributes:
        lanes (np.ndarray): Decoded x-grid coordinates with shape [row_anchors, num_lanes]. A value of -1 means absent.
        row_y (np.ndarray): Normalized y coordinates for each row anchor, ordered exactly like ``lanes``.
        x_grids (int): Number of valid horizontal grid positions before the no-lane class.
    """

    def __init__(self, orig_img, path, names, lanes, row_y, x_grids):
        super().__init__(orig_img=orig_img, path=path, names=names)
        self.lanes = np.asarray(lanes, dtype=np.float32)
        self.row_y = np.asarray(row_y, dtype=np.float32).reshape(-1)
        self.x_grids = int(x_grids)

    @property
    def active_lane_ids(self) -> list[int]:
        """Return lane slot IDs containing at least one visible row-anchor point."""
        return [lane_id for lane_id in range(self.lanes.shape[1]) if np.any(self.lanes[:, lane_id] >= 0)]

    def __len__(self) -> int:
        """Return the number of active lane slots."""
        return len(self.active_lane_ids)

    def new(self):
        """Return an empty-copy style result while preserving lane-specific data."""
        result = LaneResults(
            orig_img=self.orig_img,
            path=self.path,
            names=self.names,
            lanes=self.lanes.copy(),
            row_y=self.row_y.copy(),
            x_grids=self.x_grids,
        )
        result.speed = self.speed.copy()
        return result

    def verbose(self) -> str:
        """Return a compact summary used by the Ultralytics prediction logger."""
        active = self.active_lane_ids
        if not active:
            return "(no lanes), "
        lane_names = [self.names.get(i, f"lane_{i}") for i in active]
        return f"{len(active)} lane{'s' if len(active) != 1 else ''}: {', '.join(lane_names)}, "

    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
        color_mode="class",
        txt_color=(255, 255, 255),
    ):
        """Draw decoded lanes on the original BGR image."""
        plotted = deepcopy(self.orig_img if img is None else img)
        if not isinstance(plotted, np.ndarray):
            plotted = np.asarray(plotted)
        plotted = np.ascontiguousarray(plotted.copy())

        h, w = plotted.shape[:2]
        thickness = int(line_width) if line_width else max(round((h + w) / 700), 2)
        radius = max(thickness + 1, 3)
        colors = (
            (0, 0, 255),      # lane_follow: red in BGR
            (0, 255, 0),      # lead_lane: green
            (255, 128, 0),    # channel_left: blue/orange
            (0, 255, 255),    # channel_right: yellow
            (255, 0, 255),
            (255, 255, 0),
        )
        y_pixels = np.clip(self.row_y, 0.0, 1.0) * max(h - 1, 1)

        for lane_id in self.active_lane_ids:
            color = colors[lane_id % len(colors)]
            points = []
            for row_idx, y in enumerate(y_pixels):
                x_grid = float(self.lanes[row_idx, lane_id])
                if x_grid < 0 or x_grid >= self.x_grids:
                    continue
                x = int(round(x_grid / max(self.x_grids - 1, 1) * max(w - 1, 1)))
                point = (x, int(round(y)))
                points.append(point)
                cv2.circle(plotted, point, radius, color, -1, lineType=cv2.LINE_AA)

            if len(points) >= 2:
                cv2.polylines(
                    plotted,
                    [np.asarray(points, dtype=np.int32)],
                    isClosed=False,
                    color=color,
                    thickness=thickness,
                    lineType=cv2.LINE_AA,
                )

            if labels and points:
                name = self.names.get(lane_id, f"lane_{lane_id}")
                text_origin = (points[0][0] + 4, max(points[0][1] - 6, 12))
                cv2.putText(
                    plotted,
                    name,
                    text_origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    max(thickness - 1, 1),
                    cv2.LINE_AA,
                )

        if show:
            cv2.imshow(str(self.path), plotted)
            cv2.waitKey(0)
        if save:
            output_path = Path(filename or f"results_{Path(self.path).name}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), plotted)
        if pil:
            return Image.fromarray(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
        return plotted

    def save_txt(self, txt_file, save_conf=False) -> str:
        """Save lanes in the training label format: lane_id x1 y1 ... xR yR."""
        lines = []
        denominator = max(self.x_grids - 1, 1)
        for lane_id in self.active_lane_ids:
            values = [str(lane_id)]
            for x_grid, y in zip(self.lanes[:, lane_id], self.row_y):
                x = -1.0 if x_grid < 0 else float(x_grid) / denominator
                values.extend((f"{x:.6f}", f"{float(y):.6f}"))
            lines.append(" ".join(values))

        txt_file = Path(txt_file)
        if lines:
            txt_file.parent.mkdir(parents=True, exist_ok=True)
            with txt_file.open("a", encoding="utf-8") as file:
                file.write("\n".join(lines) + "\n")
        return str(txt_file)


class LaneRobotPredictor(DetectionPredictor):
    """Predictor for fixed-slot LaneRobot row-anchor models."""

    def _target_hw(self) -> tuple[int, int]:
        """Return predictor input size as (height, width)."""
        if isinstance(self.imgsz, int):
            return int(self.imgsz), int(self.imgsz)
        return int(self.imgsz[0]), int(self.imgsz[1])

    def pre_transform(self, images: list[np.ndarray]) -> list[np.ndarray]:
        """Apply the same direct-resize or top-padded LetterBox policy as LaneRobotDataset."""
        height, width = self._target_hw()
        if not bool(getattr(self.args, "lane_letterbox", False)):
            return [cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR) for image in images]

        color = parse_letterbox_color(getattr(self.args, "lane_letterbox_color", [0, 0, 0]))
        bottom_align = bool(getattr(self.args, "lane_letterbox_bottom_align", True))
        return [
            letterbox_lane_image(
                image,
                (height, width),
                # Predictor sources are BGR here; config color follows the RGB dataset convention.
                color=color[::-1],
                bottom_align=bottom_align,
            )[0]
            for image in images
        ]

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Decode raw lane logits and wrap each image in a ``LaneResults`` object."""
        logits = preds.get("cls") if isinstance(preds, dict) else preds[0].get("cls")
        x_grids = int(logits.shape[1] - 1)
        row_anchors = int(logits.shape[2])

        lane_xy = decode_lane(
            preds,
            no_lane_idx=x_grids,
            topk=int(getattr(self.args, "lane_softargmax_topk", 5)),
            exist_thr=float(getattr(self.args, "lane_exist_thr", 0.5)),
            post_smooth=bool(getattr(self.args, "lane_post_smooth", True)),
            poly_degree=int(getattr(self.args, "lane_poly_degree", 2)),
            poly_blend=float(getattr(self.args, "lane_poly_blend", 0.5)),
        )

        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        # Labels are stored bottom-to-top: y_end -> y_start.
        row_y = np.linspace(
            float(getattr(self.args, "lane_y_end", 1.0)),
            float(getattr(self.args, "lane_y_start", 0.333333)),
            row_anchors,
            dtype=np.float32,
        )
        paths = self.batch[0]
        names = getattr(self.model, "names", {})

        use_letterbox = bool(getattr(self.args, "lane_letterbox", False))
        bottom_align = bool(getattr(self.args, "lane_letterbox_bottom_align", True))
        target_hw = (int(img.shape[-2]), int(img.shape[-1]))
        results = []
        for lanes, orig_img, img_path in zip(lane_xy, orig_imgs, paths):
            result_row_y = row_y.copy()
            if use_letterbox:
                meta = compute_lane_letterbox_meta(
                    orig_img.shape[:2],
                    target_hw,
                    bottom_align=bottom_align,
                )
                lanes, result_row_y = restore_lanes_from_letterbox(
                    lanes,
                    result_row_y,
                    x_grids,
                    meta,
                )

            results.append(
                LaneResults(
                    orig_img=orig_img,
                    path=img_path,
                    names=names,
                    lanes=lanes,
                    row_y=result_row_y,
                    x_grids=x_grids,
                )
            )
        return results
