from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _split_preds(preds):
    if isinstance(preds, dict):
        return preds.get("cls"), preds.get("offset", None)
    if isinstance(preds, (list, tuple)) and preds and isinstance(preds[0], dict):
        return preds[0].get("cls"), preds[0].get("offset", None)
    return preds, None


def _poly_smooth_1d(xs, valid, degree=2, blend=0.5):
    xs = xs.astype(np.float32).copy()
    valid = valid.astype(bool)
    if valid.sum() < degree + 1:
        return xs
    ys = np.arange(xs.shape[0], dtype=np.float32)
    try:
        coef = np.polyfit(ys[valid], xs[valid], int(degree))
        fit = np.polyval(coef, ys).astype(np.float32)
        xs[valid] = (1.0 - float(blend)) * xs[valid] + float(blend) * fit[valid]
    except Exception:
        pass
    return xs


def decode_lane(preds, no_lane_idx=None, topk=5, exist_thr=0.5, post_smooth=False, poly_degree=2, poly_blend=0.5):
    """Decode LaneRobot logits to [B, R, L] float x-grid values; no-lane becomes -1.

    Accepts cls-only tensor [B, X+1, R, L] or dict {cls, offset}.
    """
    import torch

    logits, offset = _split_preds(preds)
    if no_lane_idx is None:
        no_lane_idx = logits.shape[1] - 1
    x_grids = int(no_lane_idx)
    cls_logits = logits[:, :x_grids]
    probs = cls_logits.softmax(dim=1)
    k = max(1, min(int(topk), x_grids))
    topv, topi = probs.topk(k=k, dim=1)
    pred_x = (topv * topi.float()).sum(dim=1) / topv.sum(dim=1).clamp_min(1e-6)
    if offset is not None:
        pred_x = pred_x + offset.squeeze(1).clamp(-0.5, 0.5)
    no_lane_prob = logits.softmax(dim=1)[:, no_lane_idx]
    valid = no_lane_prob < float(exist_thr)
    pred_x = torch.where(valid, pred_x, torch.full_like(pred_x, -1.0))
    arr = pred_x.detach().cpu().numpy()
    if post_smooth:
        for b in range(arr.shape[0]):
            for l in range(arr.shape[2]):
                v = arr[b, :, l] >= 0
                arr[b, :, l] = _poly_smooth_1d(arr[b, :, l], v, degree=poly_degree, blend=poly_blend)
                arr[b, ~v, l] = -1
    return arr


def _row_y_pixels(row_anchors, h, row_y=None, y_start=0.67, y_end=1.0):
    if row_y is not None:
        if hasattr(row_y, 'detach'):
            row_y = row_y.detach().cpu().numpy()
        row_y = np.asarray(row_y, dtype=np.float32).reshape(-1)
        if row_y.size >= row_anchors:
            return np.clip(row_y[:row_anchors], 0, 1) * (h - 1)
    return np.linspace(float(y_start), float(y_end), row_anchors) * (h - 1)


def draw_lanes_on_image(img, lane_xy, x_grids, row_anchors, row_y=None, y_start=0.67, y_end=1.0, radius=2):
    """Draw lane points from [Y, L] x-grid labels on an RGB CHW or HWC image."""
    if hasattr(img, "detach"):
        img = img.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] in {1, 3}:
        img = img.transpose(1, 2, 0)
    if img.max() <= 1.0:
        img = img * 255
    img = img.astype(np.uint8).copy()
    h, w = img.shape[:2]
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    colors = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    y_positions = _row_y_pixels(row_anchors, h, row_y=row_y, y_start=y_start, y_end=y_end).astype(int)
    for lane_idx in range(lane_xy.shape[1]):
        color = colors[lane_idx % len(colors)]
        pts = []
        for r, y in enumerate(y_positions):
            xg = float(lane_xy[r, lane_idx])
            if xg < 0 or xg >= x_grids:
                continue
            x = int(round(xg / max(x_grids - 1, 1) * (w - 1)))
            pts.append((x, int(y)))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=max(1, radius))
    return np.asarray(pil)


def save_lane_grid(images, preds, targets, x_grids, row_anchors, save_path: Path, max_images=8, row_y=None, y_start=0.67, y_end=1.0):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    images = images[:max_images]
    preds = preds[:max_images]
    targets = targets[:max_images]
    if row_y is not None:
        row_y = row_y[:max_images]
    panels = []
    for idx, (img, pred, target) in enumerate(zip(images, preds, targets)):
        ry = None if row_y is None else row_y[idx]
        panels.append(draw_lanes_on_image(img, target, x_grids, row_anchors, row_y=ry, y_start=y_start, y_end=y_end))
        panels.append(draw_lanes_on_image(img, pred, x_grids, row_anchors, row_y=ry, y_start=y_start, y_end=y_end))
    if not panels:
        return
    h, w = panels[0].shape[:2]
    cols = 2
    rows = int(np.ceil(len(panels) / cols))
    canvas = Image.new("RGB", (cols * w, rows * h), (255, 255, 255))
    for i, p in enumerate(panels):
        canvas.paste(Image.fromarray(p), ((i % cols) * w, (i // cols) * h))
    canvas.save(save_path)
