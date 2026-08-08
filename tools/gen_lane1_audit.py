#!/usr/bin/env python3
"""Audit grids for lane1 (class 1) mixed old/new semantics in datasets0806.

Groups lane1 files by geometric characteristics (delta vs lane0, x median,
line fit, batch) and renders annotated grids for human inspection.

Usage:
    python tools/gen_lane1_audit.py --root /root/ds0806/datasets --out /root/lane1_audit
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_COLORS = {
    0: (255, 70, 70),
    1: (70, 220, 220),
    2: (80, 140, 255),
    3: (255, 180, 60),
}


def batch_of(name: str) -> str:
    if re.match(r"^\d{3,6}_\d{6,}$", name):
        return "seq"
    if name.startswith("frame_"):
        return "frame"
    if name.startswith("ds57"):
        return "ds57"
    if name.startswith("lane_"):
        return "lane"
    if re.match(r"^\d{8}_\d{6}_", name):
        return "date"
    if re.match(r"^\d{19}$", name):
        return "pure"
    return "other"


def load_lanes(label_path: Path) -> dict[int, list[tuple[int, float, float]]]:
    lanes: dict[int, list[tuple[int, float, float]]] = {}
    if not label_path.exists():
        return lanes
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        v = [float(x) for x in line.replace(",", " ").split()]
        if len(v) < 3:
            continue
        lid = int(v[0])
        pts = []
        for i in range((len(v) - 1) // 2):
            x, y = v[1 + 2 * i], v[2 + 2 * i]
            if x >= 0 and y >= 0:
                pts.append((i, x, y))
        lanes[lid] = pts
    return lanes


def features(name: str, lanes: dict[int, list[tuple[int, float, float]]]):
    f = {"name": name, "batch": batch_of(name), "delta": math.nan, "x_med": math.nan, "line_mae": math.nan}
    if 1 not in lanes or len(lanes[1]) < 3:
        return f
    l1 = np.array([(x, y) for _, x, y in lanes[1]])
    f["x_med"] = float(np.median(l1[:, 0]))
    A = np.vstack([l1[:, 1], np.ones(len(l1))]).T
    coef, *_ = np.linalg.lstsq(A, l1[:, 0], rcond=None)
    f["line_mae"] = float(np.mean(np.abs(l1[:, 0] - (coef[0] * l1[:, 1] + coef[1]))))
    if 0 in lanes:
        d0 = {i: x for i, x, _ in lanes[0]}
        d1 = {i: x for i, x, _ in lanes[1]}
        common = sorted(set(d0) & set(d1))
        if common:
            f["delta"] = float(np.median([d1[i] - d0[i] for i in common]))
    return f


def render(img_path: Path, label_path: Path, thumb_w: int) -> Image.Image:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
    h = int(im.height * thumb_w / im.width)
    im = im.resize((thumb_w, h))
    draw = ImageDraw.Draw(im)
    for lid, pts in load_lanes(label_path).items():
        color = CLASS_COLORS.get(lid, (255, 255, 255))
        is1 = lid == 1
        if not is1:
            color = tuple(int(c * 0.4) for c in color)
        xy = [(int(x * thumb_w), int(y * h)) for _, x, y in pts]
        if len(xy) >= 2:
            draw.line(xy, fill=color, width=5 if is1 else 2)
        for px in xy:
            r = 4 if is1 else 2
            draw.ellipse((px[0] - r, px[1] - r, px[0] + r, px[1] + r), fill=color)
    return im


def make_grid(feats: list[dict], img_dir: Path, lab_dir: Path, title: str, thumb_w: int, cols: int, font) -> Image.Image:
    rows = (len(feats) + cols - 1) // cols
    th = int(thumb_w * 1088 / 1920)
    pad = 6
    cap = 46
    bar = 50
    canvas = Image.new("RGB", (cols * thumb_w + (cols + 1) * pad, bar + rows * (th + cap) + (rows + 1) * pad), (20, 20, 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), title, fill=(255, 220, 100), font=font)
    for idx, f in enumerate(feats):
        r, c = divmod(idx, cols)
        x0 = pad + c * (thumb_w + pad)
        y0 = bar + pad + r * (th + cap + pad)
        canvas.paste(render(img_dir / (f["name"] + ".jpg"), lab_dir / (f["name"] + ".txt"), thumb_w), (x0, y0))
        txt = f"{f['batch']} d={f['delta']:.3f} x={f['x_med']:.3f} {f['name'][:34]}"
        draw.text((x0 + 4, y0 + th + 3), txt, fill=(220, 230, 240), font=font)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-group", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = root / "images" / "train"
    lab_dir = root / "labels_corrected" / "train"
    font = ImageFont.load_default(size=15)
    thumb_w = 380
    cols = 4

    feats = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        lanes = load_lanes(lab_dir / (img_path.stem + ".txt"))
        if 1 not in lanes:
            continue
        feats.append(features(img_path.stem, lanes))

    groups = {
        "delta_neg_big": lambda f: not math.isnan(f["delta"]) and f["delta"] < -0.15,
        "delta_neg_mid": lambda f: not math.isnan(f["delta"]) and -0.15 <= f["delta"] < -0.05,
        "delta_near0": lambda f: not math.isnan(f["delta"]) and -0.05 <= f["delta"] <= 0.0,
        "delta_pos": lambda f: not math.isnan(f["delta"]) and f["delta"] > 0.0,
        "x_left": lambda f: not math.isnan(f["x_med"]) and f["x_med"] < 0.35,
        "single1": lambda f: math.isnan(f["delta"]),
    }
    for b in ("seq", "ds57", "lane", "frame", "date"):
        groups[f"batch_{b}"] = (lambda bb: (lambda f: f["batch"] == bb))(b)

    for gname, pred in groups.items():
        picked = [f for f in feats if pred(f)]
        if not picked:
            print(f"{gname}: empty")
            continue
        picked = picked[: args.per_group]
        grid = make_grid(picked, img_dir, lab_dir, f"lane1 audit: {gname} (n={len(picked)})", thumb_w, cols, font)
        out_path = out_dir / f"audit_{gname}.jpg"
        grid.save(out_path, quality=85)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
