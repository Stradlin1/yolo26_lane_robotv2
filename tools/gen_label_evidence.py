#!/usr/bin/env python3
"""Render one full-size label-evidence image: original frame + overlay + raw label text."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_COLORS = {
    0: (255, 70, 70),
    1: (70, 220, 220),
    2: (80, 140, 255),
    3: (255, 180, 60),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--name", required=True, help="Image basename without extension")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    img_path = root / "images" / args.split / (args.name + ".jpg")
    lab_path = root / "labels_corrected" / args.split / (args.name + ".txt")
    raw = lab_path.read_text() if lab_path.exists() else "(missing)"

    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    overlay = img.copy()
    for line in raw.splitlines():
        if not line.strip():
            continue
        vals = [float(v) for v in line.replace(",", " ").split()]
        lid = int(vals[0])
        color = CLASS_COLORS.get(lid, (255, 255, 255))
        pts = []
        for i in range((len(vals) - 1) // 2):
            x_n, y_n = vals[1 + 2 * i], vals[2 + 2 * i]
            if x_n >= 0.0 and y_n >= 0.0:
                pts.append((int(x_n * (w - 1)), int(y_n * (h - 1))))
        if len(pts) >= 2:
            cv2.polylines(overlay, [np.asarray(pts, dtype=np.int32)], False, color, 6)
        for px, py in pts:
            cv2.circle(overlay, (px, py), 10, color, -1)

    # text panel with raw label
    font = ImageFont.load_default(size=15)
    lines = raw.splitlines()[:12]
    panel_h = max(90, 24 * len(lines) + 30)
    canvas = Image.new("RGB", (w, h + panel_h), (20, 20, 24))
    canvas.paste(Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, h + 8), f"{args.split}/{args.name}.jpg  (raw label below)", fill=(255, 220, 100), font=font)
    for i, ln in enumerate(lines[:12]):
        draw.text((12, h + 30 + i * 22), ln[:180], fill=(210, 225, 240), font=font)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
