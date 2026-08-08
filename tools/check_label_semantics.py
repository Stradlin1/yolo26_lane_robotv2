#!/usr/bin/env python3
"""Compare four coordinate-semantics hypotheses for datasets0806 labels.

H1: x=vals[1], y=vals[2]  (label native x,y; y top->bottom)
H2: x=vals[2], y=vals[1]  (label native y,x; y top->bottom)
H3: x=vals[1], y=1-vals[2] (x,y; y bottom->top)
H4: x=vals[2], y=1-vals[1] (y,x; y bottom->top)

For a sample of images we score each hypothesis by how often labeled points
land on strong gradients (image 75th percentile) and by point-gradient median.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    rng = random.Random(args.seed)
    img_dir = root / "images" / "train"
    lab_dir = root / "labels_corrected" / "train"
    files = sorted(img_dir.glob("*.jpg"))
    sample = rng.sample(files, min(args.n, len(files)))

    scores = {h: [] for h in ("H1", "H2", "H3", "H4")}
    for img_path in sample:
        lab_path = lab_dir / (img_path.stem + ".txt")
        pairs = []
        if lab_path.exists():
            for line in lab_path.read_text().splitlines():
                if not line.strip():
                    continue
                vals = [float(v) for v in line.replace(",", " ").split()]
                if len(vals) < 3:
                    continue
                for i in range((len(vals) - 1) // 2):
                    a, b = vals[1 + 2 * i], vals[2 + 2 * i]
                    if a >= 0.0 and b >= 0.0:
                        pairs.append((a, b))
        if not pairs:
            continue

        gray = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        h, w = gray.shape
        thr75 = np.percentile(mag, 75)

        transforms = {
            "H1": lambda a, b: (a, b),
            "H2": lambda a, b: (b, a),
            "H3": lambda a, b: (a, 1.0 - b),
            "H4": lambda a, b: (b, 1.0 - a),
        }
        for name, transform in transforms.items():
            vals_p = []
            for a, b in pairs:
                x_n, y_n = transform(a, b)
                px = int(round(x_n * (w - 1)))
                py = int(round(y_n * (h - 1)))
                px = min(max(px, 0), w - 1)
                py = min(max(py, 0), h - 1)
                vals_p.append(mag[py, px])
            vals_p = np.asarray(vals_p)
            scores[name].append(float(np.mean(vals_p > thr75)))

    print(f"{'hyp':4s} {'pct>img75 (median)':22s} {'pct>img75 (mean)':22s}")
    for name in ("H1", "H2", "H3", "H4"):
        arr = np.asarray(scores[name])
        print(f"{name:4s} {np.median(arr):22.1%} {arr.mean():22.1%}")


if __name__ == "__main__":
    main()
