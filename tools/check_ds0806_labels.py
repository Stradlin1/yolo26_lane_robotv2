#!/usr/bin/env python3
"""Objective check that datasets0806 labels are aligned with their images.

For a random sample of images, every labeled point is compared against the
image's local gradient magnitude (Sobel). If labels are attached to the actual
frame, points should lie on edges/color boundaries far more often than random
pixels.

Usage:
    python tools/check_ds0806_labels.py --root /root/ds0806/datasets --n 40
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def load_points(label_path: Path) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    if not label_path.exists():
        return pts
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        vals = [float(v) for v in line.replace(",", " ").split()]
        if len(vals) < 3:
            continue
        for i in range((len(vals) - 1) // 2):
            x_n, y_n = vals[1 + 2 * i], vals[2 + 2 * i]
            if x_n >= 0.0 and y_n >= 0.0:
                pts.append((x_n, y_n))
    return pts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.root)
    rng = random.Random(args.seed)

    all_rows = []
    for split in ("train", "valid"):
        img_dir = root / "images" / split
        lab_dir = root / "labels_corrected" / split
        files = sorted(img_dir.glob("*.jpg"))
        sample = rng.sample(files, min(args.n // 2, len(files)))

        for img_path in sample:
            lab_path = lab_dir / (img_path.stem + ".txt")
            pts = load_points(lab_path)
            if not pts:
                print(f"{split} {img_path.name}: NO POINTS")
                continue
            gray = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.sqrt(gx * gx + gy * gy)
            h, w = gray.shape
            q = np.percentile(mag, [50, 75, 90, 99])

            pvals = []
            for x_n, y_n in pts:
                px = int(round(x_n * (w - 1)))
                py = int(round(y_n * (h - 1)))
                px = min(max(px, 0), w - 1)
                py = min(max(py, 0), h - 1)
                pvals.append(mag[py, px])
            pvals = np.asarray(pvals)
            above75 = float(np.mean(pvals > q[1]))
            above90 = float(np.mean(pvals > q[2]))
            print(
                f"{split} {img_path.name[:50]:52s} pts={len(pvals):3d} "
                f"pt-med={np.median(pvals):7.1f} img75={q[1]:7.1f} "
                f"pct>img75={above75:5.1%} pct>img90={above90:5.1%}"
            )
            all_rows.append((split, img_path.name, len(pvals), np.median(pvals), q[1], above75, above90))

    if all_rows:
        med_pct = np.median([r[5] for r in all_rows])
        med_pct90 = np.median([r[6] for r in all_rows])
        print(f"\nSUMMARY: median pct of labeled points above image-75th gradient = {med_pct:.1%}")
        print(f"SUMMARY: median pct of labeled points above image-90th gradient = {med_pct90:.1%}")


if __name__ == "__main__":
    main()
