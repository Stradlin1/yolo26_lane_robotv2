#!/usr/bin/env python3
"""Find which image a label file actually aligns with inside one seq group.

For a handful of seq prefixes (e.g. 000820), each label from that prefix is
drawn against every image from the same prefix. If labels are one-frame-off,
the best-matching image will be a sibling, not the same-named file.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


def load_pairs(label_path: Path):
    pairs = []
    if not label_path.exists():
        return pairs
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        vals = [float(v) for v in line.replace(",", " ").split()]
        if len(vals) < 3:
            continue
        for i in range((len(vals) - 1) // 2):
            x_n, y_n = vals[1 + 2 * i], vals[2 + 2 * i]
            if x_n >= 0.0 and y_n >= 0.0:
                pairs.append((x_n, y_n))
    return pairs


def hit_rate(pairs, mag, w, h, xform):
    if not pairs:
        return float("nan")
    thr75 = np.percentile(mag, 75)
    hits = 0
    for x_n, y_n in pairs:
        x_n, y_n = xform(x_n, y_n)
        px = int(round(x_n * (w - 1)))
        py = int(round(y_n * (h - 1)))
        px = min(max(px, 0), w - 1)
        py = min(max(py, 0), h - 1)
        hits += mag[py, px] > thr75
    return hits / len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--prefixes", default="000820,000124,001472,007407")
    ap.add_argument("--semantic", default="H1", choices=("H1", "H3"))
    args = ap.parse_args()

    root = Path(args.root)
    img_dir = root / "images" / "train"
    lab_dir = root / "labels_corrected" / "train"
    xform = {"H1": lambda a, b: (a, b), "H3": lambda a, b: (a, 1.0 - b)}[args.semantic]

    for prefix in args.prefixes.split(","):
        imgs = sorted(img_dir.glob(prefix + "_*.jpg"))
        labels = sorted(lab_dir.glob(prefix + "_*.txt"))
        if len(imgs) < 2:
            continue
        mags = {}
        for p in imgs:
            gray = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mags[p.stem] = (np.sqrt(gx * gx + gy * gy), gray.shape[1], gray.shape[0])

        print(f"\nprefix {prefix} ({args.semantic}): rows=labels cols=images")
        print("label\\img " + " ".join(f"{Path(i).stem[-14:]:>14}" for i in imgs))
        for lab in labels:
            pairs = load_pairs(lab)
            row = [f"{Path(lab).stem[-14:]:>14}"]
            for img in imgs:
                mag, w, h = mags[img.stem]
                row.append(f"{hit_rate(pairs, mag, w, h, xform):14.0%}")
            print(" ".join(row))


if __name__ == "__main__":
    main()
