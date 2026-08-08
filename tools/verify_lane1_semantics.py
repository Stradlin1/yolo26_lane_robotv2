#!/usr/bin/env python3
"""Verify the hypothesized lane1 semantic on datasets0806.

Hypothesis: lane1_x(row) ≈ (lane0_x(row) + L) / 2 + delta
  - L is a fixed left reference (image left edge / cone-channel boundary).
  - delta carries temporal inertia from the previous frame's lane1.

We test: ratio x1/x0, best constant L, per-row-band behavior, and frame-to-frame
stability of lane1 vs lane0 within the same seq video.
"""

from __future__ import annotations

import argparse
import collections
import math
import re
from pathlib import Path

import numpy as np


def load_lanes(label_path: Path) -> dict[int, list[tuple[float, float]]]:
    lanes: dict[int, list[tuple[float, float]]] = {}
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
                pts.append((x, y))
        lanes[lid] = pts
    return lanes


def seq_key(name: str) -> str:
    m = re.match(r"^(\d{3,6})_\d{6,}$", name)
    if m:
        return "seq_" + m.group(1)
    m = re.match(r"^lane_(\d{8}_\d{6})_\d+$", name)
    if m:
        return "lane_" + m.group(1)
    m = re.match(r"^(\d{8}_\d{6})_\d+$", name)
    if m:
        return "date_" + m.group(1)
    return "other_" + name.split("_")[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-files", type=int, default=4000)
    args = ap.parse_args()

    label_dir = Path(args.root) / "labels_corrected" / args.split
    files = sorted(label_dir.glob("*.txt"))
    files = files[: args.max_files]

    ratios: list[float] = []
    offsets: list[float] = []
    by_row: dict[str, list[float]] = collections.defaultdict(list)
    mid_err: dict[float, list[float]] = {L: [] for L in (0.0, 0.05, 0.1, 0.15, 0.2)}
    pairs = 0

    seq_lanes: dict[str, list[tuple[str, dict[int, list[tuple[float, float]]]]]] = collections.defaultdict(list)

    for f in files:
        lanes = load_lanes(f)
        if 0 in lanes and 1 in lanes:
            l0 = {round(y, 4): x for x, y in lanes[0]}
            l1 = {round(y, 4): x for x, y in lanes[1]}
            common = sorted(set(l0) & set(l1))
            if common:
                pairs += 1
                for y in common:
                    x0, x1 = l0[y], l1[y]
                    if x0 <= 0.02:
                        continue
                    ratios.append(x1 / x0)
                    offsets.append(x1 - x0)
                    band = "near" if y >= 0.9 else ("mid" if y >= 0.6 else "far")
                    by_row[band].append(x1 / x0)
                    for L in mid_err:
                        mid_err[L].append(x1 - (x0 + L) / 2.0)
        seq_lanes[seq_key(f.stem)].append((f.stem, lanes))

    r = np.asarray(ratios)
    o = np.asarray(offsets)
    print(f"0+1 label files: {pairs}")
    print(f"ratio x1/x0: p10={np.percentile(r,10):.3f} p25={np.percentile(r,25):.3f} "
          f"p50={np.percentile(r,50):.3f} p75={np.percentile(r,75):.3f} p90={np.percentile(r,90):.3f}")
    print(f"offset x1-x0: p10={np.percentile(o,10):.3f} p50={np.percentile(o,50):.3f} p90={np.percentile(o,90):.3f}")
    for band in ("near", "mid", "far"):
        a = np.asarray(by_row[band])
        if len(a):
            print(f"  {band}: n={len(a)} ratio p50={np.percentile(a,50):.3f} p10={np.percentile(a,10):.3f} p90={np.percentile(a,90):.3f}")

    print("midpoint error x1-(x0+L)/2:")
    for L in (0.0, 0.05, 0.1, 0.15, 0.2):
        e = np.abs(np.asarray(mid_err[L]))
        print(f"  L={L:.2f}: MAE={e.mean():.4f} p90={np.percentile(e,90):.4f}")

    # temporal stability: same-seq consecutive frames
    print("temporal frame-to-frame mean |dx| (normalized):")
    for skey, items in list(seq_lanes.items()):
        if len(items) < 3 or not skey.startswith(("seq_", "date_", "lane_")):
            continue
        d0, d1 = [], []
        prev = None
        for _, lanes in items:
            if prev is None:
                prev = lanes
                continue
            for lid, dxs in ((0, d0), (1, d1)):
                if lid in prev and lid in lanes:
                    p = {round(y, 4): x for x, y in prev[lid]}
                    q = {round(y, 4): x for x, y in lanes[lid]}
                    for y in set(p) & set(q):
                        dxs.append(abs(p[y] - q[y]))
            prev = lanes
        if d0 and d1:
            print(f"  {skey[:26]:28s} n={len(items)} lane0={np.mean(d0):.4f} lane1={np.mean(d1):.4f}")


if __name__ == "__main__":
    main()
