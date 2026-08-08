#!/usr/bin/env python3
"""Remove old/noisy class-1 (lane1) labels from datasets0806.

Keeps lane1 rows only when the labeled points form a real segment:
  - at least 3 points,
  - well explained by one straight line (or two straight segments),
  - vertical/horizontal span large enough (a point-cluster under a cone is
    considered old/noise and removed),
  - lane0 exists in the same file.

Usage:
    dry run:  python tools/clean_lane1.py --root ... --threshold 0.015 \
              --min-y-span 0.04 --min-x-span 0.04 --preview-out /tmp/l1_del
    apply:    python tools/clean_lane1.py --root ... --threshold 0.015 \
              --min-y-span 0.04 --min-x-span 0.04 --apply
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CLASS_COLORS = {
    0: (255, 70, 70),
    1: (70, 220, 220),
    2: (80, 140, 255),
    3: (255, 180, 60),
}


def parse_lines(text: str) -> list[list[float]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        v = [float(x) for x in line.replace(",", " ").split()]
        if len(v) >= 3:
            rows.append(v)
    return rows


def lane1_points(row: list[float]) -> list[tuple[float, float]]:
    pts = []
    for i in range((len(row) - 1) // 2):
        x, y = row[1 + 2 * i], row[2 + 2 * i]
        if x >= 0 and y >= 0:
            pts.append((x, y))
    return pts


def line_mae(pts: list[tuple[float, float]]) -> float:
    if len(pts) < 2:
        return 0.0
    X = np.asarray([p[0] for p in pts])
    Y = np.asarray([p[1] for p in pts])
    A = np.vstack([Y, np.ones(len(Y))]).T
    coef, *_ = np.linalg.lstsq(A, X, rcond=None)
    return float(np.mean(np.abs(X - (coef[0] * Y + coef[1]))))


def two_segment_mae(pts: list[tuple[float, float]]) -> float:
    """Best split into two segments, average of the two line-fit MAEs."""
    if len(pts) < 4:
        return line_mae(pts)
    best = float("inf")
    for k in range(2, len(pts) - 1):
        mae = (line_mae(pts[:k]) * k + line_mae(pts[k:]) * (len(pts) - k)) / len(pts)
        best = min(best, mae)
    return best


def keep_straight(row: list[float], threshold: float) -> bool:
    pts = lane1_points(row)
    if len(pts) < 3:
        return False
    if line_mae(pts) <= threshold:
        return True
    return two_segment_mae(pts) <= threshold


def span_sufficient(row: list[float], min_y: float, min_x: float) -> bool:
    pts = lane1_points(row)
    if len(pts) < 2:
        return False
    ys = [p[1] for p in pts]
    xs = [p[0] for p in pts]
    return (max(ys) - min(ys)) >= min_y and (max(xs) - min(xs)) >= min_x


def render_removed(removed_items: list[tuple[str, str]], img_root: Path, out_dir: Path) -> None:
    """Render a grid of removed samples for human inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=14)
    thumb_w = 380
    cols = 4
    cap_h = 44
    bar_h = 46
    pad = 6
    n = len(removed_items)
    if n == 0:
        return
    n = min(n, 96)
    rows = (n + cols - 1) // cols
    th = int(thumb_w * 1088 / 1920)
    canvas = Image.new(
        "RGB",
        (cols * thumb_w + (cols + 1) * pad, bar_h + rows * (th + cap_h) + (rows + 1) * pad),
        (20, 20, 24),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"REMOVED lane1 ({n})", fill=(255, 160, 120), font=font)
    for idx, (split, name) in enumerate(removed_items[: cols * rows]):
        img_path = img_root / "images" / split / (name + ".jpg")
        lab_path = img_root / "labels_corrected" / split / (name + ".txt")
        with Image.open(img_path) as im:
            im = im.convert("RGB")
        h = int(im.height * thumb_w / im.width)
        im = im.resize((thumb_w, h))
        d = ImageDraw.Draw(im)
        for line in lab_path.read_text().splitlines():
            if not line.strip():
                continue
            v = [float(x) for x in line.replace(",", " ").split()]
            if len(v) < 3:
                continue
            lid = int(v[0])
            color = CLASS_COLORS.get(lid, (255, 255, 255))
            is1 = lid == 1
            if not is1:
                color = tuple(int(c * 0.4) for c in color)
            pts = []
            for i in range((len(v) - 1) // 2):
                x, y = v[1 + 2 * i], v[2 + 2 * i]
                if x >= 0 and y >= 0:
                    pts.append((int(x * thumb_w), int(y * h)))
            if len(pts) >= 2:
                d.line(pts, fill=color, width=5 if is1 else 2)
            for px in pts:
                r = 4 if is1 else 2
                d.ellipse((px[0] - r, px[1] - r, px[0] + r, px[1] + r), fill=color)
        r, c = divmod(idx, cols)
        x0 = pad + c * (thumb_w + pad)
        y0 = bar_h + pad + r * (th + cap_h + pad)
        canvas.paste(im, (x0, y0))
        draw.text((x0 + 4, y0 + th + 3), f"{split} {name[:38]}", fill=(230, 220, 220), font=font)
    out_path = out_dir / f"removed_{n}.jpg"
    canvas.save(out_path, quality=85)
    print(f"preview saved {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--threshold", type=float, default=0.015)
    ap.add_argument("--min-y-span", type=float, default=0.04)
    ap.add_argument("--min-x-span", type=float, default=0.04)
    ap.add_argument("--apply", action="store_true", help="Actually rewrite label files")
    ap.add_argument("--preview-out", type=Path, default=None, help="Render removed samples grid (dry run)")
    args = ap.parse_args()

    root = Path(args.root)
    total = kept = removed = lone = span_removed = 0
    removed_files: list[tuple[str, str]] = []
    for split in ("train", "valid"):
        lab_dir = root / "labels_corrected" / split
        for f in sorted(lab_dir.glob("*.txt")):
            rows = parse_lines(f.read_text())
            has0 = any(int(r[0]) == 0 for r in rows)
            new_rows = []
            for r in rows:
                if int(r[0]) != 1:
                    new_rows.append(r)
                    continue
                total += 1
                if not has0:
                    lone += 1
                    removed_files.append((split, f.stem))
                    continue
                if keep_straight(r, args.threshold) and span_sufficient(r, args.min_y_span, args.min_x_span):
                    new_rows.append(r)
                    kept += 1
                else:
                    removed += 1
                    if keep_straight(r, args.threshold):
                        span_removed += 1
                    removed_files.append((split, f.stem))
            if args.apply and len(new_rows) != len(rows):
                f.write_text("\n".join(" ".join(f"{v:.6f}" for v in r) for r in new_rows) + "\n")

    print(f"lane1 rows total: {total}")
    print(f"kept: {kept} ({100*kept/max(total,1):.1f}%)")
    print(f"removed (curvy/sparse): {removed - span_removed} ({100*(removed - span_removed)/max(total,1):.1f}%)")
    print(f"removed (small span): {span_removed} ({100*span_removed/max(total,1):.1f}%)")
    print(f"removed (no lane0): {lone} ({100*lone/max(total,1):.1f}%)")
    print("applied" if args.apply else "DRY RUN (no files changed)")
    if not args.apply:
        if args.preview_out is not None:
            render_removed(removed_files, root, args.preview_out)
        else:
            for x in removed_files[:10]:
                print("  example removed:", x)


if __name__ == "__main__":
    main()
