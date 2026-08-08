#!/usr/bin/env python3
"""Generate annotated preview grids for the datasets0806 package.

Samples images from train/valid stratified by lane class combination,
overlays the row-anchor labels (using the label file's native (y, x)
normalized semantics), and stitches them into a grid image.

Usage:
    python tools/gen_ds0806_preview.py --root /root/ds0806/datasets --out /root/ds0806_preview
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


CLASS_COLORS = {
    0: (255, 70, 70),    # lane_follow
    1: (70, 220, 220),   # lead_lane
    2: (80, 140, 255),   # channel_left
    3: (255, 180, 60),   # channel_right
}


def class_combo(label_path: Path) -> str:
    try:
        cls = sorted({line.split()[0] for line in label_path.read_text().splitlines() if line.strip()})
        return ",".join(cls) if cls else "(empty)"
    except Exception:
        return "?"


def load_lanes(label_path: Path):
    """Return {lane_id: [(x_norm, y_norm), ...]} using label's native (x, y) semantics."""
    lanes: dict[int, list[tuple[float, float]]] = defaultdict(list)
    if not label_path.exists():
        return lanes
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        vals = [float(v) for v in line.replace(",", " ").split()]
        if len(vals) < 3:
            continue
        lid = int(vals[0])
        if lid < 0 or lid > 3:
            continue
        pts = []
        for i in range((len(vals) - 1) // 2):
            x_n, y_n = vals[1 + 2 * i], vals[2 + 2 * i]
            if x_n >= 0.0 and y_n >= 0.0:
                pts.append((x_n, y_n))
        lanes[lid] = pts
    return lanes


def render_frame(img_path: Path, label_path: Path, thumb_w: int, highlight: int | None = None) -> Image.Image:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
    thumb_h = int(im.height * thumb_w / im.width)
    im = im.resize((thumb_w, thumb_h))
    draw = ImageDraw.Draw(im)
    lanes = load_lanes(label_path)
    for lid, pts in lanes.items():
        color = CLASS_COLORS.get(lid, (255, 255, 255))
        is_hl = highlight is not None and lid == highlight
        if not is_hl:
            color = tuple(int(c * 0.35) for c in color)
        xy = [(int(x * thumb_w), int(y * thumb_h)) for x, y in pts]
        if len(xy) >= 2:
            draw.line(xy, fill=color, width=5 if is_hl else 2)
        for px in xy:
            r = 4 if is_hl else 2
            draw.ellipse((px[0] - r, px[1] - r, px[0] + r, px[1] + r), fill=color)
    return im


def make_grid(
    items: list[tuple[Path, Path]],
    thumb_w: int,
    cols: int,
    title: str,
    label_font,
    highlight: int | None = None,
) -> Image.Image:
    rows = (len(items) + cols - 1) // cols
    thumb_h = int(thumb_w * 1088 / 1920)
    pad = 6
    cap_h = 34
    bar_h = 52
    canvas_w = cols * thumb_w + (cols + 1) * pad
    canvas_h = bar_h + rows * (thumb_h + cap_h) + (rows + 1) * pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (22, 22, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill=(255, 220, 100), font=label_font)

    for idx, (img_path, label_path) in enumerate(items):
        r, c = divmod(idx, cols)
        x0 = pad + c * (thumb_w + pad)
        y0 = bar_h + pad + r * (thumb_h + cap_h + pad)
        frame = render_frame(img_path, label_path, thumb_w, highlight)
        canvas.paste(frame, (x0, y0))
        combo = class_combo(label_path)
        fname = img_path.name
        draw.text((x0 + 4, y0 + thumb_h + 4), f"{combo}  {fname[:44]}", fill=(220, 230, 240), font=label_font)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-combo", type=int, default=4)
    ap.add_argument("--valid-count", type=int, default=20)
    ap.add_argument("--filter-class", type=int, default=None, help="Only sample images containing this class")
    ap.add_argument("--highlight-class", type=int, default=None, help="Draw this class brightly, dim others")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_font = ImageFont.load_default(size=15)
    thumb_w = 380
    cols = 5

    for split in ("train", "valid"):
        img_dir = root / "images" / split
        lab_dir = root / "labels_corrected" / split
        files = sorted(img_dir.glob("*.jpg"))
        combos: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
        for img_path in files:
            lab_path = lab_dir / (img_path.stem + ".txt")
            combo = class_combo(lab_path)
            if args.filter_class is not None and str(args.filter_class) not in combo.split(","):
                continue
            combos[combo].append((img_path, lab_path))

        combo_counts = {k: len(v) for k, v in combos.items()}
        print(f"[{split}] {len(files)} images; combos: {combo_counts}")

        if split == "train":
            # Stratified sample: one row per dominant combo
            order = ["0", "0,1", "2,3", "2", "3", "0,3"]
            items: list[tuple[Path, Path]] = []
            for combo in order:
                pool = combos.get(combo, [])
                items.extend(pool[: args.per_combo])
            title = f"datasets0806 TRAIN preview ({len(files)} imgs)"
        else:
            # Stratified by ratio, cap per combo
            total = sum(len(v) for v in combos.values())
            items = []
            for combo, pool in combos.items():
                n = max(1, round(args.valid_count * len(pool) / total))
                items.extend(pool[:n])
            items = items[: args.valid_count]
            title = f"datasets0806 VALID preview ({len(files)} imgs)"

        if args.filter_class is not None:
            title += f" | class {args.filter_class} only"
        title += " | lane0/1/2/3 red/cyan/blue/orange"

        grid = make_grid(items, thumb_w, cols, title, label_font, args.highlight_class)
        suffix = f"_class{args.filter_class}" if args.filter_class is not None else ""
        out_path = out_dir / f"preview_ds0806_{split}{suffix}.jpg"
        grid.save(out_path, quality=86)
        print(f"saved {out_path} ({grid.width}x{grid.height})")


if __name__ == "__main__":
    main()
