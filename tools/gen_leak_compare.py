#!/usr/bin/env python3
"""Generate side-by-side train/valid leak evidence images for datasets0806.

Picks shared video segments (same seq+day / date / lane key present in both
train and valid), stitches a few frames per segment into one annotated image,
and writes them to OUT_DIR.

Usage:
    python tools/gen_leak_compare.py --root /root/ds0806/datasets --out /root/ds0806_leak_check
"""

from __future__ import annotations

import argparse
import datetime
import re
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def day_of(ns_str: str) -> str:
    try:
        dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=int(ns_str) / 1e9)
        return dt.strftime("%Y%m%d")
    except Exception:
        return "?"


def video_key(name: str) -> tuple[str, str]:
    m = re.match(r"^(\d{3,6})_(\d{6,})$", name)
    if m:
        return ("seq", m.group(1) + "_" + day_of(m.group(2)))
    m = re.match(r"^frame_(\d+)_(\d{6,})$", name)
    if m:
        return ("frame", m.group(1) + "_" + day_of(m.group(2)))
    m = re.match(r"^ds57_([a-z]*)_?(\d{6,})$", name)
    if m:
        return ("ds57", m.group(1) + "_" + m.group(2))
    m = re.match(r"^lane_(\d{8}_\d{6})_\d+$", name)
    if m:
        return ("lane", m.group(1))
    m = re.match(r"^(\d{8}_\d{6})_\d+$", name)
    if m:
        return ("date", m.group(1))
    if re.match(r"^\d{19}$", name):
        return ("pure", name)
    return ("other", name)


def ns_of(name: str) -> int | None:
    m = re.search(r"(\d{10,})", name)
    return int(m.group(1)) if m else None


def pick_examples(shared: dict[tuple[str, str], tuple[list[str], list[str]]]) -> list[tuple[str, str, list[str], list[str]]]:
    """Pick up to 6 shared segments with good coverage across types."""
    by_type: dict[str, list[tuple[str, str, list[str], list[str]]]] = defaultdict(list)
    for key, (tr, va) in shared.items():
        by_type[key[0]].append((key[0], key[1], tr, va))

    wanted: list[tuple[str, str, list[str], list[str]]] = []
    for t in ("seq", "date", "lane"):
        items = sorted(by_type.get(t, []), key=lambda x: len(x[2]) + len(x[3]), reverse=True)
        for item in items:
            if len(item[2]) + len(item[3]) >= 3:
                wanted.append(item)
            if len([w for w in wanted if w[0] == t]) >= 2:
                break
    return wanted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Dataset root containing images/train and images/valid")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr_dir = root / "images" / "train"
    va_dir = root / "images" / "valid"

    tr_names = [p.stem for p in sorted(tr_dir.iterdir()) if p.suffix.lower() == ".jpg"]
    va_names = [p.stem for p in sorted(va_dir.iterdir()) if p.suffix.lower() == ".jpg"]

    tr_key = {n: video_key(n) for n in tr_names}
    va_key = {n: video_key(n) for n in va_names}
    tr_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    va_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for n, k in tr_key.items():
        tr_groups[k].append(n)
    for n, k in va_key.items():
        va_groups[k].append(n)

    shared = {}
    for k in set(tr_groups) & set(va_groups):
        shared[k] = (sorted(tr_groups[k]), sorted(va_groups[k]))

    examples = pick_examples(shared)
    print(f"shared segments: {len(shared)}; picked {len(examples)} for display")

    title_font = ImageFont.load_default(size=26)
    label_font = ImageFont.load_default(size=17)
    bar_h = 64

    for t, seg, tr_frames, va_frames in examples:
        frames = [(f, "train") for f in tr_frames[:2]] + [(f, "valid") for f in va_frames[:2]]
        # order by timestamp so the sequence reads naturally
        frames.sort(key=lambda x: ns_of(x[0]) or 0)

        imgs = []
        for fname, _split in frames:
            img_path = (tr_dir if _split == "train" else va_dir) / (fname + ".jpg")
            with Image.open(img_path) as im:
                im = im.convert("RGB")
            h = 220
            w = int(im.width * h / im.height)
            imgs.append(im.resize((w, h)))

        canvas_w = sum(im.width for im in imgs) + 8 * (len(imgs) - 1)
        canvas = Image.new("RGB", (canvas_w, bar_h + 220 + 28), (24, 24, 28))
        draw = ImageDraw.Draw(canvas)
        title = f"{t} | segment {seg}"
        draw.text((10, 12), title, fill=(255, 220, 100), font=title_font)

        x = 0
        prev_ns = None
        for idx, (fname, _split) in enumerate(frames):
            im = imgs[idx]
            canvas.paste(im, (x, bar_h))
            ns = ns_of(fname) or 0
            dtxt = ""
            if prev_ns is not None:
                dtxt = f"  dT={abs(ns - prev_ns) / 1e9:.1f}s"
            prev_ns = ns
            label = f"{_split} {fname}{dtxt}"
            draw.text((x + 4, bar_h + 220 + 5), label[:60], fill=(200, 220, 255), font=label_font)
            x += im.width + 8

        safe = re.sub(r"[^0-9A-Za-z]+", "_", seg)[:50]
        out_path = out_dir / f"leak_{t}_{safe}.jpg"
        canvas.save(out_path, quality=88)
        print(f"saved {out_path} ({canvas.width}x{canvas.height})")


if __name__ == "__main__":
    main()
