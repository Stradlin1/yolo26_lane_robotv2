#!/usr/bin/env python3
"""Stitch preview images into overview grids."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--thumb", type=int, default=280)
    args = ap.parse_args()

    src = Path(args.source)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob("*.jpg"))
    pad = 4
    per_grid = args.cols * args.cols
    for gi in range((len(files) + per_grid - 1) // per_grid):
        chunk = files[gi * per_grid : (gi + 1) * per_grid]
        thumbs = []
        for f in chunk:
            with Image.open(f) as im:
                im = im.convert("RGB")
            h = int(im.height * args.thumb / im.width)
            thumbs.append(im.resize((args.thumb, h)))
        th = max(t.height for t in thumbs)
        canvas = Image.new("RGB", (args.cols * args.thumb + (args.cols + 1) * pad, args.cols * th + (args.cols + 1) * pad), (20, 20, 24))
        for i, t in enumerate(thumbs):
            r, c = divmod(i, args.cols)
            canvas.paste(t, (pad + c * (args.thumb + pad), pad + r * (th + pad)))
        out_path = out_dir / f"grid_{gi + 1:02d}.jpg"
        canvas.save(out_path, quality=82)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
