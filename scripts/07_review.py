"""Overlay sheets for visual review of a geometry variant on any map set.

  python scripts/07_review.py --maps MAPS --set test --images /home/claude/test_imgs --out /tmp/review \
      [--ids IMG_00163 IMG_00058] [--param source='"img"']
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from umud.evalsets import load_maps  # noqa: E402
from umud.geometry import draw, measure  # noqa: E402
from umud.geometry2 import draw2, measure2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True)
    ap.add_argument("--set", default="test")
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", nargs="*")
    ap.add_argument("--param", action="append", default=[])
    ap.add_argument("--per-sheet", type=int, default=6)
    args = ap.parse_args()
    params = {k: json.loads(v) for k, v in (p.split("=", 1) for p in args.param)}
    maps = Path(args.maps)
    idx = pd.read_csv(maps / "index.csv")
    idx = idx[idx.set == args.set]
    if args.ids:
        idx = idx[idx.stem.str.split("__").str[-1].isin(args.ids)]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    imgs = {p.stem: p for p in Path(args.images).rglob("*") if p.suffix.lower() in (".tif", ".tiff", ".png")}
    tiles = []
    for stem, h, w in zip(idx.stem, idx.h, idx.w):
        key = stem.split("__")[-1]
        if key not in imgs:
            continue
        color = cv2.imread(str(imgs[key]), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        apo, fasc = load_maps(maps, args.set, stem, int(h), int(w))
        r1 = measure(apo, fasc)
        r2 = measure2(apo, fasc, gray, **params)
        a = draw(color, r1)
        b = draw2(color, r2)
        heat = cv2.applyColorMap((np.clip(fasc, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        heat = cv2.addWeighted(heat, 0.6, cv2.applyColorMap((np.clip(apo, 0, 1) * 255).astype(np.uint8),
                                                            cv2.COLORMAP_WINTER), 0.4, 0)
        row = np.hstack([cv2.resize(x, (480, int(480 * h / w))) for x in (a, b, heat)])
        cv2.putText(row, key, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        tiles.append(row)
    for s in range(0, len(tiles), args.per_sheet):
        chunk = tiles[s:s + args.per_sheet]
        wmax = max(t.shape[1] for t in chunk)
        sheet = np.vstack([cv2.copyMakeBorder(t, 0, 4, 0, wmax - t.shape[1], cv2.BORDER_CONSTANT) for t in chunk])
        cv2.imwrite(str(out / f"sheet_{s // args.per_sheet:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(len(tiles), "tiles")


if __name__ == "__main__":
    main()
