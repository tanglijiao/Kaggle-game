"""Stage 1: px/mm for every test image from the on-screen ruler (see src/umud/scale.py).

Writes resources/test_scale.csv (committed: derived metadata, no image content) and optional check strips.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from umud.paths import find_data_root, find_subdir  # noqa: E402
from umud.scale import detect_scale  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="resources/test_scale.csv")
    args = ap.parse_args()

    root = find_data_root(args.data)
    tdir = find_subdir(root, "test_images")
    files = sorted(p for p in tdir.rglob("*") if p.suffix.lower() in (".tif", ".tiff", ".png"))
    rows = []
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        r = detect_scale(img, p.suffix)
        rows.append(dict(image_id=p.name, device=r.device, px_per_mm=round(r.px_per_mm, 4),
                         period_px=round(r.period_px, 3), mm_per_period=r.mm_per_period,
                         n_ticks=len(r.ticks), ok=r.ok, note=r.note,
                         height=img.shape[0], width=img.shape[1]))
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.groupby("device").px_per_mm.agg(["count", "min", "median", "max"]))
    bad = df[~df.ok]
    print(f"{len(df)} images, {len(bad)} failed")
    if len(bad):
        print(bad)


if __name__ == "__main__":
    main()
