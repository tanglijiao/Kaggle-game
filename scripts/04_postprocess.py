"""Stage 4: post-process measurements.csv into submission variants.

--smooth a   : within each 5-frame video clip (resources/test_clips.json), value = (1-a)*own + a*clip median
--clamp      : physiological ranges from the data description (PA 5-45, FL 30-200, MT 10-50)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

COLS = ["pa_deg", "fl_mm", "mt_mm"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meas", default="/kaggle/working/measurements.csv")
    ap.add_argument("--clips", default=str(Path(__file__).resolve().parents[1] / "resources/test_clips.json"))
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--out", default="/kaggle/working/submission_smooth.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.meas)
    for c in COLS:
        df[c] = df[c].fillna(df.groupby("device")[c].transform("median")).fillna(df[c].median())
    clips = json.loads(Path(args.clips).read_text())
    idx = df.set_index("image_id")
    for clip in clips:
        clip = [c for c in clip if c in idx.index]
        if len(clip) < 2:
            continue
        med = idx.loc[clip, COLS].median()
        for c in COLS:
            idx.loc[clip, c] = (1 - args.smooth) * idx.loc[clip, c] + args.smooth * med[c]
    sub = idx.reset_index()[["image_id"] + COLS]
    sub["pa_deg"] = sub["pa_deg"].clip(5, 45)
    sub["fl_mm"] = sub["fl_mm"].clip(30, 200)
    sub["mt_mm"] = sub["mt_mm"].clip(10, 50)
    sub.round(3).to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(sub)} rows, smooth={args.smooth})")


if __name__ == "__main__":
    main()
