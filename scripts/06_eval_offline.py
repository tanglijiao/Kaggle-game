"""Measure PA/FL/MT from saved probability maps and score against the public expert-labelled sets.

  python scripts/06_eval_offline.py --maps /home/claude/maps_v1/maps [--param extrap_max=0.5 ...] [--test-out sub.csv]

Prints MAE per variable on bench (35 imgs, mean of 7 raters) and gm (167 frames, PA only + FL in px),
optionally writes a test submission with the same geometry.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from umud.evalsets import bench_labels, gm_labels, load_maps  # noqa: E402
from umud.geometry import measure  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _one(args):
    maps, set_name, stem, h, w, params = args
    apo, fasc = load_maps(Path(maps), set_name, stem, h, w)
    r = measure(apo, fasc, **params)
    return {"stem": stem, "pa_deg": r.pa_deg, "fl_px": r.fl_px, "mt_px": r.mt_px, "n_fasc": r.n_fasc, "note": r.note}


def run_set(maps: Path, idx: pd.DataFrame, set_name: str, params: dict, workers: int = 2) -> pd.DataFrame:
    sub = idx[idx.set == set_name]
    jobs = [(str(maps), set_name, s, int(h), int(w), params) for s, h, w in zip(sub.stem, sub.h, sub.w)]
    with ProcessPoolExecutor(workers) as ex:
        rows = list(ex.map(_one, jobs, chunksize=4))
    df = pd.DataFrame(rows)
    df["key"] = df["stem"].str.split("__").str[-1]
    return df


def parse_params(items):
    p = {}
    for it in items or []:
        k, v = it.split("=")
        p[k] = json.loads(v) if v[:1] in "[{" else float(v)
    return p


def evaluate(maps: Path, params: dict, sets=("bench", "gm"), verbose=True) -> dict:
    idx = pd.read_csv(maps / "index.csv")
    res = {}
    if "bench" in sets:
        lab = bench_labels()
        d = run_set(maps, idx, "bench", params).merge(lab, on="key")
        d["fl_mm_p"] = d.fl_px / d.px_per_mm
        d["mt_mm_p"] = d.mt_px / d.px_per_mm
        e = {"PA": (d.pa_deg_x - d.pa_deg_y), "FL": (d.fl_mm_p - d.fl_mm), "MT": (d.mt_mm_p - d.mt_mm)}
        res["bench"] = {k: (float(v.abs().mean()), float(v.mean()), int(v.isna().sum())) for k, v in e.items()}
        res["bench_df"] = d
    if "gm" in sets:
        lab = gm_labels()
        d = run_set(maps, idx, "gm", params).merge(lab, on="key")
        e = {"PA": d.pa_deg_x - d.pa_deg_y, "FLpx": d.fl_px_x - d.fl_px_y}
        res["gm"] = {k: (float(v.abs().mean()), float(v.mean()), int(v.isna().sum())) for k, v in e.items()}
        res["gm_df"] = d
    if verbose:
        for s in ("bench", "gm"):
            if s in res:
                print(s, "  ".join(f"{k}: MAE {m:.2f} bias {b:+.2f} nan {n}" for k, (m, b, n) in res[s].items()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--maps", required=True)
    ap.add_argument("--param", action="append")
    ap.add_argument("--test-out")
    args = ap.parse_args()
    maps = Path(args.maps)
    params = parse_params(args.param)
    evaluate(maps, params)
    if args.test_out:
        idx = pd.read_csv(maps / "index.csv")
        d = run_set(maps, idx, "test", params)
        sc = pd.read_csv(ROOT / "resources/test_scale.csv")
        sc["key"] = sc["image_id"].str.rsplit(".", n=1).str[0]
        d = d.merge(sc[["key", "image_id", "px_per_mm"]], on="key")
        d["fl_mm"] = d.fl_px / d.px_per_mm
        d["mt_mm"] = d.mt_px / d.px_per_mm
        d.to_csv(Path(args.test_out).with_suffix(".diag.csv"), index=False)
        sub = d[["image_id", "pa_deg", "fl_mm", "mt_mm"]].copy()
        for c in ("pa_deg", "fl_mm", "mt_mm"):
            sub[c] = sub[c].fillna(sub[c].median())
        sub["pa_deg"] = sub.pa_deg.clip(5, 45)
        sub["fl_mm"] = sub.fl_mm.clip(30, 200)
        sub["mt_mm"] = sub.mt_mm.clip(10, 50)
        sub.sort_values("image_id").round(3).to_csv(args.test_out, index=False)
        print("wrote", args.test_out, len(sub))


if __name__ == "__main__":
    main()
