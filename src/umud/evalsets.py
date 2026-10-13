"""Offline evaluation on saved probability maps (scripts/05_infer_maps.py output) against the public
UMUD expert-labelled sets from OSF (xbawc):

  bench : 35 images, 7 raters (MT/FL mm, PA deg), px/cm per image      -> labels = mean of raters
  gm    : 168 Telemed GM calf-raise frames, 3 raters, PA deg + FL px    -> labels = rater mean
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

EXT = Path("/home/claude/ext")
BENCH_XLSX = EXT / "bench_arch/benchmark_dataset_architecture_v0.1.0/Results_benchmark_architecture_v0.1.0.xlsx"
GM_XLSX = (EXT / "benchmark_dataset_architecture_GM_dynamic_v0.1.0/benchmark_dataset_architecture_GM_dynamic_v0.1.0"
           / "benchmark_architecture_GM_calf_raise_v0.1.0.xlsx")


def bench_labels() -> pd.DataFrame:
    x = pd.read_excel(BENCH_XLSX)
    out = pd.DataFrame({"key": x["ImageID"], "px_per_mm": x["Scale_pixel_per_cm"] / 10.0})
    for v, col in (("MT", "mt_mm"), ("FL", "fl_mm"), ("PA", "pa_deg")):
        m = x[[f"R{i}_{v}" for i in range(1, 8)]].astype(float)
        if v == "MT":
            m = m.where(m < 60)  # one obvious typo (80.1 mm)
        out[col] = m.mean(axis=1)
        out[col + "_sd"] = m.std(axis=1)
        out["dl_" + col] = x[f"DLTrack_{v}"]
    return out


def gm_labels() -> pd.DataFrame:
    x = pd.read_excel(GM_XLSX, sheet_name=None)
    pa = x["PA"].dropna(subset=["frame"])
    fl = x["FL"]
    df = pd.DataFrame({"frame": pa["frame"].astype(int), "pa_deg": pa["Mean"].values})
    df = df.merge(pd.DataFrame({"frame": fl["frame"].astype(int), "fl_px": fl["Mean"].values}), on="frame", how="left")
    df["key"] = df["frame"].map(lambda i: f"frame_{i:05d}")
    return df


def load_maps(maps: Path, set_name: str, stem: str, h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    out = []
    for task in ("apo", "fasc"):
        a = cv2.imread(str(maps / set_name / task / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        out.append(cv2.resize(a.astype(np.float32) / 255.0, (w, h), interpolation=cv2.INTER_LINEAR))
    return out[0], out[1]
