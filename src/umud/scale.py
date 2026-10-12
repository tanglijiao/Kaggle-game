"""Pixel-to-millimetre calibration from on-screen depth rulers.

The organisers measured with square pixels using the *vertical* scale bar (discussions 689019, 748126),
so every image gets a single px/mm value. Five screen layouts occur in the test set:

  siemens   1200x800, black background, ruler on the right (x~1146-1156).
            Long ticks (11 px wide) are 1 cm apart regardless of depth; short ticks are 2 or 5 mm.
  lumify    1200x800 PNG, black background, ruler on the far left (x~2-14), ticks every 5 mm
            from 0 to the displayed depth.
  telemed   1200x800 grey UI, ruler left of the image (vertical line x~66, ticks to the right), 10 mm ticks.
  tel1088   1088x644 grey UI, ruler at x~0-8, 10 mm ticks.
  cropped   older cropped images (~1069x853, 959x853, ~465x513) with tick marks along the bottom edge only;
            the host states these ticks are 1 cm apart.

Each detector returns px/mm plus diagnostics (tick positions) so results can be checked visually.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScaleResult:
    device: str
    px_per_mm: float
    period_px: float
    mm_per_period: float
    ticks: list = field(default_factory=list)  # tick centre coordinates along the ruler axis
    axis: str = "y"                             # "y" = vertical ruler, "x" = bottom ruler
    ruler_pos: int = 0                          # x (vertical ruler) or y (bottom ruler) of the ruler
    ok: bool = True
    note: str = ""


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    return img[..., :3].astype(np.float32).mean(-1)


def classify_device(img: np.ndarray, ext: str) -> str:
    g = to_gray(img)
    h, w = g.shape
    if (w, h) == (1200, 800):
        grey_ui = g[300:500, 60:150].mean() > 30
        if grey_ui:
            return "telemed"
        return "lumify" if ext.lower() == ".png" else "siemens"
    if (w, h) == (1088, 644):
        return "tel1088"
    return "cropped"


def _runs(mask: np.ndarray, gap: int = 1) -> list[np.ndarray]:
    idx = np.where(mask)[0]
    if idx.size == 0:
        return []
    return np.split(idx, np.where(np.diff(idx) > gap)[0] + 1)


def _robust_period(ticks: list[float], lo: float, hi: float) -> float:
    """Period of a (possibly gappy) regular tick sequence: fit P so every gap is ~k*P."""
    t = np.asarray(sorted(ticks), dtype=float)
    d = np.diff(t)
    d = d[d > lo * 0.6]
    if d.size == 0:
        return float("nan")
    best, best_err = float("nan"), np.inf
    for p in np.arange(lo, hi, 0.05):
        k = np.maximum(1, np.round(d / p))
        err = np.abs(d - k * p).sum() + 0.5 * (k - 1).sum()  # mild penalty for assumed missing ticks
        if err < best_err:
            best, best_err = p, err
    # refine by least squares on k*P
    k = np.maximum(1, np.round(d / best))
    return float((d * k).sum() / (k * k).sum())


def _grid_period(ticks: list[float], lo: float, hi: float, tol: float = 3.0) -> tuple[float, list[float]]:
    """RANSAC-style fit of a regular grid t = t0 + k*P to tick candidates that may contain false
    positives (bright tissue near the ruler). Returns (P, inlier ticks). Prefers the largest P among
    fits with the most inliers, so a P/2 grid does not win by also matching every tick."""
    t = np.asarray(sorted(ticks), dtype=float)
    best = (-np.inf, 0.0, float("nan"), [])
    for i in range(len(t)):
        for j in range(i + 1, len(t)):
            for k in range(1, 8):
                p = (t[j] - t[i]) / k
                if not lo <= p <= hi:
                    continue
                r = (t - t[i]) / p
                inl = np.abs(r - np.round(r)) * p <= tol
                n = int(inl.sum())
                if n < 3:
                    continue
                kk_in = np.round(r[inl])
                expected = int(kk_in.max() - kk_in.min()) + 1      # grid points spanned by the inliers
                score = n - 0.5 * (expected - n)                  # penalise grids with many missing ticks
                if (score, p) > (best[0], best[1]):
                    # least-squares refine on inliers
                    kk = np.round(r[inl])
                    A = np.vstack([np.ones_like(kk), kk]).T
                    t0, pp = np.linalg.lstsq(A, t[inl], rcond=None)[0]
                    best = (score, p, float(pp), list(t[inl]))
    return best[2], best[3]


def _siemens(g: np.ndarray) -> ScaleResult:
    band = g[:, 1135:1165]
    rows = _runs(band.max(1) > 60)
    long_ticks, all_ticks = [], []
    for r in rows:
        seg = band[r[0]:r[-1] + 1]
        cols = np.where(seg.max(0) > 60)[0]
        width = cols.max() - cols.min() + 1
        y = float(r.mean())
        # ignore the yellow focus marker (starts further left, x<1145)
        if cols.min() + 1135 < 1144:
            continue
        all_ticks.append(y)
        if width >= 10:
            long_ticks.append(y)
    p = _robust_period(long_ticks, 80, 260) if len(long_ticks) >= 2 else float("nan")
    ok = np.isfinite(p)
    return ScaleResult("siemens", p / 10 if ok else float("nan"), p, 10.0, long_ticks, "y", 1151, ok,
                       f"{len(long_ticks)} long / {len(all_ticks)} ticks")


def _vertical_ruler(g: np.ndarray, device: str, x0: int, x1: int, mm_per_tick: float,
                    lo: float, hi: float) -> ScaleResult:
    band = g[:, x0:x1]
    bg = np.median(band)
    prof = band.max(1) - bg
    rows = _runs(prof > max(40, 0.35 * prof.max()))
    ticks = [float(r.mean()) for r in rows if len(r) <= 6]  # ticks are thin; skip text/labels
    p = _robust_period(ticks, lo, hi) if len(ticks) >= 2 else float("nan")
    ok = np.isfinite(p)
    return ScaleResult(device, p / mm_per_tick if ok else float("nan"), p, mm_per_tick, ticks, "y",
                       (x0 + x1) // 2, ok, f"{len(ticks)} ticks")


# Binarised "Tiefe" (depth) read-outs seen on the Lumify screen (x 1110-1200, y 330-362, >120),
# keyed by md5[:8]. The ruler uses 5 mm ticks up to 5 cm depth but 10 mm ticks at 6 cm, and the
# two cases look identical on the ruler, so the depth text is needed to tell them apart.
LUMIFY_DEPTH_MM = {"9881d635": 30, "a0b20107": 35, "518f3156": 40, "c6883b17": 45,
                   "2d6193ba": 50, "60cbe879": 60}


def lumify_depth_mm(g: np.ndarray) -> float | None:
    import hashlib

    crop = (g[330:362, 1110:1200] > 120).astype(np.uint8)
    return LUMIFY_DEPTH_MM.get(hashlib.md5(crop.tobytes()).hexdigest()[:8])


def _lumify(g: np.ndarray) -> ScaleResult:
    """Pure tick rows only: start at x>=4 (excludes the focus marker at x 0-3) and <=5 px tall
    (excludes the '0', '2', '4cm' label blocks that merge with their tick)."""
    band = g[:, :30]
    rows = _runs(band.max(1) > 80)
    ticks = []
    for r in rows:
        seg = band[r[0]:r[-1] + 1]
        cols = np.where(seg.max(0) > 80)[0]
        if len(r) <= 5 and cols.min() >= 4 and cols.max() <= 12:
            ticks.append(float(r.mean()))
    p = _robust_period(ticks, 40, 160) if len(ticks) >= 2 else float("nan")
    ok = np.isfinite(p)
    depth = lumify_depth_mm(g)
    mm = 10.0 if (depth is not None and depth >= 60) else 5.0
    note = f"{len(ticks)} ticks, depth text={depth}"
    return ScaleResult("lumify", p / mm if ok else float("nan"), p, mm, ticks, "y", 6, ok, note)


def _bottom_ruler(g: np.ndarray) -> ScaleResult:
    h, w = g.shape
    best = None
    for y in range(h - 1, h - 12, -1):        # ticks sit in the last few rows
        row = g[y]
        bg = np.median(row)
        prof = row - bg
        cols = _runs(prof > 50)
        ticks = [float(c.mean()) for c in cols if len(c) <= 4]
        if len(ticks) >= 2 and (best is None or len(ticks) > len(best[1])):
            best = (y, ticks)
    if best is None:
        return ScaleResult("cropped", float("nan"), float("nan"), 10.0, [], "x", h - 1, False, "no ticks")
    y, cand = best
    p, ticks = _grid_period(cand, 50, 400)
    ok = np.isfinite(p) and len(ticks) >= 3
    return ScaleResult("cropped", p / 10, p, 10.0, ticks, "x", y, ok,
                       f"{len(ticks)}/{len(cand)} grid ticks @y={y}")


def detect_scale(img: np.ndarray, ext: str) -> ScaleResult:
    g = to_gray(img)
    device = classify_device(img, ext)
    if device == "siemens":
        return _siemens(g)
    if device == "lumify":
        return _lumify(g)
    if device == "telemed":
        return _vertical_ruler(g, "telemed", 67, 74, 10.0, 80, 260)
    if device == "tel1088":
        return _vertical_ruler(g, "tel1088", 1, 8, 10.0, 80, 260)
    return _bottom_ruler(g)
