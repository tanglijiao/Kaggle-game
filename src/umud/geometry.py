"""From segmentation probability maps to muscle architecture (PA, FL, MT).

Measurement conventions (organisers, data page + discussions 744818 / 748126):
  * only the most superficial muscle: superficial aponeurosis = upper line, deep = the next line below;
  * PA = angle between the (extrapolated) fascicle and the deep aponeurosis where they meet;
  * FL = length of the fascicle line between the two aponeuroses (linear extrapolation outside the frame);
  * MT = perpendicular distance between the aponeuroses, averaged over three positions across the width;
  * automated methods use as many fascicles as possible but limit extrapolation (DL_Track style).
All geometry is done in original pixel coordinates; px/mm converts at the end (square pixels).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Apo:
    coef: np.ndarray          # y = polyval(coef, x)  (deg 1 or 2)
    x0: float
    x1: float
    mean_y: float
    support: int              # number of columns with mask pixels

    def y(self, x):
        return np.polyval(self.coef, x)

    def slope(self, x):
        return np.polyval(np.polyder(self.coef), x)


@dataclass
class Fascicle:
    p: np.ndarray             # point on the line (x, y)
    d: np.ndarray             # unit direction (dx, dy), dx >= 0
    length: float             # visible fragment length (px)
    weight: float


@dataclass
class Result:
    pa_deg: float = float("nan")
    fl_px: float = float("nan")
    mt_px: float = float("nan")
    n_fasc: int = 0
    sup: Apo | None = None
    deep: Apo | None = None
    fascicles: list = field(default_factory=list)   # (Fascicle, x_sup, x_deep, pa, fl)
    note: str = ""


# ---------------------------------------------------------------- aponeuroses
def _fit_component(mask: np.ndarray, deg: int) -> Apo | None:
    ys, xs = np.nonzero(mask)
    if xs.size < 20:
        return None
    # column-wise centre line (robust to band thickness)
    cols = np.unique(xs)
    cy = np.array([ys[xs == c].mean() for c in cols])
    deg = deg if cols.size > 50 else 1
    coef = np.polyfit(cols, cy, deg)
    return Apo(coef, float(cols.min()), float(cols.max()), float(cy.mean()), int(cols.size))


def find_aponeuroses(prob: np.ndarray, thr: float = 0.5, min_frac: float = 0.15,
                     deg: int = 2) -> list[Apo]:
    """Connected components of the aponeurosis map, merged when they continue each other horizontally,
    returned sorted top to bottom. Components spanning < min_frac of the width are ignored."""
    h, w = prob.shape
    m = (prob > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    comps = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_WIDTH] < 0.05 * w:
            continue
        a = _fit_component(lab == i, 1)
        if a is not None:
            comps.append((i, a))
    # greedy merge: two pieces belong together if one's line predicts the other's centre within tolerance
    groups: list[list[int]] = []
    used = set()
    comps.sort(key=lambda t: -(t[1].x1 - t[1].x0))
    for i, a in comps:
        if i in used:
            continue
        g = [i]
        used.add(i)
        for j, b in comps:
            if j in used:
                continue
            xm = (b.x0 + b.x1) / 2
            if abs(a.y(xm) - b.y(xm)) < max(12, 0.02 * h):
                g.append(j)
                used.add(j)
        groups.append(g)
    apos = []
    for g in groups:
        a = _fit_component(np.isin(lab, g), deg)
        if a is not None and (a.x1 - a.x0) >= min_frac * w:
            apos.append(a)
    apos.sort(key=lambda a: a.mean_y)
    return apos


def pick_sup_deep(apos: list[Apo], h: int) -> tuple[Apo | None, Apo | None]:
    """Superficial = top-most aponeurosis; deep = the next one below with a plausible gap
    (>= ~4% of the image height so double-lined fascia is not mistaken for a muscle)."""
    if len(apos) < 2:
        return (apos[0] if apos else None), None
    sup = apos[0]
    for a in apos[1:]:
        xs = np.linspace(max(sup.x0, a.x0), min(sup.x1, a.x1), 5)
        if xs[-1] <= xs[0]:
            xs = np.linspace(0, 1, 5) * (a.x1 - a.x0) + a.x0
        gap = np.median(a.y(xs) - sup.y(xs))
        if gap > 0.04 * h:
            return sup, a
    return sup, apos[1]


# ---------------------------------------------------------------- fascicles
def find_fascicles(prob: np.ndarray, thr: float = 0.4, min_len: float = 25.0) -> list[Fascicle]:
    m = (prob > thr).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    out = []
    for i in range(1, n):
        if max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) < min_len:
            continue
        ys, xs = np.nonzero(lab == i)
        pts = np.stack([xs, ys], 1).astype(np.float64)
        c = pts.mean(0)
        u, s, vt = np.linalg.svd(pts - c, full_matrices=False)
        d = vt[0]
        if d[0] < 0:
            d = -d
        proj = (pts - c) @ d
        length = float(proj.max() - proj.min())
        # elongation check: fragments should be line-like
        if s[1] > 0 and s[0] / s[1] < 3:
            continue
        w = float(prob[ys, xs].mean()) * length
        out.append(Fascicle(c, d, length, w))
    return out


def _intersect(f: Fascicle, a: Apo, x_lo: float, x_hi: float) -> float | None:
    """x where the fascicle line meets the aponeurosis curve (Newton on g(t) = y_f(x) - y_a(x))."""
    if abs(f.d[0]) < 1e-6:
        return None
    k = f.d[1] / f.d[0]
    x = f.p[0]
    for _ in range(20):
        g = f.p[1] + k * (x - f.p[0]) - a.y(x)
        dg = k - a.slope(x)
        if abs(dg) < 1e-9:
            return None
        x -= g / dg
    if not (x_lo <= x <= x_hi) or abs(f.p[1] + k * (x - f.p[0]) - a.y(x)) > 1:
        return None
    return float(x)


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    a = np.degrees(np.arctan2(abs(v1[0] * v2[1] - v1[1] * v2[0]), abs(v1 @ v2)))
    return float(a)


def measure(apo_prob: np.ndarray, fasc_prob: np.ndarray, *, extrap_max: float = 1.0,
            pa_range=(3.0, 50.0), apo_thr: float = 0.5, fasc_thr: float = 0.4) -> Result:
    """extrap_max: allowed intersection range beyond the image width, as a fraction of the width."""
    h, w = apo_prob.shape
    r = Result()
    apos = find_aponeuroses(apo_prob, apo_thr)
    sup, deep = pick_sup_deep(apos, h)
    r.sup, r.deep = sup, deep
    if sup is None or deep is None:
        r.note = f"aponeuroses found: {len(apos)}"
        return r

    # MT: perpendicular distance at 25/50/75 % of the common span
    x_lo, x_hi = max(sup.x0, deep.x0), min(sup.x1, deep.x1)
    if x_hi - x_lo < 0.2 * w:
        x_lo, x_hi = 0.2 * w, 0.8 * w
    mts = []
    for t in (0.25, 0.5, 0.75):
        x = x_lo + t * (x_hi - x_lo)
        # distance from the deep point to the superficial curve, perpendicular to the superficial tangent
        s = sup.slope(x)
        n = np.array([-s, 1.0]) / np.hypot(s, 1.0)
        mts.append(abs((deep.y(x) - sup.y(x)) * n[1]))
    r.mt_px = float(np.mean(mts))

    # fascicles strictly between the aponeuroses
    fas = []
    lo, hi = -extrap_max * w, (1 + extrap_max) * w
    for f in find_fascicles(fasc_prob, fasc_thr):
        cx, cy = f.p
        if not (sup.y(cx) + 3 < cy < deep.y(cx) - 3):
            continue
        xs = _intersect(f, sup, lo, hi)
        xd = _intersect(f, deep, lo, hi)
        if xs is None or xd is None:
            continue
        ps = np.array([xs, sup.y(xs)])
        pd_ = np.array([xd, deep.y(xd)])
        fl = float(np.linalg.norm(ps - pd_))
        tang = np.array([1.0, deep.slope(xd)])
        pa = _angle_between(f.d, tang)
        if not pa_range[0] <= pa <= pa_range[1]:
            continue
        fas.append((f, xs, xd, pa, fl))
    if not fas:
        r.note = "no valid fascicles"
        return r
    # dominant orientation: fascicles in one image share the slope sign
    sgn = np.sign([f.d[1] for f, *_ in fas])
    major = 1.0 if (sgn * [f.weight for f, *_ in fas]).sum() >= 0 else -1.0
    fas = [t for t in fas if np.sign(t[0].d[1]) == major] or fas
    wts = np.array([t[0].weight for t in fas])
    pas = np.array([t[3] for t in fas])
    fls = np.array([t[4] for t in fas])
    r.pa_deg = weighted_median(pas, wts)
    r.fl_px = weighted_median(fls, wts)
    r.n_fasc = len(fas)
    r.fascicles = fas
    return r


def weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    v, w = v[o], w[o]
    c = np.cumsum(w)
    return float(v[np.searchsorted(c, 0.5 * c[-1])])


def draw(img: np.ndarray, r: Result, px_per_mm: float | None = None) -> np.ndarray:
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    xs = np.arange(0, w)
    for a, col in ((r.sup, (0, 255, 255)), (r.deep, (255, 0, 255))):
        if a is not None:
            pts = np.stack([xs, a.y(xs)], 1).astype(np.int32)
            cv2.polylines(out, [pts], False, col, 2)
    for f, xs_, xd, pa, fl in r.fascicles:
        p1 = (int(xs_), int(r.sup.y(xs_)))
        p2 = (int(xd), int(r.deep.y(xd)))
        cv2.line(out, p1, p2, (0, 200, 0), 1)
    if px_per_mm:
        txt = f"PA {r.pa_deg:.1f}  FL {r.fl_px / px_per_mm:.1f}mm  MT {r.mt_px / px_per_mm:.1f}mm  n={r.n_fasc}"
    else:
        txt = f"PA {r.pa_deg:.1f}  FL {r.fl_px:.0f}px  MT {r.mt_px:.0f}px  n={r.n_fasc}"
    cv2.putText(out, txt, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out
