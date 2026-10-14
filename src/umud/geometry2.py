"""Geometry v2: dominant fascicle orientation from a structure tensor instead of individual fragments.

Idea: experts draw ~3 straight fascicle lines and extrapolate them to both aponeuroses. With (almost)
straight fascicles, all of them share one orientation, so a robust *field* estimate of that orientation
inside the muscle (structure tensor of the fascicle-probability map, or of the image itself) is less
noisy than intersecting individual short fragments. FL is then the length of a straight line with that
orientation through the muscle at a few positions, PA its angle to the deep aponeurosis tangent there.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .geometry import Apo, Fascicle, _intersect, find_aponeuroses, pick_sup_deep


class LinExtApo(Apo):
    """Aponeurosis that follows its polynomial fit inside the observed span and continues along the
    end tangent outside it (experts extrapolate aponeuroses linearly)."""

    def __init__(self, a: Apo, margin: float = 0.0):
        super().__init__(a.coef.copy(), a.x0, a.x1, a.mean_y, a.support)
        self.lo, self.hi = a.x0 + margin, a.x1 - margin

    def _poly(self, x):
        return np.polyval(self.coef, x)

    def _dpoly(self, x):
        return np.polyval(np.polyder(self.coef), x)

    def y(self, x):
        x = np.asarray(x, dtype=float)
        xc = np.clip(x, self.lo, self.hi)
        return self._poly(xc) + self._dpoly(xc) * (x - xc)

    def slope(self, x):
        return self._dpoly(np.clip(np.asarray(x, dtype=float), self.lo, self.hi))


@dataclass
class Result2:
    pa_deg: float = float("nan")
    fl_px: float = float("nan")
    mt_px: float = float("nan")
    angle_deg: float = float("nan")      # fascicle orientation in image coords (deg, y down)
    coherence: float = float("nan")
    sup: Apo | None = None
    deep: Apo | None = None
    lines: list = field(default_factory=list)   # [(p_sup, p_deep)]
    note: str = ""


def structure_orientation(img: np.ndarray, weight: np.ndarray, sigma_grad: float = 1.5, sigma_int: float = 6.0):
    """Weighted dominant line orientation (deg in [-90, 90), image coords) and its coherence.

    Uses the doubled-angle representation so opposite directions agree."""
    f = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma_grad)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma_int)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma_int)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma_int)
    # gradient orientation (doubled) ; line orientation = gradient + 90 deg
    c2 = (jxx - jyy)
    s2 = 2 * jxy
    energy = jxx + jyy
    coh = np.sqrt(c2 ** 2 + s2 ** 2) / (energy + 1e-6)
    w = weight * coh * np.sqrt(energy)
    C = float((w * c2 / (np.sqrt(c2 ** 2 + s2 ** 2) + 1e-6)).sum())
    S = float((w * s2 / (np.sqrt(c2 ** 2 + s2 ** 2) + 1e-6)).sum())
    grad_ang = 0.5 * np.degrees(np.arctan2(S, C))
    line_ang = grad_ang + 90.0
    if line_ang >= 90:
        line_ang -= 180
    mean_coh = float((w * coh).sum() / (w.sum() + 1e-9))
    return float(line_ang), float(np.hypot(C, S) / (w.sum() + 1e-9)), mean_coh


def angle_histogram_mode(img, weight, sigma_grad=1.5, sigma_int=6.0, lo=-60, hi=60, bin_deg=0.5, smooth=2.0):
    """Mode of the weighted local orientation histogram (robust to a second structure, e.g. a vessel)."""
    f = cv2.GaussianBlur(img.astype(np.float32), (0, 0), sigma_grad)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    jxx = cv2.GaussianBlur(gx * gx, (0, 0), sigma_int)
    jyy = cv2.GaussianBlur(gy * gy, (0, 0), sigma_int)
    jxy = cv2.GaussianBlur(gx * gy, (0, 0), sigma_int)
    c2, s2, energy = jxx - jyy, 2 * jxy, jxx + jyy
    coh = np.sqrt(c2 ** 2 + s2 ** 2) / (energy + 1e-6)
    ang = 0.5 * np.degrees(np.arctan2(s2, c2)) + 90.0
    ang = np.where(ang >= 90, ang - 180, ang)
    w = (weight * coh * np.sqrt(energy)).ravel()
    a = ang.ravel()
    m = (w > 0) & (a >= lo) & (a < hi)
    bins = np.arange(lo, hi + bin_deg, bin_deg)
    h, _ = np.histogram(a[m], bins=bins, weights=w[m])
    if h.sum() <= 0:
        return float("nan"), 0.0
    k = int(np.ceil(3 * smooth / bin_deg))
    g = np.exp(-0.5 * (np.arange(-k, k + 1) * bin_deg / smooth) ** 2)
    hs = np.convolve(h, g / g.sum(), mode="same")
    i = int(hs.argmax())
    # parabolic refinement
    if 0 < i < len(hs) - 1:
        d = (hs[i - 1] - hs[i + 1]) / (2 * (hs[i - 1] - 2 * hs[i] + hs[i + 1]) + 1e-12)
    else:
        d = 0.0
    centre = bins[i] + bin_deg / 2 + d * bin_deg
    return float(centre), float(hs[i] / hs.sum())


def band_edges(apo_mask: np.ndarray, a: Apo, x: float, which: str) -> float:
    """Inner edge of an aponeurosis band at column x: 'lower' edge for the superficial, 'upper' for the deep."""
    h = apo_mask.shape[0]
    xi = int(np.clip(round(x), 0, apo_mask.shape[1] - 1))
    yc = int(np.clip(round(a.y(x)), 0, h - 1))
    col = apo_mask[:, xi]
    if not col[yc]:
        # nearest mask pixel within 6 px
        near = np.nonzero(col[max(0, yc - 6): yc + 7])[0]
        if near.size == 0:
            return float(a.y(x))
        yc = max(0, yc - 6) + int(near[np.argmin(np.abs(near - 6))])
    y = yc
    if which == "lower":
        while y + 1 < h and col[y + 1]:
            y += 1
    else:
        while y - 1 >= 0 and col[y - 1]:
            y -= 1
    return float(y)


def inner_apos(apo_mask: np.ndarray, sup: Apo, deep: Apo, x_lo: float, x_hi: float, n: int = 15):
    """Shift the centre-line fits to the muscle-side edges of the aponeurosis bands
    (experts measure between the inner borders)."""
    xs = np.linspace(x_lo, x_hi, n)
    ds = np.median([band_edges(apo_mask, sup, x, "lower") - sup.y(x) for x in xs])
    dd = np.median([deep.y(x) - band_edges(apo_mask, deep, x, "upper") for x in xs])
    ds, dd = max(0.0, float(ds)), max(0.0, float(dd))
    import copy
    s2 = copy.deepcopy(sup)
    s2.coef = s2.coef.copy(); s2.coef[-1] += ds; s2.mean_y += ds
    d2 = copy.deepcopy(deep)
    d2.coef = d2.coef.copy(); d2.coef[-1] -= dd; d2.mean_y -= dd
    return s2, d2


def pick_pair_robust(apo_prob, fasc_prob, thr, deg):
    """Like pick_sup_deep, but (1) lowers the threshold when fewer than two aponeuroses are found,
    (2) drops short fragments when there are 3+ candidates, (3) prefers the top-most pair that actually
    contains fascicles (at least half of the best pair's mean fascicle probability)."""
    h, w = apo_prob.shape
    apos = []
    for t in (thr, 0.3, 0.15):
        apos = find_aponeuroses(apo_prob, t, deg=deg)
        if len(apos) >= 2:
            break
    if len(apos) < 2:
        return (apos[0] if apos else None), None, apos
    if len(apos) >= 3:
        mx = max(a.x1 - a.x0 for a in apos)
        if apos[0].x1 - apos[0].x0 < 0.7 * mx:   # short fragment above the muscle (skin / fascia)
            apos = apos[1:]
    pairs = []
    for i in range(len(apos) - 1):
        a = apos[i]
        for b in apos[i + 1:]:
            x0, x1 = max(a.x0, b.x0), min(a.x1, b.x1)
            if x1 - x0 < 0.1 * w:
                continue
            xs = np.linspace(x0, x1, 7)
            if np.median(b.y(xs) - a.y(xs)) <= 0.04 * h:
                continue
            xi = np.arange(int(x0), int(x1))
            ya, yb = a.y(xi), b.y(xi)
            yy = np.arange(h)[:, None]
            m = (yy > ya[None, :]) & (yy < yb[None, :])
            sc = float(fasc_prob[:, int(x0):int(x0) + len(xi)][m].mean()) if m.any() else 0.0
            pairs.append((a, b, sc))
            break  # only the next aponeurosis below a
    if not pairs:
        return pick_sup_deep(apos, h) + (apos,)
    best = max(p[2] for p in pairs)
    for a, b, sc in pairs:
        if sc >= 0.5 * best:
            return a, b, apos
    return pairs[0][0], pairs[0][1], apos


def measure2(apo_prob: np.ndarray, fasc_prob: np.ndarray, gray: np.ndarray | None = None, *,
             apo_thr: float = 0.5, source: str = "fasc", estimator: str = "mean", region: str = "all",
             margin: float = 0.1, positions=(0.25, 0.5, 0.75), mt_mode: str = "center",
             sigma_int: float = 6.0, pa_range=(2.0, 60.0), inner: bool = False,
             pa_offset: float = 0.0, pa_scale: float = 1.0, linext: bool = False, apo_deg: int = 2,
             fl_extra: float = 0.0, robust_pick: bool = False) -> Result2:
    """source: 'fasc' (orientation of the fascicle-probability map), 'img' (grey image weighted by
    fascicle probability), 'img_raw' (grey image only).  region: 'all' | 'deep_half' | 'mid'."""
    h, w = apo_prob.shape
    r = Result2()
    if robust_pick:
        sup, deep, apos = pick_pair_robust(apo_prob, fasc_prob, apo_thr, apo_deg)
    else:
        apos = find_aponeuroses(apo_prob, apo_thr, deg=apo_deg)
        sup, deep = pick_sup_deep(apos, h)
    r.sup, r.deep = sup, deep
    if sup is None or deep is None:
        r.note = f"aponeuroses: {len(apos)}"
        return r
    if linext:
        sup, deep = LinExtApo(sup), LinExtApo(deep)
        r.sup, r.deep = sup, deep
    x_lo, x_hi = max(sup.x0, deep.x0), min(sup.x1, deep.x1)
    if x_hi - x_lo < 0.2 * w:
        x_lo, x_hi = 0.2 * w, 0.8 * w

    # ---- MT
    apo_mask = apo_prob > apo_thr
    if inner:
        sup, deep = inner_apos(apo_mask, sup, deep, x_lo, x_hi)
        r.sup, r.deep = sup, deep
        mt_mode = "center"  # already on the inner borders
    mts = []
    for t in positions:
        x = x_lo + t * (x_hi - x_lo)
        if mt_mode == "inner":
            ys, yd = band_edges(apo_mask, sup, x, "lower"), band_edges(apo_mask, deep, x, "upper")
        else:
            ys, yd = sup.y(x), deep.y(x)
        s = sup.slope(x)
        mts.append(abs(yd - ys) / np.hypot(s, 1.0))
    r.mt_px = float(np.mean(mts))

    # ---- muscle region mask
    xs = np.arange(w)
    ysup, ydeep = sup.y(xs), deep.y(xs)
    thick = np.maximum(ydeep - ysup, 1)
    yy = np.arange(h)[:, None]
    top = ysup + margin * thick
    bot = ydeep - margin * thick
    if region == "deep_half":
        top = ysup + 0.5 * thick
    elif region == "mid":
        top, bot = ysup + 0.25 * thick, ydeep - 0.25 * thick
    reg = (yy > top[None, :]) & (yy < bot[None, :])
    reg &= (xs[None, :] >= x_lo) & (xs[None, :] <= x_hi)
    if reg.sum() < 500:
        r.note = "tiny region"
        return r

    if source == "fasc":
        src, wgt = fasc_prob, reg.astype(np.float32)
    elif source == "img":
        src, wgt = gray, reg * np.clip(fasc_prob, 0, 1)
    else:
        src, wgt = gray, reg.astype(np.float32)
    if estimator == "mode":
        ang, coh = angle_histogram_mode(src, wgt, sigma_int=sigma_int)
    else:
        ang, _, coh = structure_orientation(src, wgt, sigma_int=sigma_int)
    r.angle_deg, r.coherence = ang, coh
    if not np.isfinite(ang):
        r.note = "no orientation"
        return r
    d = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang))])
    if d[0] < 0:
        d = -d
    xm = 0.5 * (x_lo + x_hi)
    a_deep = np.degrees(np.arctan(deep.slope(xm)))
    rel = ang - a_deep                      # signed fascicle angle relative to the deep apo
    sgn = 1.0 if rel >= 0 else -1.0

    def direction(extra: float) -> np.ndarray:
        # rotate about the deep-aponeurosis direction so that PA_new = pa_scale * PA + pa_offset + extra
        new = a_deep + sgn * (abs(rel) * pa_scale + pa_offset + extra)
        dd = np.array([np.cos(np.radians(new)), np.sin(np.radians(new))])
        return -dd if dd[0] < 0 else dd

    d = direction(0.0)
    d_fl = direction(fl_extra)

    pas, fls = [], []
    lo, hi = -3.0 * w, 4.0 * w
    for t in positions:
        x0 = x_lo + t * (x_hi - x_lo)
        y0 = 0.5 * (sup.y(x0) + deep.y(x0))
        f = Fascicle(np.array([x0, y0]), d, 0.0, 1.0)
        xsup = _intersect(f, sup, lo, hi)
        xdeep = _intersect(f, deep, lo, hi)
        if xsup is None or xdeep is None:
            continue
        ps = np.array([xsup, sup.y(xsup)])
        pd_ = np.array([xdeep, deep.y(xdeep)])
        tang = np.array([1.0, deep.slope(xdeep)])
        tang /= np.linalg.norm(tang)
        pas.append(float(np.degrees(np.arccos(min(1.0, abs(d @ tang))))))
        if fl_extra:
            f2 = Fascicle(np.array([x0, y0]), d_fl, 0.0, 1.0)
            xs2, xd2 = _intersect(f2, sup, lo, hi), _intersect(f2, deep, lo, hi)
            if xs2 is None or xd2 is None:
                continue
            ps, pd_ = np.array([xs2, sup.y(xs2)]), np.array([xd2, deep.y(xd2)])
        fls.append(float(np.linalg.norm(ps - pd_)))
        r.lines.append((ps, pd_))
    if not pas or not fls:
        r.note = "no intersections"
        return r
    r.pa_deg = float(np.mean(pas))
    r.fl_px = float(np.mean(fls))
    if not pa_range[0] <= r.pa_deg <= pa_range[1]:
        r.note = f"pa out of range {r.pa_deg:.1f}"
    return r


def draw2(img: np.ndarray, r: Result2, px_per_mm: float | None = None) -> np.ndarray:
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    xs = np.arange(0, w)
    for a, col in ((r.sup, (0, 255, 255)), (r.deep, (255, 0, 255))):
        if a is not None:
            cv2.polylines(out, [np.stack([xs, a.y(xs)], 1).astype(np.int32)], False, col, 2)
    for ps, pd_ in r.lines:
        cv2.line(out, tuple(int(v) for v in ps), tuple(int(v) for v in pd_), (0, 220, 0), 2)
    s = px_per_mm or 1.0
    u = "mm" if px_per_mm else "px"
    cv2.putText(out, f"PA {r.pa_deg:.1f} FL {r.fl_px / s:.1f}{u} MT {r.mt_px / s:.1f}{u} coh {r.coherence:.2f}",
                (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out
