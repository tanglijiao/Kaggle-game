"""Derive PA labels from the red expert drawings in the UMUD 'Example Annotated' sets (OSF xbawc).

Each participant/timepoint folder holds annotated canvases (ANON_k.tif, red lines on a wider canvas) and the
raw images (ANON_j.tiff).  For each annotated canvas we
  1. find the raw image it was drawn on and the horizontal offset (template match),
  2. extract red line segments (Hough), merge collinear ones,
  3. pick the expert's deep / superficial aponeurosis lines as the red lines closest to OUR aponeurosis
     segmentation of the raw image (maps/anno), and
  4. PA = length-weighted mean angle between the deep line and the red segments that end on it.
Output: resources/anno_labels.csv (stem of the raw image in maps/anno, pa_deg, n_seg, sup/deep line params).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from umud.evalsets import load_maps  # noqa: E402
from umud.geometry import find_aponeuroses, pick_sup_deep  # noqa: E402


def red_mask(a):
    b, g, r = [a[:, :, i].astype(int) for i in range(3)]
    m = ((r > 110) & (r - g > 50) & (r - b > 50)).astype(np.uint8)
    m[:110] = 0
    return m


def segments(m):
    L = cv2.HoughLinesP(m, 1, np.pi / 720, 30, minLineLength=25, maxLineGap=4)
    return np.zeros((0, 4)) if L is None else L.reshape(-1, 4).astype(float)


def merge_lines(L, ang_tol=1.0, dist_tol=3.0):
    """Greedy merge of collinear segments -> list of (p0, p1, angle_deg, total_len)."""
    out = []
    used = np.zeros(len(L), bool)
    order = np.argsort(-np.hypot(L[:, 2] - L[:, 0], L[:, 3] - L[:, 1]))
    for i in order:
        if used[i]:
            continue
        x0, y0, x1, y1 = L[i]
        d = np.array([x1 - x0, y1 - y0]); d /= np.linalg.norm(d)
        nrm = np.array([-d[1], d[0]])
        grp = [i]
        used[i] = True
        for j in order:
            if used[j]:
                continue
            a0, b0, a1, b1 = L[j]
            e = np.array([a1 - a0, b1 - b0]); e /= np.linalg.norm(e)
            if abs(np.degrees(np.arcsin(np.clip(d[0] * e[1] - d[1] * e[0], -1, 1)))) > ang_tol:
                continue
            if max(abs((np.array([a0, b0]) - [x0, y0]) @ nrm), abs((np.array([a1, b1]) - [x0, y0]) @ nrm)) > dist_tol:
                continue
            grp.append(j)
            used[j] = True
        pts = np.concatenate([L[grp][:, :2], L[grp][:, 2:]])
        t = (pts - [x0, y0]) @ d
        p0 = np.array([x0, y0]) + t.min() * d
        p1 = np.array([x0, y0]) + t.max() * d
        if p1[0] < p0[0]:
            p0, p1 = p1, p0
        tot = float(np.hypot(L[grp][:, 2] - L[grp][:, 0], L[grp][:, 3] - L[grp][:, 1]).sum())
        out.append((p0, p1, float(np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))), tot))
    return out


def line_y(line, x):
    p0, p1 = line[0], line[1]
    return p0[1] + (p1[1] - p0[1]) * (x - p0[0]) / (p1[0] - p0[0] + 1e-9)


def find_offset(canvas, raw):
    cg = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    rg = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    if cg.shape[0] != rg.shape[0] or cg.shape[1] < rg.shape[1]:
        return None, 1e9
    # ignore red pixels: compare green channel (lines are red)
    res = cv2.matchTemplate(canvas[:, :, 1], raw[:, :, 1], cv2.TM_SQDIFF_NORMED)
    ox = int(res.argmin())
    return ox, float(res.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext", default="/home/claude/ext")
    ap.add_argument("--maps", default="/home/claude/maps_v4/maps")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "resources/anno_labels.csv"))
    ap.add_argument("--debug", default=None)
    args = ap.parse_args()
    maps = Path(args.maps)
    idx = pd.read_csv(maps / "index.csv")
    idx = idx[idx.set == "anno"].set_index("stem")
    rows = []
    folders = sorted({p.parent for p in Path(args.ext).glob("*_ANONYMIZED_annotated_v0.1.0/*/*/*/*.tif")})
    for fo in folders:
        raws = sorted(fo.glob("*.tiff"))
        rawimgs = {p: cv2.imread(str(p)) for p in raws}
        for ca in sorted(fo.glob("*.tif")):
            canvas = cv2.imread(str(ca))
            best = min(((find_offset(canvas, im), p) for p, im in rawimgs.items()), key=lambda t: t[0][1])
            (ox, err), rp = best
            if ox is None or err > 0.05:
                continue
            stem = f"{fo.parent.name}__{fo.name}__{rp.stem}"
            if stem not in idx.index:
                continue
            h, w = int(idx.loc[stem, "h"]), int(idx.loc[stem, "w"])
            apo, _ = load_maps(maps, "anno", stem, h, w)
            sup, deep = pick_sup_deep(find_aponeuroses(apo), h)
            if sup is None or deep is None:
                continue
            lines = [l for l in merge_lines(segments(red_mask(canvas))) if l[3] > 40]
            xs = np.linspace(max(deep.x0, sup.x0), min(deep.x1, sup.x1), 9)

            def dist(line, a):
                return float(np.mean(np.abs(line_y(line, xs + ox) - a.y(xs))))

            long_lines = [l for l in lines if (l[1][0] - l[0][0]) > 150]
            if len(long_lines) < 2:
                continue
            dl = min(long_lines, key=lambda l: dist(l, deep))
            sl = min(long_lines, key=lambda l: dist(l, sup))
            if dist(dl, deep) > 25 or dist(sl, sup) > 25 or dl is sl:
                continue
            ddir = np.array([1.0, np.tan(np.radians(dl[2]))]); ddir /= np.linalg.norm(ddir)
            angs, wts, fls = [], [], []
            sdir = np.array([1.0, np.tan(np.radians(sl[2]))])
            for l in lines:
                if l is dl or l is sl:
                    continue
                for pnt in (l[0], l[1]):
                    if abs(pnt[1] - line_y(dl, pnt[0])) < 6:
                        f = l[1] - l[0]; f = f / np.linalg.norm(f)
                        a = float(np.degrees(np.arccos(min(1.0, abs(f @ ddir)))))
                        if 3 <= a <= 45:
                            angs.append(a); wts.append(l[3])
                            # straight extrapolation from the deep contact point to the superficial line
                            A = np.array([[f[0], -sdir[0]], [f[1], -sdir[1]]])
                            try:
                                t, _ = np.linalg.solve(A, np.array(sl[0]) - pnt)
                                fls.append(abs(float(t)))
                            except np.linalg.LinAlgError:
                                pass
                        break
            if not angs:
                continue
            rows.append({"stem": stem, "key": stem, "pa_deg": float(np.average(angs, weights=wts)),
                         "pa_med": float(np.median(angs)), "n_seg": len(angs),
                         "fl_px": float(np.median(fls)) if fls else np.nan,
                         "mt_px": float(abs(line_y(dl, ox + w / 2) - line_y(sl, ox + w / 2)) * np.cos(np.radians(sl[2]))), "ox": ox, "match_err": err,
                         "deep_ang": dl[2], "sup_ang": sl[2]})
            if args.debug and len(rows) <= 12:
                dbg = canvas.copy()
                for l, c in ((dl, (255, 0, 255)), (sl, (0, 255, 255))):
                    cv2.line(dbg, tuple(int(v) for v in l[0]), tuple(int(v) for v in l[1]), c, 2)
                cv2.putText(dbg, f"PA {rows[-1]['pa_deg']:.1f} n={len(angs)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                Path(args.debug).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(Path(args.debug) / f"{stem}.jpg"), dbg)
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(len(df), "labelled;", df.pa_deg.describe().round(2).to_dict() if len(df) else "")


if __name__ == "__main__":
    main()
