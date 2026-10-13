"""Stage 3: segment the test images, measure PA / FL / MT, write the submission.

  python scripts/03_predict.py --apo models/apo_*.pt --fasc models/fasc_*.pt --out /kaggle/working

Outputs: submission.csv (comma-separated, image_id with extension; the grader rejects ';' even though sample_submission.csv uses it),
measurements.csv (diagnostics), pred/{apo,fasc}/*.png (probability maps, uint8),
overlay/*.jpg and overlay_sheet_*.jpg (visual check).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from umud.data import read_gray  # noqa: E402
from umud.geometry import draw, measure  # noqa: E402
from umud.models import load_checkpoint  # noqa: E402
from umud.paths import find_data_root, find_subdir  # noqa: E402
from umud.scale import detect_scale  # noqa: E402

MEAN, STD = 0.45, 0.25
FALLBACK = {"pa_deg": 17.0, "fl_mm": 80.0, "mt_mm": 22.0}  # used only if a measurement fails


@torch.no_grad()
def predict_prob(models, img: np.ndarray, device) -> np.ndarray:
    h, w = img.shape
    acc = None
    for m, ck in models:
        x = cv2.resize(img, (ck["w"], ck["h"]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = torch.from_numpy(((x - MEAN) / STD)[None, None]).to(device)
        xb = torch.cat([x, torch.flip(x, [3])])  # hflip TTA
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            p = torch.sigmoid(m(xb).float())
        p = 0.5 * (p[0, 0] + torch.flip(p[1, 0], [1]))
        p = cv2.resize(p.cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
        acc = p if acc is None else acc + p
    return acc / len(models)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--apo", nargs="+", required=True)
    ap.add_argument("--fasc", nargs="+", required=True)
    ap.add_argument("--out", default="/kaggle/working")
    ap.add_argument("--extrap", type=float, default=1.0)
    ap.add_argument("--save-prob", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    apo_models = [load_checkpoint(p, device) for p in args.apo]
    fasc_models = [load_checkpoint(p, device) for p in args.fasc]
    out = Path(args.out)
    (out / "overlay").mkdir(parents=True, exist_ok=True)
    if args.save_prob:
        (out / "pred" / "apo").mkdir(parents=True, exist_ok=True)
        (out / "pred" / "fasc").mkdir(parents=True, exist_ok=True)

    root = find_data_root(args.data)
    tdir = find_subdir(root, "test_images")
    files = sorted(p for p in tdir.rglob("*") if p.suffix.lower() in (".tif", ".tiff", ".png"))
    rows, thumbs = [], []
    for i, p in enumerate(files):
        color = cv2.imread(str(p), cv2.IMREAD_COLOR)
        gray = read_gray(p)
        sc = detect_scale(color, p.suffix)
        pa_prob = predict_prob(apo_models, gray, device)
        fa_prob = predict_prob(fasc_models, gray, device)
        if args.save_prob:
            cv2.imwrite(str(out / "pred" / "apo" / f"{p.stem}.png"), (pa_prob * 255).astype(np.uint8))
            cv2.imwrite(str(out / "pred" / "fasc" / f"{p.stem}.png"), (fa_prob * 255).astype(np.uint8))
        r = measure(pa_prob, fa_prob, extrap_max=args.extrap)
        ppm = sc.px_per_mm
        row = {"image_id": p.name, "device": sc.device, "px_per_mm": ppm,
               "pa_deg": r.pa_deg, "fl_mm": r.fl_px / ppm, "mt_mm": r.mt_px / ppm,
               "n_fasc": r.n_fasc, "note": r.note}
        rows.append(row)
        ov = draw(color, r, ppm)
        cv2.imwrite(str(out / "overlay" / f"{p.stem}.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 80])
        t = cv2.resize(ov, (400, int(400 * ov.shape[0] / ov.shape[1])))
        cv2.putText(t, p.stem, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        thumbs.append(cv2.copyMakeBorder(t, 0, 300 - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) if t.shape[0] < 300
                      else t[:300])
        if i % 25 == 0:
            print(f"{i}/{len(files)} {p.name} PA {r.pa_deg:.1f} FL {row['fl_mm']:.1f} MT {row['mt_mm']:.1f} "
                  f"n={r.n_fasc} {r.note}", flush=True)
    for s in range(0, len(thumbs), 30):
        chunk = thumbs[s:s + 30]
        chunk += [np.zeros_like(thumbs[0])] * (30 - len(chunk))
        sheet = np.vstack([np.hstack(chunk[k:k + 5]) for k in range(0, 30, 5)])
        cv2.imwrite(str(out / f"overlay_sheet_{s // 30:02d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 70])

    df = pd.DataFrame(rows)
    df.to_csv(out / "measurements.csv", index=False)
    sub = df[["image_id", "pa_deg", "fl_mm", "mt_mm"]].copy()
    bad = sub[["pa_deg", "fl_mm", "mt_mm"]].isna()
    for c, v in FALLBACK.items():
        med = sub[c].median() if sub[c].notna().any() else v
        sub[c] = sub[c].fillna(med)
    # physiological clamps from the data description (test ranges)
    sub["pa_deg"] = sub["pa_deg"].clip(5, 45)
    sub["fl_mm"] = sub["fl_mm"].clip(30, 200)
    sub["mt_mm"] = sub["mt_mm"].clip(10, 50)
    sub.round(3).to_csv(out / "submission.csv", index=False)  # grader expects commas (despite the ";" sample)
    print(f"submission: {len(sub)} rows, filled NaN: {bad.sum().to_dict()}")
    print(df.groupby("device")[["pa_deg", "fl_mm", "mt_mm", "n_fasc"]].median())


if __name__ == "__main__":
    main()
