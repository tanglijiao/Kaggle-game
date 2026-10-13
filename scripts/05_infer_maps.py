"""Run the segmenters on image folders and save the probability maps at model resolution (uint8 PNG).

Geometry is then tuned offline on these maps (no GPU needed):

  python scripts/05_infer_maps.py --apo A.pt [A2.pt ...] --fasc F.pt [...] \
      --set test=/kaggle/input/.../test_images_v2 --set bench=/kaggle/tmp/ext/bench ... \
      --out /kaggle/working/maps

Writes maps/<set>/apo/<stem>.png, maps/<set>/fasc/<stem>.png and maps/index.csv (set, file, stem, h, w).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from umud.data import read_gray  # noqa: E402
from umud.models import load_checkpoint  # noqa: E402
from umud.paths import find_data_root, find_subdir  # noqa: E402

MEAN, STD = 0.45, 0.25
EXTS = (".tif", ".tiff", ".png", ".jpg", ".bmp")


@torch.no_grad()
def predict_small(models, img: np.ndarray, device) -> np.ndarray:
    """Mean of sigmoid(model) over models and hflip, at the model's own resolution."""
    acc = None
    for m, ck in models:
        x = cv2.resize(img, (ck["w"], ck["h"]), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        x = torch.from_numpy(((x - MEAN) / STD)[None, None]).to(device)
        xb = torch.cat([x, torch.flip(x, [3])])
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            p = torch.sigmoid(m(xb).float())
        p = 0.5 * (p[0, 0] + torch.flip(p[1, 0], [1]))
        p = p.cpu().numpy()
        acc = p if acc is None else acc + cv2.resize(p, (acc.shape[1], acc.shape[0]))
    return acc / len(models)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apo", nargs="+", required=True)
    ap.add_argument("--fasc", nargs="+", required=True)
    ap.add_argument("--set", action="append", default=[], help="name=folder (recursive); 'test' alone = competition test")
    ap.add_argument("--out", default="/kaggle/working/maps")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    apo = [load_checkpoint(p, device) for p in args.apo]
    fasc = [load_checkpoint(p, device) for p in args.fasc]
    out = Path(args.out)
    rows = []
    for spec in args.set:
        name, _, folder = spec.partition("=")
        folder = Path(folder) if folder else find_subdir(find_data_root(), "test_images")
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in EXTS)
        (out / name / "apo").mkdir(parents=True, exist_ok=True)
        (out / name / "fasc").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        for p in files:
            stem = str(p.relative_to(folder).with_suffix("")).replace("/", "__")
            try:
                g = read_gray(p)
            except IOError as e:
                print("skip", e, flush=True)
                continue
            for task, ms in (("apo", apo), ("fasc", fasc)):
                pr = predict_small(ms, g, device)
                cv2.imwrite(str(out / name / task / f"{stem}.png"), np.round(pr * 255).astype(np.uint8))
            rows.append({"set": name, "file": str(p.relative_to(folder)), "stem": stem, "h": g.shape[0], "w": g.shape[1]})
        print(f"{name}: {len(files)} images in {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame(rows).to_csv(out / "index.csv", index=False)


if __name__ == "__main__":
    main()
