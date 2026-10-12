"""Stage 0: data inventory for the UMUD challenge.

Produces (in --out):
  inventory.csv      one row per file: split, name, size, dtype, channels, md5 of pixels, ...
  duplicates.csv     groups of files with identical pixel content
  pairs.csv          image/mask pairing check (missing partners, shape mismatches)
  test_sequences.csv consecutive test frames that look like the same video clip
  summary.json       headline numbers (also printed)
  montage_test_*.png thumbnails of every test image, labelled, for eyeballing scale bars/devices
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from umud.paths import SUBDIRS, find_data_root, find_subdir  # noqa: E402

IMG_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}


def read_image(path: Path) -> np.ndarray:
    """Read an image without altering bit depth or channels."""
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        try:
            import tifffile

            return np.asarray(tifffile.imread(str(path)))
        except Exception:
            pass
    import cv2

    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        from PIL import Image

        arr = np.asarray(Image.open(path))
    return arr


def describe(arr: np.ndarray) -> dict:
    h, w = arr.shape[:2]
    ch = 1 if arr.ndim == 2 else arr.shape[2]
    info = {"h": h, "w": w, "channels": ch, "dtype": str(arr.dtype),
            "min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean())}
    if ch >= 3:
        rgb = arr[..., :3].astype(np.int16)
        # Are colour channels identical (pure greyscale stored as RGB)?
        diff = np.abs(rgb[..., 0] - rgb[..., 1]).max() + np.abs(rgb[..., 1] - rgb[..., 2]).max()
        info["rgb_is_grey"] = bool(diff == 0)
        # Fraction of clearly coloured pixels (overlays, coloured scale bars, doppler)
        sat = rgb.max(-1) - rgb.min(-1)
        info["colour_frac"] = float((sat > 30).mean())
    if ch == 1 or ch == 2:
        info["n_unique"] = int(np.unique(arr).size) if arr.size < 5e6 else -1
    return info


def thumb(arr: np.ndarray, size: int = 64) -> np.ndarray:
    import cv2

    g = arr if arr.ndim == 2 else arr[..., :3].mean(-1)
    g = g.astype(np.float32)
    t = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    t -= t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def make_montages(paths: list[Path], out: Path, per_sheet: int = 48, cols: int = 8, tile: int = 220) -> None:
    import cv2

    for s in range(0, len(paths), per_sheet):
        chunk = paths[s:s + per_sheet]
        rows = (len(chunk) + cols - 1) // cols
        sheet = np.full((rows * (tile + 18), cols * tile, 3), 30, np.uint8)
        for i, p in enumerate(chunk):
            a = read_image(p)
            if a.ndim == 2:
                a = np.stack([a] * 3, -1)
            a = a[..., :3]
            if a.dtype != np.uint8:
                a = cv2.normalize(a, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            h, w = a.shape[:2]
            sc = tile / max(h, w)
            a = cv2.resize(a, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
            r, c = divmod(i, cols)
            y0, x0 = r * (tile + 18) + 18, c * tile
            sheet[y0:y0 + a.shape[0], x0:x0 + a.shape[1]] = a
            label = f"{p.stem} {w}x{h}"
            cv2.putText(sheet, label, (x0 + 2, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 0), 1)
        cv2.imwrite(str(out / f"montage_test_{s // per_sheet:02d}.png"), sheet)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="outputs/explore")
    ap.add_argument("--no-montage", action="store_true")
    args = ap.parse_args()

    root = find_data_root(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"data root: {root}")
    print("top level:", sorted(p.name for p in root.iterdir()))

    rows = []
    split_dirs = {}
    for prefix in SUBDIRS:
        d = find_subdir(root, prefix)
        split_dirs[prefix] = d
        if d is None:
            print(f"!! missing folder for {prefix}")
            continue
        files = sorted(p for p in d.rglob("*") if p.is_file())
        exts = Counter(p.suffix.lower() for p in files)
        print(f"{prefix:12s} -> {d.name}: {len(files)} files {dict(exts)}")
        for k, p in enumerate(files):
            if k % 500 == 0:
                print(f"  {prefix}: {k}/{len(files)}", flush=True)
            if p.suffix.lower() not in IMG_EXT:
                continue
            row = {"split": prefix, "folder": d.name, "rel": str(p.relative_to(d)), "stem": p.stem,
                   "bytes": p.stat().st_size}
            try:
                a = read_image(p)
                row.update(describe(a))
                row["md5_pix"] = hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()
            except Exception as e:  # keep going, record the failure
                row["error"] = repr(e)
            rows.append(row)

    inv = pd.DataFrame(rows)
    inv.to_csv(out / "inventory.csv", index=False)

    # ---- duplicates (identical pixels) within and across splits
    dup = inv[inv.duplicated("md5_pix", keep=False)].sort_values(["md5_pix", "split", "stem"])
    dup.to_csv(out / "duplicates.csv", index=False)
    dup_groups = dup.groupby("md5_pix")["split"].agg(lambda s: "+".join(sorted(set(s))))

    # ---- image/mask pairing
    pair_rows = []
    for img_key, mask_key in (("apo_imgs", "apo_masks"), ("fasc_imgs", "fasc_masks")):
        im = inv[inv.split == img_key].set_index("stem")
        mk = inv[inv.split == mask_key].set_index("stem")
        for stem in sorted(set(im.index) | set(mk.index)):
            r = {"set": img_key.split("_")[0], "stem": stem,
                 "has_img": stem in im.index, "has_mask": stem in mk.index}
            if r["has_img"] and r["has_mask"]:
                r["img_hw"] = f"{im.loc[stem, 'h']}x{im.loc[stem, 'w']}"
                r["mask_hw"] = f"{mk.loc[stem, 'h']}x{mk.loc[stem, 'w']}"
                r["shape_match"] = r["img_hw"] == r["mask_hw"]
                r["mask_values"] = mk.loc[stem, "n_unique"] if "n_unique" in mk.columns else None
            pair_rows.append(r)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(out / "pairs.csv", index=False)

    # ---- test sequences: consecutive frames that look alike
    test = inv[inv.split == "test_images"].sort_values("stem").reset_index(drop=True)
    seq_rows = []
    tdir = split_dirs["test_images"]
    if tdir is not None and len(test):
        thumbs = [thumb(read_image(tdir / r)) for r in test.rel]
        for i in range(1, len(test)):
            same_size = (test.h[i], test.w[i]) == (test.h[i - 1], test.w[i - 1])
            corr = float((thumbs[i] * thumbs[i - 1]).sum()) if same_size else np.nan
            seq_rows.append({"prev": test.stem[i - 1], "cur": test.stem[i], "corr": corr})
    seqs = pd.DataFrame(seq_rows)
    seqs.to_csv(out / "test_sequences.csv", index=False)

    # ---- summary
    def size_table(split: str) -> dict:
        s = inv[inv.split == split]
        return {f"{w}x{h}": int(n) for (w, h), n in s.groupby(["w", "h"]).size().sort_values(ascending=False).items()}

    summary = {
        "data_root": str(root),
        "counts": inv.groupby("split").size().to_dict(),
        "sizes": {s: size_table(s) for s in inv.split.unique()},
        "dtypes": inv.groupby(["split", "dtype", "channels"]).size().rename("n").reset_index().to_dict("records"),
        "duplicate_groups": int(dup_groups.size),
        "duplicate_groups_by_splits": dup_groups.value_counts().to_dict(),
        "duplicate_files": int(len(dup)),
        "pairs_missing_mask": int((~pairs.has_mask).sum()) if len(pairs) else 0,
        "pairs_missing_img": int((~pairs.has_img).sum()) if len(pairs) else 0,
        "pairs_shape_mismatch": int((pairs.get("shape_match") == False).sum()) if len(pairs) else 0,  # noqa: E712
        "test_consecutive_corr_gt_0.8": int((seqs["corr"] > 0.8).sum()) if len(seqs) else 0,
        "errors": int(inv["error"].notna().sum()) if "error" in inv else 0,
    }
    if "colour_frac" in inv:
        summary["test_colour_frac_gt_0.001"] = int((inv[inv.split == "test_images"].colour_frac > 0.001).sum())
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))

    if len(seqs):
        print("\nconsecutive test frames, corr distribution:")
        print(seqs["corr"].describe().to_string())

    if not args.no_montage and tdir is not None:
        make_montages([tdir / r for r in test.rel], out)
        print(f"montages written to {out}")


if __name__ == "__main__":
    main()
