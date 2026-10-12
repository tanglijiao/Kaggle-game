"""Dataset helpers: image/mask loading, de-duplication, grouped hold-out split."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np


def read_gray(path: str | Path) -> np.ndarray:
    """Read an image as uint8 grey (OpenCV handles the LZW-compressed TIFFs)."""
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if a is None:
        raise IOError(f"cannot read {path}")
    return a


def read_mask(path: str | Path, shape_hw: tuple[int, int]) -> np.ndarray:
    """Masks are stored at a different resolution from their image but cover the same field of view:
    resize the whole mask to the image size (nearest), never crop (host, discussion 702487)."""
    m = read_gray(path)
    if m.shape != shape_hw:
        m = cv2.resize(m, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def list_pairs(img_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    imgs = {p.stem: p for p in img_dir.rglob("*") if p.suffix.lower() in (".tif", ".tiff", ".png")}
    masks = {p.stem: p for p in mask_dir.rglob("*") if p.suffix.lower() in (".tif", ".tiff", ".png")}
    return [(imgs[k], masks[k]) for k in sorted(imgs) if k in masks]


def pixel_hash(a: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def dedup_and_split(items: list[dict], val_frac: float = 0.12, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Drop exact pixel duplicates (keeping the first), then split by a hash of a coarse thumbnail so that
    near-identical frames (same video / flipped copies) land on the same side."""
    seen, uniq = set(), []
    for it in items:
        if it["hash"] in seen:
            continue
        seen.add(it["hash"])
        uniq.append(it)
    rng = np.random.default_rng(seed)
    groups: dict[str, list[dict]] = {}
    for it in uniq:
        groups.setdefault(it["group"], []).append(it)
    keys = sorted(groups)
    rng.shuffle(keys)
    n_val = int(round(len(uniq) * val_frac))
    val, tr = [], []
    for k in keys:
        (val if len(val) < n_val else tr).extend(groups[k])
    return tr, val


def coarse_group(a: np.ndarray) -> str:
    """Flip-invariant coarse signature: 8x8 thumbnail of the image and its mirror, quantised."""
    t = cv2.resize(a, (8, 8), interpolation=cv2.INTER_AREA)
    t = np.minimum(t, t[:, ::-1])  # same for flipped copies
    q = (t // 32).astype(np.uint8)
    return hashlib.md5(q.tobytes()).hexdigest()[:10]
