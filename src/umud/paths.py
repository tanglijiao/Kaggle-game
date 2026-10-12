"""Locate the competition data, wherever it is mounted."""
from __future__ import annotations

import os
from pathlib import Path

SLUG = "umud-challenge-muscle-architecture-in-ultrasound-data"
SUBDIRS = ("apo_imgs", "apo_masks", "fasc_imgs", "fasc_masks", "test_images")


def find_data_root(explicit: str | os.PathLike | None = None) -> Path:
    """Return the directory that contains sample_submission.csv.

    Search order: explicit argument, $UMUD_DATA, common Kaggle mount points, ./data.
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("UMUD_DATA"):
        candidates.append(Path(os.environ["UMUD_DATA"]))
    candidates += [
        Path("/kaggle/input") / SLUG,
        Path("/kaggle/input/competitions") / SLUG,
        Path("data"),
    ]
    for c in candidates:
        if c.exists():
            hits = list(c.rglob("sample_submission.csv"))
            if hits:
                return hits[0].parent
    # last resort: scan /kaggle/input
    kin = Path("/kaggle/input")
    if kin.exists():
        hits = list(kin.rglob("sample_submission.csv"))
        if hits:
            return hits[0].parent
    raise FileNotFoundError("Could not find competition data (sample_submission.csv).")


def find_subdir(root: Path, prefix: str) -> Path | None:
    """Folders are versioned on Kaggle (e.g. test_images_v2); match by prefix."""
    matches = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    return matches[-1] if matches else None
