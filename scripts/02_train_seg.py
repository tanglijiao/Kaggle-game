"""Stage 2: train a binary segmentation model for aponeuroses (--task apo) or fascicle fragments (--task fasc).

Full-frame resize to H x W (default 512 x 768: most images are 3:2); masks are resized the same way
(they share the image's field of view). Exact duplicates are dropped and the hold-out split is grouped by a
flip-invariant thumbnail signature so mirrored copies / near-identical frames stay on one side.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from umud.data import coarse_group, dedup_and_split, list_pairs, pixel_hash, read_gray, read_mask  # noqa: E402
from umud.models import build_model  # noqa: E402
from umud.paths import find_data_root, find_subdir  # noqa: E402

MEAN, STD = 0.45, 0.25


def resize_mask(m: np.ndarray, w: int, h: int, thin: bool) -> np.ndarray:
    if thin:  # keep 1-2 px fascicle lines alive when downsampling
        f = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        return (f > 0.2).astype(np.uint8)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)


def augment(img: np.ndarray, msk: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape
    if rng.random() < 0.5:
        img, msk = img[:, ::-1], msk[:, ::-1]
    if rng.random() < 0.8:
        ang = rng.uniform(-10, 10)
        sc = rng.uniform(0.85, 1.2)
        tx, ty = rng.uniform(-0.06, 0.06) * w, rng.uniform(-0.06, 0.06) * h
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        M[:, 2] += (tx, ty)
        img = cv2.warpAffine(np.ascontiguousarray(img), M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
        msk = cv2.warpAffine(np.ascontiguousarray(msk), M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    img = img.astype(np.float32) / 255.0
    if rng.random() < 0.8:  # contrast / brightness / gamma (device & gain variation)
        img = np.clip(img * rng.uniform(0.7, 1.4) + rng.uniform(-0.12, 0.12), 0, 1)
        img = img ** rng.uniform(0.7, 1.5)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.3:  # speckle-like multiplicative noise
        img = np.clip(img * rng.normal(1.0, rng.uniform(0.05, 0.2), img.shape).astype(np.float32), 0, 1)
    return img, msk


class SegDS(Dataset):
    def __init__(self, items, train: bool, seed: int = 0):
        self.items, self.train = items, train
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img, msk = it["img"], it["msk"]
        if self.train:
            img, msk = augment(img, msk, np.random.default_rng(self.rng.integers(1 << 31)))
        else:
            img = img.astype(np.float32) / 255.0
        x = (np.ascontiguousarray(img) - MEAN) / STD
        return torch.from_numpy(x[None]).float(), torch.from_numpy(np.ascontiguousarray(msk)[None]).float()


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((2, 3)) + eps
    den = p.sum((2, 3)) + target.sum((2, 3)) + eps
    return 1 - (num / den).mean()


@torch.no_grad()
def evaluate(model, dl, device):
    model.eval()
    inter = union = 0.0
    tp = fp = fn = 0.0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            lg = model(x)
        p = (torch.sigmoid(lg.float()) > 0.5).float()
        tp += (p * y).sum().item()
        fp += (p * (1 - y)).sum().item()
        fn += ((1 - p) * y).sum().item()
    return {"dice": 2 * tp / max(1, 2 * tp + fp + fn), "prec": tp / max(1, tp + fp), "rec": tp / max(1, tp + fn)}


def save_preview(model, items, device, path: Path, n: int = 6):
    model.eval()
    tiles = []
    for it in items[:n]:
        x = torch.from_numpy(((it["img"].astype(np.float32) / 255 - MEAN) / STD)[None, None]).to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(x).float())[0, 0].cpu().numpy()
        rgb = cv2.cvtColor(it["img"], cv2.COLOR_GRAY2BGR)
        rgb[it["msk"] > 0] = (0, 255, 0)
        pr = cv2.cvtColor(it["img"], cv2.COLOR_GRAY2BGR)
        pr[p > 0.5] = (0, 0, 255)
        tiles.append(np.hstack([rgb, pr]))
    cv2.imwrite(str(path), np.vstack(tiles))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["apo", "fasc"], required=True)
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default="/kaggle/working/models")
    ap.add_argument("--encoder", default="resnet34")
    ap.add_argument("--arch", default="unet")
    ap.add_argument("--h", type=int, default=512)
    ap.add_argument("--w", type=int, default=768)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--full", action="store_true", help="train on all data (no hold-out), save last epoch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only N images")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "", flush=True)

    root = find_data_root(args.data)
    pairs = list_pairs(find_subdir(root, f"{args.task}_imgs"), find_subdir(root, f"{args.task}_masks"))
    if args.limit:
        pairs = pairs[: args.limit]
    t0 = time.time()
    items = []
    for ip, mp in pairs:
        im = read_gray(ip)
        m = read_mask(mp, im.shape)
        items.append({
            "name": ip.stem, "hash": pixel_hash(im), "group": coarse_group(im),
            "img": cv2.resize(im, (args.w, args.h), interpolation=cv2.INTER_AREA),
            "msk": resize_mask(m, args.w, args.h, thin=args.task == "fasc"),
        })
    print(f"loaded {len(items)} pairs in {time.time() - t0:.0f}s", flush=True)
    if args.full:
        seen, tr = set(), []
        for it in items:
            if it["hash"] not in seen:
                seen.add(it["hash"]); tr.append(it)
        va = tr[:64]  # monitoring only (seen in training)
    else:
        tr, va = dedup_and_split(items, args.val_frac, args.seed)
    print(f"train {len(tr)}  val {len(va)}  (dropped {len(items) - len(tr) - (0 if args.full else len(va))} dups)",
          flush=True)
    pos = np.mean([it["msk"].mean() for it in tr])
    print(f"mask positive fraction {pos:.4f}", flush=True)

    dl_tr = DataLoader(SegDS(tr, True, args.seed), batch_size=args.bs, shuffle=True, num_workers=4,
                       drop_last=True, pin_memory=True, persistent_workers=True)
    dl_va = DataLoader(SegDS(va, False), batch_size=args.bs, shuffle=False, num_workers=2)

    model = build_model(args.arch, args.encoder, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    pos_w = torch.tensor(min(20.0, 0.1 / max(pos, 1e-4)) if args.task == "fasc" else 1.0, device=device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.task}_{args.arch}_{args.encoder}_{args.h}x{args.w}" + ("_full" if args.full else "")
    best, log = -1.0, []
    for ep in range(args.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for x, y in dl_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                lg = model(x)
            lg = lg.float()
            loss = F.binary_cross_entropy_with_logits(lg, y, pos_weight=pos_w) + dice_loss(lg, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        m = evaluate(model, dl_va, device)
        row = {"epoch": ep + 1, "loss": tot / len(dl_tr), **m, "sec": time.time() - t0}
        log.append(row)
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}), flush=True)
        if args.full or m["dice"] > best:
            best = m["dice"]
            torch.save({"state_dict": model.state_dict(), "arch": args.arch, "encoder": args.encoder,
                        "h": args.h, "w": args.w, "task": args.task, "val": m, "epoch": ep + 1},
                       out / f"{tag}.pt")
    (out / f"{tag}_log.json").write_text(json.dumps(log, indent=1))
    ck = torch.load(out / f"{tag}.pt", map_location=device)
    model.load_state_dict(ck["state_dict"])
    save_preview(model, va, device, out / f"{tag}_preview.png")
    print(f"best dice {best:.4f} -> {out / tag}.pt", flush=True)


if __name__ == "__main__":
    main()
