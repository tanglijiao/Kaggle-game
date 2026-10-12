"""Segmentation model factory (segmentation_models_pytorch, single grey input channel)."""
from __future__ import annotations

import torch


def build_model(arch: str = "unet", encoder: str = "resnet34", pretrained: bool = True) -> torch.nn.Module:
    import segmentation_models_pytorch as smp

    weights = "imagenet" if pretrained else None
    cls = {"unet": smp.Unet, "unetpp": smp.UnetPlusPlus, "fpn": smp.FPN, "manet": smp.MAnet}[arch]
    kw = {"decoder_attention_type": "scse"} if arch in ("unet", "unetpp") else {}
    return cls(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=1, **kw)


def load_checkpoint(path: str, device: torch.device) -> tuple[torch.nn.Module, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = build_model(ck["arch"], ck["encoder"], pretrained=False)
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval(), ck
