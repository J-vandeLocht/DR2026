"""Segmentation model builder and checkpoint I/O for the MONAI pipeline.

MONAI UNet (configurable channels/strides) or MONAI FlexibleUNet (ImageNet-
pretrained EfficientNet encoder backbone -- the counterpart of cropping/'s
smp.UnetPlusPlus(encoder_name="efficientnet-b4", encoder_weights="imagenet")).

Checkpoints are saved as seg_fold_<i>.pth with a <name>.json config sidecar, so
any loader can rebuild the exact architecture (same naming as cropping/models).
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import UNet, FlexibleUNet

from .configs import PreprocConfig


def build_seg_model(cfg: PreprocConfig, pretrained=None) -> nn.Module:
    pre = cfg.pretrained if pretrained is None else pretrained
    if cfg.seg_arch == "unet":
        # MONAI expects len(strides) == len(channels) - 1 and ignores extra strides
        # (with a UserWarning); truncate so legacy config sidecars stay quiet too.
        strides = list(cfg.seg_strides)[: len(cfg.seg_channels) - 1]
        return UNet(spatial_dims=2, in_channels=3, out_channels=1,
                    channels=list(cfg.seg_channels), strides=strides,
                    num_res_units=2)
    if cfg.seg_arch == "flexunet":
        return FlexibleUNet(in_channels=3, out_channels=1, backbone=cfg.seg_backbone,
                            pretrained=pre, spatial_dims=2)
    raise ValueError(f"unknown seg_arch '{cfg.seg_arch}' (choose 'unet' or 'flexunet')")


def save_checkpoint(model: nn.Module, cfg: PreprocConfig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    path.with_suffix(".json").write_text(json.dumps(cfg.to_dict(), indent=2))


def load_config_for(path, cfg=None):
    path = Path(path)
    if cfg is not None:
        return cfg
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        return PreprocConfig.from_dict(json.loads(sidecar.read_text()))
    raise FileNotFoundError(f"no config given and no sidecar {sidecar} next to {path}")


def load_seg_model(path, device, cfg=None) -> nn.Module:
    cfg = load_config_for(path, cfg)
    model = build_seg_model(cfg, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def find_fold_checkpoint(models_dir, prefix, fold):
    """seg_fold_<fold>*.pth (preferred) or *fold_<fold>* -- same glob logic as the
    existing loaders in informative_frames/crop_frames.py."""
    models_dir = Path(models_dir)
    paths = sorted(models_dir.glob(f"{prefix}fold_{fold}*.pth"))
    if not paths:
        paths = sorted(models_dir.glob(f"*{prefix[:-1] if prefix.endswith('_') else prefix}fold_{fold}*.pth"))
    if not paths and prefix == "seg_":
        paths = sorted(models_dir.glob(f"*fold_{fold}*.pth"))
    if not paths:
        raise FileNotFoundError(f"no checkpoint for fold {fold} in {models_dir} (prefix {prefix!r})")
    return paths[0]


def measure_cost(model: nn.Module, img_size: int, device, batch_size: int = 1, warmup: int = 2,
                 iters: int = 5):
    """Cost of one forward pass: parameter count, FLOPs (if torch flop counter is
    available) and wall-clock latency per batch on the given device."""
    import time

    model.eval()
    params = sum(p.numel() for p in model.parameters())
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    flops = None
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.no_grad(), FlopCounterMode(model, inputs=(x,)) as fcm:
            model(x)
        flops = int(fcm.get_total_flops())
    except Exception:
        flops = None
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) / iters * 1000.0
    return {"params": params, "flops": flops, "latency_ms": latency_ms,
            "img_size": img_size, "batch_size": batch_size}
