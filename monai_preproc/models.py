"""Model builders and checkpoint I/O for the MONAI pre-processing pipeline.

Seg stage: MONAI UNet (configurable channels/strides) or MONAI FlexibleUNet
(timm/EfficientNet encoder backbone). Cls stage: MONAI EfficientNetBN with a
single-logit head, same role as the torchvision efficientnet_b4 classifier in
informative_frames/classify_frames.py.

Checkpoints are saved as <name>.pth with a <name>.json config sidecar, so any
loader can rebuild the exact architecture: seg_fold_<i>.pth / cls_fold_<i>.pth
under --models_dir, mirroring the cropping/ and informative_frames/ layout.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from monai.networks.nets import UNet, FlexibleUNet, EfficientNetBN

from .configs import PreprocConfig


def build_seg_model(cfg: PreprocConfig) -> nn.Module:
    if cfg.seg_arch == "unet":
        return UNet(spatial_dims=2, in_channels=3, out_channels=1,
                    channels=list(cfg.seg_channels), strides=list(cfg.seg_strides),
                    num_res_units=2)
    if cfg.seg_arch == "flexunet":
        return FlexibleUNet(in_channels=3, out_channels=1, backbone=cfg.seg_backbone,
                            pretrained=False, spatial_dims=2)
    raise ValueError(f"unknown seg_arch '{cfg.seg_arch}' (choose 'unet' or 'flexunet')")


def build_cls_model(cfg: PreprocConfig) -> nn.Module:
    return EfficientNetBN(model_name=cfg.cls_backbone, pretrained=False,
                          spatial_dims=2, in_channels=3, num_classes=1)


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
    model = build_seg_model(cfg)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def load_cls_model(path, device, cfg=None) -> nn.Module:
    cfg = load_config_for(path, cfg)
    model = build_cls_model(cfg)
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
