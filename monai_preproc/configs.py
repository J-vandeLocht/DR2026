"""Named segmentation configurations for the MONAI pipeline and the cost/performance sweep.

A PreprocConfig fully determines the segmentation stage (the informative-frame
classifier is a separate pipeline and is re-added later):
  arch: "unet" (MONAI UNet, configurable channels/strides) or "flexunet"
        (MONAI FlexibleUNet with an ImageNet-pretrained EfficientNet encoder --
        the MONAI counterpart of cropping/'s smp.UnetPlusPlus(EfficientNet-B4))
  seg_shape_loss_weight: weight of the ellipse shape-prior loss on top of the
        Dice loss (0.0 = plain Dice, exactly like cropping/train_segmentation.py)

Training hyperparameters mirror cropping/train_segmentation.py (the original
UNet++ pipeline): Adam(lr=1e-4), batch_size=4, epochs=40, Dice loss from logits,
best-val-micro-IoU checkpoint seg_fold_<fold>.pth; early stopping (patience=3,
0 disables) and TensorBoard logging are added on top in train.py.

DEFAULT_GRID is the original mixed grid (UNet vs FlexibleUNet comparison).
FLEXUNET_GRID is the FlexibleUNet-only architecture search used by
monai_preproc.sweep by default (--preset flexunet). load_grid selects a named
grid (GRIDS[preset]) and optionally extends/overrides it from a JSON file
({name: {field: value, ...}}).
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class PreprocConfig:
    name: str = "default"
    seg_arch: str = "unet"
    seg_backbone: str = "efficientnet-b0"
    pretrained: bool = False
    seg_channels: tuple = (32, 64, 128, 256, 512)
    seg_strides: tuple = (2, 2, 2, 2)
    seg_img_size: int = 768
    seg_shape_loss_weight: float = 0.0
    batch_size: int = 4
    lr: float = 1e-4
    epochs: int = 40
    patience: int = 3

    def to_dict(self):
        d = asdict(self)
        d["seg_channels"] = list(self.seg_channels)
        d["seg_strides"] = list(self.seg_strides)
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        if "seg_channels" in d:
            d["seg_channels"] = tuple(d["seg_channels"])
        if "seg_strides" in d:
            d["seg_strides"] = tuple(d["seg_strides"])
        known = {f for f in PreprocConfig.__dataclass_fields__}
        return PreprocConfig(**{k: v for k, v in d.items() if k in known})


DEFAULT_GRID = {
    "baseline_flexunet_b4_768": PreprocConfig(
        name="baseline_flexunet_b4_768",
        seg_arch="flexunet", seg_backbone="efficientnet-b4", seg_img_size=768,
        pretrained=True,
    ),
    "flexunet_b1_512": PreprocConfig(
        name="flexunet_b1_512",
        seg_arch="flexunet", seg_backbone="efficientnet-b1", seg_img_size=512,
        pretrained=True,
    ),
    "unet_medium_512": PreprocConfig(
        name="unet_medium_512",
        seg_arch="unet", seg_channels=(32, 64, 128, 256, 512), seg_strides=(2, 2, 2, 2),
        seg_img_size=512, pretrained=True,
    ),
    "unet_small_384": PreprocConfig(
        name="unet_small_384",
        seg_arch="unet", seg_channels=(16, 32, 64, 128), seg_strides=(2, 2, 2),
        seg_img_size=384, batch_size=8, pretrained=True,
    ),
    "unet_tiny_256": PreprocConfig(
        name="unet_tiny_256",
        seg_arch="unet", seg_channels=(8, 16, 32, 64), seg_strides=(2, 2, 2),
        seg_img_size=256, batch_size=16, pretrained=True,
    ),
}

FLEXUNET_GRID = {
    "flexunet_b0_256": PreprocConfig(
        name="flexunet_b0_256",
        seg_arch="flexunet", seg_backbone="efficientnet-b0", seg_img_size=256,
        batch_size=16, pretrained=True,
    ),
    "flexunet_b0_384": PreprocConfig(
        name="flexunet_b0_384",
        seg_arch="flexunet", seg_backbone="efficientnet-b0", seg_img_size=384,
        batch_size=8, pretrained=True,
    ),
    "flexunet_b1_384": PreprocConfig(
        name="flexunet_b1_384",
        seg_arch="flexunet", seg_backbone="efficientnet-b1", seg_img_size=384,
        batch_size=8, pretrained=True,
    ),
    "flexunet_b1_512": PreprocConfig(
        name="flexunet_b1_512",
        seg_arch="flexunet", seg_backbone="efficientnet-b1", seg_img_size=512,
        pretrained=True,
    ),
    "flexunet_b2_512": PreprocConfig(
        name="flexunet_b2_512",
        seg_arch="flexunet", seg_backbone="efficientnet-b2", seg_img_size=512,
        pretrained=True,
    ),
    "flexunet_b4_768": PreprocConfig(
        name="flexunet_b4_768",
        seg_arch="flexunet", seg_backbone="efficientnet-b4", seg_img_size=768,
        pretrained=True,
    ),
}

GRIDS = {"default": DEFAULT_GRID, "flexunet": FLEXUNET_GRID}


def load_grid(path=None, preset="default"):
    """GRIDS[preset] (default: DEFAULT_GRID), optionally overridden/extended
    by a JSON file of {name: {...}}."""
    grid = {k: PreprocConfig(**v.to_dict()) for k, v in GRIDS[preset].items()}
    if path:
        extra = json.loads(Path(path).read_text())
        for name, fields in extra.items():
            base = grid[name].to_dict() if name in grid else PreprocConfig(name=name).to_dict()
            base.update(fields)
            base["name"] = name
            grid[name] = PreprocConfig.from_dict(base)
    return grid
