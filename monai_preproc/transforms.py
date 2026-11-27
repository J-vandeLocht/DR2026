"""MONAI transform chains for the segmentation stage.

The eval/inference chain reproduces cropping.segmentation_pipeline.get_transforms(
train=False) exactly (Resize + ImageNet Normalize; parity-tested), so models
trained here are evaluated on the same inputs as cropping/. The train chain is
the MONAI equivalent of get_transforms(train=True): random 80-100% square crop
resized to img_size, flips, affine (full +/-180 deg rotation, +/-5% translation,
0.95-1.05 scale), Gaussian blur, brightness/contrast, normalize. The mask uses
nearest interpolation everywhere.

All chains consume dicts with keys "image" (HWC uint8 RGB ndarray) and "mask"
(HW float ndarray), and return torch tensors (image: [3, S, S] float32
normalized, mask: [1, S, S] float32).
"""

import math

import numpy as np
from monai.transforms import (Compose, Lambdad, Resized, ToTensord,
                              NormalizeIntensityd, RandFlipd, RandSpatialCropd,
                              RandAffined, RandGaussianSmoothd, RandScaleIntensityd,
                              RandAdjustContrastd)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def hwc_to_chw(img):
    img = np.asarray(img)
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def mask_add_channel(mask):
    return mask[None].astype(np.float32)


def to_float_01(x):
    return x.float() / 255.0


def _normalize(keys=("image",)):
    return NormalizeIntensityd(keys=keys, channel_wise=True,
                               subtrahend=list(IMAGENET_MEAN), divisor=list(IMAGENET_STD))


def seg_val_transforms(img_size):
    """Identical to cropping.segmentation_pipeline.get_transforms(train=False):
    Resize(img_size) + ImageNet Normalize (+ ToTensor)."""
    return Compose([
        Lambdad(keys="image", func=hwc_to_chw),
        Lambdad(keys="mask", func=mask_add_channel),
        Resized(keys=("image", "mask"), spatial_size=(img_size, img_size),
                mode=("bilinear", "nearest")),
        ToTensord(keys=("image", "mask")),
        Lambdad(keys="image", func=to_float_01),
        Lambdad(keys="mask", func=lambda m: m.float()),
        _normalize(),
    ])


def seg_train_transforms(img_size):
    """MONAI equivalent of cropping.segmentation_pipeline.get_transforms(train=True):
    RandomResizedCrop(0.8-1.0, square) approximated by random 80-100% spatial crop +
    resize, horizontal/vertical flips (p=0.5), affine rotate +/-180 deg /
    translate +/-5% / scale 0.95-1.05 (p=0.7), Gaussian blur (p=0.3),
    brightness/contrast +/-15% (p=0.5), ImageNet normalize."""
    s = int(img_size)
    return Compose([
        Lambdad(keys="image", func=hwc_to_chw),
        Lambdad(keys="mask", func=mask_add_channel),
        RandSpatialCropd(keys=("image", "mask"), roi_size=(int(0.8 * s), int(0.8 * s)),
                         max_roi_size=(s, s), random_size=True, random_center=True),
        Resized(keys=("image", "mask"), spatial_size=(s, s),
                mode=("bilinear", "nearest")),
        RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=0),
        RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=1),
        RandAffined(keys=("image", "mask"), prob=0.7,
                    rotate_range=(math.pi, math.pi),
                    translate_range=(0.05 * s, 0.05 * s),
                    scale_range=(0.95, 1.05),
                    mode=("bilinear", "nearest"), padding_mode="border"),
        RandGaussianSmoothd(keys="image", prob=0.3, sigma_x=(0.25, 1.0)),
        RandScaleIntensityd(keys="image", prob=0.5, factors=0.15),
        RandAdjustContrastd(keys="image", prob=0.5, gamma=(0.85, 1.15)),
        ToTensord(keys=("image", "mask")),
        Lambdad(keys="image", func=to_float_01),
        Lambdad(keys="mask", func=lambda m: m.float()),
        _normalize(),
    ])
