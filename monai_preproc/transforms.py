"""MONAI transform chains for the MONAI pre-processing pipeline.

The *eval/inference* chains reproduce the existing albumentations/torchvision
chains exactly (Resize + ImageNet Normalize; for the classifier additionally
CLAHE between resize and normalize), so models trained here are evaluated on
the same inputs as cropping/ and informative_frames/. The *train* chains are
MONAI-native equivalents of the albumentations augmentation stacks.

All chains consume dicts with keys "image" (HWC uint8 RGB ndarray) and, for
segmentation, "mask" (HW float ndarray), and return torch tensors
(image: [3, S, S] float32 normalized, mask: [1, S, S] float32).
"""

import cv2
import numpy as np
from monai.transforms import (Compose, Lambdad, Resized, ToTensord,
                              NormalizeIntensityd, RandFlipd, RandRotate90d,
                              RandSpatialCropd, RandAdjustContrastd, RandGaussianSmoothd)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def hwc_to_chw(img):
    img = np.asarray(img)
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def mask_add_channel(mask):
    return mask[None].astype(np.float32)


def to_float_01(x):
    return x.float() / 255.0


def clahe_np(img_chw):
    """CLAHE on the L channel, identical math to misc.utils.apply_clahe_cv2 (which
    works on a PIL image); keeps black pixels black. Input/output CHW uint8/float."""
    img = np.asarray(img_chw).transpose(1, 2, 0)
    if img.dtype != np.uint8:
        img = (img * 255).round().astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    valid = img.max(axis=2) > 0
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    out[~valid] = 0
    return np.ascontiguousarray(out.transpose(2, 0, 1))


def _normalize(keys=("image",)):
    return NormalizeIntensityd(keys=keys, channel_wise=True,
                               subtrahend=list(IMAGENET_MEAN), divisor=list(IMAGENET_STD))


def _base_keys():
    return Compose([
        Lambdad(keys="image", func=hwc_to_chw),
        Lambdad(keys="mask", func=mask_add_channel),
        ToTensord(keys=("image", "mask")),
    ])


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
    """MONAI equivalent of get_transforms(train=True): random 80-100% square crop
    resized to img_size, flips, 90-degree rotations, blur, brightness/contrast,
    normalize. Mask uses nearest interpolation everywhere."""
    return Compose([
        Lambdad(keys="image", func=hwc_to_chw),
        Lambdad(keys="mask", func=mask_add_channel),
        RandSpatialCropd(keys=("image", "mask"), roi_size=(int(0.8 * img_size),) * 2,
                         max_roi_size=(img_size, img_size), random_size=True, random_center=True),
        Resized(keys=("image", "mask"), spatial_size=(img_size, img_size),
                mode=("bilinear", "nearest")),
        RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=0),
        RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=1),
        RandRotate90d(keys=("image", "mask"), prob=0.7, max_k=3, spatial_axes=(0, 1)),
        RandGaussianSmoothd(keys="image", prob=0.3, sigma_x=(0.25, 1.0)),
        RandAdjustContrastd(keys="image", prob=0.5, gamma=(0.85, 1.15)),
        ToTensord(keys=("image", "mask")),
        Lambdad(keys="image", func=to_float_01),
        Lambdad(keys="mask", func=lambda m: m.float()),
        _normalize(),
    ])


def cls_val_transforms(img_size):
    """Identical to informative_frames.classify_frames.get_val_transforms:
    Resize + CLAHE + ToTensor + ImageNet Normalize."""
    return Compose([
        Lambdad(keys="image", func=hwc_to_chw),
        Resized(keys="image", spatial_size=(img_size, img_size), mode="bilinear"),
        Lambdad(keys="image", func=clahe_np),
        ToTensord(keys="image"),
        Lambdad(keys="image", func=to_float_01),
        _normalize(),
    ])


def cls_train_transforms(img_size):
    """MONAI equivalent of get_train_transforms: Resize + CLAHE + flips/90-rot + normalize."""
    return Compose([
        Lambdad(keys="image", func=hwc_to_chw),
        Resized(keys="image", spatial_size=(img_size, img_size), mode="bilinear"),
        Lambdad(keys="image", func=clahe_np),
        RandFlipd(keys="image", prob=0.5, spatial_axis=0),
        RandFlipd(keys="image", prob=0.5, spatial_axis=1),
        RandRotate90d(keys="image", prob=0.5, max_k=3, spatial_axes=(0, 1)),
        ToTensord(keys="image"),
        Lambdad(keys="image", func=to_float_01),
        _normalize(),
    ])


def seg_infer_transforms(img_size):
    """Single-image dict transform for the video extraction pipeline (seg stage)."""
    return seg_val_transforms(img_size)


def cls_infer_transforms(img_size):
    """Single-image dict transform for the video extraction pipeline (cls stage)."""
    return cls_val_transforms(img_size)
