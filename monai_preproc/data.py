"""Data loading for the MONAI segmentation stage.

Same data source as cropping/train_segmentation.py (cropping/annotations_cropping_2025.csv
+ the q5_v3-derived split CSVs), so the anti-leakage val-fold protocol is
identical. Unlike the original FundusDataset, rows whose files are not present
on disk are skipped with a warning instead of raising -- the annotation CSV can
list frames that are not downloaded locally, and that must never abort a run.
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


def load_seg_fold(ann_csv, splits_csv, fold):
    """(train_df, val_df) of the cropping split CSV for one fold, exactly like
    cropping.train_segmentation.load_fold."""
    df = pd.read_csv(ann_csv)
    sp = pd.read_csv(splits_csv)
    sp = sp[sp["fold"] == fold]
    train_files = sp[sp["split"] == "train"]["filename"]
    val_files = sp[sp["split"] == "val"]["filename"]
    return df[df["filename"].isin(train_files)], df[df["filename"].isin(val_files)]


def make_fundus_mask(row, shape):
    """Binary fundus mask from the ellipse annotation columns, as in
    cropping.segmentation_pipeline.FundusDataset."""
    mask = np.zeros(shape[:2], dtype=np.float32)
    center = (int(row["center_x"]), int(row["center_y"]))
    axes = (int(row["radius_x"]), int(row["radius_y"]))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    return mask


class MonaiFundusDataset(Dataset):
    """Dict dataset over the cropping annotations: {"image": HWC uint8 RGB,
    "mask": HW float32, "filename": str}. Rows whose image file is missing on
    disk are dropped with a warning (never an error)."""

    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        keep = []
        missing = []
        for i, row in self.df.iterrows():
            if (self.img_dir / row["filename"]).exists():
                keep.append(i)
            else:
                missing.append(row["filename"])
        if missing:
            print(f"WARNING: {len(missing)}/{len(self.df)} annotated frames not found in "
                  f"{self.img_dir} -- skipped, e.g. {missing[:3]}")
        self.df = self.df.loc[keep].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["filename"]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"unreadable image {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        item = {"image": image, "mask": make_fundus_mask(row, image.shape),
                "filename": row["filename"]}
        if self.transform:
            item = self.transform(item)
        return item
