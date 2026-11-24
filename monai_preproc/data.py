"""Datasets for the MONAI pre-processing pipeline.

Same data sources as the existing pipeline (cropping/annotations_cropping_2025.csv
+ the q5_v3-derived split CSVs; informative_frames split CSV + cropped frames),
so the anti-leakage val-fold protocol is identical. Unlike the existing
datasets, rows whose files are not present on disk are skipped with a warning
instead of raising -- the annotation CSVs can list frames/videos that are not
downloaded locally, and that must never abort a run.
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


def load_cls_samples(fold, split, img_base, split_dir):
    """(path, label) pairs for one fold+split of the informative-frame CSV, over
    <img_base>/frames_cropped/<label>/<filename>. Missing files are skipped with
    a warning (mirrors informative_frames.classify_frames.build_samples but
    never fails on missing files)."""
    split_csv_path = Path(split_dir) / "split_info_5fold_paxos2025_from_q5v3.csv"
    df = pd.read_csv(split_csv_path)
    df = df[(df["fold"] == fold) & (df["split"] == split)]
    folder = Path(img_base) / "frames_cropped"
    samples, missing = [], []
    for row in df.itertuples():
        path = folder / str(int(row.label)) / row.filename
        (samples.append((path, int(row.label))) if path.exists() else missing.append(path.name))
    if missing:
        print(f"WARNING: {len(missing)}/{len(df)} {split} rows of {split_csv_path.name} (fold {fold}) "
              f"not found under {folder} -- skipped, e.g. {missing[:3]}")
    return samples


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


class MonaiFrameDataset(Dataset):
    """(path, label) samples with a dict transform, mirroring
    informative_frames.classify_frames.FrameDataset."""

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, y = self.samples[index]
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"unreadable image {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        item = {"image": image, "filename": path.name}
        if self.transform:
            item = self.transform(item)
        return item["image"], y
