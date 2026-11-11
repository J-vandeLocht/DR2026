"""Train the fundus-cropping segmentation model (Unet++ / EfficientNet-B4) for the
cross-dataset experiment: one run per dataset combo (2020 / 2025 / both) and fold.

Splits come from cropping/splits_<year>.json (cropping/make_splits_simple.py). For
--dataset both, fold i of 2020 and fold i of 2025 are merged (train = train_2020 +
train_2025, val = val_2020 + val_2025). Single-dataset runs should use double the
epochs of 'both' runs (the cohorts are about the same size) -- scripts/
train_cropping_xdomain.sh does this automatically.

Saves the best-val-IoU model to <out_dir>/seg_<dataset>_fold_<fold>.pth and a
per-epoch log to <...>_log.json.

Run from the repo root:
    python -u -m cropping.train_segmentation --dataset 2025 --fold 0 --epochs 40
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from cropping.segmentation_pipeline import FundusDataset, get_transforms


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_fold(ann_dir, dataset, fold):
    """Return (train_df, val_df) for the given dataset combo and fold."""
    ann_dir = Path(ann_dir)
    if dataset == "both":
        dfs, train_files, val_files = [], [], []
        for year in ("2020", "2025"):
            dfs.append(pd.read_csv(ann_dir / f"annotations_cropping_{year}.csv"))
            with open(ann_dir / f"splits_{year}.json") as f:
                s = json.load(f)[f"fold_{fold}"]
            train_files += s["train"]
            val_files += s["val"]
        df = pd.concat(dfs, ignore_index=True)
    else:
        df = pd.read_csv(ann_dir / f"annotations_cropping_{dataset}.csv")
        with open(ann_dir / f"splits_{dataset}.json") as f:
            s = json.load(f)[f"fold_{fold}"]
        train_files, val_files = s["train"], s["val"]
    return df[df["filename"].isin(train_files)], df[df["filename"].isin(val_files)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["2020", "2025", "both"], default="both")
    parser.add_argument("--fold", type=int, default=0, help="Fold index 0..4 (its val part is held out).")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_dir", type=str, default="cropping/frames_to_annotate_larger")
    parser.add_argument("--ann_dir", type=str, default="cropping")
    parser.add_argument("--out_dir", type=str, default="cropping/models_xdomain")
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b4")
    parser.add_argument("--encoder_weights", type=str, default="imagenet",
                        help="'imagenet' or 'none' for random init.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_fold(args.ann_dir, args.dataset, args.fold)
    print(f"dataset={args.dataset} fold={args.fold}: {len(train_df)} train / {len(val_df)} val frames")

    train_ds = FundusDataset(train_df, args.img_dir, get_transforms(train=True, img_size=args.img_size))
    val_ds = FundusDataset(val_df, args.img_dir, get_transforms(train=False, img_size=args.img_size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    model = smp.UnetPlusPlus(
        encoder_name=args.encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    model_save_path = out_dir / f"seg_{args.dataset}_fold_{args.fold}.pth"
    log_path = out_dir / f"seg_{args.dataset}_fold_{args.fold}_log.json"

    best_iou = -1.0  # guarantees a checkpoint is saved even if val IoU stays 0
    log = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = []

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = loss_fn(output, masks)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        model.eval()
        tp, fp, fn, tn = 0, 0, 0, 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                output = model(imgs)
                pred = (torch.sigmoid(output) > 0.5).long()
                s = smp.metrics.get_stats(pred, masks.long(), mode="binary")
                tp += s[0].sum()
                fp += s[1].sum()
                fn += s[2].sum()
                tn += s[3].sum()

        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro").item()
        epoch_log = {"epoch": epoch, "train_loss": float(np.mean(train_loss)), "val_iou": iou}
        log.append(epoch_log)
        print(f"dataset={args.dataset} Fold {args.fold} | Epoch {epoch + 1}/{args.epochs} | "
              f"Train Loss: {epoch_log['train_loss']:.4f} | Val IoU: {iou:.4f}")

        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), model_save_path)
            print(f"   -> New Best IoU! Saved {model_save_path}")

    with open(log_path, "w") as f:
        json.dump({"dataset": args.dataset, "fold": args.fold, "epochs": args.epochs,
                   "best_val_iou": best_iou, "history": log}, f, indent=2)
    print(f"Finished dataset={args.dataset} fold={args.fold}. Best Val IoU: {best_iou:.4f}")


if __name__ == "__main__":
    main()
