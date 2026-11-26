"""Train the MONAI pre-processing models, one fold at a time.

Same protocol as cropping/train_segmentation.py and informative_frames/classify_frames.py:
fold i trains on rows fold==i & split=="train" of the same q5_v3-derived split
CSVs and validates on fold==i & split=="val" (never seen in training), so the
anti-leakage evaluation of the existing pipeline carries over unchanged.

    python -u -m monai_preproc.train seg --config unet_small_384 --fold 0
    python -u -m monai_preproc.train cls --config unet_small_384 --fold 0

Checkpoints: <out_dir>/seg_fold_<fold>.pth (+ .json config sidecar, + _log.json),
respectively cls_fold_<fold>.pth. Seg selects best val IoU, cls best val PR-AUC.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from monai.losses import DiceCELoss
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from .configs import PreprocConfig, DEFAULT_GRID, GRIDS, load_grid
from .data import MonaiFundusDataset, MonaiFrameDataset, load_seg_fold, load_cls_samples
from .models import build_seg_model, build_cls_model, save_checkpoint
from .transforms import seg_train_transforms, seg_val_transforms, cls_train_transforms, cls_val_transforms


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def binary_counts(pred, target):
    """Pixel-level tp/fp/fn/tn for binary masks (replaces smp.metrics.get_stats)."""
    pred = pred.bool()
    target = target.bool()
    tp = (pred & target).sum(dim=(1, 2, 3))
    fp = (pred & ~target).sum(dim=(1, 2, 3))
    fn = (~pred & target).sum(dim=(1, 2, 3))
    tn = (~pred & ~target).sum(dim=(1, 2, 3))
    return tp, fp, fn, tn


def train_seg(cfg, fold, ann_csv, splits_csv, img_dir, out_dir, epochs=None, batch_size=None,
              lr=None, num_workers=4, device=None, seed=42, log=print):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    epochs = cfg.epochs if epochs is None else epochs
    batch_size = cfg.batch_size if batch_size is None else batch_size
    lr = cfg.lr if lr is None else lr
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_seg_fold(ann_csv, splits_csv, fold)
    train_ds = MonaiFundusDataset(train_df, img_dir, seg_train_transforms(cfg.seg_img_size))
    val_ds = MonaiFundusDataset(val_df, img_dir, seg_val_transforms(cfg.seg_img_size))
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"fold {fold}: empty train ({len(train_ds)}) or val ({len(val_ds)}) "
                           f"after dropping missing files -- check --img_dir {img_dir}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    log(f"fold={fold}: {len(train_ds)} train / {len(val_ds)} val frames (img_size={cfg.seg_img_size}, arch={cfg.seg_arch})")

    model = build_seg_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = DiceCELoss(sigmoid=True)

    model_path = out_dir / f"seg_fold_{fold}.pth"
    log_path = out_dir / f"seg_fold_{fold}_log.json"
    best_iou, best_epoch, history = -1.0, -1, []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(imgs), masks)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                pred = (torch.sigmoid(model(imgs)) > 0.5).long()
                t, f, n, nn_ = binary_counts(pred, masks.long())
                tp += t.sum().item(); fp += f.sum().item(); fn += n.sum().item(); tn += nn_.sum().item()
        iou = tp / (tp + fp + fn + 1e-7)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_iou": float(iou)})
        log(f"fold {fold} epoch {epoch + 1}/{epochs} | train loss {np.mean(losses):.4f} | val IoU {iou:.4f}")
        if iou > best_iou:
            best_iou, best_epoch = float(iou), epoch
            save_checkpoint(model, cfg, model_path)
    log_path.write_text(json.dumps({"config": cfg.to_dict(), "fold": fold, "epochs_run": len(history),
                                    "best_epoch": best_epoch, "best_val_iou": best_iou,
                                    "history": history}, indent=2))
    return {"fold": fold, "best_val_iou": best_iou, "best_epoch": best_epoch, "model": str(model_path)}


def train_cls(cfg, fold, img_base, split_dir, out_dir, epochs=None, batch_size=None, lr=None,
              num_workers=4, device=None, seed=42, log=print):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(seed)
    epochs = cfg.cls_epochs if epochs is None else epochs
    batch_size = cfg.cls_batch_size if batch_size is None else batch_size
    lr = cfg.cls_lr if lr is None else lr
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_samples = load_cls_samples(fold, "train", img_base, split_dir)
    val_samples = load_cls_samples(fold, "val", img_base, split_dir)
    if not train_samples or not val_samples:
        raise RuntimeError(f"fold {fold}: empty train ({len(train_samples)}) or val "
                           f"({len(val_samples)}) samples -- check --img_base {img_base}")
    train_labels = np.array([y for _, y in train_samples])
    counts = np.array([(train_labels == t).sum() for t in np.unique(train_labels)])
    weights = 1.0 / counts
    sampler = WeightedRandomSampler(torch.DoubleTensor([weights[t] for t in train_labels]), len(train_samples))
    train_loader = DataLoader(MonaiFrameDataset(train_samples, cls_train_transforms(cfg.cls_img_size)),
                              batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(MonaiFrameDataset(val_samples, cls_val_transforms(cfg.cls_img_size)),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    log(f"fold={fold}: {len(train_samples)} train / {len(val_samples)} val frames "
        f"(backbone={cfg.cls_backbone}, img_size={cfg.cls_img_size})")

    model = build_cls_model(cfg).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    model_path = out_dir / f"cls_fold_{fold}.pth"
    log_path = out_dir / f"cls_fold_{fold}_log.json"
    best_pr_auc, best_epoch, history = 0.0, -1, []
    for epoch in range(epochs):
        model.train()
        losses = []
        for imgs, lbls in train_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device).float().view(-1, 1)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        y_true, y_probs = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device)
                y_probs.extend(torch.sigmoid(model(imgs)).view(-1).cpu().numpy().tolist())
                y_true.extend(lbls.numpy().tolist())
        y_true = np.asarray(y_true)
        y_probs = np.asarray(y_probs)
        pr_auc = auc(*reversed(precision_recall_curve(y_true, y_probs)[:2])) if len(np.unique(y_true)) > 1 else 0.5
        roc = roc_auc_score(y_true, y_probs) if len(np.unique(y_true)) > 1 else 0.5
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "roc_auc": float(roc), "pr_auc": float(pr_auc)})
        log(f"fold {fold} epoch {epoch + 1}/{epochs} | train loss {np.mean(losses):.4f} | "
            f"val ROC-AUC {roc:.4f} | PR-AUC {pr_auc:.4f}")
        if pr_auc > best_pr_auc:
            best_pr_auc, best_epoch = float(pr_auc), epoch
            save_checkpoint(model, cfg, model_path)
    log_path.write_text(json.dumps({"config": cfg.to_dict(), "fold": fold, "epochs_run": len(history),
                                    "best_epoch": best_epoch, "best_val_pr_auc": best_pr_auc,
                                    "history": history}, indent=2))
    return {"fold": fold, "best_val_pr_auc": best_pr_auc, "best_epoch": best_epoch, "model": str(model_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["seg", "cls"])
    parser.add_argument("--config", type=str, default=None, help="Name from the grid (default: first grid entry).")
    parser.add_argument("--preset", choices=sorted(GRIDS), default="default", help="Named grid to take --config from.")
    parser.add_argument("--grid", type=str, default=None, help="Optional JSON file overriding/extending the grid.")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ann_csv", type=str, default="cropping/annotations_cropping_2025.csv")
    parser.add_argument("--splits_csv", type=str, default="cropping/split_cropping_5fold_paxos2025_from_q5v3.csv")
    parser.add_argument("--img_dir", type=str, default="cropping/frames_to_annotate_larger")
    parser.add_argument("--img_base", type=str, default="data/informative_frames")
    parser.add_argument("--split_dir", type=str, default="informative_frames")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="default monai_preproc/models/seg or .../cls")
    args = parser.parse_args(argv)

    grid = load_grid(args.grid, preset=args.preset)
    cfg = grid[args.config] if args.config else next(iter(grid.values()))
    out_dir = args.out_dir or f"monai_preproc/models/{args.stage}"
    if args.stage == "seg":
        return train_seg(cfg, args.fold, args.ann_csv, args.splits_csv, args.img_dir, out_dir,
                         epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                         num_workers=args.num_workers, seed=args.seed)
    return train_cls(cfg, args.fold, args.img_base, args.split_dir, out_dir,
                     epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                     num_workers=args.num_workers, seed=args.seed)


if __name__ == "__main__":
    main()
