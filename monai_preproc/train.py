"""Train the MONAI fundus-cropping segmentation model.

Mirrors cropping/train_segmentation.py (the original UNet++ pipeline): fold i
trains on rows fold==i & split=="train" of the same q5_v3-derived split CSVs and
validates on fold==i & split=="val" (never seen in training); Adam(lr),
Dice loss from logits (smooth=1.0, i.e. smp.losses.DiceLoss(BINARY_MODE,
from_logits=True)), optional AMP, best-val-micro-IoU checkpoint saved as
<out_dir>/seg_fold_<fold>.pth (+ .json config sidecar). On top of the original:
TensorBoard logging (<out_dir>/runs/fold_<i>), early stopping (patience), and a
2x5 visualization (top row: original val images, bottom row: image with
everything outside the predicted mask blacked out) saved on every new best val
IoU under <out_dir>/vis/.

    python -u -m monai_preproc.train --config flexunet_b4_768 --fold 0
    tensorboard --logdir monai_preproc/models/seg/runs
"""

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai.losses import DiceLoss

from .configs import GRIDS, load_grid
from .data import MonaiFundusDataset, load_seg_fold
from .losses import ellipse_shape_loss
from .models import build_seg_model, save_checkpoint
from .transforms import seg_train_transforms, seg_val_transforms, IMAGENET_MEAN, IMAGENET_STD


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


def save_improvement_plot(model, val_loader, out_path, n_images=5, device=None, writer=None, epoch=None):
    """Save the 2 x n_images visualization for a new best val IoU: top row the
    original val images, bottom row the same images with everything outside the
    predicted mask blacked out. Also logged to TensorBoard if a writer is given."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    batch = next(iter(val_loader))
    imgs = batch["image"][:n_images].to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(imgs)) > 0.5
    mean = torch.tensor(IMAGENET_MEAN, device=imgs.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=imgs.device).view(1, 3, 1, 1)
    orig = (imgs * std + mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    masked = orig * pred[:, 0].cpu().numpy()[..., None]
    fig, axes = plt.subplots(2, n_images, figsize=(3 * n_images, 6.2), squeeze=False)
    for i in range(n_images):
        has = i < len(orig)
        axes[0][i].imshow(orig[i] if has else np.zeros((imgs.shape[-1], imgs.shape[-1], 3)))
        axes[0][i].set_title(batch["filename"][i] if has else "", fontsize=8)
        axes[0][i].axis("off")
        axes[1][i].imshow(masked[i] if has else np.zeros((imgs.shape[-1], imgs.shape[-1], 3)))
        axes[1][i].axis("off")
    fig.suptitle("top: original val image | bottom: prediction mask (outside black)")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if writer is not None and epoch is not None:
        writer.add_figure("val/best_masked", fig, global_step=epoch)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def train_seg(cfg, fold, ann_csv, splits_csv, img_dir, out_dir, epochs=None, batch_size=None,
              lr=None, patience=None, amp=False, num_workers=4, device=None, seed=42, log=print):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    seed_everything(seed)
    epochs = cfg.epochs if epochs is None else epochs
    batch_size = cfg.batch_size if batch_size is None else batch_size
    lr = cfg.lr if lr is None else lr
    patience = cfg.patience if patience is None else patience
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_seg_fold(ann_csv, splits_csv, fold)
    train_ds = MonaiFundusDataset(train_df, img_dir, seg_train_transforms(cfg.seg_img_size))
    val_ds = MonaiFundusDataset(val_df, img_dir, seg_val_transforms(cfg.seg_img_size))
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(f"fold {fold}: empty train ({len(train_ds)}) or val ({len(val_ds)}) "
                           f"after dropping missing files -- check --img_dir {img_dir}")
    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin)
    log(f"fold={fold}: {len(train_ds)} train / {len(val_ds)} val frames "
        f"(img_size={cfg.seg_img_size}, arch={cfg.seg_arch}, shape_loss_weight={cfg.seg_shape_loss_weight})")

    model = build_seg_model(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # same form as smp.losses.DiceLoss(BINARY_MODE, from_logits=True) in cropping/train_segmentation.py
    loss_fn = DiceLoss(sigmoid=True, smooth_nr=1.0, smooth_dr=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    writer = SummaryWriter(log_dir=str(out_dir / "runs" / f"fold_{fold}"))
    model_path = out_dir / f"seg_fold_{fold}.pth"
    best_iou, best_epoch, since_best, history = -1.0, -1, 0, []
    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                logits = model(imgs)
                loss = loss_fn(logits, masks)
                if cfg.seg_shape_loss_weight > 0:
                    loss = loss + cfg.seg_shape_loss_weight * ellipse_shape_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())

        model.eval()
        tp = fp = fn = tn = 0
        with torch.no_grad(), torch.autocast(device_type="cuda", enabled=use_amp):
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                pred = (torch.sigmoid(model(imgs)) > 0.5).long()
                t, f, n, nn_ = binary_counts(pred, masks.long())
                tp += t.sum().item(); fp += f.sum().item(); fn += n.sum().item(); tn += nn_.sum().item()
        iou = tp / (tp + fp + fn + 1e-7)
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch, "train_loss": mean_loss, "val_iou": float(iou)})
        writer.add_scalar("train/loss", mean_loss, epoch)
        writer.add_scalar("val/iou", float(iou), epoch)
        log(f"fold {fold} epoch {epoch + 1}/{epochs} | train loss {mean_loss:.4f} | val IoU {iou:.4f}")

        if iou > best_iou:
            best_iou, best_epoch, since_best = float(iou), epoch, 0
            save_checkpoint(model, cfg, model_path)
            vis_path = out_dir / "vis" / f"fold{fold}_best_epoch{epoch:03d}.png"
            save_improvement_plot(model, val_loader, vis_path, n_images=5,
                                  device=device, writer=writer, epoch=epoch)
            writer.add_scalar("best/val_iou", best_iou, epoch)
            log(f"   -> new best val IoU {best_iou:.4f}: saved {model_path.name} and {vis_path.name}")
        else:
            since_best += 1
            if patience > 0 and since_best >= patience:
                log(f"   -> early stopping: val IoU did not improve for {patience} epochs")
                break
    writer.close()
    log(f"fold {fold} finished after {len(history)}/{epochs} epochs | best val IoU {best_iou:.4f} "
        f"(epoch {best_epoch + 1})")
    return {"fold": fold, "best_val_iou": best_iou, "best_epoch": best_epoch,
            "epochs_run": len(history), "model": str(model_path), "history": history}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default=None, help="Name from the grid (default: first grid entry).")
    parser.add_argument("--preset", choices=sorted(GRIDS), default="flexunet", help="Named grid to take --config from.")
    parser.add_argument("--grid", type=str, default=None, help="Optional JSON file overriding/extending the grid.")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience; 0 disables (default: config value, 3).")
    parser.add_argument("--amp", action="store_true", help="Mixed precision (autocast + GradScaler), like cropping/.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ann_csv", type=str, default="cropping/annotations_cropping_2025.csv")
    parser.add_argument("--splits_csv", type=str, default="cropping/split_cropping_5fold_paxos2025_from_q5v3.csv")
    parser.add_argument("--img_dir", type=str, default="cropping/frames_to_annotate_larger")
    parser.add_argument("--out_dir", type=str, default="monai_preproc/models/seg")
    args = parser.parse_args(argv)

    grid = load_grid(args.grid, preset=args.preset)
    cfg = grid[args.config] if args.config else next(iter(grid.values()))
    return train_seg(cfg, args.fold, args.ann_csv, args.splits_csv, args.img_dir, args.out_dir,
                     epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                     patience=args.patience, amp=args.amp, num_workers=args.num_workers,
                     seed=args.seed)


if __name__ == "__main__":
    main()
