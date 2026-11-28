"""Visualize seg-stage predictions of a swept config on val images.

Loads the per-fold checkpoints that monai_preproc.sweep wrote under
<models_root>/<config>/seg (seg_fold_<i>.pth + .json sidecar), runs fold-i's
model on fold-i's val split (the same leakage-free protocol as the sweep), and
saves one figure per fold: per image [input | GT mask + GT ellipse | pred mask
+ fitted pred ellipse | TP/FP/FN], plus a CSV with per-image IoU and the
fitted ellipse parameters (center/radii in pixels) of prediction vs GT -- the
quantities the ellipse shape-prior loss penalizes.

    python -u -m monai_preproc.vis_seg --config flexunet_b0_256 --preset flexunet \
        --models_root results/monai_sweep/models --out_dir results/monai_sweep/vis \
        --folds 0 1 --n_images 4

Outputs: <out_dir>/<config>_fold<i>_val.png and <config>_fold<i>_val_metrics.csv
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .configs import load_grid, GRIDS
from .data import MonaiFundusDataset, load_seg_fold
from .models import find_fold_checkpoint, load_seg_model
from .transforms import seg_val_transforms, IMAGENET_MEAN, IMAGENET_STD
from .train import binary_counts


def ellipse_moments(prob, eps=1e-6):
    """Soft ellipse parameters of a map prob (B, 1, H, W) on a normalized [0, 1]
    coordinate grid. Returns (B, 4) rows (cx, cy, a, b): centroid and equivalent
    axis-aligned ellipse semi-axes a = 2*sqrt(var_x), b = 2*sqrt(var_y). Used only
    to draw the fitted ellipse of a prediction in the figure."""
    b, _, h, w = prob.shape
    ys = torch.linspace(0.0, 1.0, h, device=prob.device, dtype=prob.dtype).view(1, h, 1)
    xs = torch.linspace(0.0, 1.0, w, device=prob.device, dtype=prob.dtype).view(1, 1, w)
    p = prob[:, 0]
    m00 = p.sum(dim=(1, 2)) + eps
    cx = (p * xs).sum(dim=(1, 2)) / m00
    cy = (p * ys).sum(dim=(1, 2)) / m00
    var_x = (p * (xs - cx.view(b, 1, 1)) ** 2).sum(dim=(1, 2)) / m00
    var_y = (p * (ys - cy.view(b, 1, 1)) ** 2).sum(dim=(1, 2)) / m00
    a = 2.0 * torch.sqrt(var_x.clamp_min(eps))
    bb = 2.0 * torch.sqrt(var_y.clamp_min(eps))
    return torch.stack([cx, cy, a, bb], dim=1)


def denorm_image(img_chw):
    """Model input tensor (3,S,S, ImageNet-normalized) -> HWC float in [0,1]."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = (img_chw.cpu() * std + mean).clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def overlay(img01, mask, color, alpha=0.45):
    out = img01.copy()
    m = mask.astype(bool)
    out[m] = (1 - alpha) * out[m] + alpha * np.asarray(color, dtype=np.float64)
    return out


def draw_ellipse(img, params_px, color, thickness=2):
    cx, cy, rx, ry = [float(v) for v in params_px]
    cv2.ellipse(img, (int(round(cx)), int(round(cy))), (max(1, int(round(rx))), max(1, int(round(ry)))),
                0, 0, 360, color, thickness)


def vis_seg_fold(cfg, fold, models_dir, ann_csv, splits_csv, img_dir, out_dir, n_images=4,
                 batch_size=8, num_workers=0, device=None, log=print):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = find_fold_checkpoint(models_dir, "seg_", fold)  # FileNotFoundError -> caller warns
    model = load_seg_model(ckpt, device, cfg=cfg)
    _, val_df = load_seg_fold(ann_csv, splits_csv, fold)
    ds = MonaiFundusDataset(val_df, img_dir, seg_val_transforms(cfg.seg_img_size))
    if len(ds) == 0:
        raise RuntimeError(f"fold {fold}: empty val set after dropping missing files (--img_dir {img_dir})")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    stride = max(1, len(ds) // max(1, n_images))  # evenly spaced val images

    picked, seen = [], 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            probs = torch.sigmoid(model(imgs))
            pred = (probs > 0.5).long()
            tp, fp, fn, tn = binary_counts(pred, masks.long())
            for i in range(imgs.size(0)):
                seen += 1
                if (seen - 1) % stride != 0:
                    continue
                iou = float(tp[i]) / (float(tp[i]) + float(fp[i]) + float(fn[i]) + 1e-7)
                pred_ell = (ellipse_moments(probs[i:i + 1].cpu())[0] * cfg.seg_img_size).tolist()
                gt_ell = (ellipse_moments(masks[i:i + 1].cpu())[0] * cfg.seg_img_size).tolist()
                picked.append({"filename": batch["filename"][i], "iou": iou,
                               "img": denorm_image(imgs[i]), "mask": masks[i, 0].cpu().numpy(),
                               "prob": probs[i, 0].cpu().numpy(), "pred": pred[i, 0].cpu().numpy(),
                               "pred_ellipse": pred_ell, "gt_ellipse": gt_ell})
            if len(picked) >= n_images:
                break
    if not picked:
        raise RuntimeError(f"fold {fold}: no val images selected")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(picked)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n), squeeze=False)
    for row, r in enumerate(picked):
        img = r["img"]
        gt, pred = r["mask"], r["pred"]
        tp_m = (pred == 1) & (gt == 1)
        fp_m = (pred == 1) & (gt == 0)
        fn_m = (pred == 0) & (gt == 1)
        diff = np.zeros_like(img)
        diff[tp_m] = (0.0, 1.0, 0.0)
        diff[fp_m] = (1.0, 0.0, 0.0)
        diff[fn_m] = (0.0, 0.0, 1.0)
        gt_vis = overlay(img, gt, (0.0, 1.0, 0.0))
        draw_ellipse(gt_vis, r["gt_ellipse"], (0.0, 1.0, 1.0))
        pred_vis = overlay(img, pred, (1.0, 0.0, 0.0))
        draw_ellipse(pred_vis, r["pred_ellipse"], (1.0, 1.0, 0.0))
        panels = [(img, f"{r['filename']} | IoU {r['iou']:.3f}"),
                  (gt_vis, "GT mask (green) + GT ellipse (cyan)"),
                  (pred_vis, "pred mask (red) + fitted pred ellipse (yellow)"),
                  (diff, "TP green / FP red / FN blue")]
        for col, (panel, title) in enumerate(panels):
            ax = axes[row][col]
            ax.imshow(np.clip(panel, 0, 1))
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.suptitle(f"{cfg.name} | fold {fold} val | checkpoint {ckpt.name} | "
                 f"arch={cfg.seg_arch}/{cfg.seg_backbone}, pretrained={cfg.pretrained}")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    png_path = out_dir / f"{cfg.name}_fold{fold}_val.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)

    rows = [{"filename": r["filename"], "iou": r["iou"],
             "pred_cx": r["pred_ellipse"][0], "pred_cy": r["pred_ellipse"][1],
             "pred_rx": r["pred_ellipse"][2], "pred_ry": r["pred_ellipse"][3],
             "gt_cx": r["gt_ellipse"][0], "gt_cy": r["gt_ellipse"][1],
             "gt_rx": r["gt_ellipse"][2], "gt_ry": r["gt_ellipse"][3]} for r in picked]
    csv_path = out_dir / f"{cfg.name}_fold{fold}_val_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    log(f"fold {fold}: wrote {png_path} and {csv_path} ({n} val images)")
    return png_path, csv_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True, help="Grid config name, e.g. unetpp_b1_512.")
    parser.add_argument("--preset", choices=sorted(GRIDS), default="arch")
    parser.add_argument("--grid", type=str, default=None, help="JSON file overriding/extending the preset grid.")
    parser.add_argument("--models_root", type=str, default="results/monai_sweep/models",
                        help="Sweep models root; fold checkpoints are read from <models_root>/<config>/seg.")
    parser.add_argument("--out_dir", type=str, default="results/monai_sweep/vis")
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
    parser.add_argument("--n_images", type=int, default=4, help="Val images shown per fold (evenly spaced).")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ann_csv", type=str, default="cropping/annotations_cropping_2025.csv")
    parser.add_argument("--splits_csv", type=str, default="cropping/split_cropping_5fold_paxos2025_from_q5v3.csv")
    parser.add_argument("--img_dir", type=str, default="cropping/frames_to_annotate_larger")
    args = parser.parse_args(argv)

    cfg = load_grid(args.grid, preset=args.preset)[args.config]
    models_dir = Path(args.models_root) / args.config / "seg"
    for fold in args.folds:
        try:
            vis_seg_fold(cfg, fold, models_dir, args.ann_csv, args.splits_csv, args.img_dir,
                         args.out_dir, n_images=args.n_images, batch_size=args.batch_size,
                         num_workers=args.num_workers)
        except FileNotFoundError as e:
            print(f"WARNING: fold {fold} skipped: {e}")


if __name__ == "__main__":
    main()
