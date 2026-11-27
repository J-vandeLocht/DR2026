"""Evaluate the MONAI segmentation models, mirroring cropping/evaluate_segmentation.py.

Every checkpoint <models_dir>/seg_fold_<i>.pth is evaluated on the fold-i *val*
rows of the q5_v3-derived split CSV (never seen in training); together the 5
folds cover every annotated frame exactly once. Metrics per image at the
config's input size: IoU, Dice, pixel accuracy; per fold micro (over all pixels)
and macro (per-image mean), like the original. summary.csv additionally carries
cost columns (params, flops, latency_ms, seg_img_size, seg_arch) for the sweep.

Outputs in --out_dir: per_image_metrics.csv, summary.csv, bar_summary.csv,
segmentation_eval.png -- the same files as cropping/eval/, so the two pipelines
are directly comparable.

    python -u -m monai_preproc.evaluate --config flexunet_b4_768 \
        --models_dir results/monai_sweep/models/flexunet_b4_768/seg \
        --out_dir results/monai_sweep/eval/flexunet_b4_768/seg
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .configs import GRIDS, load_grid
from .data import MonaiFundusDataset, load_seg_fold
from .models import load_seg_model, find_fold_checkpoint, measure_cost
from .train import binary_counts
from .transforms import seg_val_transforms

SEG_METRICS = ["iou", "dice", "acc"]


def metrics_from_counts(tp, fp, fn, tn, eps=1e-7):
    return {"iou": tp / (tp + fp + fn + eps),
            "dice": 2 * tp / (2 * tp + fp + fn + eps),
            "acc": (tp + tn) / (tp + fp + fn + tn + eps)}


def evaluate_seg(cfg, models_dir, ann_csv, splits_csv, img_dir, out_dir, n_splits=5,
                 batch_size=8, num_workers=4, device=None, log=print):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_rows, summary_rows = [], []
    for fold in range(n_splits):
        try:
            ckpt = find_fold_checkpoint(models_dir, "seg_", fold)
        except FileNotFoundError:
            log(f"WARNING: no seg checkpoint for fold {fold} in {models_dir} -- fold skipped")
            continue
        _, val_df = load_seg_fold(ann_csv, splits_csv, fold)
        ds = MonaiFundusDataset(val_df, img_dir, seg_val_transforms(cfg.seg_img_size))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        model = load_seg_model(ckpt, device, cfg=cfg)
        model.eval()
        fold_tp = fold_fp = fold_fn = fold_tn = 0
        with torch.no_grad():
            for batch in loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                pred = (torch.sigmoid(model(imgs)) > 0.5).long()
                tp, fp, fn, tn = binary_counts(pred, masks.long())
                for i in range(imgs.size(0)):
                    m = metrics_from_counts(int(tp[i]), int(fp[i]), int(fn[i]), int(tn[i]))
                    per_image_rows.append({"fold": fold, "filename": batch["filename"][i],
                                           "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i]),
                                           "tn": int(tn[i]), **m})
                fold_tp += int(tp.sum()); fold_fp += int(fp.sum()); fold_fn += int(fn.sum()); fold_tn += int(tn.sum())
        micro = metrics_from_counts(fold_tp, fold_fp, fold_fn, fold_tn)
        macro = {m: float(np.nanmean([r[m] for r in per_image_rows if r["fold"] == fold])) for m in SEG_METRICS}
        row = {"fold": fold, "n_images": len(ds), "config": cfg.name,
               "tp": fold_tp, "fp": fold_fp, "fn": fold_fn, "tn": fold_tn,
               **{f"micro_{m}": v for m, v in micro.items()},
               **{f"macro_{m}": v for m, v in macro.items()}}
        summary_rows.append(row)
        log(f"fold={fold} (val fold): " + " ".join(f"micro_{m}={micro[m]:.4f}" for m in SEG_METRICS))
    if not summary_rows:
        raise RuntimeError(f"no seg checkpoints found in {models_dir} -- train first (monai_preproc.train)")
    write_eval_outputs(out_dir, pd.DataFrame(per_image_rows), pd.DataFrame(summary_rows),
                       SEG_METRICS, "segmentation_eval.png",
                       "Cropping (segmentation, MONAI) on Paxos2025 -- fold-i model on fold-i val")
    cost = measure_cost(load_seg_model(find_fold_checkpoint(models_dir, "seg_", 0), device, cfg=cfg),
                        cfg.seg_img_size, device)
    summary = pd.DataFrame(summary_rows)
    summary["params"] = cost["params"]
    summary["flops"] = cost["flops"]
    summary["latency_ms"] = cost["latency_ms"]
    summary["seg_img_size"] = cfg.seg_img_size
    summary["seg_arch"] = cfg.seg_arch
    summary.to_csv(out_dir / "summary.csv", index=False)
    return summary


def write_eval_outputs(out_dir, per_image, summary, metrics, plot_name, title):
    per_image.to_csv(out_dir / "per_image_metrics.csv", index=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    bar_rows = []
    for m in metrics:
        key = m if m in summary.columns else f"micro_{m}"
        v = summary[key].to_numpy(dtype=float)
        bar_rows.append({"metric": m, "mean": float(np.nanmean(v)),
                         "std": float(np.nanstd(v)), "n_points": int(np.sum(~np.isnan(v)))})
    pd.DataFrame(bar_rows).to_csv(out_dir / "bar_summary.csv", index=False)
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 7))
    if len(metrics) == 1:
        axes = [axes]
    for ax, m in zip(axes, metrics):
        key = m if m in summary.columns else f"micro_{m}"
        vals = summary[key].to_numpy(dtype=float)
        mean, std = float(np.nanmean(vals)), float(np.nanstd(vals))
        ax.bar(0, mean, yerr=std, capsize=6, color="#DD8452", alpha=0.85, edgecolor="black")
        jitter = np.linspace(-0.12, 0.12, len(vals))
        ax.scatter(jitter, vals, color="black", zorder=5, s=30)
        ax.set_xticks([0])
        ax.set_xticklabels(["Paxos2025"], fontsize=11)
        ax.set_title(m)
        ax.set_ylim(0, 1.12)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(title + " -- one bar = mean over folds, points = fold values")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_dir / plot_name, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--preset", choices=sorted(GRIDS), default="flexunet", help="Named grid to take --config from.")
    parser.add_argument("--grid", type=str, default=None)
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ann_csv", type=str, default="cropping/annotations_cropping_2025.csv")
    parser.add_argument("--splits_csv", type=str, default="cropping/split_cropping_5fold_paxos2025_from_q5v3.csv")
    parser.add_argument("--img_dir", type=str, default="cropping/frames_to_annotate_larger")
    args = parser.parse_args(argv)

    cfg = load_grid(args.grid, preset=args.preset)[args.config]
    return evaluate_seg(cfg, args.models_dir, args.ann_csv, args.splits_csv, args.img_dir,
                        args.out_dir, n_splits=args.n_splits, batch_size=args.batch_size,
                        num_workers=args.num_workers)


if __name__ == "__main__":
    main()
