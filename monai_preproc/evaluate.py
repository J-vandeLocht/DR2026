"""Evaluate MONAI pre-processing models on the same leakage-free val folds as the
existing pipeline, and measure per-config cost (params / FLOPs / latency).

    python -u -m monai_preproc.evaluate seg --config unet_small_384 \
        --models_dir monai_preproc/models/seg --out_dir monai_preproc/eval/seg/unet_small_384
    python -u -m monai_preproc.evaluate cls --config unet_small_384 \
        --models_dir monai_preproc/models/cls --out_dir monai_preproc/eval/cls/unet_small_384

Outputs mirror cropping/evaluate_segmentation.py and
informative_frames/evaluate_classification.py (per_image_metrics.csv,
summary.csv, bar_summary.csv + a bar plot) so results are directly comparable
with cropping/eval/ and informative_frames/eval/. summary.csv additionally
carries cost columns (params, flops, latency_ms, img_size) for the sweep.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from .configs import PreprocConfig, load_grid
from .data import MonaiFundusDataset, MonaiFrameDataset, load_seg_fold, load_cls_samples
from .models import load_seg_model, load_cls_model, find_fold_checkpoint, measure_cost
from .train import binary_counts
from .transforms import seg_val_transforms, cls_val_transforms

SEG_METRICS = ["iou", "dice", "acc"]
CLS_METRICS = ["roc_auc", "pr_auc", "f1"]


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
        row = {"fold": fold, "n_images": len(ds), "config": cfg.name, **{f"micro_{m}": v for m, v in micro.items()}}
        summary_rows.append(row)
        log(f"fold={fold} (val fold): " + " ".join(f"{m}={micro[m]:.4f}" for m in SEG_METRICS))
    if not summary_rows:
        raise RuntimeError(f"no seg checkpoints found in {models_dir} -- train first (monai_preproc.train seg)")
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


def evaluate_cls(cfg, models_dir, img_base, split_dir, out_dir, n_splits=5, batch_size=32,
                 num_workers=4, device=None, log=print):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image_rows, summary_rows = [], []
    for fold in range(n_splits):
        try:
            ckpt = find_fold_checkpoint(models_dir, "cls_", fold)
        except FileNotFoundError:
            log(f"WARNING: no cls checkpoint for fold {fold} in {models_dir} -- fold skipped")
            continue
        val_samples = load_cls_samples(fold, "val", img_base, split_dir)
        loader = DataLoader(MonaiFrameDataset(val_samples, cls_val_transforms(cfg.cls_img_size)),
                            batch_size=batch_size, shuffle=False, num_workers=num_workers)
        model = load_cls_model(ckpt, device, cfg=cfg)
        probs = []
        with torch.no_grad():
            for imgs, _ in loader:
                probs.extend(torch.sigmoid(model(imgs.to(device))).view(-1).cpu().numpy().tolist())
        y_true = np.array([y for _, y in val_samples])
        probs = np.asarray(probs)
        pred = probs > 0.5
        tp = int((pred & (y_true == 1)).sum()); fp = int((pred & (y_true == 0)).sum())
        fn = int(((~pred) & (y_true == 1)).sum()); tn = int(((~pred) & (y_true == 0)).sum())
        pr_auc = auc(*reversed(precision_recall_curve(y_true, probs)[:2])) if len(np.unique(y_true)) > 1 else np.nan
        roc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)
        summary_rows.append({"fold": fold, "n_images": len(y_true), "config": cfg.name,
                             "roc_auc": roc, "pr_auc": pr_auc, "f1": f1,
                             "acc": (tp + tn) / (tp + fp + fn + tn + 1e-7)})
        for (p, y), prob in zip(val_samples, probs):
            per_image_rows.append({"fold": fold, "filename": p.name, "label": y,
                                   "prob": float(prob), "pred": int(prob > 0.5)})
        log(f"fold={fold} (val fold): roc_auc={roc:.4f} pr_auc={pr_auc:.4f} f1={f1:.4f}")
    if not summary_rows:
        raise RuntimeError(f"no cls checkpoints found in {models_dir} -- train first (monai_preproc.train cls)")
    write_eval_outputs(out_dir, pd.DataFrame(per_image_rows), pd.DataFrame(summary_rows),
                       CLS_METRICS, "classification_eval.png",
                       "Informative-frame classifier (MONAI) on Paxos2025 -- fold-i model on fold-i val")
    cost = measure_cost(load_cls_model(find_fold_checkpoint(models_dir, "cls_", 0), device, cfg=cfg),
                        cfg.cls_img_size, device)
    summary = pd.DataFrame(summary_rows)
    summary["params"] = cost["params"]
    summary["flops"] = cost["flops"]
    summary["latency_ms"] = cost["latency_ms"]
    summary["cls_img_size"] = cfg.cls_img_size
    summary["cls_backbone"] = cfg.cls_backbone
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
    parser.add_argument("stage", choices=["seg", "cls"])
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--grid", type=str, default=None)
    parser.add_argument("--models_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ann_csv", type=str, default="cropping/annotations_cropping_2025.csv")
    parser.add_argument("--splits_csv", type=str, default="cropping/split_cropping_5fold_paxos2025_from_q5v3.csv")
    parser.add_argument("--img_dir", type=str, default="cropping/frames_to_annotate_larger")
    parser.add_argument("--img_base", type=str, default="data/informative_frames")
    parser.add_argument("--split_dir", type=str, default="informative_frames")
    args = parser.parse_args(argv)

    cfg = load_grid(args.grid)[args.config]
    if args.stage == "seg":
        return evaluate_seg(cfg, args.models_dir, args.ann_csv, args.splits_csv, args.img_dir,
                            args.out_dir, n_splits=args.n_splits, batch_size=args.batch_size,
                            num_workers=args.num_workers)
    return evaluate_cls(cfg, args.models_dir, args.img_base, args.split_dir, args.out_dir,
                        n_splits=args.n_splits, batch_size=args.batch_size, num_workers=args.num_workers)


if __name__ == "__main__":
    main()
