"""Evaluate the cross-dataset cropping experiments.

For every training config (2020 / 2025 / both) and fold i, the model
<models_dir>/seg_<config>_fold_<i>.pth is evaluated on the fold-i *val* split of
BOTH datasets (cropping/splits_2020.json and splits_2025.json). So all configs are
compared on identical data: a model trained with fold i held out is always
evaluated on fold-i val frames only (e.g. the split-1 Paxos2025 model is evaluated
on split-1 Paxos2020 val frames, not on other splits).

Metrics per image (binary fundus mask, at --img_size): IoU, Dice, pixel accuracy;
aggregated micro over all val images of the fold, plus per-image means.

Outputs in --out_dir:
  per_image_metrics.csv  one row per (config, fold, eval_ds, image)
  summary.csv            one row per (config, fold, eval_ds): micro + macro metrics
  bar_summary.csv        one row per bar of the plot: mean/std over the 5 folds
  segmentation_eval.png  3 subplots (IoU, Dice, pixel acc), each with 6 bars
                         [train 2020/2025/both x eval 2020/2025]; bar = mean over
                         the 5 folds, the 5 individual fold values overlaid as points

Run from the repo root:
    python -u -m cropping.evaluate_segmentation
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
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from cropping.segmentation_pipeline import FundusDataset, get_transforms

CONFIGS = ["2020", "2025", "both"]
METRICS = ["iou", "dice", "acc"]
METRIC_TITLES = {"iou": "IoU (Jaccard)", "dice": "Dice", "acc": "Pixel accuracy"}


def load_val_df(ann_dir, eval_ds, fold):
    ann_dir = Path(ann_dir)
    df = pd.read_csv(ann_dir / f"annotations_cropping_{eval_ds}.csv")
    with open(ann_dir / f"splits_{eval_ds}.json") as f:
        val_files = json.load(f)[f"fold_{fold}"]["val"]
    return df[df["filename"].isin(val_files)]


def evaluate_model(model, loader, device):
    """Return per-image (tp, fp, fn, tn) rows aligned with loader.dataset order."""
    model.eval()
    per_image = []
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            pred = (torch.sigmoid(model(imgs)) > 0.5).long()
            s = smp.metrics.get_stats(pred, masks.long(), mode="binary")
            for i in range(imgs.size(0)):
                per_image.append((s[0][i].item(), s[1][i].item(), s[2][i].item(), s[3][i].item()))
    return per_image


def metrics_from_counts(tp, fp, fn, tn, eps=1e-7):
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "acc": (tp + tn) / (tp + fp + fn + tn + eps),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models_dir", type=str, default="cropping/models_xdomain")
    parser.add_argument("--ann_dir", type=str, default="cropping")
    parser.add_argument("--img_dir", type=str, default="cropping/frames_to_annotate_larger")
    parser.add_argument("--out_dir", type=str, default="cropping/eval_xdomain")
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b4")
    parser.add_argument("--configs", nargs="+", default=CONFIGS)
    parser.add_argument("--eval_datasets", nargs="+", default=["2020", "2025"], choices=["2020", "2025"])
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cache one val dataset per (eval_ds, fold)
    val_loaders = {}
    for eval_ds in args.eval_datasets:
        for fold in range(args.n_splits):
            df = load_val_df(args.ann_dir, eval_ds, fold)
            ds = FundusDataset(df, args.img_dir, get_transforms(train=False, img_size=args.img_size))
            val_loaders[(eval_ds, fold)] = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                                      num_workers=args.num_workers)
            print(f"eval set {eval_ds} fold {fold}: {len(df)} images")

    per_image_rows, summary_rows = [], []
    for config in args.configs:
        for fold in range(args.n_splits):
            model_path = Path(args.models_dir) / f"seg_{config}_fold_{fold}.pth"
            if not model_path.exists():
                print(f"WARNING: missing model {model_path} -- skipping (config={config}, fold={fold})")
                continue
            model = smp.UnetPlusPlus(encoder_name=args.encoder_name, encoder_weights=None,
                                     in_channels=3, classes=1).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))

            for eval_ds in args.eval_datasets:
                loader = val_loaders[(eval_ds, fold)]
                per_image = evaluate_model(model, loader, device)
                filenames = loader.dataset.df["filename"].tolist()
                assert len(filenames) == len(per_image)
                tp = sum(p[0] for p in per_image)
                fp = sum(p[1] for p in per_image)
                fn = sum(p[2] for p in per_image)
                tn = sum(p[3] for p in per_image)
                micro = metrics_from_counts(tp, fp, fn, tn)
                per_img_metrics = [metrics_from_counts(*p) for p in per_image]
                macro = {m: float(np.mean([pm[m] for pm in per_img_metrics])) for m in METRICS}
                for fn_name, p, pm in zip(filenames, per_image, per_img_metrics):
                    per_image_rows.append({"config": config, "fold": fold, "eval_ds": eval_ds,
                                           "filename": fn_name, "tp": p[0], "fp": p[1], "fn": p[2],
                                           "tn": p[3], **pm})
                summary_rows.append({
                    "config": config, "fold": fold, "eval_ds": eval_ds, "n_images": len(per_image),
                    "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                    **{f"micro_{m}": v for m, v in micro.items()},
                    **{f"macro_{m}": v for m, v in macro.items()},
                })
                print(f"config={config} fold={fold} eval={eval_ds}: "
                      + " ".join(f"{m}={micro[m]:.4f}" for m in METRICS)
                      + f" (micro over {len(per_image)} images)")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    pd.DataFrame(per_image_rows).to_csv(out_dir / "per_image_metrics.csv", index=False)
    print(f"Wrote {out_dir / 'per_image_metrics.csv'} ({len(per_image_rows)} rows)")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote {out_dir / 'summary.csv'} ({len(summary)} rows)")

    # Bar summary: mean/std over folds per (config, eval_ds)
    bar_rows = []
    for config in args.configs:
        for eval_ds in args.eval_datasets:
            sub = summary[(summary["config"] == config) & (summary["eval_ds"] == eval_ds)]
            for m in METRICS:
                vals = sub[f"micro_{m}"].to_numpy()
                bar_rows.append({"config": config, "eval_ds": eval_ds, "metric": m,
                                 "mean": float(np.mean(vals)) if len(vals) else np.nan,
                                 "std": float(np.std(vals)) if len(vals) else np.nan,
                                 "folds": len(vals)})
    bar_summary = pd.DataFrame(bar_rows)
    bar_summary.to_csv(out_dir / "bar_summary.csv", index=False)
    print(f"Wrote {out_dir / 'bar_summary.csv'}")

    # Plot: 3 subplots (one per metric), 6 bars each, 5 fold points overlaid
    fig, axes = plt.subplots(1, len(METRICS), figsize=(8.5 * len(METRICS), 7), sharey=False)
    bar_labels, bar_colors = [], []
    palette = {"2020": "#4C72B0", "2025": "#DD8452", "both": "#55A868"}
    for ax, m in zip(axes, METRICS):
        x, labels = 0, []
        for config in args.configs:
            for eval_ds in args.eval_datasets:
                sub = summary[(summary["config"] == config) & (summary["eval_ds"] == eval_ds)]
                vals = sub[f"micro_{m}"].to_numpy()
                mean = np.mean(vals) if len(vals) else np.nan
                std = np.std(vals) if len(vals) else np.nan
                ax.bar(x, mean, yerr=std, capsize=4, color=palette[config], alpha=0.85,
                       edgecolor="black")
                jitter = np.linspace(-0.18, 0.18, len(vals)) if len(vals) else []
                ax.scatter(np.full(len(vals), x) + jitter, vals, color="black", zorder=5, s=28)
                for xi, v in zip(np.full(len(vals), x) + jitter, vals):
                    ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 7),
                                ha="center", fontsize=9)
                labels.append(f"{config}→{eval_ds}")
                x += 1
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11)
        ax.set_title(METRIC_TITLES[m])
        ax.set_ylim(0, 1.12)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("micro " + METRIC_TITLES[m])
    # legend for train config colors
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=palette[c], label=f"train {c}") for c in args.configs],
               loc="upper right")
    fig.suptitle("Cropping (segmentation) cross-dataset evaluation — bars: mean over 5 folds, "
                 "points: individual folds\n(each model evaluated on the fold-i val split of both datasets)")
    fig.tight_layout(rect=[0, 0, 0.9, 0.93])
    plot_path = out_dir / "segmentation_eval.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    print("\nBar summary (mean ± std over folds, micro):")
    print(bar_summary.to_string(index=False))


if __name__ == "__main__":
    main()
