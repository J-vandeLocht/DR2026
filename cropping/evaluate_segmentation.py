"""Evaluate the cross-dataset cropping experiments.

Two evaluation modes per (train config, eval dataset) pair:

- in-dataset (config == eval_ds, or config == "both"): model seg_<config>_fold_<i>
  is evaluated on the fold-i *val* split of the eval dataset only (the part it never
  saw during training) -> 5 points in the plot, bar = mean over the 5 folds.
- cross-dataset (config != eval_ds, config != "both"): the model never saw the eval
  dataset, so ALL of its frames are used, and the 5 fold models are ensembled by
  averaging sigmoid PROBABILITIES (threshold 0.5) -- the same ensembling as
  informative_frames/extract_inf_frames.py, which averages probabilities, not
  logits -> ONE bar (mean of the ensemble over all data), hatched.

Metrics per image (binary fundus mask, at --img_size): IoU, Dice, pixel accuracy;
micro over all evaluated images, plus per-image means.

Outputs in --out_dir:
  per_image_metrics.csv  one row per (config, eval_ds, mode, fold, image)
  summary.csv            one row per (config, eval_ds, mode, fold): micro + macro
  bar_summary.csv        one row per bar of the plot
  segmentation_eval.png  3 subplots (IoU, Dice, pixel acc), 6 bars each
                         [train 2020/2025/both x eval 2020/2025]; in-dataset bars
                         show the 5 fold values as points, cross-dataset bars are
                         single hatched bars (5-model ensemble on all data)

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


def load_full_df(ann_dir, eval_ds):
    return pd.read_csv(Path(ann_dir) / f"annotations_cropping_{eval_ds}.csv")


def evaluate_model(model, loader, device):
    """Per-image (tp, fp, fn, tn) rows aligned with loader order (single model)."""
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


def evaluate_ensemble(models, loader, device):
    """Ensemble of the fold models: mean of sigmoid probabilities, threshold 0.5
    (probabilities, not logits -- same as extract_inf_frames.py).

    Returns (per_image_ensemble, per_model) where per_image_ensemble holds
    (tp, fp, fn, tn, mean_prob) per image and per_model[fold] holds (tp, fp, fn, tn)
    per image for that single model."""
    for m in models:
        m.eval()
    per_image_ens, per_model = [], [[] for _ in models]
    with torch.no_grad():
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            probs = torch.stack([torch.sigmoid(m(imgs)) for m in models])  # [M, B, 1, H, W]
            ens = (probs.mean(dim=0) > 0.5).long()
            s = smp.metrics.get_stats(ens, masks.long(), mode="binary")
            mp = probs.mean(dim=0)  # [B, 1, H, W]
            for i in range(imgs.size(0)):
                per_image_ens.append((s[0][i].item(), s[1][i].item(), s[2][i].item(),
                                      s[3][i].item(), mp[i].mean().item()))
            for mi, m in enumerate(models):
                sm = smp.metrics.get_stats((torch.sigmoid(m(imgs)) > 0.5).long(), masks.long(), mode="binary")
                for i in range(imgs.size(0)):
                    per_model[mi].append((sm[0][i].item(), sm[1][i].item(), sm[2][i].item(), sm[3][i].item()))
    return per_image_ens, per_model


def metrics_from_counts(tp, fp, fn, tn, eps=1e-7):
    return {
        "iou": tp / (tp + fp + fn + eps),
        "dice": 2 * tp / (2 * tp + fp + fn + eps),
        "acc": (tp + tn) / (tp + fp + fn + tn + eps),
    }


def summarize(filenames, per_image, config, eval_ds, mode, fold):
    """per_image: list of (tp, fp, fn, tn) per image, aligned with filenames."""
    tp = sum(p[0] for p in per_image)
    fp = sum(p[1] for p in per_image)
    fn = sum(p[2] for p in per_image)
    tn = sum(p[3] for p in per_image)
    micro = metrics_from_counts(tp, fp, fn, tn)
    per_img_metrics = [metrics_from_counts(*p[:4]) for p in per_image]
    macro = {m: float(np.mean([pm[m] for pm in per_img_metrics])) for m in METRICS}
    row = {"config": config, "eval_ds": eval_ds, "mode": mode, "fold": fold,
           "n_images": len(per_image), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
           **{f"micro_{m}": v for m, v in micro.items()},
           **{f"macro_{m}": v for m, v in macro.items()}}
    img_rows = [{"config": config, "eval_ds": eval_ds, "mode": mode, "fold": fold,
                 "filename": fn_name, "tp": p[0], "fp": p[1], "fn": p[2], "tn": p[3],
                 **metrics_from_counts(*p[:4])}
                for fn_name, p in zip(filenames, per_image)]
    return row, img_rows


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

    val_loaders, full_loaders = {}, {}
    for eval_ds in args.eval_datasets:
        full_df = load_full_df(args.ann_dir, eval_ds)
        full_ds = FundusDataset(full_df, args.img_dir, get_transforms(train=False, img_size=args.img_size))
        full_loaders[eval_ds] = DataLoader(full_ds, batch_size=args.batch_size, shuffle=False,
                                           num_workers=args.num_workers)
        print(f"eval set {eval_ds} (all): {len(full_df)} images")
        for fold in range(args.n_splits):
            df = load_val_df(args.ann_dir, eval_ds, fold)
            ds = FundusDataset(df, args.img_dir, get_transforms(train=False, img_size=args.img_size))
            val_loaders[(eval_ds, fold)] = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                                      num_workers=args.num_workers)
            print(f"eval set {eval_ds} fold {fold} val: {len(df)} images")

    per_image_rows, summary_rows = [], []

    for config in args.configs:
        fold_models = {}
        for fold in range(args.n_splits):
            model_path = Path(args.models_dir) / f"seg_{config}_fold_{fold}.pth"
            if not model_path.exists():
                print(f"WARNING: missing model {model_path} -- fold {fold} skipped for config={config}")
                continue
            model = smp.UnetPlusPlus(encoder_name=args.encoder_name, encoder_weights=None,
                                     in_channels=3, classes=1).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            fold_models[fold] = model
        if not fold_models:
            print(f"WARNING: no models for config={config} -- skipped")
            continue

        for eval_ds in args.eval_datasets:
            in_dataset = (config == "both") or (config == eval_ds)
            if in_dataset:
                # fold-i model on fold-i val only -> 5 points
                for fold, model in fold_models.items():
                    loader = val_loaders[(eval_ds, fold)]
                    per_image = evaluate_model(model, loader, device)
                    filenames = loader.dataset.df["filename"].tolist()
                    assert len(filenames) == len(per_image)
                    row, img_rows = summarize(filenames, per_image, config, eval_ds, "val_fold", fold)
                    summary_rows.append(row)
                    per_image_rows.extend(img_rows)
                    print(f"config={config} fold={fold} eval={eval_ds} (val fold): "
                          + " ".join(f"{m}={row[f'micro_{m}']:.4f}" for m in METRICS))
            else:
                # cross-dataset: ALL frames of eval_ds, 5-model probability ensemble -> one bar
                loader = full_loaders[eval_ds]
                models = [fold_models[f] for f in sorted(fold_models)]
                ens_per_image, per_model = evaluate_ensemble(models, loader, device)
                filenames = loader.dataset.df["filename"].tolist()
                assert len(filenames) == len(ens_per_image)
                row, img_rows = summarize(filenames, [p[:4] for p in ens_per_image],
                                          config, eval_ds, "cross_ensemble", -1)
                summary_rows.append(row)
                per_image_rows.extend(img_rows)
                for fold, pmi in zip(sorted(fold_models), per_model):
                    mrow, _ = summarize(filenames, pmi, config, eval_ds, "cross_model", fold)
                    summary_rows.append(mrow)
                print(f"config={config} eval={eval_ds} (cross, all {len(filenames)} images, "
                      f"{len(models)}-model prob ensemble): "
                      + " ".join(f"{m}={row[f'micro_{m}']:.4f}" for m in METRICS))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(per_image_rows).to_csv(out_dir / "per_image_metrics.csv", index=False)
    print(f"Wrote {out_dir / 'per_image_metrics.csv'} ({len(per_image_rows)} rows)")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote {out_dir / 'summary.csv'} ({len(summary)} rows)")

    # Bar summary: one row per (config, eval_ds, metric)
    bar_rows = []
    for config in args.configs:
        for eval_ds in args.eval_datasets:
            sub = summary[(summary["config"] == config) & (summary["eval_ds"] == eval_ds)]
            val = sub[sub["mode"] == "val_fold"]
            cross = sub[sub["mode"] == "cross_ensemble"]
            if len(val):
                mode, vals = "val_fold", val
            elif len(cross):
                mode, vals = "cross_ensemble", cross
            else:
                mode, vals = "missing", sub
            for m in METRICS:
                v = vals[f"micro_{m}"].to_numpy()
                bar_rows.append({"config": config, "eval_ds": eval_ds, "metric": m, "mode": mode,
                                 "mean": float(np.mean(v)) if len(v) else np.nan,
                                 "std": float(np.std(v)) if len(v) else np.nan,
                                 "n_points": len(v)})
    bar_summary = pd.DataFrame(bar_rows)
    bar_summary.to_csv(out_dir / "bar_summary.csv", index=False)
    print(f"Wrote {out_dir / 'bar_summary.csv'}")

    # Plot: 3 subplots (one per metric), 6 bars each; fold points only for in-dataset bars
    fig, axes = plt.subplots(1, len(METRICS), figsize=(8.5 * len(METRICS), 7), sharey=False)
    palette = {"2020": "#4C72B0", "2025": "#DD8452", "both": "#55A868"}
    for ax, m in zip(axes, METRICS):
        x, labels = 0, []
        for config in args.configs:
            for eval_ds in args.eval_datasets:
                bar = bar_summary[(bar_summary["config"] == config)
                                  & (bar_summary["eval_ds"] == eval_ds)
                                  & (bar_summary["metric"] == m)].iloc[0]
                mean, std, n_pts, mode = bar["mean"], bar["std"], int(bar["n_points"]), bar["mode"]
                hatch = "//" if mode == "cross_ensemble" else None
                ax.bar(x, mean, yerr=std if n_pts > 1 else 0, capsize=4, color=palette[config],
                       alpha=0.85, edgecolor="black", hatch=hatch)
                if n_pts > 1:
                    sub = summary[(summary["config"] == config) & (summary["eval_ds"] == eval_ds)
                                  & (summary["mode"] == "val_fold")]
                    vals = sub[f"micro_{m}"].to_numpy()
                    jitter = np.linspace(-0.18, 0.18, len(vals))
                    ax.scatter(np.full(len(vals), x) + jitter, vals, color="black", zorder=5, s=28)
                    for xi, v in zip(np.full(len(vals), x) + jitter, vals):
                        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 7),
                                    ha="center", fontsize=9)
                else:
                    ax.annotate(f"{mean:.3f}", (x, mean), textcoords="offset points", xytext=(0, 7),
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
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=palette[c], label=f"train {c}") for c in args.configs]
               + [Patch(facecolor="white", edgecolor="black", hatch="//",
                        label="cross-dataset: one bar = 5-model prob-ensemble on ALL data")],
               loc="upper right")
    fig.suptitle("Cropping (segmentation) evaluation — in-dataset: mean over 5 folds, points = fold-i val; "
                 "cross-dataset: single bar, 5-model probability ensemble on all data\n"
                 "(ensembling averages sigmoid probabilities, not logits — same as extract_inf_frames.py)")
    fig.tight_layout(rect=[0, 0, 0.9, 0.92])
    plot_path = out_dir / "segmentation_eval.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    print("\nBar summary (micro):")
    print(bar_summary.to_string(index=False))


if __name__ == "__main__":
    main()
