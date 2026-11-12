"""Evaluate the cross-dataset informative-frame classifier experiment.

Mirrors cropping/evaluate_segmentation.py. Every trained classifier
(<models_dir>/cls_<config>_fold_<i>.pth, config in 2020/2025/both) is evaluated on both
datasets, using the frames cropped by the SAME config:
< img_base>/frames_cropped_paxos<eval_ds>_by_<config>/<label>/<filename>,
with frame lists and labels from
<split_dir>/split_info_5fold_paxos<eval_ds>_from_cropping.csv.

Two evaluation modes per (config, eval_ds) pair:

- in-dataset (config == eval_ds, or config == "both"): classifier cls_<config>_fold_<i>
  is evaluated on the fold-i *val* rows of the eval dataset only (the part it never saw
  during training) -> 5 points in the plot, bar = mean over the 5 folds.
- cross-dataset (config != eval_ds, config != "both"): the config never saw the eval
  dataset, so ALL of its frames are used, and the 5 fold classifiers are ensembled by
  averaging sigmoid PROBABILITIES (threshold 0.5) -- the same ensembling as
  informative_frames/extract_inf_frames.py -> ONE bar, hatched.

Metrics per (config, eval_ds, mode, fold) group: micro accuracy / F1 / precision / recall
at threshold 0.5 over all evaluated images, plus ROC-AUC and PR-AUC over the probabilities.

Outputs in --out_dir:
  per_image_metrics.csv  one row per (config, eval_ds, mode, fold, image): label, prob, pred
  summary.csv            one row per (config, eval_ds, mode, fold)
  bar_summary.csv        one row per bar of the plot
  classification_eval.png  3 subplots (ROC-AUC, PR-AUC, F1), 6 bars each
                         [train 2020/2025/both x eval 2020/2025]; in-dataset bars show the
                         5 fold values as points, cross-dataset bars are single hatched bars
                         (5-model ensemble on all data)

Run from the repo root:
    python -u -m informative_frames.evaluate_classification
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
from torchvision import models
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from informative_frames.classify_frames import (FrameDataset, cropped_dir,
                                                split_csv, get_val_transforms)

CONFIGS = ["2020", "2025", "both"]
METRICS = ["roc_auc", "pr_auc", "f1"]
METRIC_TITLES = {"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "f1": "F1 @ 0.5"}


def build_loader(eval_ds, config, img_base, split_dir, transform, batch_size, num_workers,
                 fold=None, split=None):
    """One row per frame: fold+split given -> those rows; otherwise the val rows
    (each frame appears in exactly one val fold, so this covers all frames once)."""
    df = pd.read_csv(split_csv(split_dir, eval_ds))
    if fold is not None:
        df = df[(df["fold"] == fold) & (df["split"] == split)]
    else:
        df = df[df["split"] == "val"]
    folder = cropped_dir(img_base, eval_ds, config)
    samples = []
    for row in df.itertuples():
        path = folder / str(int(row.label)) / row.filename
        if path.exists():
            samples.append((path, int(row.label)))
    filenames = [p.name for p, _ in samples]
    ds = FrameDataset(samples, transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return loader, filenames


def load_model(path, device):
    model = models.efficientnet_b4(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def predict(model, loader, device):
    """Sigmoid prob per image, aligned with loader order (single model)."""
    probs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            probs.extend(torch.sigmoid(model(imgs)).view(-1).cpu().numpy().tolist())
    return probs


def predict_ensemble(models_, loader, device):
    """Mean of sigmoid PROBABILITIES over the fold models (not logits -- same as
    extract_inf_frames.py). Returns (ensemble_probs, per_model_probs)."""
    per_model = [[] for _ in models_]
    ens_probs = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            probs = torch.stack([torch.sigmoid(m(imgs)).view(-1) for m in models_])  # [M, B]
            ens_probs.extend(probs.mean(dim=0).cpu().numpy().tolist())
            for mi in range(len(models_)):
                per_model[mi].extend(probs[mi].cpu().numpy().tolist())
    return ens_probs, per_model


def group_metrics(y_true, probs, eps=1e-7):
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    pred = probs > 0.5
    tp = int(((pred) & (y_true == 1)).sum())
    fp = int(((pred) & (y_true == 0)).sum())
    fn = int(((~pred) & (y_true == 1)).sum())
    tn = int(((~pred) & (y_true == 0)).sum())
    row = {
        "n_images": len(y_true), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "acc": (tp + tn) / (tp + fp + fn + tn + eps),
        "f1": 2 * tp / (2 * tp + fp + fn + eps),
        "precision": tp / (tp + fp + eps),
        "recall": tp / (tp + fn + eps),
        "roc_auc": roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else np.nan,
    }
    p_curve, r_curve, _ = precision_recall_curve(y_true, probs)
    row["pr_auc"] = auc(r_curve, p_curve) if len(np.unique(y_true)) > 1 else np.nan
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models_dir", type=str, default="informative_frames/models_xdomain")
    parser.add_argument("--img_base", type=str, default="data/informative_frames")
    parser.add_argument("--split_dir", type=str, default="informative_frames")
    parser.add_argument("--out_dir", type=str, default="informative_frames/eval_xdomain")
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--configs", nargs="+", default=CONFIGS)
    parser.add_argument("--eval_datasets", nargs="+", default=["2020", "2025"], choices=["2020", "2025"])
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = get_val_transforms(args.img_size)

    per_image_rows, summary_rows = [], []

    for config in args.configs:
        fold_models = {}
        for fold in range(args.n_splits):
            model_path = Path(args.models_dir) / f"cls_{config}_fold_{fold}.pth"
            if not model_path.exists():
                print(f"WARNING: missing model {model_path} -- fold {fold} skipped for config={config}")
                continue
            fold_models[fold] = load_model(model_path, device)
        if not fold_models:
            print(f"WARNING: no models for config={config} -- skipped")
            continue

        for eval_ds in args.eval_datasets:
            in_dataset = (config == "both") or (config == eval_ds)
            if in_dataset:
                for fold, model in fold_models.items():
                    loader, filenames = build_loader(eval_ds, config, args.img_base, args.split_dir,
                                                     transform, args.batch_size, args.num_workers,
                                                     fold=fold, split="val")
                    probs = predict(model, loader, device)
                    y_true = [lbl for _, lbl in loader.dataset.samples]
                    assert len(filenames) == len(probs) == len(y_true)
                    m = group_metrics(y_true, probs)
                    m.update({"config": config, "eval_ds": eval_ds, "mode": "val_fold", "fold": fold})
                    summary_rows.append(m)
                    for fn, y, p in zip(filenames, y_true, probs):
                        per_image_rows.append({"config": config, "eval_ds": eval_ds, "mode": "val_fold",
                                               "fold": fold, "filename": fn, "label": y, "prob": p,
                                               "pred": int(p > 0.5)})
                    print(f"config={config} fold={fold} eval={eval_ds} (val fold): "
                          + " ".join(f"{k}={m[k]:.4f}" for k in METRICS))
            else:
                loader, filenames = build_loader(eval_ds, config, args.img_base, args.split_dir,
                                                 transform, args.batch_size, args.num_workers)
                models_ = [fold_models[f] for f in sorted(fold_models)]
                ens_probs, per_model = predict_ensemble(models_, loader, device)
                y_true = [lbl for _, lbl in loader.dataset.samples]
                assert len(filenames) == len(ens_probs) == len(y_true)
                m = group_metrics(y_true, ens_probs)
                m.update({"config": config, "eval_ds": eval_ds, "mode": "cross_ensemble", "fold": -1})
                summary_rows.append(m)
                for fn, y, p in zip(filenames, y_true, ens_probs):
                    per_image_rows.append({"config": config, "eval_ds": eval_ds, "mode": "cross_ensemble",
                                           "fold": -1, "filename": fn, "label": y, "prob": p,
                                           "pred": int(p > 0.5)})
                for fold, probs in zip(sorted(fold_models), per_model):
                    mm = group_metrics(y_true, probs)
                    mm.update({"config": config, "eval_ds": eval_ds, "mode": "cross_model", "fold": fold})
                    summary_rows.append(mm)
                print(f"config={config} eval={eval_ds} (cross, all {len(filenames)} images, "
                      f"{len(models_)}-model prob ensemble): "
                      + " ".join(f"{k}={m[k]:.4f}" for k in METRICS))
        if device.type == "cuda":
            torch.cuda.empty_cache()

    pd.DataFrame(per_image_rows).to_csv(out_dir / "per_image_metrics.csv", index=False)
    print(f"Wrote {out_dir / 'per_image_metrics.csv'} ({len(per_image_rows)} rows)")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "summary.csv", index=False)
    print(f"Wrote {out_dir / 'summary.csv'} ({len(summary)} rows)")

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
                v = vals[m].to_numpy(dtype=float)
                bar_rows.append({"config": config, "eval_ds": eval_ds, "metric": m, "mode": mode,
                                 "mean": float(np.nanmean(v)) if len(v) else np.nan,
                                 "std": float(np.nanstd(v)) if len(v) else np.nan,
                                 "n_points": len(v)})
    bar_summary = pd.DataFrame(bar_rows)
    bar_summary.to_csv(out_dir / "bar_summary.csv", index=False)
    print(f"Wrote {out_dir / 'bar_summary.csv'}")

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
                    vals = sub[m].to_numpy(dtype=float)
                    jitter = np.linspace(-0.18, 0.18, len(vals))
                    ax.scatter(np.full(len(vals), x) + jitter, vals, color="black", zorder=5, s=28)
                    for xi, v in zip(np.full(len(vals), x) + jitter, vals):
                        ax.annotate(f"{v:.3f}", (xi, v), textcoords="offset points", xytext=(0, 7),
                                    ha="center", fontsize=9)
                else:
                    ax.annotate(f"{mean:.3f}", (x, mean), textcoords="offset points", xytext=(0, 7),
                                ha="center", fontsize=9)
                labels.append(f"{config}\u2192{eval_ds}")
                x += 1
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=11)
        ax.set_title(METRIC_TITLES[m])
        ax.set_ylim(0, 1.12)
        ax.tick_params(axis="y", labelsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel(METRIC_TITLES[m])
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=palette[c], label=f"train {c}") for c in args.configs]
               + [Patch(facecolor="white", edgecolor="black", hatch="//",
                        label="cross-dataset: one bar = 5-model prob-ensemble on ALL data")],
               loc="upper right")
    fig.suptitle("Informative-frame selection (classification) evaluation \u2014 in-dataset: mean over 5 folds, "
                 "points = fold-i val; cross-dataset: single bar, 5-model probability ensemble on all data\n"
                 "(ensembling averages sigmoid probabilities, not logits \u2014 same as extract_inf_frames.py; "
                 "eval frames cropped by the same config: frames_cropped_paxos<eval>_by_<config>)")
    fig.tight_layout(rect=[0, 0, 0.9, 0.92])
    plot_path = out_dir / "classification_eval.png"
    fig.savefig(plot_path, dpi=170)
    plt.close(fig)
    print(f"Wrote {plot_path}")

    print("\nBar summary:")
    print(bar_summary.to_string(index=False))


if __name__ == "__main__":
    main()
