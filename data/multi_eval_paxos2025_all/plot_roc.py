"""Referable-DR ROC curves of all models in one figure, pooled over all splits (one curve per model), from the binary
prediction CSVs of make_binary.py. Legend: pooled AUC and per-split AUC mean ± SE; marker = the fixed 0.5 threshold.
Colour = model family (classifier / DINO / ConvNeXt), line style = variant, so no pair relies on colour alone.
Usage: python plot_roc.py --input_dir ../multi_eval_X/binary"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from sklearn.metrics import roc_auc_score, roc_curve

FIXED_THRESH = 0.5  # as in analyse_runs.py
# Model-name prefix -> family colour (CVD-safe triple); unknown families fall back to violet
FAMILY_COLORS = {"Clf": "#52514e", "DINO": "#2a78d6", "CNX": "#eb6834"}
FALLBACK_COLOR = "#6250d6"
# Per variant within a family: line style + operating-point marker (solid/circle, dashed/square, dotted/triangle)
LINESTYLES = ["-", (0, (5, 2)), (0, (1.5, 1.5))]
MARKERS = ["o", "s", "^"]


def load_predictions(input_dir):
    """-> {model: DataFrame(split, label, p)} over all prediction CSVs, models in CSV column order."""
    per_model = {}
    for f in sorted(glob.glob(os.path.join(input_dir, "predictions_split_*.csv"))):
        split = int(re.search(r"split_(\d+)", os.path.basename(f)).group(1))
        df = pd.read_csv(f)
        for col in df.columns:
            m = re.match(r"^(.+)_split_\d+$", col)
            if not m:
                continue
            ok = df[col].notna()  # e.g. Clf on videos without hand-picked frames
            part = pd.DataFrame(dict(split=split, label=df.label[ok].astype(int).values, p=df[col][ok].values))
            per_model.setdefault(m.group(1), []).append(part)
    return {name: pd.concat(parts, ignore_index=True) for name, parts in per_model.items()}


def model_styles(models):
    """Colour by family (longest matching prefix), line style by order within the family."""
    styles, seen = {}, {}
    for name in models:
        family = max((f for f in FAMILY_COLORS if name.startswith(f)), key=len, default=None)
        idx = seen.get(family, 0)
        seen[family] = idx + 1
        styles[name] = dict(color=FAMILY_COLORS.get(family, FALLBACK_COLOR), linestyle=LINESTYLES[idx % len(LINESTYLES)],
                            marker=MARKERS[idx % len(MARKERS)])
    return styles


def main(args):
    preds = load_predictions(args.input_dir)
    if not preds:
        raise SystemExit(f"No binary prediction CSVs in {args.input_dir} (run make_binary.py first).")
    styles = model_styles(preds)
    out_dir = args.output_dir or os.path.join(args.input_dir, "split_analysis_results", "binary")
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams.update({"font.size": 11, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b",
                         "xtick.color": "#52514e", "ytick.color": "#52514e"})
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.6), gridspec_kw=dict(wspace=0.22))
    rows, handles, labels = [], [], []
    for name, d in preds.items():
        fpr, tpr, thr = roc_curve(d.label, d.p)
        auc_pooled = roc_auc_score(d.label, d.p)
        split_aucs = d.groupby("split").apply(lambda g: roc_auc_score(g.label, g.p))
        se = split_aucs.std() / np.sqrt(len(split_aucs))
        # operating point of the fixed threshold
        pred = d.p >= FIXED_THRESH
        pos, neg = d.label == 1, d.label == 0
        sens, fpr_op = (pred & pos).sum() / pos.sum(), (pred & neg).sum() / neg.sum()
        rows.append(dict(model=name, n=len(d), auc_pooled=auc_pooled, auc_split_mean=split_aucs.mean(), auc_split_se=se,
                         sens_at_thresh=sens, spec_at_thresh=1 - fpr_op))

        st = styles[name]
        marker_kw = dict(marker=st["marker"], ms=8, color=st["color"], markeredgecolor="white", markeredgewidth=1.5)
        for ax in axes:
            ax.plot(fpr, tpr, lw=2, color=st["color"], linestyle=st["linestyle"], drawstyle="steps-post")
            ax.plot(fpr_op, sens, linestyle="none", zorder=5, **marker_kw)
        # line sample and marker side by side (a marker on the line would split a solid line into "dashes")
        handles.append((plt.Line2D([], [], lw=2, color=st["color"], linestyle=st["linestyle"]),
                        plt.Line2D([], [], linestyle="none", **marker_kw)))
        labels.append(f"{name}: AUC {auc_pooled:.3f}  (per split {split_aucs.mean():.3f} ± {se:.3f})")

    for ax, (xlim, ylim), title in zip(axes, [((0, 1), (0, 1)), ((0, 0.4), (0.6, 1))],
                                       ["All thresholds", "Zoom: top-left corner"]):
        ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1, linestyle=(0, (2, 2)), zorder=0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal")
        ax.grid(color="#e9e8e4", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xlabel("False positive rate (1 − specificity)")
        ax.set_ylabel("True positive rate (sensitivity)")
        ax.set_title(title, fontsize=12, fontweight="bold", color="#0b0b0b")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 0.0),
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.6)}, handlelength=5,
               title=f"Pooled AUC (per-split mean ± SE); marker = operating point at P(referable) ≥ {FIXED_THRESH}",
               title_fontsize=10)
    n_splits = preds[next(iter(preds))].split.nunique()
    fig.suptitle(f"Referable DR (grade ≥ 2): ROC curves pooled over {n_splits} splits", fontsize=15, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.9, bottom=0.24)

    png = os.path.join(out_dir, "roc_curves.png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(out_dir, "roc_auc_summary.csv"), index=False)
    print(table.round(4).to_string(index=False))
    print(f"Saved {png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help="Folder with the binary prediction CSVs (make_binary.py output, e.g. <eval_dir>/binary).")
    parser.add_argument('--output_dir', type=str, default=None,
                        help="Default: <input_dir>/split_analysis_results/binary.")
    main(parser.parse_args())
