"""Cost/performance sweep over the pre-processing configs.

For every config in the grid (DEFAULT_GRID or --grid JSON): train the 5 folds
of both stages (unless --no-train and checkpoints already exist), evaluate on
the leakage-free val folds, measure cost (params / FLOPs / latency per frame),
and append one row per config+stage to <out_dir>/sweep_results.csv. At the end
the Pareto-optimal configs (best performance per cost) are written to
<out_dir>/pareto.csv and printed.

    python -u -m monai_preproc.sweep --out_dir results/monai_sweep [--grid my_grid.json] \
        [--epochs 40] [--no-train] [--folds 0 1 2 3 4]

Performance metric: seg -> mean micro IoU over folds; cls -> mean PR-AUC.
Cost metric: seg/cls latency_ms per frame at the config's input size
(falls back to params if latency is unavailable), plus params and FLOPs.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .configs import load_grid
from .train import train_seg, train_cls
from .evaluate import evaluate_seg, evaluate_cls


def pareto_front(rows, perf_key, cost_key, maximize=True):
    """Indices of rows not dominated in (cost, perf): a row is dominated if another
    row is at least as good in perf and strictly better in cost (or equal perf and
    lower cost)."""
    idx = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            better_perf = (b[perf_key] > a[perf_key]) if maximize else (b[perf_key] < a[perf_key])
            same_perf = b[perf_key] == a[perf_key]
            lower_cost = b[cost_key] < a[cost_key]
            if (better_perf and lower_cost) or (same_perf and lower_cost) or (better_perf and b[cost_key] == a[cost_key]):
                dominated = True
                break
        if not dominated:
            idx.append(i)
    return idx


def pareto_per_stage(df):
    """Pareto-optimal rows per stage (cost = latency_ms, falling back to params).

    Selection happens on each stage's subset, but the returned rows must keep
    their original df index: sub.index[...] maps Pareto positions back to the
    full frame, so df.loc[...] picks the right rows (reset_index here would
    misalign subset positions with full-frame rows).
    """
    idx = []
    for stage in df["stage"].unique():
        sub = df[df["stage"] == stage]
        records = [{"perf": r["perf"], "cost": c}
                   for r, c in zip(sub.to_dict("records"), sub["latency_ms"].fillna(sub["params"]))]
        idx.extend(sub.index[pareto_front(records, "perf", "cost")].tolist())
    return df.loc[idx].sort_values(["stage", "latency_ms"])


def run_sweep(grid, out_dir, folds, epochs_seg=None, epochs_cls=None, train=True,
              ann_csv="cropping/annotations_cropping_2025.csv",
              splits_csv="cropping/split_cropping_5fold_paxos2025_from_q5v3.csv",
              img_dir="cropping/frames_to_annotate_larger",
              img_base="data/informative_frames", split_dir="informative_frames",
              num_workers=4, device=None, log=print):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "sweep_results.csv"
    rows = []
    for name, cfg in grid.items():
        log(f"=== config {name} (seg: {cfg.seg_arch} @ {cfg.seg_img_size}, cls: {cfg.cls_backbone} @ {cfg.cls_img_size}) ===")
        seg_models_dir = out_dir / "models" / name / "seg"
        cls_models_dir = out_dir / "models" / name / "cls"
        if train:
            for fold in folds:
                train_seg(cfg, fold, ann_csv, splits_csv, img_dir, seg_models_dir,
                          epochs=epochs_seg, num_workers=num_workers, device=device, log=log)
                train_cls(cfg, fold, img_base, split_dir, cls_models_dir,
                          epochs=epochs_cls, num_workers=num_workers, device=device, log=log)
        try:
            seg_summary = evaluate_seg(cfg, seg_models_dir, ann_csv, splits_csv, img_dir,
                                       out_dir / "eval" / name / "seg", n_splits=len(folds),
                                       num_workers=num_workers, device=device, log=log)
            seg_row = {"config": name, "stage": "seg",
                       "perf": float(seg_summary["micro_iou"].mean()),
                       "params": int(seg_summary["params"].iloc[0]),
                       "flops": seg_summary["flops"].iloc[0],
                       "latency_ms": float(seg_summary["latency_ms"].iloc[0]),
                       "img_size": int(seg_summary["seg_img_size"].iloc[0]),
                       "arch": cfg.seg_arch}
            rows.append(seg_row)
        except (RuntimeError, FileNotFoundError) as e:
            log(f"WARNING: seg evaluation failed for {name}: {e}")
        try:
            cls_summary = evaluate_cls(cfg, cls_models_dir, img_base, split_dir,
                                       out_dir / "eval" / name / "cls", n_splits=len(folds),
                                       num_workers=num_workers, device=device, log=log)
            cls_row = {"config": name, "stage": "cls",
                       "perf": float(cls_summary["pr_auc"].mean()),
                       "params": int(cls_summary["params"].iloc[0]),
                       "flops": cls_summary["flops"].iloc[0],
                       "latency_ms": float(cls_summary["latency_ms"].iloc[0]),
                       "img_size": int(cls_summary["cls_img_size"].iloc[0]),
                       "arch": cfg.cls_backbone}
            rows.append(cls_row)
        except (RuntimeError, FileNotFoundError) as e:
            log(f"WARNING: cls evaluation failed for {name}: {e}")
        pd.DataFrame(rows).to_csv(results_path, index=False)
        log(f"wrote {results_path} ({len(rows)} rows so far)")
    df = pd.DataFrame(rows)
    if not df.empty:
        pareto = pareto_per_stage(df)
        pareto.to_csv(out_dir / "pareto.csv", index=False)
        log("Pareto-optimal configs (per stage):")
        log(pareto.to_string(index=False))
    return df


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", type=str, default="results/monai_sweep")
    parser.add_argument("--grid", type=str, default=None, help="JSON file overriding/extending the default grid.")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated config names to run.")
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs_seg", type=int, default=None)
    parser.add_argument("--epochs_cls", type=int, default=None)
    parser.add_argument("--no-train", action="store_true", help="Only evaluate existing checkpoints.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ann_csv", type=str, default="cropping/annotations_cropping_2025.csv")
    parser.add_argument("--splits_csv", type=str, default="cropping/split_cropping_5fold_paxos2025_from_q5v3.csv")
    parser.add_argument("--img_dir", type=str, default="cropping/frames_to_annotate_larger")
    parser.add_argument("--img_base", type=str, default="data/informative_frames")
    parser.add_argument("--split_dir", type=str, default="informative_frames")
    args = parser.parse_args(argv)

    grid = load_grid(args.grid)
    if args.only:
        names = [n.strip() for n in args.only.split(",")]
        grid = {n: grid[n] for n in names if n in grid}
        missing = [n for n in names if n not in grid]
        if missing:
            raise SystemExit(f"unknown config(s) {missing}; available: {sorted(grid)}")
    return run_sweep(grid, args.out_dir, args.folds, epochs_seg=args.epochs_seg, epochs_cls=args.epochs_cls,
                     train=not args.no_train, ann_csv=args.ann_csv, splits_csv=args.splits_csv,
                     img_dir=args.img_dir, img_base=args.img_base, split_dir=args.split_dir,
                     num_workers=args.num_workers)


if __name__ == "__main__":
    main()
