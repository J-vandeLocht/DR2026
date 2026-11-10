"""5-class prediction CSVs (dino/vis_video_results.py) -> referable-DR (grade >= 2) CSVs for analyse_runs.py, with
P(referable) = P(2) + P(3) + P(4), plus fixed-decision metrics (referred = argmax grade >= 2) per split and pooled.
Run binary analysis afterwards: python analyse_runs.py --input_dir <output_dir> --output_dir <output_dir>/split_analysis_results"""
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score


def fixed_decision_metrics(g):
    y, yh = g.y.values, g.yh.values
    tp, tn, fp, fn = (y & yh).sum(), (~y & ~yh).sum(), (~y & yh).sum(), (y & ~yh).sum()
    sens, spec, ppv = tp / (tp + fn), tn / (tn + fp), tp / max(tp + fp, 1)
    return pd.Series(dict(n=len(y), sens=sens, spec=spec, ppv=ppv, npv=tn / max(tn + fn, 1), acc=(tp + tn) / len(y),
                          f1=2 * ppv * sens / (ppv + sens), bal_acc=(sens + spec) / 2, roc_auc=roc_auc_score(y, g.p),
                          missed=fn, false_ref=fp))


def main(args):
    src = Path(args.input_dir)
    out = Path(args.output_dir) if args.output_dir else src / "binary"
    out.mkdir(parents=True, exist_ok=True)

    models, pooled = [], []
    for f in sorted(src.glob("predictions_split_*.csv")):
        split = int(re.search(r"split_(\d+)", f.name).group(1))
        df = pd.read_csv(f)
        binary = df[["video", "label", "grade"]].copy()
        for col in df.columns:
            m = re.match(r"^(.+)_pred_grade_split_\d+$", col)
            if not m:
                continue
            name = m.group(1)
            if name not in models:
                models.append(name)
            # min_count: NaN (no hand-picked frames for this video) stays NaN and is dropped from the metrics
            p = df[[f"{name}_prob_{k}" for k in (2, 3, 4)]].sum(axis=1, min_count=3)
            binary[f"{name}_split_{split}"] = p
            ok = p.notna()
            pooled.append(pd.DataFrame(dict(model=name, split=split, y=df.grade.values[ok] >= 2,
                                            yh=df[col].values[ok] >= 2, p=p[ok].values)))
        binary.to_csv(out / f.name, index=False)

    a = pd.concat(pooled)
    per_split = a.groupby(["split", "model"]).apply(fixed_decision_metrics).reset_index()
    total = a.groupby("model").apply(fixed_decision_metrics).reindex(models)
    res_dir = out / "split_analysis_results" / "binary"
    res_dir.mkdir(parents=True, exist_ok=True)
    per_split.to_csv(res_dir / "fixed_decision_metrics.csv", index=False)
    total.to_csv(res_dir / "fixed_decision_pooled.csv")
    print(total.round(3).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help="Folder with the 5-class prediction CSVs.")
    parser.add_argument('--output_dir', type=str, default=None, help="Default: <input_dir>/binary.")
    main(parser.parse_args())
