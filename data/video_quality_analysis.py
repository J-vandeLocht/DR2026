"""Correlate per-video extraction quality (informative-frame threshold/count/confidence, frame_quality.py metrics)
with wrong video-level predictions. Analysis only; reads the extraction txts and the multi-eval prediction CSVs."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score

# extract_inf_frames.py: start 0.5, relax by 0.05 down to 0.2, else top-N fallback
THRESHOLD_GRID = np.round(np.arange(0.5, 0.2 - 1e-9, -0.05), 2)
FEATURES = ["n_frames", "n_inf", "frac_inf", "mean_prob_inf", "median_prob_inf", "mean_prob_all",
            "n_segments", "longest_run", "sharp_mean", "light_mean"]
# Low values = worse quality for every feature used in a flag
FLAG_FEATURES = {"few_inf": "n_inf", "low_frac": "frac_inf", "low_conf": "mean_prob_inf"}
TARGETS = ["wrong", "off2", "missed_ref"]


def runs(mask):
    """Lengths of consecutive True runs."""
    m = np.concatenate([[0], mask.astype(int), [0]])
    d = np.diff(m)
    return np.where(d == -1)[0] - np.where(d == 1)[0]


def video_features(txt_dir, quality_dir):
    rows = []
    for f in sorted(Path(txt_dir).glob("*.txt")):
        d = pd.read_csv(f, sep="\t")
        inf = d["informative"].to_numpy(bool)
        p_inf = d.loc[inf, "prob"]
        min_p = p_inf.min()
        ok = THRESHOLD_GRID[THRESHOLD_GRID <= min_p + 1e-9]
        thr = float(ok[0]) if len(ok) else np.nan  # nan = top-N fallback
        r = runs(inf)
        row = dict(stem=f.stem, n_frames=len(d), n_inf=int(inf.sum()), frac_inf=inf.mean(), threshold_used=thr,
                   fallback=bool(np.isnan(thr)), relaxed=bool(thr < 0.5) if not np.isnan(thr) else False,
                   mean_prob_inf=p_inf.mean(), median_prob_inf=p_inf.median(), mean_prob_all=d["prob"].mean(),
                   n_segments=len(r), longest_run=int(r.max()) if len(r) else 0)
        q = Path(quality_dir) / f"CLEAN_{f.stem}.npz"
        if q.exists():
            q = np.load(q)
            row.update(sharp_mean=float(q["sharp"].mean()), light_mean=float(q["light"].mean()))
        rows.append(row)
    df = pd.DataFrame(rows)
    df["fallback_or_relaxed"] = df["fallback"] | df["relaxed"]
    for flag, feat in FLAG_FEATURES.items():
        df[flag] = df[feat] <= df[feat].quantile(0.10)
    df["any_flag"] = df[["fallback_or_relaxed", *FLAG_FEATURES]].any(axis=1)
    return df


def load_predictions(pred_dir, models):
    parts = []
    for f in sorted(Path(pred_dir).glob("predictions_split_*_*.csv")):
        split = int(f.name.split("_")[2])
        d = pd.read_csv(f)
        out = pd.DataFrame({"stem": d["video"].str.replace(r"\.mp4$", "", regex=True), "grade": d["grade"], "split": split})
        for m in models:
            out[m] = d[f"{m}_pred_grade_split_{split}"]
        parts.append(out)
    return pd.concat(parts, ignore_index=True)


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    valid = ~np.isnan(p)
    pv = p[valid]
    order = np.argsort(pv)
    adj = np.maximum.accumulate((len(pv) - np.arange(len(pv))) * pv[order])
    res = np.empty_like(pv)
    res[order] = np.minimum(adj, 1)
    out[valid] = res
    return out


def metrics(g, p):
    ref = g >= 2
    return dict(qwk=cohen_kappa_score(g, p, weights="quadratic"), acc=accuracy_score(g, p),
                rr=((p >= 2) & ref).sum() / max(ref.sum(), 1))


def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    feats = video_features(args.txt_dir, args.quality_dir)
    feats.to_csv(out / "video_features.csv", index=False)
    preds = load_predictions(args.pred_dir, args.models)
    missing = set(preds["stem"]) - set(feats["stem"])
    print(f"{len(preds)} val videos, {len(missing)} without extraction txt: {sorted(missing)}")
    print(f"{feats['sharp_mean'].isna().sum()} videos without frame_quality npz")
    df = preds.merge(feats, on="stem", how="left")

    flags = ["fallback_or_relaxed", *FLAG_FEATURES, "any_flag"]
    low = df[df["any_flag"]][["stem", "split", "grade", *flags, "n_inf", "frac_inf", "mean_prob_inf", "threshold_used"]]
    low.sort_values(["split", "stem"]).to_csv(out / "low_quality_videos.csv", index=False)
    print(f"Flagged among val videos: " + ", ".join(f"{f} {int(df[f].sum())}" for f in flags))

    stats, summary = [], []
    for m in args.models:
        d = df[df[m].notna()].copy()
        g, p = d["grade"].to_numpy(int), d[m].to_numpy(int)
        d["abs_err"] = np.abs(p - g)
        d["wrong"] = d["abs_err"] > 0
        d["off2"] = d["abs_err"] >= 2
        d["missed_ref"] = (g >= 2) & (p < 2)
        d["false_ref"] = (g < 2) & (p >= 2)
        rows = []
        for feat in FEATURES:
            x = d[feat]
            ok = x.notna()
            rho, prho = spearmanr(x[ok], d.loc[ok, "abs_err"])
            row = dict(model=m, kind="feature", name=feat, rho_abs_err=rho, p=prho)
            for t in TARGETS:
                y = d.loc[ok, t]
                sub = (y.index if t != "missed_ref" else d[ok & (d["grade"] >= 2)].index)
                yy, xx = d.loc[sub, t], x[sub]
                if yy.nunique() == 2:
                    # AUROC < 0.5: errors have LOWER feature values
                    row[f"auroc_{t}"] = roc_auc_score(yy, xx)
                    row[f"p_{t}"] = mannwhitneyu(xx[yy], xx[~yy]).pvalue
            rows.append(row)
        for flag in flags:
            f = d[flag].astype(bool)
            row = dict(model=m, kind="flag", name=flag, n_flagged=int(f.sum()))
            for t in TARGETS + ["false_ref"]:
                base = d if t not in ("missed_ref", "false_ref") else d[(d["grade"] >= 2) == (t == "missed_ref")]
                fb = base[flag].astype(bool)
                a, b = base.loc[fb, t].sum(), (~base.loc[fb, t]).sum()
                c, e = base.loc[~fb, t].sum(), (~base.loc[~fb, t]).sum()
                row[f"{t}_flagged"] = a / max(a + b, 1)
                row[f"{t}_unflagged"] = c / max(c + e, 1)
                if t != "false_ref":
                    orr, pf = fisher_exact([[a, b], [c, e]])
                    row[f"or_{t}"], row[f"p_{t}"] = orr, pf
            kept = d[~f]
            mk = metrics(kept["grade"].to_numpy(int), kept[m].to_numpy(int))
            row.update({f"{k}_kept": v for k, v in mk.items()})
            row["p"] = row["p_wrong"]
            rows.append(row)
        ma = metrics(g, p)
        rows.append(dict(model=m, kind="all", name="all videos", n_flagged=0, **{f"{k}_kept": v for k, v in ma.items()}))
        r = pd.DataFrame(rows)
        # Holm over every test of this model
        pcols = [c for c in r.columns if c == "p" or c.startswith("p_")]
        flat = r[pcols].to_numpy().ravel()
        r[[c + "_holm" for c in pcols]] = holm(flat).reshape(r[pcols].shape)
        stats.append(r)

        # Per-grade error rates (confounder check)
        pg = d.groupby(["grade", "any_flag"])["wrong"].agg(["mean", "size"]).unstack()
        pg.to_csv(out / f"per_grade_wrong_{m}.csv")

        # Plots: features by abs_err group, error rate per feature quintile
        d["err_group"] = np.minimum(d["abs_err"], 2).map({0: "0", 1: "1", 2: "≥2"})
        fig, axes = plt.subplots(2, len(FEATURES) // 2, figsize=(18, 7))
        for ax, feat in zip(axes.ravel(), FEATURES):
            groups = [d.loc[d["err_group"] == k, feat].dropna() for k in ["0", "1", "≥2"]]
            ax.boxplot(groups, tick_labels=["0", "1", "≥2"], showfliers=False)
            for i, gv in enumerate(groups):
                ax.scatter(np.random.normal(i + 1, 0.06, len(gv)), gv, s=5, alpha=0.4)
            if feat in ("n_frames", "n_inf", "longest_run"):
                ax.set_yscale("log")
            ax.set_title(feat, fontsize=9)
            ax.set_xlabel("|pred − grade|", fontsize=8)
        fig.suptitle(f"{m}: extraction features by error size (n={len(d)})")
        fig.tight_layout()
        fig.savefig(out / f"features_by_error_{m}.png", dpi=120)
        plt.close(fig)

        fig, axes = plt.subplots(2, len(FEATURES) // 2, figsize=(18, 7))
        for ax, feat in zip(axes.ravel(), FEATURES):
            q = pd.qcut(d[feat].rank(method="first"), 5, labels=False)
            rate = d.groupby(q)[["wrong", "off2"]].mean()
            ax.plot(rate.index + 1, rate["wrong"], "o-", label="wrong")
            ax.plot(rate.index + 1, rate["off2"], "s-", label="≥2 off")
            ax.set_ylim(0, 1)
            ax.set_title(feat, fontsize=9)
            ax.set_xlabel("quintile (1 = lowest)", fontsize=8)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{m}: error rate per feature quintile")
        fig.tight_layout()
        fig.savefig(out / f"error_by_quintile_{m}.png", dpi=120)
        plt.close(fig)

        best = r[r["kind"] == "feature"].assign(a=lambda x: (x["auroc_wrong"] - 0.5).abs()).nlargest(3, "a")
        summary.append(f"{m}: all QWK {ma['qwk']:.3f} RR {ma['rr']:.3f} | strongest features (AUROC wrong, Holm p): "
                       + "; ".join(f"{n} {a:.2f} ({ph:.2g})" for n, a, ph in
                                   zip(best["name"], best["auroc_wrong"], best["p_wrong_holm"])))
        for _, fr in r[r["kind"] == "flag"].iterrows():
            summary.append(f"   {fr['name']:<20} n={fr['n_flagged']:>3} wrong {fr['wrong_flagged']:.2f} vs {fr['wrong_unflagged']:.2f}"
                           f" (OR {fr['or_wrong']:.2f}, Holm p {fr['p_wrong_holm']:.2g}), ≥2 off {fr['off2_flagged']:.2f} vs "
                           f"{fr['off2_unflagged']:.2f}, missed {fr['missed_ref_flagged']:.2f} vs {fr['missed_ref_unflagged']:.2f}"
                           f" | kept QWK {fr['qwk_kept']:.3f} RR {fr['rr_kept']:.3f}")
    pd.concat(stats, ignore_index=True).to_csv(out / "stats.csv", index=False)
    print("\n".join(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--txt_dir", default="data/ensemble_results_paxos2025/txt_files")
    parser.add_argument("--quality_dir", default="data/ensemble_results_paxos2025/frame_quality")
    parser.add_argument("--pred_dir", default="data/multi_eval_paxos2025_best/artifact")
    parser.add_argument("--models", nargs="+", default=["DINO_FT", "CNX_mix", "Clf", "Clf32"])
    parser.add_argument("--output_dir", default="data/multi_eval_paxos2025_best/video_quality")
    main(parser.parse_args())
