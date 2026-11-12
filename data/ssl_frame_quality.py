"""Quality filter for the SSL frames (data/extract_ssl_frames.py: 512x512, border-cropped): scores every frame with
data/frame_quality.py's sharp (green DoG) and light (Mertens well-exposedness) metrics, takes the --percentile-th
percentile of each over --n_ref random frames as thresholds and keeps frames at or above both. With several --frame_dirs
the thresholds are computed per dir (cohort; Paxos2020 frames come from higher-res clips and score much sharper), with
--pooled over all frames. Scores are cached per frame dir in <frame_dir>/../ssl_frame_quality/scores.csv. Writes <out_dir>/{thresholds_p<pct>.json,
keep_p<pct>.txt}; keep paths are relative to the frame dirs' common parent (one dir: the frame dir itself), which is the
--frame_dir of python -m dino.ssl_lora --frame_list <keep_p<pct>.txt>. CPU only."""
import argparse
import glob
import json
import os
import multiprocessing as mp

import cv2
import numpy as np
import pandas as pd

from data.frame_quality import ERODE_PX, SIZE, frame_metrics


def score(path):
    cv2.setNumThreads(1)
    img = cv2.imread(path)  # BGR, as frame_metrics expects
    assert img.shape[:2] == (SIZE, SIZE), f"{path}: {img.shape}, expected {SIZE}x{SIZE}"  # no resampling
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE_PX + 1, 2 * ERODE_PX + 1))
    sharp, light, mean_l, frac_dark, frac_glare = frame_metrics(img, kernel)
    return path, sharp, light, mean_l, frac_dark, frac_glare


def load_scores(frame_dir, workers):
    """Scores of all frames in frame_dir (paths relative to it), cached next to it."""
    cache = os.path.join(os.path.dirname(os.path.normpath(frame_dir)), "ssl_frame_quality", "scores.csv")
    if os.path.exists(cache):
        return pd.read_csv(cache)
    paths = sorted(glob.glob(os.path.join(frame_dir, "*", "*.jpg")))
    # spawn: fresh workers, no inherited OpenCV/OpenMP thread pools (fork + thread-count changes can deadlock)
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = pool.map(score, paths, chunksize=64)
    df = pd.DataFrame(rows, columns=["path", "sharp", "light", "mean_L", "frac_dark", "frac_glare"])
    df["path"] = df["path"].map(lambda p: os.path.relpath(p, frame_dir))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    df.to_csv(cache, index=False)
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_dirs", nargs="+", default=["data/ensemble_results_paxos2025/ssl_frames"])
    parser.add_argument("--out_dir", default="data/ensemble_results_paxos2025/ssl_frame_quality")
    parser.add_argument("--n_ref", type=int, default=1000)
    parser.add_argument("--percentile", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pooled", action="store_true", help="One threshold pair over all frame dirs, not one per dir.")
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dirs = [os.path.normpath(d) for d in args.frame_dirs]
    root = dirs[0] if len(dirs) == 1 else os.path.commonpath(dirs)
    dfs = []
    for d in dirs:
        df = load_scores(d, args.workers)
        df["source"] = os.path.relpath(d, os.path.dirname(root) if len(dirs) == 1 else root)
        df["path"] = df["path"].map(lambda p, d=d: os.path.relpath(os.path.join(d, p), root))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    groups = {"pooled": df} if args.pooled else dict(iter(df.groupby("source", sort=False)))
    thr = {g: {m: float(np.percentile(gdf.sample(n=args.n_ref, random_state=args.seed)[m], args.percentile))
               for m in ("sharp", "light")} for g, gdf in groups.items()}
    row_thr = {m: df["source"].map(lambda s, m=m: thr["pooled" if args.pooled else s][m]) for m in ("sharp", "light")}
    low_sharp, low_light = df["sharp"] < row_thr["sharp"], df["light"] < row_thr["light"]
    keep = df[~low_sharp & ~low_light]
    tag = f"p{args.percentile:g}"
    with open(os.path.join(args.out_dir, f"keep_{tag}.txt"), "w") as f:
        f.write("\n".join(keep["path"]) + "\n")

    def stats(sel):
        clip = df["path"].str.rsplit("/", n=1).str[0]
        return dict(n_frames=int(sel.sum()), n_low_sharp=int((low_sharp & sel).sum()),
                    n_low_light=int((low_light & sel).sum()), n_low_both=int((low_sharp & low_light & sel).sum()),
                    n_kept=int((~low_sharp & ~low_light & sel).sum()), n_clips=int(clip[sel].nunique()),
                    n_clips_kept=int(clip[sel & ~low_sharp & ~low_light].nunique()))
    thr = next(iter(thr.values())) if len(thr) == 1 else thr  # pooled or single dir: one pair (old format)
    summary = dict(thresholds=thr, pooled=args.pooled, percentile=args.percentile, n_ref=args.n_ref, seed=args.seed, root=root,
                   **stats(df["path"].notna()), per_source={s: stats(df["source"] == s) for s in df["source"].unique()})
    json.dump(summary, open(os.path.join(args.out_dir, f"thresholds_{tag}.json"), "w"), indent=2)
    print(json.dumps(summary, indent=2))
