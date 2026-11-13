"""Build the per-dataset 5-fold split files for the informative-frame annotations from the cropping splits.

The frames sampled for the informative-frame annotation (frames_raw_extract = Paxos2025,
frames_raw_extract_paxos2020 = Paxos2020) were chosen from the same videos that were
annotated for cropping, so the video-level folds of cropping/splits_<year>.json (subject-grouped,
made by cropping/make_splits_simple.py) can be reused unchanged: a frame's fold is the fold in
which its video sits in the *val* split of the cropping split; in that fold the frame is 'val',
in the other four folds 'train'. This gives exactly the train/val semantics used by
cropping/evaluate_segmentation.py, so the same evaluation logic applies to the informative-frame
classifier: same-dataset -> fold-i model on the fold-i val rows (the part it never saw in
training); other dataset -> ensemble of all 5 fold models on all rows of that dataset.

One output CSV per dataset, same format as the split_info_5fold_<timestamp>.csv files written by
informative_frames/classify_frames.py (one row per frame per fold):

    filename,video_id,label,fold,split

Inputs per dataset:
    2025: cropping/splits_2025.json + a label CSV with columns filename,video_id,label
          (default informative_frames/split_info_full.csv)
    2020: cropping/splits_2020.json + a label CSV with columns filename,label
          (default informative_frames/manual_labels_binary_paxos2020.csv;
           video_id is derived from the filename by stripping the trailing _f<idx>)

Outputs in --out_dir:
    split_info_5fold_paxos<year>_<suffix>.csv   (default suffix: from_cropping)

Stdlib only (csv/json), so it runs on a fresh checkout without the ML stack.

Run from the repo root:
    python -u -m informative_frames.make_splits_from_cropping
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

FRAME_RE = re.compile(r"^(?P<stem>.*)_f(?P<idx>\d+)\.(?P<ext>png|jpe?g)$", re.IGNORECASE)
CROP_FRAME_RE = re.compile(r"^(?P<stem>.*)_frame_(?P<idx>\d+)\.(?P<ext>png|jpe?g)$", re.IGNORECASE)


def video_stem(filename):
    m = FRAME_RE.match(filename)
    return m.group("stem") if m else Path(filename).stem


def crop_stem(filename):
    m = CROP_FRAME_RE.match(filename)
    return m.group("stem") if m else Path(filename).stem


def load_stem_folds(splits_path, n_splits):
    """Map video stem -> fold whose val split contains it (from a cropping splits json).

    Used to pick, per video, the fold model that was NOT trained on that video."""
    with open(splits_path) as f:
        splits = json.load(f)
    stem_fold = {}
    for fold in range(n_splits):
        for fn in splits[f"fold_{fold}"]["val"]:
            s = crop_stem(fn)
            assert s not in stem_fold, f"{splits_path}: video {s} in the val split of multiple folds"
            stem_fold[s] = fold
    return stem_fold


def load_cropping_folds(splits_path, n_splits):
    with open(splits_path) as f:
        splits = json.load(f)
    all_stems = set()
    for fold in range(n_splits):
        for fn in splits[f"fold_{fold}"]["train"] + splits[f"fold_{fold}"]["val"]:
            all_stems.add(crop_stem(fn))
    return load_stem_folds(splits_path, n_splits), all_stems


def load_labels(label_csv):
    with open(label_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        fn = r["filename"]
        vid = r.get("video_id") or video_stem(fn)
        out.append({"filename": fn, "video_id": vid, "label": r["label"]})
    return out


def build_split(year, cropping_dir, label_csv, n_splits):
    stem_fold, all_stems = load_cropping_folds(Path(cropping_dir) / f"splits_{year}.json", n_splits)
    labels = load_labels(label_csv)

    missing = sorted({r["video_id"] for r in labels if r["video_id"] not in all_stems})
    assert not missing, (
        f"{label_csv}: {len(missing)} videos not present in cropping/splits_{year}.json, "
        f"e.g. {missing[:5]} -- the cropping split cannot be reused for these frames"
    )

    rows = []
    for r in labels:
        fold = stem_fold[r["video_id"]]
        for f in range(n_splits):
            rows.append({
                "filename": r["filename"],
                "video_id": r["video_id"],
                "label": r["label"],
                "fold": f,
                "split": "val" if f == fold else "train",
            })

    val_per_fold = {f: [r for r in rows if r["fold"] == f and r["split"] == "val"] for f in range(n_splits)}
    val_all = [r["filename"] for f in range(n_splits) for r in val_per_fold[f]]
    assert len(val_all) == len(set(val_all)) == len(labels), \
        f"{label_csv}: val rows across folds must cover each frame exactly once"
    for f in range(n_splits):
        train_stems = {r["video_id"] for r in rows if r["fold"] == f and r["split"] == "train"}
        val_stems = {r["video_id"] for r in val_per_fold[f]}
        assert not train_stems & val_stems, f"{label_csv} fold {f}: video in train and val"

    return rows, val_per_fold


def report(year, rows, val_per_fold, n_splits):
    n_frames = len(rows) // n_splits
    n_videos = len({r["video_id"] for r in rows})
    print(f"--- paxos{year}: {n_frames} frames, {n_videos} videos, folds taken from "
          f"cropping/splits_{year}.json ---")
    for f in range(n_splits):
        val = val_per_fold[f]
        dist = Counter(r["label"] for r in val)
        print(f"  fold_{f}: val {len(val)} frames / {len({r['video_id'] for r in val})} videos "
              f"(labels {dict(sorted(dist.items()))}), "
              f"train {n_frames - len(val)} frames")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cropping_dir", type=str, default="cropping")
    parser.add_argument("--out_dir", type=str, default="informative_frames")
    parser.add_argument("--suffix", type=str, default="from_cropping",
                        help="Output files are split_info_5fold_paxos<year>_<suffix>.csv")
    parser.add_argument("--labels_2025", type=str, default="informative_frames/split_info_full.csv",
                        help="Paxos2025 label CSV (columns filename,video_id,label)")
    parser.add_argument("--labels_2020", type=str, default="informative_frames/manual_labels_binary_paxos2020.csv",
                        help="Paxos2020 label CSV (columns filename,label)")
    parser.add_argument("--frames_dir_2025", type=str, default="data/informative_frames/frames_raw_extract",
                        help="Raw Paxos2025 frame folder; used only to report unlabeled frames when it exists.")
    parser.add_argument("--frames_dir_2020", type=str, default="data/informative_frames/frames_raw_extract_paxos2020",
                        help="Raw Paxos2020 frame folder; used only to report unlabeled frames when it exists.")
    parser.add_argument("--n_splits", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for year, label_csv, frames_dir in [
        ("2025", args.labels_2025, args.frames_dir_2025),
        ("2020", args.labels_2020, args.frames_dir_2020),
    ]:
        rows, val_per_fold = build_split(year, args.cropping_dir, label_csv, args.n_splits)

        out_path = out_dir / f"split_info_5fold_paxos{year}_{args.suffix}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "video_id", "label", "fold", "split"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Saved {out_path} ({len(rows)} rows = {len(rows) // args.n_splits} frames x {args.n_splits} folds)")
        report(year, rows, val_per_fold, args.n_splits)

        frames_path = Path(frames_dir)
        if frames_path.is_dir():
            on_disk = {p.name for p in frames_path.iterdir()
                       if p.suffix.lower() in {".png", ".jpg", ".jpeg"}}
            labeled = {r["filename"] for r in rows}
            unlabeled = sorted(on_disk - labeled)
            print(f"  frames in {frames_path}: {len(on_disk)}, labeled: {len(labeled & on_disk)}, "
                  f"unlabeled (excluded, no label): {len(unlabeled)}")
        else:
            print(f"  frames dir {frames_path} not found -- skipped unlabeled-frame check")


if __name__ == "__main__":
    main()
