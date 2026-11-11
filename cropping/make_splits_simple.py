"""Video/subject-level 5-fold splits for the cropping annotations, per dataset.

No DR grades are involved anywhere in the cropping stage: the subject is derived
purely from the video id, so neither the grading sheet (DR_Grading_Summary_v3.csv,
DR_Grading_campR.csv) nor the clip labels (data/clip_labels_<year>.json) are read:

    R008R / R008R2 / R008L        -> subject "R008"   (regex ^([A-Za-z]+\\d+)[LR]\\d*$)
    ..._IMG_4608 ... / IMG_2503   -> subject "IMG_4608" / "IMG_2503"
    anything else                 -> the video stem itself

--group_by subject (default) keeps all videos of one subject in the same fold;
--group_by video keeps each video whole. Either way frames of one video are never
split across train/val, and the val folds cover every frame exactly once.

Outputs (same format as cropping/splits.json, consumed by cropping/train_segmentation.py):
    cropping/splits_2020.json, cropping/splits_2025.json
    cropping/splits_<year>_metadata.csv   # per-fold totals only (no grades)

Run from the repo root:
    python -u -m cropping.make_splits_simple
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold

FRAME_RE = re.compile(r"^(?P<stem>.*)_frame_(?P<idx>\d+)\.(?P<ext>png|jpe?g)$", re.IGNORECASE)
IMG_RE = re.compile(r"IMG_(\d+)", re.IGNORECASE)
EYE_RE = re.compile(r"^([A-Za-z]+\d+)[LR]\d*$")


def frame_stem(filename):
    m = FRAME_RE.match(filename)
    return m.group("stem") if m else Path(filename).stem


def subject_of(stem):
    m = IMG_RE.search(stem)
    if m:
        return f"IMG_{m.group(1)}"
    m = EYE_RE.match(stem)
    if m:
        return m.group(1).upper()
    return stem


def make_dataset_splits(ann_csv, group_by, n_splits, seed):
    df = pd.read_csv(ann_csv)
    filenames = df["filename"].tolist()
    stems = [frame_stem(fn) for fn in filenames]

    unique_stems = sorted(set(stems))
    subject = {s: subject_of(s) if group_by == "subject" else s for s in unique_stems}

    rng = random.Random(seed)
    shuffled = unique_stems[:]
    rng.shuffle(shuffled)

    stem_fold = {}
    gkf = GroupKFold(n_splits=n_splits)
    for fold, (train_idx, val_idx) in enumerate(gkf.split(shuffled, None, [subject[s] for s in shuffled])):
        for i in val_idx:
            stem_fold[shuffled[i]] = fold
        shared = {subject[shuffled[i]] for i in train_idx} & {subject[shuffled[i]] for i in val_idx}
        assert not shared, f"{ann_csv} fold {fold}: subjects in train and val: {sorted(shared)[:5]}"

    splits = {f"fold_{f}": {"train": [], "val": []} for f in range(n_splits)}
    for fn, s in zip(filenames, stems):
        splits[f"fold_{stem_fold[s]}"]["val"].append(fn)
    for f in range(n_splits):
        val_stems = {s for s in unique_stems if stem_fold[s] == f}
        splits[f"fold_{f}"]["train"] = sorted(fn for fn, s in zip(filenames, stems) if s not in val_stems)
        splits[f"fold_{f}"]["val"] = sorted(splits[f"fold_{f}"]["val"])
        train_stems = {frame_stem(fn) for fn in splits[f"fold_{f}"]["train"]}
        assert not train_stems & val_stems, f"{ann_csv} fold {f}: video in train and val"

    val_all = [fn for f in range(n_splits) for fn in splits[f"fold_{f}"]["val"]]
    assert len(val_all) == len(set(val_all)) == len(set(filenames)), f"{ann_csv}: val folds must cover each frame once"

    n_subjects = len(set(subject.values()))
    print(f"  {ann_csv}: {len(filenames)} frames, {len(unique_stems)} videos, {n_subjects} subjects, "
          f"no train/val subject overlap, videos never split")

    meta_rows = []
    for f in range(n_splits):
        for set_name in ("train", "val"):
            fold_stems = {s for s in unique_stems if (stem_fold[s] == f) == (set_name == "val")}
            meta_rows.append({
                "fold": f + 1, "set": set_name,
                "total": sum(1 for s in stems if s in fold_stems),
                "n_videos": len(fold_stems),
                "n_subjects": len({subject[s] for s in fold_stems}),
            })
    return splits, pd.DataFrame(meta_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ann_dir", type=str, default="cropping")
    parser.add_argument("--out_dir", type=str, default="cropping")
    parser.add_argument("--years", nargs="+", default=["2020", "2025"], choices=["2020", "2025"])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group_by", choices=["subject", "video"], default="subject",
                        help="'subject': all videos of one subject share a fold; 'video': each video is its own group.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in args.years:
        ann_csv = Path(args.ann_dir) / f"annotations_cropping_{year}.csv"
        if not ann_csv.exists():
            print(f"Error: {ann_csv} not found. Run cropping/make_dataset_annotations.py first.")
            continue
        print(f"--- {year} (group_by={args.group_by}) ---")
        splits, meta = make_dataset_splits(ann_csv, args.group_by, args.n_splits, args.seed)
        splits_path = out_dir / f"splits_{year}.json"
        with open(splits_path, "w") as f:
            json.dump(splits, f, indent=4)
        meta_path = out_dir / f"splits_{year}_metadata.csv"
        meta.to_csv(meta_path, index=False)
        print(f"  Saved {splits_path} and {meta_path}")
        print(meta.to_string(index=False))


if __name__ == "__main__":
    main()
