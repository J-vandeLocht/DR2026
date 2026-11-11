"""Simple video/subject-level 5-fold splits for the cropping annotations, per dataset.

Unlike cropping/make_splits.py (which groups patients via the Eye_ID mapping in
data/DR_Grading_Summary_v3.csv), this script derives the subject purely from the
video id, so it does not need the grading sheet (or DR_Grading_campR.csv):

    R008R / R008R2 / R008L        -> subject "R008"   (regex ^([A-Za-z]+\\d+)[LR]\\d*$)
    ..._IMG_4608 ... / IMG_2503   -> subject "IMG_4608" / "IMG_2503"
    anything else                 -> the video stem itself

--group_by subject (default) keeps all videos of one subject in the same fold;
--group_by video keeps each video whole (frames of one video never split across
train/val). If the clip labels exist (data/clip_labels_<year>.json), folds are
still stratified by DR grade (ungraded videos form their own class) so the splits
stay reusable for the DR classifiers; --no_stratify switches to plain GroupKFold.

Outputs (same format as cropping/splits.json, consumed by cropping/train_segmentation.py):
    cropping/splits_2020.json, cropping/splits_2025.json
    cropping/splits_<year>_metadata.csv

Run from the repo root:
    python -u -m cropping.make_splits_simple
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from cropping.make_splits import frame_stem, load_grade_lookup

IMG_RE = re.compile(r"IMG_(\d+)", re.IGNORECASE)
EYE_RE = re.compile(r"^([A-Za-z]+\d+)[LR]\d*$")


def subject_of(stem):
    m = IMG_RE.search(stem)
    if m:
        return f"IMG_{m.group(1)}"
    m = EYE_RE.match(stem)
    if m:
        return m.group(1).upper()
    return stem


def make_dataset_splits(ann_csv, grade_lookup, group_by, stratify, n_splits, seed):
    df = pd.read_csv(ann_csv)
    filenames = df["filename"].tolist()
    stems = [frame_stem(fn) for fn in filenames]

    unique_stems = sorted(set(stems))
    stem_subject = {s: subject_of(s) if group_by == "subject" else s for s in unique_stems}
    stem_grade = {s: grade_lookup.get(s.lower().strip(), -1) for s in unique_stems}
    n_ungraded = sum(1 for s in unique_stems if stem_grade[s] == -1)

    subjects = [stem_subject[s] for s in unique_stems]
    grades = [stem_grade[s] for s in unique_stems]

    if stratify:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = list(splitter.split(unique_stems, grades, subjects))
    else:
        splitter = GroupKFold(n_splits=n_splits)
        folds = list(splitter.split(unique_stems, None, subjects))

    stem_fold = {}
    for fold, (train_idx, val_idx) in enumerate(folds):
        for i in val_idx:
            stem_fold[unique_stems[i]] = fold
        shared = set(subjects[i] for i in train_idx) & set(subjects[i] for i in val_idx)
        assert not shared, f"{ann_csv} fold {fold}: subjects in train and val: {sorted(shared)[:5]}"

    splits = {f"fold_{f}": {"train": [], "val": []} for f in range(n_splits)}
    for fn, s in zip(filenames, stems):
        splits[f"fold_{stem_fold[s]}"]["val"].append(fn)
    for f in range(n_splits):
        val_stems = {s for s in unique_stems if stem_fold[s] == f}
        splits[f"fold_{f}"]["train"] = sorted(fn for fn, s in zip(filenames, stems) if s not in val_stems)
        splits[f"fold_{f}"]["val"] = sorted(splits[f"fold_{f}"]["val"])
        # video-level guarantee: no video stem in both train and val of a fold
        train_stems = {frame_stem(fn) for fn in splits[f"fold_{f}"]["train"]}
        assert not train_stems & val_stems

    n_subjects = len(set(subjects))
    print(f"  {ann_csv}: {len(filenames)} frames, {len(unique_stems)} videos, {n_subjects} subjects "
          f"({n_ungraded} ungraded), no train/val subject overlap, videos never split")

    meta_rows = []
    for f in range(n_splits):
        for set_name in ("train", "val"):
            fold_stems = [s for s in unique_stems if (stem_fold[s] == f) == (set_name == "val")]
            counts = Counter(stem_grade[s] for s in fold_stems)
            meta_rows.append({
                "fold": f + 1, "set": set_name, "total": len(fold_stems),
                **{f"grade_{i}": counts.get(i, 0) for i in range(5)},
                "ungraded": counts.get(-1, 0),
            })
    return splits, pd.DataFrame(meta_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ann_dir", type=str, default="cropping")
    parser.add_argument("--out_dir", type=str, default="cropping")
    parser.add_argument("--clip_labels_2025", type=str, default="data/clip_labels_2025.json")
    parser.add_argument("--clip_labels_2020", type=str, default="data/clip_labels_2020.json")
    parser.add_argument("--years", nargs="+", default=["2020", "2025"], choices=["2020", "2025"])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group_by", choices=["subject", "video"], default="subject",
                        help="'subject': all videos of one subject share a fold; 'video': each video is its own group.")
    parser.add_argument("--no_stratify", action="store_true",
                        help="Plain GroupKFold instead of grade-stratified StratifiedGroupKFold.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    grade_lookup = load_grade_lookup([args.clip_labels_2025, args.clip_labels_2020])

    for year in args.years:
        ann_csv = Path(args.ann_dir) / f"annotations_cropping_{year}.csv"
        if not ann_csv.exists():
            print(f"Error: {ann_csv} not found. Run cropping/make_dataset_annotations.py first.")
            continue
        print(f"--- {year} (group_by={args.group_by}, stratify={not args.no_stratify}) ---")
        splits, meta = make_dataset_splits(ann_csv, grade_lookup, args.group_by,
                                           stratify=not args.no_stratify, n_splits=args.n_splits, seed=args.seed)
        splits_path = out_dir / f"splits_{year}.json"
        with open(splits_path, "w") as f:
            json.dump(splits, f, indent=4)
        meta_path = out_dir / f"splits_{year}_metadata.csv"
        meta.to_csv(meta_path, index=False)
        print(f"  Saved {splits_path} and {meta_path}")
        print(meta.to_string(index=False))


if __name__ == "__main__":
    main()
