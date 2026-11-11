"""Eye/patient-level stratified 5-fold splits for the cropping annotations, per dataset.

Mirrors data/split_data.py (the DR classifier splits): StratifiedGroupKFold over
videos grouped by patient (make_patient_id_fn), stratified by DR grade taken from
data/clip_labels_<year>.json. All frames of one video/eye land in the same fold,
and no patient appears in both train and val of a fold -- so the same splits can
later be reused for training the DR classifiers on these frames.

Inputs (produced by cropping/make_dataset_annotations.py):
    cropping/annotations_cropping_2020.csv, cropping/annotations_cropping_2025.csv
Outputs (same format as the old cropping/splits.json):
    cropping/splits_2020.json, cropping/splits_2025.json   # {"fold_i": {"train": [...], "val": [...]}}
    cropping/splits_<year>_metadata.csv                    # per-fold grade/group counts

Run from the repo root:
    python -u -m cropping.make_splits
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from data.split_data import make_patient_id_fn

FRAME_RE = re.compile(r"^(?P<stem>.*)_frame_(?P<idx>\d+)\.(?P<ext>png|jpe?g)$", re.IGNORECASE)


def load_grade_lookup(clip_labels_paths):
    """clip_labels json ({'name.mp4': grade}) -> {lowercase stem: grade}."""
    lookup = {}
    for path in clip_labels_paths:
        with open(path) as f:
            labels = json.load(f)
        for name, grade in labels.items():
            lookup[Path(name).stem.lower().strip()] = grade
    return lookup


def frame_stem(filename):
    m = FRAME_RE.match(filename)
    return m.group("stem") if m else Path(filename).stem


def make_dataset_splits(ann_csv, grade_lookup, group_fn, n_splits, seed):
    df = pd.read_csv(ann_csv)
    filenames = df["filename"].tolist()
    stems = [frame_stem(fn) for fn in filenames]

    unique_stems = sorted(set(stems))
    stem_grade, stem_group, ungraded = {}, {}, []
    for s in unique_stems:
        key = s.lower().strip()
        if key in grade_lookup:
            stem_grade[s] = grade_lookup[key]
        else:
            stem_grade[s] = -1
            ungraded.append(s)
        stem_group[s] = group_fn(s)
    if ungraded:
        print(f"  {ann_csv}: {len(ungraded)}/{len(unique_stems)} videos have no grade in the clip labels "
              f"(stratified as their own class -1): {ungraded[:5]}{'...' if len(ungraded) > 5 else ''}")

    grades = [stem_grade[s] for s in unique_stems]
    groups = [stem_group[s] for s in unique_stems]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    stem_fold = {}
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(unique_stems, grades, groups)):
        for i in val_idx:
            stem_fold[unique_stems[i]] = fold
        shared = set(groups[i] for i in train_idx) & set(groups[i] for i in val_idx)
        assert not shared, f"{ann_csv} fold {fold}: groups in train and val: {sorted(shared)[:5]}"

    splits = {f"fold_{f}": {"train": [], "val": []} for f in range(n_splits)}
    for fn, s in zip(filenames, stems):
        splits[f"fold_{stem_fold[s]}"]["val"].append(fn)
    for f in range(n_splits):
        val_stems = {s for s in unique_stems if stem_fold[s] == f}
        splits[f"fold_{f}"]["train"] = sorted(fn for fn, s in zip(filenames, stems) if s not in val_stems)
        splits[f"fold_{f}"]["val"] = sorted(splits[f"fold_{f}"]["val"])

    n_groups = len(set(groups))
    print(f"  {ann_csv}: {len(filenames)} frames, {len(unique_stems)} videos, {n_groups} groups, "
          f"no train/val group overlap in any fold")

    meta_rows = []
    for f in range(n_splits):
        for set_name in ("train", "val"):
            fold_stems = [s for s in unique_stems
                          if (stem_fold[s] == f) == (set_name == "val")]
            g = [stem_grade[s] for s in fold_stems]
            counts = Counter(g)
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
    parser.add_argument("--grading_summary", type=str, default="data/DR_Grading_Summary_v3.csv")
    parser.add_argument("--years", nargs="+", default=["2020", "2025"], choices=["2020", "2025"])
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group_by", choices=["patient", "eye"], default="patient",
                        help="'patient' = both eyes of a patient share a fold (same as the DR splits in "
                             "data/split_data.py); 'eye' = every video is its own group.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grade_lookup = load_grade_lookup([args.clip_labels_2025, args.clip_labels_2020])
    patient_id = make_patient_id_fn(args.grading_summary)

    for year in args.years:
        ann_csv = Path(args.ann_dir) / f"annotations_cropping_{year}.csv"
        if not ann_csv.exists():
            print(f"Error: {ann_csv} not found. Run cropping/make_dataset_annotations.py first.")
            continue
        group_fn = patient_id if args.group_by == "patient" else (lambda s: s)
        print(f"--- {year} (group_by={args.group_by}) ---")
        splits, meta = make_dataset_splits(ann_csv, grade_lookup, group_fn, args.n_splits, args.seed)

        splits_path = out_dir / f"splits_{year}.json"
        with open(splits_path, "w") as f:
            json.dump(splits, f, indent=4)
        meta_path = out_dir / f"splits_{year}_metadata.csv"
        meta.to_csv(meta_path, index=False)
        print(f"  Saved {splits_path} and {meta_path}")
        print(meta.to_string(index=False))


if __name__ == "__main__":
    main()
