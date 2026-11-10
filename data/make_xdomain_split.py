"""Cross-domain split: train on all Paxos2025 q5 videos/frames, test on the filtered Paxos2020 cohort.

Writes <output_dir>/split_1/ (split_2..N are symlinks = one pseudo-split per seed):
mil_train.json / frame_train.json = train + val of the q5 split_1, frame_val.json = q5 frame_val (dummy, inside train),
mil_val.json = mil_val_2020.json = Paxos2020 videos with Paxos_sharpness 3 and > --min_frames cleaned-clip frames.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import pandas as pd


def load(path):
    with open(path) as f:
        return json.load(f)


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def main(args):
    src = Path(args.source_split)
    out = Path(args.output_dir) / "split_1"
    out.mkdir(parents=True, exist_ok=True)

    mil_train = {**load(src / "mil_train.json"), **load(src / "mil_val.json")}
    frame_train = {**load(src / "frame_train.json"), **load(src / "frame_val.json")}

    sheet = pd.read_excel(args.annotation_xlsx, sheet_name="Paxos 4.7")
    sharpness = dict(zip(sheet.Eye_ID.astype(str).str.strip(), sheet.Paxos_sharpness))
    test, removed = {}, []
    for name, grade in sorted(load(args.labels_2020).items()):
        vid = Path(name).stem
        clip = Path(args.clips_2020) / f"CLEAN_{vid}.mp4"
        assert clip.exists(), clip
        cap = cv2.VideoCapture(str(clip))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        reasons = []
        if sharpness[vid] != 3:
            reasons.append(f"sharpness {sharpness[vid]}")
        if n_frames <= args.min_frames:
            reasons.append(f"{n_frames} frames")
        if reasons:
            removed.append(f"{vid}\t{grade}\t{', '.join(reasons)}")
        else:
            test[name] = grade

    assert not set(test) & set(mil_train), "Paxos2020 video in train"
    dump(mil_train, out / "mil_train.json")
    dump(frame_train, out / "frame_train.json")
    dump(load(src / "frame_val.json"), out / "frame_val.json")
    dump(test, out / "mil_val.json")
    dump(test, out / "mil_val_2020.json")
    with open(Path(args.output_dir) / "removed_videos.txt", "w") as f:
        f.write(f"Paxos2020 videos removed from the test set (Paxos_sharpness != 3 or <= {args.min_frames} cleaned-clip frames).\n"
                "video\tgrade\treason\n" + "\n".join(removed) + "\n")

    for i in range(2, args.num_seeds + 1):
        link = Path(args.output_dir) / f"split_{i}"
        if not link.exists():
            os.symlink("split_1", link)

    grades = pd.Series(test).value_counts().sort_index().to_dict()
    print(f"train: {len(mil_train)} videos, {len(frame_train)} frames; test (Paxos2020): {len(test)} videos {grades}, "
          f"removed {len(removed)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_split", default="data/stratified_splits_paxos2025_patient_q5/split_1")
    parser.add_argument("--output_dir", default="data/stratified_splits_paxos2025_q5_full_xdom")
    parser.add_argument("--labels_2020", default="data/clip_labels_2020.json")
    parser.add_argument("--clips_2020", default="data/ensemble_results_paxos2020/cleaned_videos")
    parser.add_argument("--annotation_xlsx", default=os.path.expanduser(
        "~/Data/2020_Paxos_Videos/Patient list + Analysis Sankara Smartphone DR 4.7 - f R.xlsx"))
    parser.add_argument("--min_frames", type=int, default=139, help="Same cut as the q5 splits (5th pct of Paxos2025).")
    parser.add_argument("--num_seeds", type=int, default=1, help="split_2..N = symlinks to split_1, one per seed.")
    main(parser.parse_args())
