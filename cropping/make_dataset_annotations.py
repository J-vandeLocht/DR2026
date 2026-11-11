"""Split the combined cropping annotations into per-dataset CSVs.

The filename column of annotations_video_frames_larger.csv holds frames from
both cohorts (e.g. 'R069R_frame_561.png' or
'2024_02_15_14_20_IMG_4659 LE MILD NPDR_frame_906.png'). Each frame belongs to
the video '<stem>.<ext>' in one of the two video folders, so dataset membership
is decided by looking the stem up in the video folders:

    <stem>_frame_<N>.png  ->  video <stem>.mp4/.MOV/... in --video_dir_2025 or --video_dir_2020

Rows whose stem matches no video are reported ('no video found') and rows that
match videos in both folders (or several files) are reported ('multiple matches
found') and skipped. Output: annotations_cropping_2025.csv and
annotations_cropping_2020.csv next to the input CSV.

Run from the repo root:
    python -u -m cropping.make_dataset_annotations \
        --video_dir_2025 data/dr_videos --video_dir_2020 data/Paxos_2020_videos
"""

import argparse
import csv
import re
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
FRAME_RE = re.compile(r"^(?P<stem>.*)_frame_(?P<idx>\d+)\.(?P<ext>png|jpe?g)$", re.IGNORECASE)


def video_key(path):
    """Comparable video stem: lowercase, no extension, no CLEAN_ prefix."""
    s = path.stem
    s = re.sub(r"^CLEAN_", "", s, flags=re.IGNORECASE)
    return s.lower().strip()


def index_video_dir(video_dir):
    """video_key -> list of matching video paths (only real video extensions)."""
    index = {}
    d = Path(video_dir)
    if not d.is_dir():
        print(f"WARNING: video folder not found: {d}")
        return index
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            index.setdefault(video_key(p), []).append(p)
    return index


def frame_stem(filename):
    m = FRAME_RE.match(filename)
    if m:
        return m.group("stem")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=str, default="cropping/annotations_video_frames_larger.csv")
    parser.add_argument("--video_dir_2025", type=str, default="data/dr_videos",
                        help="Folder with the Paxos2025 videos (e.g. R069R.mp4, IMG_....mp4).")
    parser.add_argument("--video_dir_2020", type=str, default="data/Paxos_2020_videos",
                        help="Folder with the Paxos2020 videos.")
    parser.add_argument("--out_dir", type=str, default="cropping")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.csv, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Read {len(rows)} annotated frames from {args.csv}")

    index_2025 = index_video_dir(args.video_dir_2025)
    index_2020 = index_video_dir(args.video_dir_2020)
    print(f"Indexed {sum(len(v) for v in index_2025.values())} videos in {args.video_dir_2025} "
          f"({len(index_2025)} unique stems)")
    print(f"Indexed {sum(len(v) for v in index_2020.values())} videos in {args.video_dir_2020} "
          f"({len(index_2020)} unique stems)")

    per_dataset = {"2025": [], "2020": []}
    skipped = []

    for row in rows:
        filename = row["filename"]
        stem = frame_stem(filename)
        if stem is None:
            print(f"no video found for '{filename}' (filename has no '_frame_<N>.<ext>' suffix)")
            skipped.append(filename)
            continue
        key = stem.lower().strip()
        matches = [(ds, p) for ds, index in (("2025", index_2025), ("2020", index_2020))
                   for p in index.get(key, [])]
        if not matches:
            print(f"no video found for '{filename}' (stem '{stem}')")
            skipped.append(filename)
            continue
        datasets_hit = {ds for ds, _ in matches}
        if len(matches) > 1 or len(datasets_hit) > 1:
            print(f"multiple matches found for '{filename}': "
                  + ", ".join(f"{ds}:{p}" for ds, p in matches))
            if len(datasets_hit) > 1:
                skipped.append(filename)
                continue
        per_dataset[matches[0][0]].append(row)

    for year, out_rows in per_dataset.items():
        out_path = out_dir / f"annotations_cropping_{year}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sorted(out_rows, key=lambda r: r["filename"]))
        print(f"Wrote {len(out_rows)} rows to {out_path}")

    if skipped:
        skipped_path = out_dir / "annotations_cropping_skipped.txt"
        with open(skipped_path, "w") as f:
            f.write("\n".join(skipped) + "\n")
        print(f"Skipped {len(skipped)} rows (see {skipped_path})")
    else:
        print("No rows skipped.")


if __name__ == "__main__":
    main()
