"""Convert informative_frames/split_info_full.csv into the plain annotation CSV.

Drops the split and video_id columns and keeps filename,label, writing
informative_frames/annotations_informative_2025.csv -- the label source for the
informative-frame stage (replacement for split_info_full.csv).

Run from the repo root:
    python -u -m informative_frames.make_informative_annotations
"""

import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=str, default="informative_frames/split_info_full.csv",
                        help="Source CSV with columns filename,video_id,label,split.")
    parser.add_argument("--output", type=str, default="informative_frames/annotations_informative_2025.csv",
                        help="Output CSV with columns filename,label.")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    missing_cols = [c for c in ("filename", "label") if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{args.input} is missing required columns: {missing_cols}")

    out = df[["filename", "label"]]
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(out)} rows (columns: {', '.join(out.columns)})")


if __name__ == "__main__":
    main()
