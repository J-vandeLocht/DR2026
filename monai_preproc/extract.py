"""MONAI version of the Paxos2025 production pre-processing: videos -> cleaned clips.

Mirrors informative_frames/extract_inf_frames.py (same q5_v3 video set, same
per-video val-fold model choice, same outputs) but runs MONAI models/transforms:

    <out_base>/cleaned_videos/CLEAN_<video>.mp4   (name data/match_clips.py,
                                                   data/split_data.py and data/dataset.py expect)
    <out_base>/txt_files/<video>.txt              (per-frame prob + informative flag, tab-separated)
    <out_base>/plots/<video>.png

--out_base must not be the in-use data/ensemble_results_paxos2025 (same guard
as the existing extractor). Videos listed in the q5_v3 splits but missing (or
unreadable) in --video_dir are skipped with a warning, never an error -- the
split lists can contain videos that are not downloaded locally.

    python -u -m monai_preproc.extract \
      --video_dir data/dr_videos \
      --seg_model_dir monai_preproc/models/seg --cls_model_dir monai_preproc/models/cls \
      --config unet_small_384 \
      --out_base data/ensemble_results_paxos2025_monai
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from informative_frames.crop_frames import crop_and_pad, get_largest_component_mask, pad_to_square_info
from informative_frames.extract_inf_frames import select_informative_frames
from .configs import load_grid
from .models import find_fold_checkpoint, load_seg_model, load_cls_model
from .transforms import seg_infer_transforms, cls_infer_transforms

PROTECTED_OUT_BASE = Path("data/ensemble_results_paxos2025")
CONF_THRESHOLD = 0.5
MIN_CONF_THRESHOLD = 0.2
THRESHOLD_STEP = 0.05
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--total_gpus", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Max videos per worker (smoke tests).")
    parser.add_argument("--video_dir", type=str, default="data/dr_videos")
    parser.add_argument("--seg_model_dir", type=str, default="monai_preproc/models/seg")
    parser.add_argument("--cls_model_dir", type=str, default="monai_preproc/models/cls")
    parser.add_argument("--q5_dir", type=str, default="data/stratified_splits_paxos2025_patient_q5_v3")
    parser.add_argument("--seg_split_csv", type=str, default="cropping/split_cropping_5fold_paxos2025_from_q5v3.csv")
    parser.add_argument("--cls_split_csv", type=str, default="informative_frames/split_info_5fold_paxos2025_from_q5v3.csv")
    parser.add_argument("--out_base", type=str, default="data/ensemble_results_paxos2025_monai")
    parser.add_argument("--config", type=str, default=None, help="Grid name; default: first grid entry.")
    parser.add_argument("--grid", type=str, default=None)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--min_informative_frames", type=int, default=32)
    parser.add_argument("--infer_batch_size", type=int, default=16)
    return parser.parse_args(argv)


def load_q5_videos(q5_dir, n_splits):
    """Stems of all videos in the q5_v3 mil_val lists. A missing split file is a
    warning, not an error."""
    stems = set()
    for fold in range(1, n_splits + 1):
        path = Path(q5_dir) / f"split_{fold}" / "mil_val_2025.json"
        if not path.exists():
            print(f"WARNING: {path} not found -- skipped")
            continue
        stems |= {Path(k).stem for k in json.load(open(path))}
    return stems


def load_val_folds(split_csv, log=print):
    """Video stem -> the fold whose val part contains it in a training split CSV.
    Videos in the val part of several folds get the lowest fold, with a warning
    (never an error)."""
    df = pd.read_csv(split_csv)
    folds = df[df["split"] == "val"].groupby("video_id")["fold"].unique()
    mapping = {}
    for vid, f in folds.items():
        if len(f) != 1:
            log(f"WARNING: {split_csv}: video {vid!r} in the val part of several folds "
                f"{sorted(f.tolist())} -- using fold {int(f.min())}")
        mapping[vid] = int(f.min())
    return mapping


def process_video(v_path, seg_models, cls_models, seg_fold, cls_fold, n_splits,
                  seg_trans, cls_trans, out_vid_path, txt_path, plot_path, args, device):
    vid_id = v_path.stem
    sf, cf = seg_fold.get(vid_id), cls_fold.get(vid_id)
    seg_sel = [seg_models[sf]] if sf is not None else seg_models
    cls_sel = [cls_models[cf]] if cf is not None else cls_models

    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        print(f"WARNING: cannot open video {v_path} -- skipped")
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_results = []
    with tqdm(total=total_f, desc=f"Analysis: {vid_id}", leave=False) as pbar:
        f_idx = 0
        while True:
            batch_frames, batch_f_indices = [], []
            for _ in range(args.infer_batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                batch_frames.append(frame)
                batch_f_indices.append(f_idx)
                f_idx += 1
            if not batch_frames:
                break
            h, w = batch_frames[0].shape[:2]
            padded = [pad_to_square_info(fr) for fr in batch_frames]
            seg_in = torch.stack([
                seg_trans({"image": cv2.cvtColor(sq, cv2.COLOR_BGR2RGB), "mask": np.zeros((sq.shape[0], sq.shape[1]), np.float32)})["image"]
                for sq, _, _, _ in padded
            ]).to(device)
            with torch.no_grad():
                mask_batch = torch.mean(torch.stack([torch.sigmoid(m(seg_in)) for m in seg_sel]), dim=0)
            mask_batch = mask_batch.squeeze(1).float().cpu().numpy()

            per_frame_info = [(None, None, None, 0)] * len(batch_frames)
            crops, crop_slots = [], []
            for bi, mask_small in enumerate(mask_batch):
                _, y_off, x_off, side = padded[bi]
                full_mask = cv2.resize(mask_small, (side, side))[y_off:y_off + h, x_off:x_off + w]
                clean_mask, box = get_largest_component_mask(full_mask > 0.5)
                if clean_mask is None:
                    continue
                x1, y1, x2, y2 = box
                cx, cy, size = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
                per_frame_info[bi] = (clean_mask, cx, cy, size)
                masked = cv2.bitwise_and(batch_frames[bi], batch_frames[bi], mask=clean_mask)
                crop = crop_and_pad(masked, cx, cy, size)
                if crop.size > 0:
                    crops.append(cls_trans({"image": cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)})["image"])
                    crop_slots.append(bi)
            probs = [0.0] * len(batch_frames)
            if crops:
                cls_in = torch.stack(crops).to(device)
                with torch.no_grad():
                    mean_probs = torch.mean(torch.stack([torch.sigmoid(m(cls_in)) for m in cls_sel]), dim=0)
                for slot_idx, bi in enumerate(crop_slots):
                    probs[bi] = float(mean_probs.view(-1)[slot_idx])
            for bi in range(len(batch_frames)):
                clean_mask, cx, cy, size = per_frame_info[bi]
                frame_results.append({"f": batch_f_indices[bi], "prob": probs[bi], "cx": cx,
                                      "cy": cy, "size": size, "mask": clean_mask, "frame": batch_frames[bi]})
            pbar.update(len(batch_frames))
    cap.release()

    if not frame_results:
        print(f"WARNING: no readable frames in {v_path} -- skipped")
        return False

    all_probs = [r["prob"] for r in frame_results]
    informative_idx, threshold_used, used_fallback = select_informative_frames(
        all_probs, CONF_THRESHOLD, MIN_CONF_THRESHOLD, THRESHOLD_STEP, args.min_informative_frames)
    informative_idx_set = set(int(i) for i in informative_idx)
    status = "fallback top-N" if used_fallback else f"threshold={threshold_used}"
    print(f"{vid_id}: {len(informative_idx_set)}/{len(frame_results)} informative frames ({status})")

    informative_sizes = [r["size"] for r in frame_results if r["f"] in informative_idx_set and r["mask"] is not None]
    final_sz = int(max(informative_sizes)) if informative_sizes else 0
    if final_sz > 0 and informative_idx_set:
        out = cv2.VideoWriter(str(out_vid_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (final_sz, final_sz))
        for r in frame_results:
            if r["f"] not in informative_idx_set:
                continue
            if r["mask"] is not None:
                masked = cv2.bitwise_and(r["frame"], r["frame"], mask=r["mask"])
                out.write(crop_and_pad(masked, r["cx"], r["cy"], final_sz))
            else:
                out.write(np.zeros((final_sz, final_sz, 3), dtype=np.uint8))
        out.release()

    df_res = pd.DataFrame([{"frame": r["f"], "prob": r["prob"]} for r in frame_results])
    df_res["informative"] = df_res["frame"].isin(informative_idx_set)
    df_res.to_csv(txt_path, sep="\t", index=False)

    plt.figure(figsize=(10, 4))
    plt.plot(df_res["frame"], df_res["prob"], label="Mean Prob")
    plt.axhline(y=CONF_THRESHOLD, color="r", linestyle="--", alpha=0.5, label=f"Preferred ({CONF_THRESHOLD})")
    plt.fill_between(df_res["frame"], 0, 1.1, where=df_res["informative"], color="green", alpha=0.1, label="Informative")
    plt.title(f"Informative Frames: {vid_id} ({len(informative_idx_set)} frames, {status})")
    plt.xlabel("Frame Index")
    plt.ylabel("Avg Probability")
    plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig(plot_path)
    plt.close()
    return True


def main(argv=None):
    args = parse_args(argv)
    if Path(args.out_base).resolve() == PROTECTED_OUT_BASE.resolve():
        raise SystemExit(f"--out_base {args.out_base} is the in-use clip dir; write to a new dir instead.")
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    grid = load_grid(args.grid)
    cfg = grid[args.config] if args.config else next(iter(grid.values()))

    q5_stems = load_q5_videos(args.q5_dir, args.n_splits)
    seg_fold = load_val_folds(args.seg_split_csv)
    cls_fold = load_val_folds(args.cls_split_csv)

    out_base = Path(args.out_base)
    txt_dir, plot_dir, vid_dir = out_base / "txt_files", out_base / "plots", out_base / "cleaned_videos"
    for d in (txt_dir, plot_dir, vid_dir):
        d.mkdir(parents=True, exist_ok=True)

    seg_models = [load_seg_model(find_fold_checkpoint(args.seg_model_dir, "seg_", f), device, cfg=cfg)
                  for f in range(args.n_splits)]
    cls_models = [load_cls_model(find_fold_checkpoint(args.cls_model_dir, "cls_", f), device, cfg=cfg)
                  for f in range(args.n_splits)]
    seg_trans = seg_infer_transforms(cfg.seg_img_size)
    cls_trans = cls_infer_transforms(cfg.cls_img_size)

    video_dir = Path(args.video_dir)
    if not video_dir.exists():
        raise SystemExit(f"--video_dir {video_dir} does not exist")
    on_disk = {p.stem: p for p in sorted(video_dir.iterdir())
               if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES}
    missing = sorted(q5_stems - on_disk.keys())
    if missing:
        print(f"WARNING: {len(missing)}/{len(q5_stems)} q5_v3 videos are not present in {video_dir} "
              f"-- skipped (no error), e.g. {missing[:5]}")
    all_vids = [on_disk[s] for s in sorted(q5_stems & on_disk.keys())]
    my_vids = [v for i, v in enumerate(all_vids) if i % args.total_gpus == args.gpu_id]
    if args.limit:
        my_vids = my_vids[:args.limit]
    print(f"Worker {args.gpu_id}: processing {len(my_vids)} of {len(all_vids)} locally available "
          f"q5_v3 videos ({len(missing)} missing skipped) with config '{cfg.name}' on {device}")

    done = skipped = 0
    for v_path in tqdm(my_vids, desc=f"Overall GPU {args.gpu_id}"):
        vid_id = v_path.stem
        out_vid_path = vid_dir / f"CLEAN_{vid_id}.mp4"
        txt_path = txt_dir / f"{vid_id}.txt"
        plot_path = plot_dir / f"{vid_id}.png"
        if out_vid_path.exists() and txt_path.exists() and plot_path.exists():
            print(f"Skipping {vid_id} (outputs exist)")
            continue
        if process_video(v_path, seg_models, cls_models, seg_fold, cls_fold, args.n_splits,
                         seg_trans, cls_trans, out_vid_path, txt_path, plot_path, args, device):
            done += 1
        else:
            skipped += 1
    print(f"Done: {done} videos written to {vid_dir}, {skipped} skipped (missing/unreadable), "
          f"{len(missing)} q5_v3 videos not available locally.")
    return {"done": done, "skipped": skipped, "missing_locally": len(missing)}


if __name__ == "__main__":
    main()
