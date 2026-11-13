"""Full ensemble production pipeline: videos -> cleaned clips + per-frame prob txt + plot.

Model choice per video follows the same rule as cropping/evaluate_segmentation.py:
with --dataset given, a video is scored by the fold-i model of its val fold (from
<splits_dir>/splits_<dataset>.json, the cropping splits) when the model config matches
the dataset (--seg_config/--cls_config == --dataset) -- i.e. ONLY the model that was
NOT trained on that video; otherwise (config is the other dataset, or the video is
unknown to the splits) the full n-fold ensemble is used, averaging sigmoid PROBABILITIES
(not logits). Without --dataset the full ensemble is used for every video (old behaviour).

Run from the repo root, e.g. Paxos2025 videos with the Paxos2025 models:
    python -u -m informative_frames.extract_inf_frames \
      --dataset 2025 --video_dir data/dr_videos \
      --seg_model_dir cropping/models/2025 --cls_model_dir informative_frames/models_xdomain \
      --seg_config 2025 --cls_config 2025 --out_base data/ensemble_results_paxos2025_by_2025
See scripts/extract_inf_frames_xdomain.sh for all three x-domain runs.
"""

import cv2
import re
import torch
import numpy as np
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from misc.utils import apply_clahe_cv2
from .crop_frames import crop_and_pad, get_largest_component_mask, pad_to_square_info
from .make_splits_from_cropping import load_stem_folds


def parse_args():
    parser = argparse.ArgumentParser(description="Full Ensemble Production Pipeline")
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--total_gpus', type=int, default=1)
    parser.add_argument('--limit', type=int, default=None, help="Max videos per worker.")

    parser.add_argument('--video_dir', type=str, default='data/dr_videos')
    parser.add_argument('--seg_model_dir', type=str, default='cropping/models')
    parser.add_argument('--cls_model_dir', type=str, default='informative_frames/models')
    parser.add_argument('--out_base', type=str, default='data/ensemble_results')
    parser.add_argument('--min_informative_frames', type=int, default=32,
                        help="Guaranteed minimum length of the cleaned clip.")
    parser.add_argument('--infer_batch_size', type=int, default=16,
                        help="Frames batched per ensemble forward pass in PASS 1.")
    parser.add_argument('--dataset', type=str, default=None, choices=["2020", "2025"],
                        help="Cohort of the videos in --video_dir. If given, each video is scored by the "
                             "fold model of its val fold when the model config matches the dataset, else by "
                             "the full ensemble. Default None: always full ensemble (old behaviour).")
    parser.add_argument('--seg_config', type=str, default=None, choices=["2020", "2025", "both"],
                        help="Config of the seg models in --seg_model_dir; default: derived from the dir name.")
    parser.add_argument('--cls_config', type=str, default=None, choices=["2020", "2025", "both"],
                        help="Config of the cls models in --cls_model_dir; default: derived from cls_<config>_fold_<i>.pth names.")
    parser.add_argument('--splits_dir', type=str, default='cropping',
                        help="Dir with splits_<dataset>.json (cropping splits) used for the per-video val fold.")
    parser.add_argument('--n_splits', type=int, default=5)
    return parser.parse_args()


def get_seg_transforms(size=768):
    return albu.Compose([
        albu.Resize(size, size),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])


def select_informative_frames(probs, start_threshold, min_threshold, step, min_frames):
    probs = np.asarray(probs)
    n_total = len(probs)

    threshold = start_threshold
    idx = np.where(probs >= threshold)[0]

    while len(idx) < min_frames and threshold > min_threshold:
        threshold = round(max(threshold - step, min_threshold), 4)
        idx = np.where(probs >= threshold)[0]

    used_fallback = False
    if len(idx) < min_frames:
        n_take = min(min_frames, n_total)
        idx = np.sort(np.argsort(probs)[::-1][:n_take])
        used_fallback = True

    return idx, threshold, used_fallback


def main():
    args = parse_args()
    DEVICE = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    VIDEO_DIR = Path(args.video_dir)
    SEG_MODEL_DIR = Path(args.seg_model_dir)
    CLS_MODEL_DIR = Path(args.cls_model_dir)

    if args.seg_config is None:
        name = SEG_MODEL_DIR.name
        args.seg_config = name if name in ("2020", "2025", "both") else None
    if args.cls_config is None:
        for p in sorted(CLS_MODEL_DIR.glob("cls_*_fold_*.pth")):
            m = re.fullmatch(r"cls_(.+)_fold_\d+\.pth", p.name)
            if m:
                args.cls_config = m.group(1)
                break

    stem_fold = {}
    if args.dataset:
        stem_fold = load_stem_folds(Path(args.splits_dir) / f"splits_{args.dataset}.json", args.n_splits)
        print(f"dataset={args.dataset}: mapped {len(stem_fold)} videos to their val fold "
              f"from {args.splits_dir}/splits_{args.dataset}.json "
              f"(seg_config={args.seg_config}, cls_config={args.cls_config})")

    OUT_BASE = Path(args.out_base)
    TXT_DIR = OUT_BASE / 'txt_files'
    PLOT_DIR = OUT_BASE / 'plots'
    VID_DIR = OUT_BASE / 'cleaned_videos'
    for d in [TXT_DIR, PLOT_DIR, VID_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    SEG_SIZE = 768
    CLS_SIZE = 512
    CONF_THRESHOLD = 0.5
    MIN_CONF_THRESHOLD = 0.2
    THRESHOLD_STEP = 0.05
    MIN_INFORMATIVE_FRAMES = args.min_informative_frames
    INFER_BATCH_SIZE = args.infer_batch_size

    print(f"Loading {args.n_splits}-fold Unet++ (B4) and Classification (B4) on {DEVICE}...")
    seg_ensemble = []
    cls_ensemble = []
    for fold in range(args.n_splits):
        seg_paths = sorted(SEG_MODEL_DIR.glob(f"*fold_{fold}*.pth"))
        if not seg_paths:
            raise FileNotFoundError(f"Could not find seg model for fold {fold} in {SEG_MODEL_DIR}")
        sm = smp.UnetPlusPlus(encoder_name="efficientnet-b4", in_channels=3, classes=1).to(DEVICE).eval()
        sm.load_state_dict(torch.load(seg_paths[0], map_location=DEVICE))
        seg_ensemble.append(sm)

        cls_paths = sorted(CLS_MODEL_DIR.glob(f"cls_{args.cls_config}_fold_{fold}*.pth")) if args.cls_config else []
        if not cls_paths:
            cls_paths = sorted(CLS_MODEL_DIR.glob(f"*Fold{fold}*.pth"))
        if not cls_paths:
            raise FileNotFoundError(f"Could not find cls model for fold {fold} in {CLS_MODEL_DIR}")
        cm = models.efficientnet_b4()
        cm.classifier[1] = torch.nn.Linear(cm.classifier[1].in_features, 1)
        cm.load_state_dict(torch.load(cls_paths[0], map_location=DEVICE))
        cm.to(DEVICE).eval()
        cls_ensemble.append(cm)

    seg_trans = get_seg_transforms(SEG_SIZE)
    cls_trans = transforms.Compose([
        transforms.Resize((CLS_SIZE, CLS_SIZE)),
        transforms.Lambda(apply_clahe_cv2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    all_vids = sorted([p for p in VIDEO_DIR.glob("*") if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov"}])
    my_vids = [v for i, v in enumerate(all_vids) if i % args.total_gpus == args.gpu_id]
    if args.limit: my_vids = my_vids[:args.limit]
    print(f"Worker {args.gpu_id}: Processing {len(my_vids)} videos.")

    for v_path in tqdm(my_vids, desc=f"Overall GPU {args.gpu_id}"):
        vid_id = v_path.stem

        fold = stem_fold.get(vid_id) if stem_fold else None
        seg_single = args.dataset is not None and args.seg_config == args.dataset and fold is not None
        cls_single = args.dataset is not None and args.cls_config == args.dataset and fold is not None
        seg_models = [seg_ensemble[fold]] if seg_single else seg_ensemble
        cls_models = [cls_ensemble[fold]] if cls_single else cls_ensemble
        seg_mode = f"fold {fold} (val fold, not in its training)" if seg_single else f"{len(seg_ensemble)}-fold prob ensemble"
        cls_mode = f"fold {fold} (val fold, not in its training)" if cls_single else f"{len(cls_ensemble)}-fold prob ensemble"
        print(f"{vid_id}: seg={seg_mode} | cls={cls_mode}")

        out_vid_path = VID_DIR / f"CLEAN_{vid_id}.mp4"
        txt_path = TXT_DIR / f"{vid_id}.txt"
        plot_path = PLOT_DIR / f"{vid_id}.png"

        if out_vid_path.exists() and txt_path.exists() and plot_path.exists():
            print(f"Skipping {vid_id}")
            continue
        cap = cv2.VideoCapture(str(v_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # PASS 1: Ensemble Analysis (batched)
        frame_results = []
        global_max_dim = 0
        amp_ctx = lambda: torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=(DEVICE.type == 'cuda'))

        with tqdm(total=total_f, desc=f"Analysis: {vid_id}", leave=False) as pbar:
            f_idx = 0
            while True:
                batch_frames, batch_f_indices = [], []
                for _ in range(INFER_BATCH_SIZE):
                    ret, frame = cap.read()
                    if not ret: break
                    batch_frames.append(frame)
                    batch_f_indices.append(f_idx)
                    f_idx += 1
                if not batch_frames:
                    break
                h, w = batch_frames[0].shape[:2]

                padded = [pad_to_square_info(fr) for fr in batch_frames]
                seg_in = torch.stack([
                    seg_trans(image=cv2.cvtColor(sq, cv2.COLOR_BGR2RGB))['image'] for sq, _, _, _ in padded
                ]).to(DEVICE)

                with torch.no_grad(), amp_ctx():
                    m_preds = torch.stack([torch.sigmoid(m(seg_in)) for m in seg_models])
                    mean_mask_batch = torch.mean(m_preds, dim=0).squeeze(1).float().cpu().numpy()

                per_frame_info = [(None, None, None, 0)] * len(batch_frames)
                crops, crop_slots = [], []
                for bi, mask_small in enumerate(mean_mask_batch):
                    _, y_off, x_off, side = padded[bi]
                    full_mask = cv2.resize(mask_small, (side, side))[y_off:y_off + h, x_off:x_off + w]
                    clean_mask, box = get_largest_component_mask(full_mask > 0.5)
                    if clean_mask is None:
                        continue
                    x1, y1, x2, y2 = box
                    cx, cy, size = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1)
                    per_frame_info[bi] = (clean_mask, cx, cy, size)
                    global_max_dim = max(global_max_dim, size)
                    masked = cv2.bitwise_and(batch_frames[bi], batch_frames[bi], mask=clean_mask)
                    crop = crop_and_pad(masked, cx, cy, size)
                    if crop.size > 0:
                        crops.append(cls_trans(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
                        crop_slots.append(bi)

                probs = [0.0] * len(batch_frames)
                if crops:
                    cls_in = torch.stack(crops).to(DEVICE)
                    with torch.no_grad(), amp_ctx():
                        c_preds = torch.stack([torch.sigmoid(m(cls_in)) for m in cls_models])
                        mean_probs = torch.mean(c_preds, dim=0).squeeze(1).float().cpu().numpy()
                    for slot_idx, bi in enumerate(crop_slots):
                        probs[bi] = float(mean_probs[slot_idx])

                for bi in range(len(batch_frames)):
                    clean_mask, cx, cy, size = per_frame_info[bi]
                    frame_results.append({
                        'f': batch_f_indices[bi], 'prob': probs[bi], 'cx': cx, 'cy': cy,
                        'size': size, 'mask': clean_mask, 'frame': batch_frames[bi],
                    })
                pbar.update(len(batch_frames))

        all_probs = [r['prob'] for r in frame_results]
        informative_idx, threshold_used, used_fallback = select_informative_frames(
            all_probs, CONF_THRESHOLD, MIN_CONF_THRESHOLD, THRESHOLD_STEP, MIN_INFORMATIVE_FRAMES)
        informative_idx_set = set(int(i) for i in informative_idx)

        status = "fallback top-N" if used_fallback else f"threshold={threshold_used}"

        print(f"{vid_id}: {len(informative_idx_set)}/{len(frame_results)} informative frames ({status})")

        # PASS 2: Export Cropped Video (informative frames only)
        if global_max_dim > 0 and len(informative_idx_set) > 0:
            final_sz = int(global_max_dim)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(out_vid_path), fourcc, fps, (final_sz, final_sz))
            for res in frame_results:
                if res['f'] not in informative_idx_set:
                    continue
                if res['mask'] is not None:
                    masked = cv2.bitwise_and(res['frame'], res['frame'], mask=res['mask'])
                    cropped = crop_and_pad(masked, res['cx'], res['cy'], final_sz)
                else:
                    cropped = np.zeros((final_sz, final_sz, 3), dtype=np.uint8)
                out.write(cropped)
            out.release()

        # PASS 3: Txt & Plot
        df_res = pd.DataFrame([{'frame': r['f'], 'prob': r['prob']} for r in frame_results])
        df_res['informative'] = df_res['frame'].isin(informative_idx_set)
        df_res.to_csv(TXT_DIR / f"{vid_id}.txt", sep='\t', index=False)

        plt.figure(figsize=(10, 4))
        plt.plot(df_res['frame'], df_res['prob'], label='Ensemble Prob')
        plt.axhline(y=CONF_THRESHOLD, color='r', linestyle='--', alpha=0.5, label=f'Preferred ({CONF_THRESHOLD})')
        if threshold_used != CONF_THRESHOLD and not used_fallback:
            plt.axhline(y=threshold_used, color='orange', linestyle='--', label=f'Used ({threshold_used})')
        plt.fill_between(df_res['frame'], 0, 1.1, where=df_res['informative'], color='green', alpha=0.1, label='Informative')
        plt.title(f"Informative Frames: {vid_id} ({len(informative_idx_set)} frames, {status})")
        plt.xlabel("Frame Index")
        plt.ylabel("Avg Probability")
        plt.ylim(0, 1.1)
        plt.legend()
        plt.savefig(PLOT_DIR / f"{vid_id}.png")
        plt.close()

        cap.release()


if __name__ == "__main__":
    main()
