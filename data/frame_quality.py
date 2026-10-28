"""Per-frame quality metrics of the cleaned clips (frame sampler removed 2026-10-26, see TODO_experiments.md).

sharp: log(median |band-pass|) of the green channel (DoG, sigma 1.5 / 4: vessel scale) inside the eroded fundus mask.
The median keeps single strong edges (shadow borders, glare, crop rim) and sensor-noise speckles of dark frames from
dominating, which a plain Laplacian variance did.
light: mean Mertens well-exposedness exp(-(L/255 - 0.5)^2 / (2 * 0.2^2)) over the masked luminance
(penalizes dark and blown-out pixels alike). Both are also stored as within-video rank percentiles in [0, 1].
Writes <out_dir>/<clip stem>.npz.
"""
import argparse
import multiprocessing as mp
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import rankdata
from tqdm import tqdm

SIZE = 512
MASK_THRESHOLD = 15  # gray level separating the fundus from the black crop border
ERODE_PX = 10
EXPOSURE_SIGMA = 0.2
DOG_SIGMAS = (1.5, 4.0)


def frame_metrics(frame, kernel):
    frame = cv2.resize(frame, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.erode((gray > MASK_THRESHOLD).astype(np.uint8), kernel) > 0
    if mask.sum() < 100:  # (almost) black frame
        return float(np.log(1e-3)), 0.0, float(gray.mean()), 1.0, 0.0
    green = frame[:, :, 1].astype(np.float32)
    dog = cv2.GaussianBlur(green, (0, 0), DOG_SIGMAS[0]) - cv2.GaussianBlur(green, (0, 0), DOG_SIGMAS[1])
    sharp = np.log(np.median(np.abs(dog[mask])) + 1e-3)
    lum = gray[mask].astype(np.float32)
    light = np.exp(-((lum / 255 - 0.5) ** 2) / (2 * EXPOSURE_SIGMA ** 2)).mean()
    return float(sharp), float(light), float(lum.mean()), float((lum < 25).mean()), float((lum > 235).mean())


def rank_percentile(x):
    return ((rankdata(x) - 1) / max(len(x) - 1, 1)).astype(np.float32)


def process_clip(args):
    video_path, out_path = args
    cv2.setNumThreads(1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE_PX + 1, 2 * ERODE_PX + 1))
    cap = cv2.VideoCapture(str(video_path))
    rows = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rows.append(frame_metrics(frame, kernel))
    cap.release()
    if not rows:
        return video_path.name, 0
    sharp, light, mean_l, frac_dark, frac_glare = (np.array(c, dtype=np.float32) for c in zip(*rows))
    np.savez(out_path, sharp=sharp, light=light, mean_L=mean_l, frac_dark=frac_dark, frac_glare=frac_glare,
             sharp_rank=rank_percentile(sharp), light_rank=rank_percentile(light), num_frames=len(rows))
    return video_path.name, len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, default="data/ensemble_results_paxos2025/cleaned_videos")
    parser.add_argument('--out_dir', type=str, default="data/ensemble_results_paxos2025/frame_quality")
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--limit', type=int, default=None, help="Only the first N clips (testing).")
    parser.add_argument('--force', action='store_true', help="Recompute clips that already have a file.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = sorted(Path(args.video_dir).glob("*.mp4"))[:args.limit]
    jobs = [(v, out_dir / f"{v.stem}.npz") for v in videos]
    jobs = [j for j in jobs if args.force or not j[1].exists()]
    print(f"{len(videos)} clips, {len(jobs)} to process")

    start = time.time()
    # spawn: fresh workers, no inherited OpenCV/OpenMP thread pools (fork after their use can deadlock)
    with mp.get_context("spawn").Pool(args.num_workers) as pool:
        for name, n in tqdm(pool.imap_unordered(process_clip, jobs), total=len(jobs)):
            if n == 0:
                print(f"No frames decoded: {name}")
    print(f"Done in {time.time() - start:.0f}s -> {out_dir}")


if __name__ == "__main__":
    main()
