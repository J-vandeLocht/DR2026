"""Extract every --stride-th frame of all cleaned clips in --video_dir (Paxos2025 or Paxos2020, incl. unlabeled ones) for self-supervised pretraining
(dino/ssl_lora.py): black borders cropped (crop_black_borders, as --crop_borders in training), resized to --size, JPEG.
Output: <out_dir>/<clip stem>/<frame idx>.jpg; existing frames are skipped. CPU only."""
import argparse
import glob
import os
import multiprocessing as mp

import cv2

from data.dataset import crop_black_borders


def extract(job):
    path, out_dir, stride, size = job
    stem = os.path.basename(path)[:-len(".mp4")].removeprefix("CLEAN_")
    vid_dir = os.path.join(out_dir, stem)
    os.makedirs(vid_dir, exist_ok=True)
    cap = cv2.VideoCapture(path)
    idx = written = 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % stride == 0:
            out = os.path.join(vid_dir, f"{idx:05d}.jpg")
            if not os.path.exists(out):
                _, frame = cap.retrieve()
                rgb = crop_black_borders(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
                cv2.imwrite(out, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            written += 1
        idx += 1
    cap.release()
    return stem, idx, written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_dir", default="data/ensemble_results_paxos2025/cleaned_videos")
    parser.add_argument("--out_dir", default="data/ensemble_results_paxos2025/ssl_frames")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()

    cv2.setNumThreads(1)
    videos = sorted(glob.glob(os.path.join(args.video_dir, "CLEAN_*.mp4")))
    jobs = [(v, args.out_dir, args.stride, args.size) for v in videos]
    total = 0
    # spawn: fresh workers, no inherited OpenCV/OpenMP thread pools (fork + thread-count changes can deadlock)
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (stem, n, w) in enumerate(pool.imap_unordered(extract, jobs), 1):
            total += w
            print(f"[{i}/{len(jobs)}] {stem}: {w}/{n} frames")
    print(f"{total} frames from {len(videos)} clips -> {args.out_dir}")
