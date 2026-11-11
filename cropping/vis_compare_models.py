"""Side-by-side visual comparison of the cropping models trained on Paxos2020 vs Paxos2025.

For each eval dataset (2020 / 2025) this picks --n random frames from the val splits
(the fold is chosen randomly per frame), runs both single-dataset models
(seg_2020_fold_<f>.pth and seg_2025_fold_<f>.pth of that fold) on the frame, and
writes one PNG per dataset with one row per frame:
    frame | ground-truth mask | 2020-model prediction (IoU) | 2025-model prediction (IoU)

Outputs: <out_dir>/vis_compare_2020.png, <out_dir>/vis_compare_2025.png

Run from the repo root:
    python -u -m cropping.vis_compare_models
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import segmentation_models_pytorch as smp
import albumentations as albu
from albumentations.pytorch import ToTensorV2


def get_val_transform(img_size):
    return albu.Compose([
        albu.Resize(img_size, img_size),
        albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def build_gt_mask(row, h, w):
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (int(row["center_x"]), int(row["center_y"])),
                (int(row["radius_x"]), int(row["radius_y"])), 0, 0, 360, 1, -1)
    return mask


def iou(pred, gt, eps=1e-7):
    inter = (pred & gt).sum()
    return inter / (pred.sum() + gt.sum() - inter + eps)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models_dir", type=str, default="cropping/models_xdomain")
    parser.add_argument("--ann_dir", type=str, default="cropping")
    parser.add_argument("--img_dir", type=str, default="cropping/frames_to_annotate_larger")
    parser.add_argument("--out_dir", type=str, default="cropping/eval_xdomain")
    parser.add_argument("--img_size", type=int, default=768)
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b4")
    parser.add_argument("--n", type=int, default=15, help="frames per eval dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--configs", nargs="+", default=["2020", "2025"], help="the two models to compare")
    parser.add_argument("--eval_datasets", nargs="+", default=["2020", "2025"], choices=["2020", "2025"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = get_val_transform(args.img_size)
    rng = random.Random(args.seed)

    for eval_ds in args.eval_datasets:
        ann_csv = Path(args.ann_dir) / f"annotations_cropping_{eval_ds}.csv"
        splits_path = Path(args.ann_dir) / f"splits_{eval_ds}.json"
        df = pd.read_csv(ann_csv).set_index("filename")
        with open(splits_path) as f:
            splits = json.load(f)

        candidates = [(fold, fn) for fold in range(args.n_splits) for fn in splits[f"fold_{fold}"]["val"]]
        picks = rng.sample(candidates, min(args.n, len(candidates)))

        models = {}
        for fold, _ in picks:
            for cfg in args.configs:
                if (cfg, fold) in models:
                    continue
                path = Path(args.models_dir) / f"seg_{cfg}_fold_{fold}.pth"
                if not path.exists():
                    print(f"WARNING: missing {path} -- those frames will be skipped")
                    continue
                m = smp.UnetPlusPlus(encoder_name=args.encoder_name, encoder_weights=None,
                                     in_channels=3, classes=1).to(device).eval()
                m.load_state_dict(torch.load(path, map_location=device))
                models[(cfg, fold)] = m

        rows = []
        for fold, fn in picks:
            if any((cfg, fold) not in models for cfg in args.configs):
                continue
            row = df.loc[fn]
            img_bgr = cv2.imread(str(Path(args.img_dir) / fn))
            if img_bgr is None:
                print(f"WARNING: cannot read {args.img_dir}/{fn} -- skipped")
                continue
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            gt = build_gt_mask(row, h, w)
            inp = transform(image=img)["image"].unsqueeze(0).to(device)
            gt_r = cv2.resize(gt, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST) > 0
            preds = {}
            with torch.no_grad():
                for cfg in args.configs:
                    out = models[(cfg, fold)](inp)
                    preds[cfg] = (torch.sigmoid(out)[0, 0] > 0.5).cpu().numpy()
            rows.append((fn, fold, img, gt_r, preds))

        if not rows:
            print(f"eval_ds={eval_ds}: nothing to plot (missing models/images)")
            continue

        n = len(rows)
        fig, axes = plt.subplots(n, 2 + len(args.configs), figsize=(4.2 * (2 + len(args.configs)), 2.6 * n))
        if n == 1:
            axes = axes[None, :]
        col_titles = ["frame", "ground truth"] + [f"model {c} (IoU)" for c in args.configs]
        for j, t in enumerate(col_titles):
            axes[0, j].set_title(t, fontsize=11)
        for i, (fn, fold, img, gt_r, preds) in enumerate(rows):
            img_r = cv2.resize(img, (args.img_size, args.img_size))
            axes[i, 0].imshow(img_r)
            axes[i, 0].set_ylabel(f"{fn[:38]}\nfold {fold}", fontsize=7)
            axes[i, 1].imshow(img_r)
            axes[i, 1].imshow(np.ma.masked_where(~gt_r, gt_r), cmap="Greens", alpha=0.45)
            for j, cfg in enumerate(args.configs):
                p = preds[cfg]
                axes[i, 2 + j].imshow(img_r)
                axes[i, 2 + j].imshow(np.ma.masked_where(~p, p), cmap="Reds", alpha=0.45)
                axes[i, 2 + j].set_title(f"IoU {iou(p, gt_r):.3f}", fontsize=9)
            for ax in axes[i]:
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f"Cropping models {args.configs[0]} vs {args.configs[1]} on Paxos{eval_ds} val frames "
                     f"(random fold per frame; red = predicted mask, green = ground truth)")
        fig.tight_layout(rect=[0, 0, 1, 0.99])
        out_path = out_dir / f"vis_compare_{eval_ds}.png"
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"Wrote {out_path} ({n} frames)")


if __name__ == "__main__":
    main()
