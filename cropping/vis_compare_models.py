"""Side-by-side pixel-wise comparison of the cropping masks: Paxos2020 vs Paxos2025 model.

For each eval dataset (2020 / 2025) this picks --n random frames from the val splits
(the fold is chosen randomly per frame), runs both single-dataset models
(seg_2020_fold_<f>.pth and seg_2025_fold_<f>.pth of that fold) on the frame, and
writes one PNG per dataset with one row per frame and three panels of EQUAL size:
    ground-truth mask | 2020-model mask (IoU) | 2025-model mask (IoU)

Each panel shows only the lens: the RGB frame under a transparent colored mask
(green = ground truth, blue/red = the two models). All three panels of a row are
cropped to one common square window at native resolution (side = largest dimension
of the three masks, centered on their joint bounding box) -- no resizing, no
skewing -- so the masks can be compared pixel by pixel.

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
import textwrap
from matplotlib.colors import ListedColormap
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
    return mask > 0


def bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def common_square_window(masks, h, w):
    """One square window shared by all masks: side = largest bbox dimension of the
    three masks, centered on their joint bounding box, clamped to the image."""
    boxes = [b for b in (bbox(m) for m in masks) if b is not None]
    if not boxes:
        side = min(h, w)
        cy, cx = h // 2, w // 2
    else:
        side = max(max(b[1] - b[0], b[3] - b[2]) for b in boxes)
        side = min(side, h, w)
        cy = (min(b[0] for b in boxes) + max(b[1] for b in boxes)) // 2
        cx = (min(b[2] for b in boxes) + max(b[3] for b in boxes)) // 2
    r0 = min(max(cy - side // 2, 0), h - side)
    c0 = min(max(cx - side // 2, 0), w - side)
    return r0, c0, side


def iou(pred, gt, eps=1e-7):
    inter = (pred & gt).sum()
    return inter / (pred.sum() + gt.sum() - inter + eps)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models_dir", type=str, default="cropping/models_xdomain")
    parser.add_argument("--ann_dir", type=str, default="cropping")
    parser.add_argument("--img_dir", type=str, default="cropping/frames_to_annotate_larger")
    parser.add_argument("--out_dir", type=str, default="cropping/eval_xdomain")
    parser.add_argument("--img_size", type=int, default=768, help="model input size (as trained)")
    parser.add_argument("--encoder_name", type=str, default="efficientnet-b4")
    parser.add_argument("--n", type=int, default=15, help="frames per eval dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--configs", nargs="+", default=["2020", "2025"], help="the two models to compare")
    parser.add_argument("--eval_datasets", nargs="+", default=["2020", "2025"], choices=["2020", "2025"])
    args = parser.parse_args()

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 9})
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
            preds = {}
            with torch.no_grad():
                for cfg in args.configs:
                    out = models[(cfg, fold)](inp)
                    p = (torch.sigmoid(out)[0, 0] > 0.5).cpu().numpy()
                    # back to native resolution (nearest) so masks are pixel-comparable
                    preds[cfg] = cv2.resize(p.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            r0, c0, side = common_square_window([gt] + list(preds.values()), h, w)
            crops = {"rgb": img[r0:r0 + side, c0:c0 + side], "gt": gt[r0:r0 + side, c0:c0 + side]}
            for cfg in args.configs:
                crops[cfg] = preds[cfg][r0:r0 + side, c0:c0 + side]
            rows.append((fn, fold, side, crops))

        if not rows:
            print(f"eval_ds={eval_ds}: nothing to plot (missing models/images)")
            continue

        n = len(rows)
        ncols = 1 + len(args.configs)
        fig, axes = plt.subplots(n, ncols, figsize=(5.2 * ncols, 4.8 * n))
        if n == 1:
            axes = axes[None, :]
        mask_cmaps = {"gt": ListedColormap(["none", "lime"])}
        pred_colors = ["dodgerblue", "red", "darkorange", "purple"]
        for j, cfg in enumerate(args.configs):
            mask_cmaps[cfg] = ListedColormap(["none", pred_colors[j % len(pred_colors)]])
        for j, t in enumerate(["ground truth"] + [f"model {cfg}" for cfg in args.configs]):
            axes[0, j].set_title(t, fontsize=12, pad=8)
        for i, (fn, fold, side, crops) in enumerate(rows):
            axes[i, 0].set_ylabel("\n".join(textwrap.wrap(fn, 32)) + f"\nfold {fold} | {side}px", fontsize=8)
            for j, key in enumerate(["gt"] + args.configs):
                ax = axes[i, j]
                ax.imshow(crops["rgb"], interpolation="nearest")
                m = crops[key]
                ax.imshow(np.ma.masked_where(~m, m), cmap=mask_cmaps[key], alpha=0.5,
                          interpolation="nearest")
                if key != "gt":
                    ax.set_title(f"IoU {iou(crops[key], crops['gt']):.3f}", fontsize=11)
                ax.set_xticks([])
                ax.set_yticks([])
        fig.suptitle(f"Cropping: model {args.configs[0]} vs model {args.configs[1]} on Paxos{eval_ds} val frames "
                     f"(random fold per frame; RGB frame under mask; one common square window per row, no resize)",
                     fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out_path = out_dir / f"vis_compare_{eval_ds}.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"Wrote {out_path} ({n} frames)")


if __name__ == "__main__":
    main()
